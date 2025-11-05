from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
from datetime import datetime
import logging
import asyncio
import sys
import os

# Configuration pour Render
app = FastAPI(
    title="Forecasting Pro API",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware pour production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://forecasting-tools.vercel.app",
        "https://forecasting-tools.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import des modules locaux
try:
    from forecasting import ForecastEngine
    from data_processing import DataProcessor
    from helpers import DataHelpers
    BACKEND_READY = True
    logger.info("✅ Modules backend chargés avec succès")
except ImportError as e:
    logger.error(f"❌ Erreur import modules: {e}")
    BACKEND_READY = False

# Initialisation des services
if BACKEND_READY:
    try:
        data_processor = DataProcessor()
        forecast_engine = ForecastEngine()
        data_helpers = DataHelpers()
        logger.info("✅ Services backend initialisés")
    except Exception as e:
        logger.error(f"❌ Erreur initialisation: {e}")
        BACKEND_READY = False

@app.get("/")
async def root():
    return {
        "message": "Macro Forecasting Pro API", 
        "status": "running",
        "version": "2.0.0",
        "backend_ready": BACKEND_READY,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "backend_ready": BACKEND_READY,
        "services": {
            "data_processing": "active",
            "forecasting": "active", 
            "visualization": "active"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/upload-data")
async def upload_data(file: UploadFile = File(...), orientation: str = Form("dates_in_rows")):
    try:
        logger.info(f"📤 Traitement fichier: {file.filename}")
        
        file_extension = file.filename.lower().split('.')[-1]
        allowed_extensions = ["csv", "xls", "xlsx"]
        
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Type de fichier non supporté. Utilisez CSV, XLS ou XLSX.")
        
        # Lecture du fichier
        if file_extension == "csv":
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file)
        
        logger.info(f"✅ Fichier lu: {df.shape}")
        
        # Traitement des données
        processed_df = data_processor.process_dataframe(df, orientation)
        validation_results = data_helpers.validate_dataframe(processed_df)
        
        if not validation_results["is_valid"]:
            raise HTTPException(status_code=400, detail=f"Données invalides: {', '.join(validation_results['errors'])}")
        
        # Conversion pour JSON
        processed_data = processed_df.replace({np.nan: None, np.inf: None, -np.inf: None})
        if 'Date' in processed_data.columns:
            processed_data['Date'] = processed_data['Date'].astype(str)
        
        # Analyse des séries temporelles
        analysis_results = {}
        numeric_columns = [col for col in processed_data.columns if col != 'Date' and processed_data[col].dtype in ['float64', 'int64']]
        
        for column in numeric_columns:
            series = processed_data[column].dropna()
            if len(series) > 0:
                analysis = forecast_engine.analyze_time_series(series)
                analysis_results[column] = analysis
        
        response_data = {
            "success": True,
            "message": "Fichier traité avec succès",
            "data": processed_data.to_dict('records'),
            "columns": list(processed_data.columns),
            "numeric_columns": numeric_columns,
            "summary": validation_results["summary"],
            "analysis": analysis_results,
            "validation": validation_results,
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(response_data)
        
    except Exception as e:
        logger.error(f"❌ Erreur traitement: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erreur lors du traitement: {str(e)}")

@app.post("/api/analyze-timeseries")
async def analyze_timeseries(request: Dict[str, Any]):
    try:
        data = request.get("data", [])
        target_variable = request.get("target_variable")
        
        if not data:
            raise HTTPException(status_code=400, detail="Aucune donnée fournie")
        
        df = pd.DataFrame(data)
        df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
        
        analysis_results = {}
        target_columns = [target_variable] if target_variable else [
            col for col in df.columns if col != 'Date' and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        for column in target_columns:
            if column in df.columns:
                series = df[column].dropna()
                if len(series) > 0:
                    analysis = forecast_engine.analyze_time_series(series)
                    analysis_results[column] = analysis
        
        return {
            "success": True,
            "analysis": analysis_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur analyse: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/generate-forecast")
async def generate_forecast(request: Dict[str, Any]):
    try:
        data = request.get("data", [])
        model_type = request.get("model_type", "NAIVE")
        periods = request.get("periods", 12)
        params = request.get("params", {})
        target_variables = request.get("target_variables", [])
        
        if not data:
            raise HTTPException(status_code=400, detail="Aucune donnée fournie")
        
        if not target_variables:
            raise HTTPException(status_code=400, detail="Aucune variable cible spécifiée")
        
        # Préparation des données
        df = pd.DataFrame(data)
        df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
        
        # Conversion des types
        for col in df.columns:
            if col != 'Date' and df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        logger.info(f"🚀 Génération prévision: {model_type}, {periods} périodes, variables: {target_variables}")
        
        # Génération des prévisions
        results = forecast_engine.forecast(
            df=df,
            model_type=model_type,
            periods=periods,
            params=params,
            target_variables=target_variables
        )
        
        return {
            "success": True,
            "message": "Prévision générée avec succès",
            "forecasts": results["forecasts"],
            "metrics": results["metrics"],
            "plots": results["plots"],
            "model_info": results.get("model_info", {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur prévision: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prévision: {str(e)}")

@app.get("/api/models")
async def get_available_models():
    try:
        models = forecast_engine.get_available_models()
        
        categories = {}
        for model_name, model_info in models.items():
            category = model_info["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append({
                "name": model_name,
                "description": model_info["description"]
            })
        
        return {
            "success": True,
            "models": models,
            "categories": categories,
            "total_models": len(models),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Erreur modèles: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(app, host="0.0.0.0", port=port)
