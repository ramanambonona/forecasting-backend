import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json

class DataHelpers:
    @staticmethod
    def format_number(value: float) -> str:
        """Format numbers in French style with thousands separators"""
        if pd.isna(value) or value is None:
            return "N/A"
        
        abs_value = abs(value)
        
        # Correction pour utiliser le format français (espace comme séparateur de milliers, virgule comme décimal)
        if abs_value >= 1_000_000_000:
            return f"{value / 1_000_000_000:,.1f} Md MGA".replace(',', 'TEMP').replace('.', ',').replace('TEMP', ' ')
        elif abs_value >= 1_000_000:
            return f"{value / 1_000_000:,.1f} M MGA".replace(',', 'TEMP').replace('.', ',').replace('TEMP', ' ')
        elif abs_value >= 1_000:
            return f"{value / 1_000:,.1f} k MGA".replace(',', 'TEMP').replace('.', ',').replace('TEMP', ' ')
        else:
            return f"{value:,.1f} MGA".replace(',', 'TEMP').replace('.', ',').replace('TEMP', ' ')
    
    @staticmethod
    def get_mape_status(mape_value: float) -> Dict[str, Any]:
        """Determine model quality based on MAPE"""
        if mape_value < 0.10:
            color = "#28a745"
            label = "🟢 Excellent"
            quality = "excellent"
        elif mape_value <= 0.20:
            color = "#ffc107"
            label = "🟠 Bon"
            quality = "good"
        else:
            color = "#dc3545"
            label = "🔴 Mauvais"
            quality = "poor"
        
        return {
            "color": color,
            "label": label,
            "quality": quality,
            "value": mape_value,
            "formatted_value": f"{mape_value:.2%}"
        }
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate DataFrame structure and content"""
        errors = []
        warnings = []
        
        if 'Date' not in df.columns:
            errors.append("La colonne 'Date' est manquante")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            errors.append("Aucune colonne numérique trouvée")
        
        if 'Date' in df.columns:
            date_series = pd.to_datetime(df['Date'], errors='coerce')
            missing_dates = date_series.isna().sum()
            if missing_dates > 0:
                warnings.append(f"{missing_dates} date(s) invalide(s) trouvée(s)")
        
        for col in numeric_columns:
            if df[col].nunique() == 1:
                warnings.append(f"La colonne '{col}' a une valeur constante")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns": len(numeric_columns),
                "date_range": {
                    "start": str(df['Date'].min()) if 'Date' in df.columns and not df['Date'].empty else None,
                    "end": str(df['Date'].max()) if 'Date' in df.columns and not df['Date'].empty else None
                } if 'Date' in df.columns else None
            }
        }
    
    @staticmethod
    def generate_future_dates(last_date: str, periods: int, freq: str = 'M') -> List[str]:
        """Generate future dates for forecasting"""
        last_date = pd.to_datetime(last_date)
        
        if freq == 'M':
            future_dates = pd.date_range(
                start=last_date + pd.offsets.MonthBegin(1),
                periods=periods,
                freq='M'
            )
        elif freq == 'Q':
            future_dates = pd.date_range(
                start=last_date + pd.offsets.QuarterBegin(1),
                periods=periods,
                freq='Q'
            )
        else:
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=periods,
                freq='D'
            )
        
        return future_dates.strftime('%Y-%m-%d').tolist()
    
    @staticmethod
    def calculate_growth_metrics(series: pd.Series) -> Dict[str, float]:
        """Calculate various growth metrics for a time series"""
        if len(series) < 2:
            return {}
        
        values = series.dropna().values
        
        latest_value = values[-1]
        previous_value = values[-2] if len(values) > 1 else values[0]
        # Dans un contexte de séries temporelles, YTD (Year-to-Date) est souvent comparé au début de la période
        ytd_value = values[0] if len(values) > 0 else 0 
        
        mom_growth = ((latest_value - previous_value) / previous_value) if previous_value != 0 else 0
        ytd_growth = ((latest_value - ytd_value) / ytd_value) if ytd_value != 0 else 0
        
        returns = np.diff(values) / values[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0] if len(values) > 1 else 0
        
        return {
            "latest_value": float(latest_value),
            "mom_growth": float(mom_growth),
            "ytd_growth": float(ytd_growth),
            "volatility": float(volatility),
            "trend_slope": float(slope),
            "is_increasing": slope > 0
        }

class ModelHelpers:
    @staticmethod
    def get_model_categories() -> Dict[str, List[str]]:
        """Get available models organized by category"""
        return {
            "Statistical": ["NAIVE", "AR(p)", "ARIMA", "SARIMA", "Exponential Smoothing"],
            "Multivariate": ["VAR", "BVAR", "ARDL", "Factor Models"],
            "Machine Learning": ["Régression Linéaire", "Random Forest", "XGBoost", "LightGBM"],
            "Neural Networks": ["MLP", "LSTM"],
            "Bayesian": ["BART", "BVAR"],
            "Specialized": ["GARCH", "MIDAS", "TSLM", "Prophet"]
        }
    
    @staticmethod
    def get_model_parameters(model_type: str) -> Dict[str, Any]:
        """Get default parameters for each model type"""
        parameters = {
            # Modèles statistiques simples
            "NAIVE": {
                "description": "Moyenne mobile simple",
                "parameters": {}
            },
            "AR(p)": {
                "description": "Modèle autorégressif d'ordre p",
                "parameters": {
                    "p": {"type": "integer", "default": 1, "min": 1, "max": 12, "description": "Ordre autorégressif"}
                }
            },
            "MA(q)": {
                "description": "Moyenne mobile d'ordre q",
                "parameters": {
                    "q": {"type": "integer", "default": 1, "min": 1, "max": 12, "description": "Ordre moyenne mobile"}
                }
            },
            "ARIMA": {
                "description": "ARIMA (Autoregressive Integrated Moving Average)",
                "parameters": {
                    "p": {"type": "integer", "default": 1, "min": 0, "max": 5, "description": "Ordre AR"},
                    "d": {"type": "integer", "default": 1, "min": 0, "max": 2, "description": "Ordre de différenciation"},
                    "q": {"type": "integer", "default": 0, "min": 0, "max": 5, "description": "Ordre MA"}
                }
            },
            "SARIMA": {
                "description": "ARIMA saisonnier",
                "parameters": {
                    "order": {"type": "tuple", "default": (1,1,1), "description": "Ordre (p,d,q)"},
                    "seasonal_order": {"type": "tuple", "default": (1,1,1,12), "description": "Ordre saisonnier (P,D,Q,s)"},
                    "seasonal_periods": {"type": "integer", "default": 12, "min": 2, "max": 24, "description": "Période saisonnière"}
                }
            },
            "Exponential Smoothing": {
                "description": "Lissage exponentiel (Holt-Winters)",
                "parameters": {
                    "trend": {"type": "string", "default": "additive", "options": ["additive", "multiplicative", None], "description": "Type de tendance"},
                    "seasonal": {"type": "string", "default": "additive", "options": ["additive", "multiplicative", None], "description": "Type de saisonnalité"},
                    "seasonal_periods": {"type": "integer", "default": 12, "min": 2, "max": 24, "description": "Période saisonnière"}
                }
            },

            # Modèles multivariés
            "VAR": {
                "description": "Vector Autoregression",
                "parameters": {
                    "maxlags": {"type": "integer", "default": 1, "min": 1, "max": 12, "description": "Ordre maximum"},
                    "ic": {"type": "string", "default": "aic", "options": ["aic", "bic", "hqic", "fpe"], "description": "Critère d'information"}
                }
            },
            "BVAR": {
                "description": "Bayesian Vector Autoregression",
                "parameters": {
                    "lags": {"type": "integer", "default": 1, "min": 1, "max": 12, "description": "Ordre"},
                    "prior": {"type": "string", "default": "minnesota", "options": ["minnesota", "conjugate"], "description": "Type de prior"}
                }
            },
            "ARDL": {
                "description": "Autoregressive Distributed Lag",
                "parameters": {
                    "lags": {"type": "integer", "default": 1, "min": 1, "max": 12, "description": "Ordre AR"},
                    "max_lead_lags": {"type": "integer", "default": 1, "min": 0, "max": 12, "description": "Ordre des variables exogènes"}
                }
            },
            "Factor Models": {
                "description": "Modèles à facteurs",
                "parameters": {
                    "n_factors": {"type": "integer", "default": 3, "min": 1, "max": 10, "description": "Nombre de facteurs"},
                    "factor_order": {"type": "integer", "default": 1, "min": 1, "max": 5, "description": "Ordre des facteurs"}
                }
            },

            # Machine Learning
            "Régression Linéaire": {
                "description": "Régression linéaire avec caractéristiques temporelles",
                "parameters": {
                    "fit_intercept": {"type": "boolean", "default": True, "description": "Inclure l'intercept"},
                    "normalize": {"type": "boolean", "default": False, "description": "Normaliser les données"}
                }
            },
            "Random Forest": {
                "description": "Random Forest Regressor",
                "parameters": {
                    "n_estimators": {"type": "integer", "default": 100, "min": 10, "max": 200, "description": "Nombre d'arbres"},
                    "max_depth": {"type": "integer", "default": 10, "min": 3, "max": 20, "description": "Profondeur maximale"},
                    "min_samples_split": {"type": "integer", "default": 2, "min": 2, "max": 20, "description": "Échantillons minimum par split"}
                }
            },
            "XGBoost": {
                "description": "Gradient Boosting extrême",
                "parameters": {
                    "n_estimators": {"type": "integer", "default": 100, "min": 10, "max": 200, "description": "Nombre d'arbres"},
                    "max_depth": {"type": "integer", "default": 6, "min": 3, "max": 15, "description": "Profondeur maximale"},
                    "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 0.3, "step": 0.01, "description": "Taux d'apprentissage"}
                }
            },
            "LightGBM": {
                "description": "Light Gradient Boosting Machine",
                "parameters": {
                    "n_estimators": {"type": "integer", "default": 100, "min": 10, "max": 200, "description": "Nombre d'arbres"},
                    "max_depth": {"type": "integer", "default": -1, "min": -1, "max": 20, "description": "Profondeur maximale (-1 = illimité)"},
                    "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 0.3, "step": 0.01, "description": "Taux d'apprentissage"}
                }
            },

            # Réseaux de neurones
            "MLP": {
                "description": "Multi-Layer Perceptron",
                "parameters": {
                    "hidden_layer_sizes": {"type": "tuple", "default": (100,), "description": "Tailles des couches cachées"},
                    "activation": {"type": "string", "default": "relu", "options": ["relu", "tanh", "logistic"], "description": "Fonction d'activation"},
                    "learning_rate": {"type": "string", "default": "constant", "options": ["constant", "invscaling", "adaptive"], "description": "Taux d'apprentissage"}
                }
            },
            "LSTM": {
                "description": "Long Short-Term Memory",
                "parameters": {
                    "units": {"type": "integer", "default": 50, "min": 10, "max": 200, "description": "Unités LSTM"},
                    "lookback": {"type": "integer", "default": 12, "min": 1, "max": 24, "description": "Périodes historiques"},
                    "dropout": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.5, "step": 0.1, "description": "Taux de dropout"}
                }
            },

            # Modèles bayésiens
            "BART": {
                "description": "Bayesian Additive Regression Trees",
                "parameters": {
                    "n_trees": {"type": "integer", "default": 50, "min": 10, "max": 200, "description": "Nombre d'arbres"},
                    "alpha": {"type": "float", "default": 0.95, "min": 0.5, "max": 0.99, "step": 0.01, "description": "Paramètre alpha"},
                    "beta": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0, "step": 0.1, "description": "Paramètre beta"}
                }
            },

            # Modèles spécialisés
            "GARCH": {
                "description": "Modèle GARCH pour la volatilité",
                "parameters": {
                    "p": {"type": "integer", "default": 1, "min": 1, "max": 3, "description": "Ordre GARCH"},
                    "q": {"type": "integer", "default": 1, "min": 1, "max": 3, "description": "Ordre ARCH"},
                    "vol": {"type": "string", "default": "GARCH", "options": ["GARCH", "EGARCH", "GJR-GARCH"], "description": "Type de volatilité"}
                }
            },
            "MIDAS": {
                "description": "Mixed Data Sampling",
                "parameters": {
                    "high_freq_lags": {"type": "integer", "default": 3, "min": 1, "max": 12, "description": "Décalages haute fréquence"},
                    "low_freq_lags": {"type": "integer", "default": 1, "min": 1, "max": 4, "description": "Décalages basse fréquence"}
                }
            },
            "TSLM": {
                "description": "Time Series Linear Model",
                "parameters": {
                    "trend": {"type": "boolean", "default": True, "description": "Inclure la tendance"},
                    "seasonality": {"type": "boolean", "default": True, "description": "Inclure la saisonnalité"},
                    "fourier_terms": {"type": "integer", "default": 0, "min": 0, "max": 10, "description": "Termes de Fourier"}
                }
            },
            "Prophet": {
                "description": "Facebook Prophet",
                "parameters": {
                    "changepoint_prior_scale": {"type": "float", "default": 0.05, "min": 0.001, "max": 0.5, "step": 0.01, "description": "Échelle des points de changement"},
                    "seasonality_prior_scale": {"type": "float", "default": 10.0, "min": 0.01, "max": 100.0, "step": 0.1, "description": "Échelle de saisonnalité"},
                    "holidays_prior_scale": {"type": "float", "default": 10.0, "min": 0.01, "max": 100.0, "step": 0.1, "description": "Échelle des jours fériés"}
                }
            }
        }
        
        return parameters.get(model_type, {
            "description": "Modèle de prévision",
            "parameters": {}
        })
    
    @staticmethod
    def recommend_models(analysis_results: Dict[str, Any]) -> List[str]:
        """Recommend models based on time series analysis"""
        recommendations = []
        
        trend = analysis_results.get('tendance', 'Non détectée')
        seasonality = analysis_results.get('saisonnalite', 'Non détectée')
        
        recommendations.extend(["NAIVE", "Régression Linéaire", "Random Forest"])
        
        if trend == 'Détectée':
            recommendations.extend(["ARIMA", "Exponential Smoothing"])
        
        if seasonality in ['Forte', 'Modérée']:
            recommendations.extend(["SARIMA", "Prophet"])
        
        if analysis_results.get('has_multiple_vars', False):
            recommendations.extend(["VAR", "ARDL"])
        
        return list(dict.fromkeys(recommendations))

class ExportHelpers:
    @staticmethod
    def format_for_excel(df: pd.DataFrame, orientation: str = "dates_in_rows") -> pd.DataFrame:
        """Format DataFrame for Excel export"""
        if orientation == "dates_in_rows":
            return df
        else:
            transposed_df = df.set_index('Date').T.reset_index()
            transposed_df.rename(columns={'index': 'Variable'}, inplace=True)
            return transposed_df
    
    @staticmethod
    def create_forecast_report(historical_data: pd.DataFrame, 
                              forecast_data: pd.DataFrame,
                              model_info: Dict[str, Any],
                              metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive forecast report"""
        
        # S'assurer que les colonnes Date existent et ne sont pas vides avant de tenter min/max
        historical_start = historical_data['Date'].min().strftime('%Y-%m-%d') if 'Date' in historical_data.columns and not historical_data.empty else None
        historical_end = historical_data['Date'].max().strftime('%Y-%m-%d') if 'Date' in historical_data.columns and not historical_data.empty else None
        forecast_start = forecast_data['Date'].min().strftime('%Y-%m-%d') if 'Date' in forecast_data.columns and not forecast_data.empty else None
        forecast_end = forecast_data['Date'].max().strftime('%Y-%m-%d') if 'Date' in forecast_data.columns and not forecast_data.empty else None

        target_var = model_info.get('target_variable', '')

        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "model_used": model_info.get('model_type', 'Unknown'),
                "forecast_periods": model_info.get('periods', 0),
                "target_variable": target_var
            },
            "summary": {
                "historical_period": {
                    "start": historical_start,
                    "end": historical_end,
                    "data_points": len(historical_data)
                },
                "forecast_period": {
                    "start": forecast_start,
                    "end": forecast_end,
                    "data_points": len(forecast_data)
                }
            },
            "model_performance": metrics,
            "key_metrics": DataHelpers.calculate_growth_metrics(
                    pd.concat([
                        historical_data[target_var],
                        forecast_data[target_var]
                    ])
                ) if target_var and target_var in historical_data.columns else {}
        }
        
        return report
