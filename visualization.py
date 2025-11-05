import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
from statsmodels.tsa.seasonal import seasonal_decompose

class VisualizationService:
    def __init__(self):
        self.color_palette = {
            'primary': '#8FBC8F',
            'secondary': '#6A8A6A',
            'accent': '#FF6B6B',
            'background': '#FFF8E7',
            'text': '#333333'
        }
    
    def create_time_series_plot(self, df: pd.DataFrame, selected_vars: List[str]) -> Dict[str, Any]:
        """Create interactive time series plot"""
        fig = go.Figure()
        
        colors = ['#8FBC8F', '#6A8A6A', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, var in enumerate(selected_vars):
            if var in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df[var],
                    mode='lines',
                    name=var,
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'<b>{var}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Valeur: %{y:.2f}<br>' +
                                 '<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(
                text="Évolution des Variables",
                font=dict(family="Palatino Linotype, Book Antiqua, Palatino, serif", size=20)
            ),
            xaxis=dict(
                title="Date",
                gridcolor='lightgray',
                showline=True,
                linewidth=1,
                linecolor='lightgray'
            ),
            yaxis=dict(
                title="Valeur",
                gridcolor='lightgray',
                showline=True,
                linewidth=1,
                linecolor='lightgray'
            ),
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Palatino Linotype, Book Antiqua, Palatino, serif"),
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig.to_dict()

    def create_forecast_plot(self, historical_data: pd.DataFrame, 
                           forecast_data: pd.DataFrame, 
                           variable: str,
                           model_name: str) -> Dict[str, Any]:
        """Create forecast visualization with historical and predicted values"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data[variable],
            mode='lines',
            name='Historique',
            line=dict(color=self.color_palette['primary'], width=2.5),
            hovertemplate='<b>Historique</b><br>Date: %{x}<br>Valeur: %{y:.2f}<extra></extra>'
        ))
        
        if len(historical_data) > 0 and len(forecast_data) > 0:
            last_historical_date = historical_data['Date'].iloc[-1]
            first_forecast_date = forecast_data['Date'].iloc[0]
            last_historical_value = historical_data[variable].iloc[-1]
            first_forecast_value = forecast_data[variable].iloc[0]
            
            fig.add_trace(go.Scatter(
                x=[last_historical_date, first_forecast_date],
                y=[last_historical_value, first_forecast_value],
                mode='lines',
                line=dict(color=self.color_palette['accent'], width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.add_trace(go.Scatter(
            x=forecast_data['Date'],
            y=forecast_data[variable],
            mode='lines',
            name='Prévision',
            line=dict(color=self.color_palette['accent'], width=2.5, dash='dash'),
            hovertemplate='<b>Prévision</b><br>Date: %{x}<br>Valeur: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f"Prévision de {variable} - {model_name}",
                font=dict(family="Palatino Linotype, Book Antiqua, Palatino, serif", size=20)
            ),
            xaxis=dict(
                title="Date",
                gridcolor='lightgray',
                showline=True,
                linewidth=1,
                linecolor='lightgray'
            ),
            yaxis=dict(
                title="Valeur",
                gridcolor='lightgray',
                showline=True,
                linewidth=1,
                linecolor='lightgray'
            ),
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Palatino Linotype, Book Antiqua, Palatino, serif"),
            height=500,
            showlegend=True
        )
        
        if len(historical_data) > 0:
            last_historical_date = historical_data['Date'].iloc[-1]
            fig.add_vline(
                x=last_historical_date,
                line_width=2,
                line_dash="dot",
                line_color="gray"
            )
            
            fig.add_annotation(
                x=last_historical_date,
                y=1,
                xref="x",
                yref="paper",
                text="Début prévision",
                showarrow=False,
                yshift=10,
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1
            )
        
        return fig.to_dict()

    def create_decomposition_plot(self, series: pd.Series, variable: str) -> Dict[str, Any]:
        """Create seasonal decomposition plot"""
        try:
            if len(series) < 24:
                raise ValueError("Insufficient data for decomposition")
            
            decomposition = seasonal_decompose(series, period=12, model='additive', extrapolate_trend='freq')
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=series.index,
                y=decomposition.observed,
                mode='lines',
                name='Série Originale',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=series.index,
                y=decomposition.trend,
                mode='lines',
                name='Tendance',
                line=dict(color='#ff7f0e', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=series.index,
                y=decomposition.seasonal,
                mode='lines',
                name='Saisonnalité',
                line=dict(color='#2ca02c', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=series.index,
                y=decomposition.resid,
                mode='lines',
                name='Résidu',
                line=dict(color='#d62728', width=2)
            ))
            
            fig.update_layout(
                title=dict(
                    text=f"Décomposition de {variable}",
                    font=dict(family="Palatino Linotype, Book Antiqua, Palatino, serif", size=20)
                ),
                xaxis=dict(title="Date"),
                yaxis=dict(title="Valeur"),
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Palatino Linotype, Book Antiqua, Palatino, serif"),
                height=600,
                showlegend=True
            )
            
            return fig.to_dict()
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Erreur lors de la décomposition: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            fig.update_layout(
                title=dict(
                    text=f"Décomposition de {variable}",
                    font=dict(family="Palatino Linotype, Book Antiqua, Palatino, serif", size=20)
                ),
                height=400
            )
            return fig.to_dict()

    def create_comparison_plot(self, forecasts: Dict[str, Any], variable: str) -> Dict[str, Any]:
        """Create comparison plot for multiple models"""
        fig = go.Figure()
        
        colors = ['#8FBC8F', '#6A8A6A', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, (model_name, forecast_data) in enumerate(forecasts.items()):
            if i < len(colors):
                fig.add_trace(go.Scatter(
                    x=forecast_data['dates'],
                    y=forecast_data['values'],
                    mode='lines',
                    name=model_name,
                    line=dict(color=colors[i], width=2),
                    hovertemplate=f'<b>{model_name}</b><br>Date: %{x}<br>Valeur: %{y:.2f}<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(
                text=f"Comparaison des Prévisions - {variable}",
                font=dict(family="Palatino Linotype, Book Antiqua, Palatino, serif", size=20)
            ),
            xaxis=dict(title="Date"),
            yaxis=dict(title="Valeur"),
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Palatino Linotype, Book Antiqua, Palatino, serif"),
            height=500,
            showlegend=True
        )
        
        return fig.to_dict()

    def create_metrics_gauge(self, mape: float) -> Dict[str, Any]:
        """Create gauge chart for MAPE metric"""
        if mape < 0.10:
            color = "green"
            level = "Excellent"
        elif mape < 0.20:
            color = "orange"
            level = "Bon"
        else:
            color = "red"
            level = "Mauvais"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = mape * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"MAPE - {level}", 'font': {'size': 20}},
            delta = {'reference': 20, 'increasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [None, 50], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 10], 'color': 'lightgreen'},
                    {'range': [10, 20], 'color': 'yellow'},
                    {'range': [20, 50], 'color': 'lightcoral'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 20}}
        ))
        
        fig.update_layout(
            font = {'family': "Palatino Linotype, Book Antiqua, Palatino, serif"},
            height=300
        )
        
        return fig.to_dict()

    def create_correlation_heatmap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create correlation heatmap for numeric variables"""
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='Viridis',
            hoverongaps=False,
            text=correlation_matrix.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=dict(
                text="Matrice de Corrélation",
                font=dict(family="Palatino Linotype, Book Antiqua, Palatino, serif", size=20)
            ),
            xaxis=dict(tickangle=-45),
            yaxis=dict(tickangle=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Palatino Linotype, Book Antiqua, Palatino, serif"),
            height=500
        )
        
        return fig.to_dict()