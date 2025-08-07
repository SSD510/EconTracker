"""
Advanced visualization components for the Economic Dashboard
This module provides sophisticated charting and analysis tools
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import streamlit as st
from datetime import datetime, timedelta

class AdvancedVisualizations:
    """Advanced visualization components for economic data"""
    
    def __init__(self):
        self.color_palette = {
            'positive': '#00C851',
            'negative': '#ff4444',
            'neutral': '#33b5e5',
            'primary': '#007bff',
            'secondary': '#6c757d',
            'success': '#28a745',
            'danger': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8'
        }
    
    def create_correlation_heatmap(self, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create a correlation heatmap between different indicators"""
        # Prepare correlation data
        correlation_data = {}
        
        for name, df in data.items():
            if not df.empty:
                if 'Close' in df.columns:
                    # Use daily returns for correlation
                    returns = df['Close'].pct_change().dropna()
                    correlation_data[name.upper()] = returns
                elif 'Value' in df.columns:
                    returns = df['Value'].pct_change().dropna()
                    correlation_data[name.upper()] = returns
        
        if not correlation_data:
            return go.Figure().add_annotation(
                text="No data available for correlation analysis",
                showarrow=False
            )
        
        # Create correlation matrix
        corr_df = pd.DataFrame(correlation_data).corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.columns,
            colorscale='RdYlBu_r',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.around(corr_df.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Economic Indicators Correlation Matrix",
            xaxis_title="Indicators",
            yaxis_title="Indicators",
            height=600,
            width=800
        )
        
        return fig
    
    def create_performance_comparison(self, data: Dict[str, pd.DataFrame], 
                                   normalize: bool = True) -> go.Figure:
        """Create a normalized performance comparison chart"""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, (name, df) in enumerate(data.items()):
            if df.empty:
                continue
                
            if 'Close' in df.columns:
                values = df['Close']
            elif 'Value' in df.columns:
                values = df['Value']
            else:
                values = df.iloc[:, 0]
            
            if normalize:
                # Normalize to percentage change from first value
                normalized_values = (values / values.iloc[0] - 1) * 100
                y_label = "Percentage Change (%)"
            else:
                normalized_values = values
                y_label = "Value"
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=normalized_values,
                mode='lines',
                name=name.upper(),
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title="Economic Indicators Performance Comparison",
            xaxis_title="Date",
            yaxis_title=y_label,
            hovermode='x unified',
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    def create_volatility_analysis(self, data: pd.DataFrame, window: int = 30) -> go.Figure:
        """Create volatility analysis chart with rolling standard deviation"""
        if data.empty or 'Close' not in data.columns:
            return go.Figure().add_annotation(
                text="No price data available for volatility analysis",
                showarrow=False
            )
        
        # Calculate returns and rolling volatility
        returns = data['Close'].pct_change()
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100  # Annualized volatility
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price', f'{window}-Day Rolling Volatility (Annualized %)')
        )
        
        # Add price chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Price',
                line=dict(color=self.color_palette['primary'])
            ),
            row=1, col=1
        )
        
        # Add volatility chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=rolling_vol,
                mode='lines',
                name='Volatility',
                line=dict(color=self.color_palette['danger']),
                fill='tozeroy',
                fillcolor='rgba(220, 53, 69, 0.3)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    def create_economic_calendar_view(self, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create a calendar-style view of economic releases"""
        # This is a simplified version - in practice, you'd integrate with economic calendar APIs
        
        fig = go.Figure()
        
        # Mock economic events (replace with real calendar API)
        events = [
            {"date": "2024-01-15", "event": "CPI Release", "importance": "High"},
            {"date": "2024-01-20", "event": "Fed Meeting", "importance": "High"},
            {"date": "2024-02-01", "event": "Jobs Report", "importance": "Medium"},
            {"date": "2024-02-15", "event": "GDP Release", "importance": "High"},
        ]
        
        # Add events to chart
        for event in events:
            color = self.color_palette['danger'] if event['importance'] == 'High' else self.color_palette['warning']
            
            fig.add_trace(go.Scatter(
                x=[event['date']],
                y=[1],
                mode='markers+text',
                text=[event['event']],
                textposition='top center',
                marker=dict(
                    size=15,
                    color=color,
                    symbol='star'
                ),
                name=event['event'],
                showlegend=False
            ))
        
        fig.update_layout(
            title="Upcoming Economic Events",
            xaxis_title="Date",
            yaxis=dict(showticklabels=False, showgrid=False),
            height=300,
            template="plotly_white"
        )
        
        return fig
    
    def create_market_sentiment_gauge(self, sentiment_score: float) -> go.Figure:
        """Create a gauge chart for market sentiment"""
        # Determine color based on sentiment
        if sentiment_score >= 0.5:
            color = self.color_palette['success']
            sentiment_text = "Bullish"
        elif sentiment_score >= 0:
            color = self.color_palette['info']
            sentiment_text = "Neutral-Positive"
        elif sentiment_score >= -0.5:
            color = self.color_palette['warning']
            sentiment_text = "Neutral-Negative"
        else:
            color = self.color_palette['danger']
            sentiment_text = "Bearish"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = sentiment_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Market Sentiment<br><span style='font-size:0.8em;color:gray'>{sentiment_text}</span>"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': color},
                'steps': [
                    {'range': [-1, -0.5], 'color': "rgba(220, 53, 69, 0.3)"},
                    {'range': [-0.5, 0], 'color': "rgba(255, 193, 7, 0.3)"},
                    {'range': [0, 0.5], 'color': "rgba(23, 162, 184, 0.3)"},
                    {'range': [0.5, 1], 'color': "rgba(40, 167, 69, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def create_economic_summary_table(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create a summary table of all economic indicators"""
        summary_data = []
        
        for name, df in data.items():
            if df.empty:
                continue
            
            # Get the appropriate column
            if 'Close' in df.columns:
                values = df['Close']
                current_value = values.iloc[-1]
                previous_value = values.iloc[-2] if len(values) > 1 else current_value
            elif 'Value' in df.columns:
                values = df['Value']
                current_value = values.iloc[-1]
                previous_value = values.iloc[-2] if len(values) > 1 else current_value
            else:
                continue
            
            # Calculate changes
            change = current_value - previous_value
            change_pct = (change / previous_value) * 100 if previous_value != 0 else 0
            
            # Get period statistics
            period_high = values.max()
            period_low = values.min()
            volatility = values.pct_change().std() * np.sqrt(252) * 100  # Annualized
            
            summary_data.append({
                'Indicator': name.upper().replace('_', ' '),
                'Current': f"{current_value:.2f}",
                'Change': f"{change:+.2f}",
                'Change %': f"{change_pct:+.2f}%",
                'Period High': f"{period_high:.2f}",
                'Period Low': f"{period_low:.2f}",
                'Volatility %': f"{volatility:.1f}%"
            })
        
        return pd.DataFrame(summary_data)
    
    def create_technical_analysis_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create advanced technical analysis chart with multiple indicators"""
        if data.empty or 'Close' not in data.columns:
            return go.Figure().add_annotation(
                text="No price data available for technical analysis",
                showarrow=False
            )
        
        # Calculate technical indicators
        data = self._add_technical_indicators(data)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Price & Moving Averages', 'RSI', 'MACD')
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color=self.color_palette['positive'],
                decreasing_line_color=self.color_palette['negative']
            ),
            row=1, col=1
        )
        
        # Moving averages
        if 'SMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if 'SMA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        # Bollinger Bands
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple')
                ),
                row=2, col=1
            )
            
            # RSI reference lines
            fig.add_hline(y=70, row=2, col=1, line_dash="dash", line_color="red", opacity=0.5)
            fig.add_hline(y=30, row=2, col=1, line_dash="dash", line_color="green", opacity=0.5)
        
        # MACD
        if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue')
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD_Signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='red')
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            height=800,
            template="plotly_white",
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        
        # Update y-axis ranges
        fig.update_yaxes(range=[0, 100], row=2, col=1)  # RSI
        
        return fig
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data"""
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        return df