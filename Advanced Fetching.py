"""
Advanced data sources module for the Economic Dashboard
This module provides enhanced data fetching capabilities with multiple sources
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from fredapi import Fred  # Requires FRED API key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Configuration for a data source"""
    name: str
    symbol: str
    source_type: str  # 'yahoo', 'fred', 'api', 'csv'
    description: str
    frequency: str  # 'daily', 'monthly', 'quarterly'

class AdvancedEconomicDataFetcher:
    """Enhanced data fetcher with multiple sources and error handling"""
    
    def __init__(self, fred_api_key: Optional[str] = None):
        self.fred_api_key = fred_api_key
        self.fred_client = Fred(api_key=fred_api_key) if fred_api_key else None
        
        # Define data sources configuration
        self.data_sources = {
            'sp500': DataSource('S&P 500', '^GSPC', 'yahoo', 'S&P 500 Index', 'daily'),
            'nasdaq': DataSource('NASDAQ', '^IXIC', 'yahoo', 'NASDAQ Composite Index', 'daily'),
            'oil': DataSource('Oil', 'CL=F', 'yahoo', 'WTI Crude Oil', 'daily'),
            'dollar_index': DataSource('DXY', 'DX-Y.NYB', 'yahoo', 'US Dollar Index', 'daily'),
            'inflation': DataSource('CPI', 'CPIAUCSL', 'fred', 'Consumer Price Index', 'monthly'),
            'unemployment': DataSource('Unemployment', 'UNRATE', 'fred', 'Unemployment Rate', 'monthly'),
            'fed_funds_rate': DataSource('Fed Funds', 'FEDFUNDS', 'fred', 'Federal Funds Rate', 'monthly'),
            'us_debt': DataSource('US Debt', 'GFDEGDQ188S', 'fred', 'Federal Debt to GDP Ratio', 'quarterly'),
            'gdp': DataSource('GDP', 'GDP', 'fred', 'Gross Domestic Product', 'quarterly'),
            'treasury_10y': DataSource('10Y Treasury', '^TNX', 'yahoo', '10-Year Treasury Yield', 'daily'),
            'vix': DataSource('VIX', '^VIX', 'yahoo', 'Volatility Index', 'daily'),
            'gold': DataSource('Gold', 'GC=F', 'yahoo', 'Gold Futures', 'daily'),
            'bitcoin': DataSource('Bitcoin', 'BTC-USD', 'yahoo', 'Bitcoin Price', 'daily'),
        }
    
    def fetch_yahoo_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetch data from Yahoo Finance with error handling"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data returned for symbol: {symbol}")
                return pd.DataFrame()
            
            # Add metadata
            data.attrs['symbol'] = symbol
            data.attrs['source'] = 'yahoo'
            data.attrs['last_update'] = datetime.now()
            
            logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_fred_data(self, series_id: str, start_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch data from FRED API"""
        if not self.fred_client:
            logger.warning("FRED API key not provided, generating mock data")
            return self._generate_mock_fred_data(series_id)
        
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
            
            data = self.fred_client.get_series(series_id, start=start_date)
            
            if data.empty:
                logger.warning(f"No data returned for FRED series: {series_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame with proper structure
            df = pd.DataFrame({
                'Date': data.index,
                'Value': data.values
            })
            df.set_index('Date', inplace=True)
            
            # Add metadata
            df.attrs['series_id'] = series_id
            df.attrs['source'] = 'fred'
            df.attrs['last_update'] = datetime.now()
            
            logger.info(f"Successfully fetched {len(df)} rows for FRED series {series_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching FRED data for {series_id}: {str(e)}")
            return self._generate_mock_fred_data(series_id)
    
    def _generate_mock_fred_data(self, series_id: str) -> pd.DataFrame:
        """Generate mock data when FRED API is unavailable"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years
        
        # Generate monthly data points
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Mock data based on series type
        if series_id == 'CPIAUCSL':  # Inflation (CPI)
            base_value = 300
            values = base_value + np.cumsum(np.random.normal(0.2, 0.5, len(date_range)))
        elif series_id == 'UNRATE':  # Unemployment rate
            values = np.random.normal(4.0, 0.8, len(date_range))
            values = np.clip(values, 2.0, 10.0)
        elif series_id == 'FEDFUNDS':  # Fed funds rate
            values = np.random.normal(5.0, 0.5, len(date_range))
            values = np.clip(values, 0.0, 8.0)
        elif series_id == 'GFDEGDQ188S':  # Debt to GDP
            values = np.random.normal(120, 5, len(date_range))
            values = np.clip(values, 100, 150)
        elif series_id == 'GDP':  # GDP
            base_value = 25000
            values = base_value + np.cumsum(np.random.normal(200, 100, len(date_range)))
        else:
            values = np.random.normal(100, 10, len(date_range))
        
        df = pd.DataFrame({
            'Date': date_range,
            'Value': values
        })
        df.set_index('Date', inplace=True)
        
        # Add metadata
        df.attrs['series_id'] = series_id
        df.attrs['source'] = 'mock'
        df.attrs['last_update'] = datetime.now()
        
        logger.info(f"Generated mock data for {series_id}: {len(df)} rows")
        return df
    
    def fetch_economic_indicator(self, indicator: str, period: str = "1y") -> pd.DataFrame:
        """Fetch data for a specific economic indicator"""
        if indicator not in self.data_sources:
            logger.error(f"Unknown indicator: {indicator}")
            return pd.DataFrame()
        
        source = self.data_sources[indicator]
        
        if source.source_type == 'yahoo':
            return self.fetch_yahoo_data(source.symbol, period)
        elif source.source_type == 'fred':
            start_date = self._get_start_date_from_period(period)
            return self.fetch_fred_data(source.symbol, start_date)
        else:
            logger.error(f"Unsupported source type: {source.source_type}")
            return pd.DataFrame()
    
    def _get_start_date_from_period(self, period: str) -> str:
        """Convert period string to start date for FRED API"""
        now = datetime.now()
        
        if period == "1mo":
            start_date = now - timedelta(days=30)
        elif period == "3mo":
            start_date = now - timedelta(days=90)
        elif period == "6mo":
            start_date = now - timedelta(days=180)
        elif period == "1y":
            start_date = now - timedelta(days=365)
        elif period == "2y":
            start_date = now - timedelta(days=730)
        elif period == "5y":
            start_date = now - timedelta(days=1825)
        else:
            start_date = now - timedelta(days=365)
        
        return start_date.strftime('%Y-%m-%d')
    
    def get_all_indicators(self, period: str = "1y") -> Dict[str, pd.DataFrame]:
        """Fetch all available indicators"""
        data = {}
        
        for indicator in self.data_sources.keys():
            try:
                df = self.fetch_economic_indicator(indicator, period)
                if not df.empty:
                    data[indicator] = df
                else:
                    logger.warning(f"No data available for {indicator}")
            except Exception as e:
                logger.error(f"Failed to fetch {indicator}: {str(e)}")
        
        return data
    
    def get_correlation_matrix(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation matrix between indicators"""
        # Prepare data for correlation analysis
        correlation_data = {}
        
        for name, df in data.items():
            if not df.empty:
                if 'Close' in df.columns:
                    values = df['Close'].resample('M').last()  # Monthly resampling
                elif 'Value' in df.columns:
                    values = df['Value']
                else:
                    values = df.iloc[:, 0]
                
                # Calculate percentage change
                pct_change = values.pct_change().dropna()
                correlation_data[name] = pct_change
        
        # Create correlation matrix
        if correlation_data:
            corr_df = pd.DataFrame(correlation_data)
            return corr_df.corr()
        else:
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data"""
        if data.empty or 'Close' not in data.columns:
            return data
        
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators (if volume data available)
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        return df
    
    def get_market_sentiment_score(self, data: Dict[str, pd.DataFrame]) -> float:
        """Calculate a simple market sentiment score based on multiple indicators"""
        sentiment_score = 0.0
        weights = {
            'sp500': 0.3,
            'nasdaq': 0.2,
            'vix': -0.2,  # Negative weight (higher VIX = lower sentiment)
            'dollar_index': 0.1,
            'treasury_10y': -0.1,
            'oil': 0.1,
            'gold': 0.1
        }
        
        for indicator, weight in weights.items():
            if indicator in data and not data[indicator].empty:
                df = data[indicator]
                
                if 'Close' in df.columns and len(df) >= 5:
                    # Calculate 5-day momentum
                    current_price = df['Close'].iloc[-1]
                    price_5_days_ago = df['Close'].iloc[-6] if len(df) >= 6 else df['Close'].iloc[0]
                    momentum = (current_price - price_5_days_ago) / price_5_days_ago
                    
                    sentiment_score += momentum * weight
        
        # Normalize to -1 to 1 scale
        sentiment_score = max(-1.0, min(1.0, sentiment_score * 10))
        
        return sentiment_score