"""
Configuration and settings module for the Economic Dashboard
This module handles all configuration settings, environment variables, and constants
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

@dataclass
class APIConfig:
    """Configuration for external APIs"""
    fred_api_key: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None
    quandl_api_key: Optional[str] = None
    news_api_key: Optional[str] = None

@dataclass
class CacheConfig:
    """Configuration for data caching"""
    enable_cache: bool = True
    cache_duration_minutes: int = 15
    max_cache_size_mb: int = 100
    cache_directory: str = "./cache"

@dataclass
class UIConfig:
    """Configuration for user interface"""
    theme: str = "light"  # light, dark
    default_period: str = "1y"
    default_indicators: List[str] = None
    auto_refresh: bool = False
    refresh_interval_minutes: int = 5
    
    def __post_init__(self):
        if self.default_indicators is None:
            self.default_indicators = ["sp500", "nasdaq", "oil", "inflation"]

@dataclass
class ChartConfig:
    """Configuration for charts and visualizations"""
    color_scheme: str = "plotly"  # plotly, seaborn, custom
    chart_height: int = 400
    enable_crossfilter: bool = True
    show_volume: bool = True
    technical_indicators: List[str] = None
    
    def __post_init__(self):
        if self.technical_indicators is None:
            self.technical_indicators = ["sma", "rsi", "macd"]

@dataclass
class DataConfig:
    """Configuration for data sources and processing"""
    primary_source: str = "yahoo"  # yahoo, fred, mixed
    backup_source: str = "mock"
    data_quality_checks: bool = True
    outlier_detection: bool = True
    fill_missing_data: bool = True
    max_data_age_hours: int = 24

class DashboardConfig:
    """Main configuration class for the Economic Dashboard"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._get_default_config_file()
        self.api = APIConfig()
        self.cache = CacheConfig()
        self.ui = UIConfig()
        self.charts = ChartConfig()
        self.data = DataConfig()
        
        # Load configuration from various sources
        self._load_from_environment()
        self._load_from_file()
        self._validate_config()
    
    def _get_default_config_file(self) -> str:
        """Get the default configuration file path"""
        return os.path.join(os.path.dirname(__file__), "config.json")
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # API Keys
        self.api.fred_api_key = os.getenv("FRED_API_KEY")
        self.api.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.api.quandl_api_key = os.getenv("QUANDL_API_KEY")
        self.api.news_api_key = os.getenv("NEWS_API_KEY")
        
        # Cache settings
        self.cache.enable_cache = os.getenv("ENABLE_CACHE", "true").lower() == "true"
        self.cache.cache_duration_minutes = int(os.getenv("CACHE_DURATION_MINUTES", "15"))
        self.cache.cache_directory = os.getenv("CACHE_DIRECTORY", "./cache")
        
        # UI settings
        self.ui.theme = os.getenv("UI_THEME", "light")
        self.ui.default_period = os.getenv("DEFAULT_PERIOD", "1y")
        self.ui.auto_refresh = os.getenv("AUTO_REFRESH", "false").lower() == "true"
        
        # Data settings
        self.data.primary_source = os.getenv("PRIMARY_DATA_SOURCE", "yahoo")
        self.data.backup_source = os.getenv("BACKUP_DATA_SOURCE", "mock")
        self.data.data_quality_checks = os.getenv("DATA_QUALITY_CHECKS", "true").lower() == "true"
    
    def _load_from_file(self):
        """Load configuration from JSON file"""
        if not os.path.exists(self.config_file):
            self._create_default_config_file()
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration with file data
            self._update_config_from_dict(config_data)
            
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_file}: {e}")
    
    def _create_default_config_file(self):
        """Create a default configuration file"""
        default_config = {
            "api": {
                "fred_api_key": "",
                "alpha_vantage_api_key": "",
                "quandl_api_key": "",
                "news_api_key": ""
            },
            "cache": {
                "enable_cache": True,
                "cache_duration_minutes": 15,
                "max_cache_size_mb": 100,
                "cache_directory": "./cache"
            },
            "ui": {
                "theme": "light",
                "default_period": "1y",
                "default_indicators": ["sp500", "nasdaq", "oil", "inflation"],
                "auto_refresh": False,
                "refresh_interval_minutes": 5
            },
            "charts": {
                "color_scheme": "plotly",
                "chart_height": 400,
                "enable_crossfilter": True,
                "show_volume": True,
                "technical_indicators": ["sma", "rsi", "macd"]
            },
            "data": {
                "primary_source": "yahoo",
                "backup_source": "mock",
                "data_quality_checks": True,
                "outlier_detection": True,
                "fill_missing_data": True,
                "max_data_age_hours": 24
            }
        }
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            print(f"Created default configuration file: {self.config_file}")
            
        except Exception as e:
            print(f"Warning: Could not create config file {self.config_file}: {e}")
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        # Update API config
        if 'api' in config_dict:
            api_config = config_dict['api']
            self.api.fred_api_key = api_config.get('fred_api_key') or self.api.fred_api_key
            self.api.alpha_vantage_api_key = api_config.get('alpha_vantage_api_key') or self.api.alpha_vantage_api_key
            self.api.quandl_api_key = api_config.get('quandl_api_key') or self.api.quandl_api_key
            self.api.news_api_key = api_config.get('news_api_key') or self.api.news_api_key
        
        # Update cache config
        if 'cache' in config_dict:
            cache_config = config_dict['cache']
            self.cache.enable_cache = cache_config.get('enable_cache', self.cache.enable_cache)
            self.cache.cache_duration_minutes = cache_config.get('cache_duration_minutes', self.cache.cache_duration_minutes)
            self.cache.max_cache_size_mb = cache_config.get('max_cache_size_mb', self.cache.max_cache_size_mb)
            self.cache.cache_directory = cache_config.get('cache_directory', self.cache.cache_directory)
        
        # Update UI config
        if 'ui' in config_dict:
            ui_config = config_dict['ui']
            self.ui.theme = ui_config.get('theme', self.ui.theme)
            self.ui.default_period = ui_config.get('default_period', self.ui.default_period)
            self.ui.default_indicators = ui_config.get('default_indicators', self.ui.default_indicators)
            self.ui.auto_refresh = ui_config.get('auto_refresh', self.ui.auto_refresh)
            self.ui.refresh_interval_minutes = ui_config.get('refresh_interval_minutes', self.ui.refresh_interval_minutes)
        
        # Update chart config
        if 'charts' in config_dict:
            chart_config = config_dict['charts']
            self.charts.color_scheme = chart_config.get('color_scheme', self.charts.color_scheme)
            self.charts.chart_height = chart_config.get('chart_height', self.charts.chart_height)
            self.charts.enable_crossfilter = chart_config.get('enable_crossfilter', self.charts.enable_crossfilter)
            self.charts.show_volume = chart_config.get('show_volume', self.charts.show_volume)
            self.charts.technical_indicators = chart_config.get('technical_indicators', self.charts.technical_indicators)
        
        # Update data config
        if 'data' in config_dict:
            data_config = config_dict['data']
            self.data.primary_source = data_config.get('primary_source', self.data.primary_source)
            self.data.backup_source = data_config.get('backup_source', self.data.backup_source)
            self.data.data_quality_checks = data_config.get('data_quality_checks', self.data.data_quality_checks)
            self.data.outlier_detection = data_config.get('outlier_detection', self.data.outlier_detection)
            self.data.fill_missing_data = data_config.get('fill_missing_data', self.data.fill_missing_data)
            self.data.max_data_age_hours = data_config.get('max_data_age_hours', self.data.max_data_age_hours)
    
    def _validate_config(self):
        """Validate configuration settings"""
        # Validate periods
        valid_periods = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
        if self.ui.default_period not in valid_periods:
            print(f"Warning: Invalid default period '{self.ui.default_period}'. Using '1y'.")
            self.ui.default_period = "1y"
        
        # Validate themes
        valid_themes = ["light", "dark"]
        if self.ui.theme not in valid_themes:
            print(f"Warning: Invalid theme '{self.ui.theme}'. Using 'light'.")
            self.ui.theme = "light"
        
        # Validate data sources
        valid_sources = ["yahoo", "fred", "mixed", "mock"]
        if self.data.primary_source not in valid_sources:
            print(f"Warning: Invalid primary source '{self.data.primary_source}'. Using 'yahoo'.")
            self.data.primary_source = "yahoo"
        
        # Validate cache directory
        if self.cache.enable_cache:
            try:
                os.makedirs(self.cache.cache_directory, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create cache directory: {e}")
                self.cache.enable_cache = False
        
        # Validate numeric settings
        if self.cache.cache_duration_minutes <= 0:
            self.cache.cache_duration_minutes = 15
        
        if self.charts.chart_height <= 0:
            self.charts.chart_height = 400
        
        if self.ui.refresh_interval_minutes <= 0:
            self.ui.refresh_interval_minutes = 5
    
    def save_config(self):
        """Save current configuration to file"""
        config_dict = {
            "api": {
                "fred_api_key": self.api.fred_api_key or "",
                "alpha_vantage_api_key": self.api.alpha_vantage_api_key or "",
                "quandl_api_key": self.api.quandl_api_key or "",
                "news_api_key": self.api.news_api_key or ""
            },
            "cache": {
                "enable_cache": self.cache.enable_cache,
                "cache_duration_minutes": self.cache.cache_duration_minutes,
                "max_cache_size_mb": self.cache.max_cache_size_mb,
                "cache_directory": self.cache.cache_directory
            },
            "ui": {
                "theme": self.ui.theme,
                "default_period": self.ui.default_period,
                "default_indicators": self.ui.default_indicators,
                "auto_refresh": self.ui.auto_refresh,
                "refresh_interval_minutes": self.ui.refresh_interval_minutes
            },
            "charts": {
                "color_scheme": self.charts.color_scheme,
                "chart_height": self.charts.chart_height,
                "enable_crossfilter": self.charts.enable_crossfilter,
                "show_volume": self.charts.show_volume,
                "technical_indicators": self.charts.technical_indicators
            },
            "data": {
                "primary_source": self.data.primary_source,
                "backup_source": self.data.backup_source,
                "data_quality_checks": self.data.data_quality_checks,
                "outlier_detection": self.data.outlier_detection,
                "fill_missing_data": self.data.fill_missing_data,
                "max_data_age_hours": self.data.max_data_age_hours
            }
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def get_color_palette(self) -> Dict[str, str]:
        """Get color palette based on current theme and color scheme"""
        if self.ui.theme == "dark":
            return {
                'background': '#1e1e1e',
                'paper': '#2d2d2d',
                'text': '#ffffff',
                'positive': '#4CAF50',
                'negative': '#F44336',
                'neutral': '#2196F3',
                'primary': '#3f51b5',
                'secondary': '#ff9800',
                'accent': '#e91e63'
            }
        else:  # light theme
            return {
                'background': '#ffffff',
                'paper': '#f8f9fa',
                'text': '#212529',
                'positive': '#28a745',
                'negative': '#dc3545',
                'neutral': '#17a2b8',
                'primary': '#007bff',
                'secondary': '#6c757d',
                'accent': '#e83e8c'
            }
    
    def get_plotly_template(self) -> str:
        """Get Plotly template based on current theme"""
        if self.ui.theme == "dark":
            return "plotly_dark"
        else:
            return "plotly_white"
    
    def is_api_available(self, api_name: str) -> bool:
        """Check if an API key is available and configured"""
        api_keys = {
            'fred': self.api.fred_api_key,
            'alpha_vantage': self.api.alpha_vantage_api_key,
            'quandl': self.api.quandl_api_key,
            'news': self.api.news_api_key
        }
        
        return api_keys.get(api_name) is not None and api_keys.get(api_name) != ""
    
    def get_cache_path(self, cache_key: str) -> str:
        """Get full path for a cache file"""
        return os.path.join(self.cache.cache_directory, f"{cache_key}.pkl")
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"""Economic Dashboard Configuration:
API Keys: FRED={'✓' if self.is_api_available('fred') else '✗'}
Cache: {'Enabled' if self.cache.enable_cache else 'Disabled'} ({self.cache.cache_duration_minutes} min)
Theme: {self.ui.theme}
Default Period: {self.ui.default_period}
Primary Data Source: {self.data.primary_source}
Chart Height: {self.charts.chart_height}px"""

# Constants and default values
class Constants:
    """Application constants"""
    
    # Supported time periods
    TIME_PERIODS = {
        "1 Month": "1mo",
        "3 Months": "3mo", 
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    }
    
    # Available indicators
    INDICATORS = {
        'sp500': {
            'name': 'S&P 500',
            'symbol': '^GSPC',
            'source': 'yahoo',
            'category': 'equity'
        },
        'nasdaq': {
            'name': 'NASDAQ',
            'symbol': '^IXIC', 
            'source': 'yahoo',
            'category': 'equity'
        },
        'oil': {
            'name': 'Oil (WTI)',
            'symbol': 'CL=F',
            'source': 'yahoo',
            'category': 'commodity'
        },
        'gold': {
            'name': 'Gold',
            'symbol': 'GC=F',
            'source': 'yahoo',
            'category': 'commodity'
        },
        'dollar_index': {
            'name': 'US Dollar Index',
            'symbol': 'DX-Y.NYB',
            'source': 'yahoo',
            'category': 'currency'
        },
        'treasury_10y': {
            'name': '10Y Treasury',
            'symbol': '^TNX',
            'source': 'yahoo',
            'category': 'bond'
        },
        'vix': {
            'name': 'VIX',
            'symbol': '^VIX',
            'source': 'yahoo',
            'category': 'volatility'
        },
        'bitcoin': {
            'name': 'Bitcoin',
            'symbol': 'BTC-USD',
            'source': 'yahoo',
            'category': 'crypto'
        },
        'inflation': {
            'name': 'Inflation Rate',
            'symbol': 'CPIAUCSL',
            'source': 'fred',
            'category': 'economic'
        },
        'unemployment': {
            'name': 'Unemployment Rate',
            'symbol': 'UNRATE',
            'source': 'fred',
            'category': 'economic'
        },
        'fed_funds_rate': {
            'name': 'Fed Funds Rate',
            'symbol': 'FEDFUNDS',
            'source': 'fred',
            'category': 'economic'
        },
        'gdp': {
            'name': 'GDP',
            'symbol': 'GDP',
            'source': 'fred',
            'category': 'economic'
        }
    }
    
    # Technical indicators
    TECHNICAL_INDICATORS = [
        'sma_20', 'sma_50', 'sma_200',
        'ema_12', 'ema_26',
        'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'bb_middle'
    ]
    
    # Chart color schemes
    COLOR_SCHEMES = {
        'plotly': [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ],
        'seaborn': [
            '#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3',
            '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd'
        ],
        'custom': [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#34495e', '#95a5a6', '#e67e22', '#16a085'
        ]
    }
    
    # Data source priorities (fallback order)
    DATA_SOURCE_PRIORITY = ['yahoo', 'fred', 'alpha_vantage', 'mock']
    
    # Cache settings
    MAX_CACHE_AGE_SECONDS = 900  # 15 minutes
    MAX_CACHE_ENTRIES = 100
    
    # Performance settings
    MAX_DATA_POINTS = 10000  # Maximum data points to display
    CHART_UPDATE_THROTTLE_MS = 100  # Minimum time between chart updates

# Global configuration instance
config = DashboardConfig()

# Utility functions
def get_indicator_info(indicator_key: str) -> Dict[str, str]:
    """Get information about a specific indicator"""
    return Constants.INDICATORS.get(indicator_key, {})

def get_indicators_by_category(category: str) -> List[str]:
    """Get all indicators in a specific category"""
    return [
        key for key, info in Constants.INDICATORS.items()
        if info.get('category') == category
    ]

def validate_period(period: str) -> bool:
    """Validate if a period string is supported"""
    return period in Constants.TIME_PERIODS.values()

def get_display_name(indicator_key: str) -> str:
    """Get the display name for an indicator"""
    info = Constants.INDICATORS.get(indicator_key, {})
    return info.get('name', indicator_key.replace('_', ' ').title())

def get_symbol(indicator_key: str) -> str:
    """Get the symbol/ticker for an indicator"""
    info = Constants.INDICATORS.get(indicator_key, {})
    return info.get('symbol', indicator_key.upper())

def get_data_source(indicator_key: str) -> str:
    """Get the data source for an indicator"""
    info = Constants.INDICATORS.get(indicator_key, {})
    return info.get('source', 'yahoo')

# Environment-specific settings
def is_production() -> bool:
    """Check if running in production environment"""
    return os.getenv('ENVIRONMENT', 'development').lower() == 'production'

def is_development() -> bool:
    """Check if running in development environment"""
    return not is_production()

def get_log_level() -> str:
    """Get the appropriate log level based on environment"""
    if is_development():
        return 'DEBUG'
    return os.getenv('LOG_LEVEL', 'INFO').upper()