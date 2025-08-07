"""
Comprehensive testing suite for the Economic Dashboard
This module includes unit tests, integration tests, and performance tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import patch, MagicMock
import time

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import EconomicDataFetcher, EconomicDashboard
from data_sources import AdvancedEconomicDataFetcher
from visualizations import AdvancedVisualizations

class TestEconomicDataFetcher:
    """Test cases for the EconomicDataFetcher class"""
    
    @pytest.fixture
    def fetcher(self):
        """Create a data fetcher instance for testing"""
        return EconomicDataFetcher()
    
    def test_initialization(self, fetcher):
        """Test that the fetcher initializes correctly"""
        assert fetcher is not None
        assert fetcher.fred_api_key is None
        assert fetcher.cached_data == {}
    
    @patch('yfinance.Ticker')
    def test_fetch_stock_data_success(self, mock_ticker, fetcher):
        """Test successful stock data fetching"""
        # Mock the yfinance response
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        result = fetcher.fetch_stock_data("AAPL", "1mo")
        
        assert not result.empty
        assert len(result) == 3
        assert 'Close' in result.columns
        mock_ticker.assert_called_once_with("AAPL")
        mock_ticker_instance.history.assert_called_once_with(period="1mo")
    
    @patch('yfinance.Ticker')
    def test_fetch_stock_data_failure(self, mock_ticker, fetcher):
        """Test stock data fetching failure handling"""
        mock_ticker.side_effect = Exception("Network error")
        
        result = fetcher.fetch_stock_data("INVALID", "1mo")
        
        assert result.empty
    
    def test_generate_mock_inflation_data(self, fetcher):
        """Test mock inflation data generation"""
        result = fetcher.generate_mock_inflation_data()
        
        assert not result.empty
        assert 'Inflation_Rate' in result.columns
        assert all(0.5 <= rate <= 8.0 for rate in result['Inflation_Rate'])
        assert len(result) > 10  # Should have multiple months of data
    
    def test_generate_mock_unemployment_data(self, fetcher):
        """Test mock unemployment data generation"""
        result = fetcher.generate_mock_unemployment_data()
        
        assert not result.empty
        assert 'Unemployment_Rate' in result.columns
        assert all(2.5 <= rate <= 10.0 for rate in result['Unemployment_Rate'])
    
    def test_generate_mock_interest_rate_data(self, fetcher):
        """Test mock interest rate data generation"""
        result = fetcher.generate_mock_interest_rate_data()
        
        assert not result.empty
        assert 'Interest_Rate' in result.columns
        assert all(0.0 <= rate <= 10.0 for rate in result['Interest_Rate'])
    
    def test_generate_mock_debt_data(self, fetcher):
        """Test mock debt data generation"""
        result = fetcher.generate_mock_debt_data()
        
        assert not result.empty
        assert 'US_Debt_Trillion' in result.columns
        assert all(debt >= 31.0 for debt in result['US_Debt_Trillion'])

class TestEconomicDashboard:
    """Test cases for the EconomicDashboard class"""
    
    @pytest.fixture
    def dashboard(self):
        """Create a dashboard instance for testing"""
        return EconomicDashboard()
    
    def test_initialization(self, dashboard):
        """Test dashboard initialization"""
        assert dashboard is not None
        assert dashboard.data_fetcher is not None
        assert dashboard.cached_data == {}
        assert dashboard.last_update is None
    
    def test_calculate_statistics_valid_data(self, dashboard):
        """Test statistics calculation with valid data"""
        # Create sample data
        data = pd.DataFrame({
            'Close': [100, 105, 103, 108, 110],
            'Volume': [1000, 1100, 900, 1200, 1050]
        }, index=pd.date_range('2024-01-01', periods=5))
        
        stats = dashboard.calculate_statistics(data, 'Close')
        
        assert 'current' in stats
        assert 'previous' in stats
        assert 'change' in stats
        assert 'change_pct' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'mean' in stats
        assert 'std' in stats
        
        assert stats['current'] == 110
        assert stats['previous'] == 108
        assert stats['change'] == 2
        assert abs(stats['change_pct'] - 1.85) < 0.01  # Approximately 1.85%
    
    def test_calculate_statistics_empty_data(self, dashboard):
        """Test statistics calculation with empty data"""
        data = pd.DataFrame()
        stats = dashboard.calculate_statistics(data)
        
        assert stats == {}
    
    def test_create_price_chart_valid_data(self, dashboard):
        """Test price chart creation with valid data"""
        data = pd.DataFrame({
            'Close': [100, 105, 103, 108, 110]
        }, index=pd.date_range('2024-01-01', periods=5))
        
        fig = dashboard.create_price_chart(data, "Test Chart", "blue")
        
        assert fig is not None
        assert len(fig.data) == 1  # One trace
        assert fig.data[0].name == "Test Chart"
    
    def test_create_price_chart_empty_data(self, dashboard):
        """Test price chart creation with empty data"""
        data = pd.DataFrame()
        fig = dashboard.create_price_chart(data, "Empty Chart")
        
        assert fig is not None
        # Should have an annotation indicating no data

class TestAdvancedEconomicDataFetcher:
    """Test cases for the AdvancedEconomicDataFetcher class"""
    
    @pytest.fixture
    def advanced_fetcher(self):
        """Create an advanced fetcher instance for testing"""
        return AdvancedEconomicDataFetcher()
    
    def test_initialization(self, advanced_fetcher):
        """Test advanced fetcher initialization"""
        assert advanced_fetcher is not None
        assert advanced_fetcher.fred_api_key is None
        assert advanced_fetcher.fred_client is None
        assert len(advanced_fetcher.data_sources) > 0
    
    def test_data_sources_configuration(self, advanced_fetcher):
        """Test that data sources are properly configured"""
        sources = advanced_fetcher.data_sources
        
        # Check required indicators are present
        required_indicators = ['sp500', 'nasdaq', 'oil', 'inflation', 'unemployment']
        for indicator in required_indicators:
            assert indicator in sources
            assert hasattr(sources[indicator], 'name')
            assert hasattr(sources[indicator], 'symbol')
            assert hasattr(sources[indicator], 'source_type')
    
    @patch('yfinance.Ticker')
    def test_fetch_yahoo_data(self, mock_ticker, advanced_fetcher):
        """Test Yahoo Finance data fetching"""
        mock_data = pd.DataFrame({
            'Close': [100, 105, 103],
        }, index=pd.date_range('2024-01-01', periods=3))
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        result = advanced_fetcher.fetch_yahoo_data("^GSPC")
        
        assert not result.empty
        assert result.attrs['symbol'] == "^GSPC"
        assert result.attrs['source'] == 'yahoo'
    
    def test_generate_mock_fred_data(self, advanced_fetcher):
        """Test FRED mock data generation"""
        result = advanced_fetcher._generate_mock_fred_data('CPIAUCSL')
        
        assert not result.empty
        assert 'Value' in result.columns
        assert result.attrs['series_id'] == 'CPIAUCSL'
        assert result.attrs['source'] == 'mock'
    
    def test_get_start_date_from_period(self, advanced_fetcher):
        """Test period to start date conversion"""
        # Test various periods
        periods = ['1mo', '3mo', '6mo', '1y', '2y', '5y']
        
        for period in periods:
            start_date = advanced_fetcher._get_start_date_from_period(period)
            assert isinstance(start_date, str)
            assert len(start_date) == 10  # YYYY-MM-DD format
            
            # Verify it's a valid date
            datetime.strptime(start_date, '%Y-%m-%d')

class TestAdvancedVisualizations:
    """Test cases for the AdvancedVisualizations class"""
    
    @pytest.fixture
    def visualizations(self):
        """Create a visualizations instance for testing"""
        return AdvancedVisualizations()
    
    def test_initialization(self, visualizations):
        """Test visualizations initialization"""
        assert visualizations is not None
        assert hasattr(visualizations, 'color_palette')
        assert 'positive' in visualizations.color_palette
        assert 'negative' in visualizations.color_palette
    
    def test_create_correlation_heatmap(self, visualizations):
        """Test correlation heatmap creation"""
        # Create sample data
        data = {
            'sp500': pd.DataFrame({
                'Close': np.random.randn(100) + 100
            }, index=pd.date_range('2024-01-01', periods=100)),
            'oil': pd.DataFrame({
                'Close': np.random.randn(100) + 50
            }, index=pd.date_range('2024-01-01', periods=100))
        }
        
        fig = visualizations.create_correlation_heatmap(data)
        
        assert fig is not None
        assert len(fig.data) >= 1
    
    def test_create_performance_comparison(self, visualizations):
        """Test performance comparison chart creation"""
        data = {
            'asset1': pd.DataFrame({
                'Close': [100, 105, 110, 108, 115]
            }, index=pd.date_range('2024-01-01', periods=5)),
            'asset2': pd.DataFrame({
                'Close': [50, 52, 51, 53, 55]
            }, index=pd.date_range('2024-01-01', periods=5))
        }
        
        fig = visualizations.create_performance_comparison(data)
        
        assert fig is not None
        assert len(fig.data) == 2  # Two traces
    
    def test_create_market_sentiment_gauge(self, visualizations):
        """Test market sentiment gauge creation"""
        # Test different sentiment levels
        sentiment_scores = [-0.8, -0.3, 0.2, 0.7]
        
        for score in sentiment_scores:
            fig = visualizations.create_market_sentiment_gauge(score)
            assert fig is not None
            assert len(fig.data) == 1
    
    def test_create_economic_summary_table(self, visualizations):
        """Test economic summary table creation"""
        data = {
            'sp500': pd.DataFrame({
                'Close': [4000, 4100, 4050, 4200]
            }, index=pd.date_range('2024-01-01', periods=4)),
            'inflation': pd.DataFrame({
                'Value': [3.2, 3.4, 3.1, 3.3]
            }, index=pd.date_range('2024-01-01', periods=4, freq='M'))
        }
        
        summary = visualizations.create_economic_summary_table(data)
        
        assert not summary.empty
        assert 'Indicator' in summary.columns
        assert 'Current' in summary.columns
        assert 'Change %' in summary.columns
        assert len(summary) == 2  # Two indicators

class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_end_to_end_workflow(self):
        """Test the complete workflow from data fetching to visualization"""
        # Initialize components
        fetcher = EconomicDataFetcher()
        dashboard = EconomicDashboard()
        
        # Generate mock data
        sp500_data = fetcher.generate_mock_inflation_data()  # Using this as sample data
        
        # Calculate statistics
        stats = dashboard.calculate_statistics(sp500_data, 'Inflation_Rate')
        
        # Create visualization
        fig = dashboard.create_price_chart(sp500_data, "Test Integration", "blue")
        
        # Assertions
        assert not sp500_data.empty
        assert stats is not None
        assert fig is not None
    
    def test_data_caching_mechanism(self):
        """Test that data caching works correctly"""
        dashboard = EconomicDashboard()
        
        # First call should fetch data
        start_time = time.time()
        data1 = dashboard.get_data_with_cache(
            'test_key', 
            lambda period: dashboard.data_fetcher.generate_mock_inflation_data()
        )
        first_call_time = time.time() - start_time
        
        # Second call should use cache (should be faster)
        start_time = time.time()
        data2 = dashboard.get_data_with_cache(
            'test_key',
            lambda period: dashboard.data_fetcher.generate_mock_inflation_data()
        )
        second_call_time = time.time() - start_time
        
        assert not data1.empty
        assert not data2.empty
        assert pd.DataFrame.equals(data1, data2)  # Should be identical from cache

class TestPerformance:
    """Performance tests for the dashboard"""
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets"""
        # Create large dataset
        large_data = pd.DataFrame({
            'Close': np.random.randn(10000) + 100,
            'Volume': np.random.randint(1000000, 10000000, 10000)
        }, index=pd.date_range('2020-01-01', periods=10000))
        
        dashboard = EconomicDashboard()
        
        # Test statistics calculation performance
        start_time = time.time()
        stats = dashboard.calculate_statistics(large_data)
        calc_time = time.time() - start_time
        
        # Should complete within reasonable time (< 1 second)
        assert calc_time < 1.0
        assert stats is not None
    
    def test_multiple_indicators_performance(self):
        """Test performance with multiple indicators"""
        fetcher = EconomicDataFetcher()
        
        # Generate multiple datasets
        indicators = ['sp500', 'nasdaq', 'oil', 'inflation', 'unemployment']
        data = {}
        
        start_time = time.time()
        for indicator in indicators:
            if indicator in ['sp500', 'nasdaq', 'oil']:
                data[indicator] = pd.DataFrame({
                    'Close': np.random.randn(252) + 100  # 1 year of data
                }, index=pd.date_range('2024-01-01', periods=252))
            else:
                data[indicator] = fetcher.generate_mock_inflation_data()
        
        processing_time = time.time() - start_time
        
        # Should process all indicators quickly
        assert processing_time < 5.0  # 5 seconds max
        assert len(data) == len(indicators)

# Test fixtures and utilities
@pytest.fixture
def sample_stock_data():
    """Fixture providing sample stock data"""
    return pd.DataFrame({
        'Open': [100, 102, 101, 105, 107],
        'High': [105, 106, 104, 108, 110],
        'Low': [99, 101, 100, 104, 106],
        'Close': [104, 105, 103, 107, 109],
        'Volume': [1000000, 1100000, 950000, 1200000, 1150000]
    }, index=pd.date_range('2024-01-01', periods=5))

@pytest.fixture
def sample_economic_data():
    """Fixture providing sample economic data"""
    return pd.DataFrame({
        'Value': [3.2, 3.4, 3.1, 3.3, 3.5]
    }, index=pd.date_range('2024-01-01', periods=5, freq='M'))

# Utility functions for testing
def assert_valid_dataframe(df, expected_columns=None):
    """Assert that a DataFrame is valid and has expected structure"""
    assert not df.empty
    assert isinstance(df, pd.DataFrame)
    if expected_columns:
        for col in expected_columns:
            assert col in df.columns

def assert_valid_plotly_figure(fig):
    """Assert that a Plotly figure is valid"""
    assert fig is not None
    assert hasattr(fig, 'data')
    assert hasattr(fig, 'layout')

# Run tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])