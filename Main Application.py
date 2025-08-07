import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Economic Indicators Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EconomicDataFetcher:
    """Class to handle data fetching from various sources"""
    
    def __init__(self):
        self.fred_api_key = None  # Users need to set this
        
    def fetch_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch stock/index data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_oil_prices(self, period: str = "1y") -> pd.DataFrame:
        """Fetch oil prices (WTI Crude)"""
        return self.fetch_stock_data("CL=F", period)
    
    def fetch_sp500(self, period: str = "1y") -> pd.DataFrame:
        """Fetch S&P 500 data"""
        return self.fetch_stock_data("^GSPC", period)
    
    def fetch_nasdaq(self, period: str = "1y") -> pd.DataFrame:
        """Fetch NASDAQ data"""
        return self.fetch_stock_data("^IXIC", period)
    
    def fetch_dollar_index(self, period: str = "1y") -> pd.DataFrame:
        """Fetch US Dollar Index"""
        return self.fetch_stock_data("DX-Y.NYB", period)
    
    def generate_mock_inflation_data(self, period: str = "1y") -> pd.DataFrame:
        """Generate mock inflation data (replace with FRED API)"""
        # Convert period to number of months for mock data generation
        period_months = self._period_to_months(period)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_months * 30)
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Ensure we have at least some data points
        if len(dates) == 0:
            dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='M')
        
        # Simulate realistic inflation data with more variation
        base_rate = 3.2
        inflation_rates = []
        for i, date in enumerate(dates):
            # Add trend and seasonal variation
            trend = np.sin(i * 0.3) * 0.8  # Seasonal variation
            noise = np.random.normal(0, 0.4)  # Random noise
            rate = base_rate + trend + noise
            inflation_rates.append(max(0.5, min(8.0, rate)))  # Clip between 0.5 and 8.0
        
        df = pd.DataFrame({
            'Inflation_Rate': inflation_rates
        }, index=dates)
        return df
    
    def generate_mock_unemployment_data(self, period: str = "1y") -> pd.DataFrame:
        """Generate mock unemployment data"""
        period_months = self._period_to_months(period)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_months * 30)
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Ensure we have at least some data points
        if len(dates) == 0:
            dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='M')
        
        # Simulate realistic unemployment data with variation
        base_rate = 3.8
        unemployment_rates = []
        for i, date in enumerate(dates):
            # Add cyclical variation
            cycle = np.cos(i * 0.2) * 0.5  # Economic cycle variation
            noise = np.random.normal(0, 0.3)  # Random noise
            rate = base_rate + cycle + noise
            unemployment_rates.append(max(2.0, min(10.0, rate)))  # Clip between 2.0 and 10.0
        
        df = pd.DataFrame({
            'Unemployment_Rate': unemployment_rates
        }, index=dates)
        return df
    
    def generate_mock_interest_rate_data(self, period: str = "1y") -> pd.DataFrame:
        """Generate mock interest rate data"""
        period_months = self._period_to_months(period)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_months * 30)
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Ensure we have at least some data points
        if len(dates) == 0:
            dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='M')
        
        # Simulate interest rate changes (more realistic stepping pattern)
        base_rate = 5.25
        interest_rates = []
        current_rate = base_rate
        
        for i, date in enumerate(dates):
            # Occasionally adjust rates (simulate Fed meetings)
            if i % 3 == 0 and np.random.random() < 0.3:  # 30% chance every 3 months
                adjustment = np.random.choice([-0.25, 0, 0.25], p=[0.3, 0.4, 0.3])
                current_rate += adjustment
                current_rate = max(0.0, min(10.0, current_rate))  # Keep within bounds
            
            # Add small random variation
            rate_with_noise = current_rate + np.random.normal(0, 0.05)
            interest_rates.append(max(0.0, min(10.0, rate_with_noise)))
        
        df = pd.DataFrame({
            'Interest_Rate': interest_rates
        }, index=dates)
        return df
    
    def generate_mock_debt_data(self, period: str = "1y") -> pd.DataFrame:
        """Generate mock US debt data"""
        period_months = self._period_to_months(period)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_months * 30)
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Ensure we have at least some data points
        if len(dates) == 0:
            dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='M')
        
        # Simulate more realistic debt growth (non-linear)
        base_debt = 31.5  # Trillion USD
        debt_values = []
        
        for i, date in enumerate(dates):
            # Non-linear growth with seasonal variations
            monthly_growth_rate = 0.003 + np.random.normal(0, 0.001)  # ~3.6% annual growth with variation
            if i == 0:
                current_debt = base_debt
            else:
                current_debt = debt_values[-1] * (1 + monthly_growth_rate)
            
            # Add some seasonal/policy-driven variations
            if i % 6 == 0:  # Every 6 months, add some policy impact
                policy_impact = np.random.normal(0, 0.2)
                current_debt += policy_impact
            
            debt_values.append(max(30.0, current_debt))  # Ensure debt doesn't go below 30T
        
        df = pd.DataFrame({
            'US_Debt_Trillion': debt_values
        }, index=dates)
        return df
    
    def _period_to_months(self, period: str) -> int:
        """Convert yfinance period string to number of months"""
        period_map = {
            "1mo": 1,
            "3mo": 3,
            "6mo": 6,
            "1y": 12,
            "2y": 24,
            "5y": 60
        }
        return period_map.get(period, 12)  # Default to 12 months

class EconomicDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.data_fetcher = EconomicDataFetcher()
        self.cached_data = {}
        self.last_update = None
        
    def get_data_with_cache(self, data_key: str, fetch_func, period: str = "1y") -> pd.DataFrame:
        """Get data with caching mechanism"""
        current_time = datetime.now()
        
        # Cache data for 15 minutes
        if (data_key not in self.cached_data or 
            self.last_update is None or 
            (current_time - self.last_update).seconds > 900):
            
            self.cached_data[data_key] = fetch_func(period)
            self.last_update = current_time
            
        return self.cached_data[data_key]
    
    def create_price_chart(self, data: pd.DataFrame, title: str, color: str = "blue") -> go.Figure:
        """Create a price/value chart"""
        if data.empty:
            return go.Figure().add_annotation(text="No data available", showarrow=False)
        
        fig = go.Figure()
        
        # Use 'Close' column if available, otherwise use the first numeric column
        if 'Close' in data.columns:
            y_values = data['Close']
            y_label = 'Price'
        else:
            y_values = data.iloc[:, 0]
            y_label = 'Value'
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=y_values,
            mode='lines',
            name=title,
            line=dict(color=color, width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=y_label,
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def create_indicator_summary(self, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create a summary chart with multiple indicators"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('S&P 500', 'Oil Prices', 'Inflation Rate', 'Unemployment Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # S&P 500
        if 'sp500' in data and not data['sp500'].empty:
            fig.add_trace(
                go.Scatter(x=data['sp500'].index, y=data['sp500']['Close'], 
                          name="S&P 500", line=dict(color='green')),
                row=1, col=1
            )
        
        # Oil Prices
        if 'oil' in data and not data['oil'].empty:
            fig.add_trace(
                go.Scatter(x=data['oil'].index, y=data['oil']['Close'], 
                          name="Oil", line=dict(color='black')),
                row=1, col=2
            )
        
        # Inflation Rate
        if 'inflation' in data and not data['inflation'].empty:
            fig.add_trace(
                go.Scatter(x=data['inflation'].index, y=data['inflation']['Inflation_Rate'], 
                          name="Inflation", line=dict(color='red')),
                row=2, col=1
            )
        
        # Unemployment Rate
        if 'unemployment' in data and not data['unemployment'].empty:
            fig.add_trace(
                go.Scatter(x=data['unemployment'].index, y=data['unemployment']['Unemployment_Rate'], 
                          name="Unemployment", line=dict(color='orange')),
                row=2, col=2
            )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Economic Indicators Overview"
        )
        
        return fig
    
    def calculate_statistics(self, data: pd.DataFrame, column: str = None) -> Dict:
        """Calculate basic statistics for a dataset"""
        if data.empty:
            return {}
        
        # Auto-detect the correct column if not specified
        if column is None or column not in data.columns:
            # For stock data, prefer 'Close', otherwise use first column
            if 'Close' in data.columns:
                column = 'Close'
            else:
                column = data.columns[0]
        
        values = data[column].dropna()
        
        return {
            'current': values.iloc[-1] if len(values) > 0 else None,
            'previous': values.iloc[-2] if len(values) > 1 else None,
            'change': values.iloc[-1] - values.iloc[-2] if len(values) > 1 else 0,
            'change_pct': ((values.iloc[-1] - values.iloc[-2]) / values.iloc[-2] * 100) if len(values) > 1 else 0,
            'min': values.min(),
            'max': values.max(),
            'mean': values.mean(),
            'std': values.std()
        }

# Initialize dashboard
@st.cache_data(ttl=900)  # Cache for 15 minutes
def load_dashboard():
    return EconomicDashboard()

def main():
    """Main application function"""
    st.title("📊 Economic Indicators Dashboard")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.title("Dashboard Controls")
    
    # Time period selection
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    }
    
    selected_period = st.sidebar.selectbox(
        "Select Time Period",
        options=list(period_options.keys()),
        index=3  # Default to 1 year
    )
    
    period = period_options[selected_period]
    
    # Indicator selection
    st.sidebar.subheader("Select Indicators")
    show_sp500 = st.sidebar.checkbox("S&P 500", value=True)
    show_nasdaq = st.sidebar.checkbox("NASDAQ", value=True)
    show_oil = st.sidebar.checkbox("Oil Prices", value=True)
    show_dollar = st.sidebar.checkbox("US Dollar Index", value=True)
    show_inflation = st.sidebar.checkbox("Inflation Rate", value=True)
    show_unemployment = st.sidebar.checkbox("Unemployment Rate", value=True)
    show_interest = st.sidebar.checkbox("Interest Rate", value=True)
    show_debt = st.sidebar.checkbox("US Debt", value=True)
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto Refresh (5 min)", value=False)
    
    if auto_refresh:
        time.sleep(300)  # 5 minutes
        st.rerun()
    
    # Load dashboard
    dashboard = load_dashboard()
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Data loading with progress
    with st.spinner("Loading economic data..."):
        data = {}
        
        if show_sp500:
            data['sp500'] = dashboard.get_data_with_cache(
                'sp500', dashboard.data_fetcher.fetch_sp500, period
            )
        
        if show_nasdaq:
            data['nasdaq'] = dashboard.get_data_with_cache(
                'nasdaq', dashboard.data_fetcher.fetch_nasdaq, period
            )
        
        if show_oil:
            data['oil'] = dashboard.get_data_with_cache(
                'oil', dashboard.data_fetcher.fetch_oil_prices, period
            )
        
        if show_dollar:
            data['dollar'] = dashboard.get_data_with_cache(
                'dollar', dashboard.data_fetcher.fetch_dollar_index, period
            )
        
        if show_inflation:
            data['inflation'] = dashboard.get_data_with_cache(
                'inflation', dashboard.data_fetcher.generate_mock_inflation_data, period
            )
            # Debug: Check inflation data
            if not data['inflation'].empty:
                st.sidebar.write(f"Inflation data shape: {data['inflation'].shape}")
                st.sidebar.write(f"Inflation columns: {list(data['inflation'].columns)}")
        
        if show_unemployment:
            data['unemployment'] = dashboard.get_data_with_cache(
                'unemployment', dashboard.data_fetcher.generate_mock_unemployment_data, period
            )
            # Debug: Check unemployment data
            if not data['unemployment'].empty:
                st.sidebar.write(f"Unemployment data shape: {data['unemployment'].shape}")
                st.sidebar.write(f"Unemployment columns: {list(data['unemployment'].columns)}")
        
        if show_interest:
            data['interest'] = dashboard.get_data_with_cache(
                'interest', dashboard.data_fetcher.generate_mock_interest_rate_data, period
            )
        
        if show_debt:
            data['debt'] = dashboard.get_data_with_cache(
                'debt', dashboard.data_fetcher.generate_mock_debt_data, period
            )
    
    # Display summary metrics
    st.subheader("📈 Key Metrics")
    
    cols = st.columns(4)
    
    # Calculate and display metrics
    metrics_data = []
    for key, df in data.items():
        if not df.empty:
            # Use appropriate column for statistics
            stats_column = None
            if 'Close' in df.columns:
                stats_column = 'Close'
            elif 'Inflation_Rate' in df.columns:
                stats_column = 'Inflation_Rate'
            elif 'Unemployment_Rate' in df.columns:
                stats_column = 'Unemployment_Rate'
            elif 'Interest_Rate' in df.columns:
                stats_column = 'Interest_Rate'
            elif 'US_Debt_Trillion' in df.columns:
                stats_column = 'US_Debt_Trillion'
            
            stats = dashboard.calculate_statistics(df, stats_column)
            metrics_data.append((key, stats))
    
    for i, (key, stats) in enumerate(metrics_data[:4]):
        with cols[i]:
            if stats and stats.get('current') is not None:
                current = stats['current']
                change_pct = stats.get('change_pct', 0)
                
                # Format based on data type
                if key in ['sp500', 'nasdaq', 'oil', 'dollar']:
                    value_str = f"${current:.2f}"
                elif key in ['inflation', 'unemployment', 'interest']:
                    value_str = f"{current:.2f}%"
                elif key == 'debt':
                    value_str = f"${current:.1f}T"
                else:
                    value_str = f"{current:.2f}"
                
                # Use "off" for very small changes, "normal" otherwise
                delta_color = "off" if abs(change_pct) < 0.01 else "normal"
                st.metric(
                    label=key.upper().replace('_', ' '),
                    value=value_str,
                    delta=f"{change_pct:.2f}%",
                    delta_color=delta_color
                )
    
    # Display overview chart if multiple indicators selected
    if len(data) >= 4:
        st.subheader("🔍 Multi-Indicator Overview")
        overview_data = {k: v for k, v in list(data.items())[:4]}
        overview_fig = dashboard.create_indicator_summary(overview_data)
        st.plotly_chart(overview_fig, use_container_width=True)
    
    # Individual charts
    st.subheader("📊 Individual Indicators")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    chart_configs = [
        ('sp500', "S&P 500 Index", "green", show_sp500),
        ('nasdaq', "NASDAQ Composite", "blue", show_nasdaq),
        ('oil', "Oil Prices (WTI Crude)", "black", show_oil),
        ('dollar', "US Dollar Index", "purple", show_dollar),
        ('inflation', "Inflation Rate", "red", show_inflation),
        ('unemployment', "Unemployment Rate", "orange", show_unemployment),
        ('interest', "Interest Rate", "brown", show_interest),
        ('debt', "US National Debt", "gray", show_debt)
    ]
    
    chart_count = 0
    for key, title, color, show in chart_configs:
        if show and key in data:
            with col1 if chart_count % 2 == 0 else col2:
                fig = dashboard.create_price_chart(data[key], title, color)
                st.plotly_chart(fig, use_container_width=True)
            chart_count += 1
    
    # Data table section
    with st.expander("📋 Raw Data"):
        selected_indicator = st.selectbox(
            "Select indicator to view data",
            options=list(data.keys())
        )
        
        if selected_indicator in data and not data[selected_indicator].empty:
            st.dataframe(data[selected_indicator].tail(20), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
        <p>📊 Economic Indicators Dashboard | Data sources: Yahoo Finance, Mock Data (for rates)</p>
        <p><small>⚠️ Note: Some indicators use simulated data. For production use, integrate with FRED API for real economic data.</small></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()