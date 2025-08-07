# 📊 Economic Indicators Dashboard

A comprehensive, real-time economic indicators dashboard built with Python and Streamlit. This tool provides interactive visualizations and analysis of key economic metrics including stock indices, commodity prices, and macroeconomic indicators.

## 🎯 Features

### Core Indicators
- **Stock Markets**: S&P 500, NASDAQ Composite
- **Commodities**: Oil (WTI Crude), Gold
- **Currency**: US Dollar Index
- **Economic Data**: Inflation Rate, Unemployment Rate, Interest Rates, US National Debt
- **Market Sentiment**: VIX (Fear Index), 10-Year Treasury Yield
- **Cryptocurrency**: Bitcoin (optional)

### Advanced Features
- 📈 **Interactive Charts**: Plotly-based interactive visualizations
- 🔄 **Real-time Data**: Auto-refresh capabilities with caching
- 📊 **Technical Analysis**: Moving averages, RSI, MACD, Bollinger Bands
- 🎛️ **Customizable Views**: Select indicators and time periods
- 📋 **Data Export**: Raw data viewing and export capabilities
- 🌐 **Multi-source Data**: Yahoo Finance, FRED API integration
- 📱 **Responsive Design**: Works on desktop and mobile devices

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/economic-dashboard.git
   cd economic-dashboard
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

### Advanced Setup with FRED API

For real economic data, obtain a free API key from [FRED (Federal Reserve Economic Data)](https://fred.stlouisfed.org/docs/api/api_key.html):

1. **Set your FRED API key**:
   ```bash
   export FRED_API_KEY="your_api_key_here"
   ```

2. **Or create a `.env` file**:
   ```env
   FRED_API_KEY=your_api_key_here
   ```

## 📊 Data Sources

### Primary Sources
1. **Yahoo Finance** (via `yfinance`)
   - Stock indices (S&P 500, NASDAQ)
   - Commodities (Oil, Gold)
   - Currency indices (USD Index)
   - Treasury yields
   - VIX

2. **FRED API** (Federal Reserve Economic Data)
   - Consumer Price Index (Inflation)
   - Unemployment Rate
   - Federal Funds Rate
   - GDP data
   - Government debt statistics

### Data Quality & Reliability
- **Update Frequency**: 
  - Stock data: Real-time (15-20 min delay)
  - Economic indicators: Monthly/Quarterly (official releases)
- **Historical Data**: Up to 10+ years available
- **Data Validation**: Built-in error handling and data quality checks

## 🎨 Dashboard Components

### 1. Overview Metrics
- Key performance indicators with percentage changes
- Color-coded trend indicators
- Current values with historical context

### 2. Multi-Indicator Charts
- Synchronized time series plots
- Comparative analysis tools
- Correlation visualizations

### 3. Individual Indicator Analysis
- Detailed charts with technical indicators
- Zoom and pan functionality
- Custom time period selection

### 4. Raw Data Access
- Exportable data tables
- CSV download capabilities
- API endpoint documentation

## 🔧 Configuration

### Time Periods
- 1 Month, 3 Months, 6 Months
- 1 Year, 2 Years, 5 Years
- Custom date ranges (advanced)

### Indicator Selection
Toggle any combination of indicators:
- Market indices
- Commodities
- Economic indicators
- Currency metrics

### Display Options
- Auto-refresh intervals
- Chart themes and colors
- Data aggregation levels

## 🚀 Deployment Options

### 1. Local Development
```bash
streamlit run dashboard.py
```

### 2. Docker
```bash
docker-compose up -d
```

### 3. Cloud Platforms

#### Streamlit Community Cloud
1. Fork this repository
2. Connect to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Deploy with one click

#### Heroku
```bash
git push heroku main
```

#### Google Cloud Platform
```bash
gcloud app deploy
```

#### AWS EC2
Use the provided Docker configuration or install directly on an EC2 instance.

#### Railway/Render
Single-click deployment with the provided configuration files.

## 📈 Technical Analysis Features

### Indicators Available
- **Trend Following**: SMA (20, 50, 200), EMA (12, 26)
- **Momentum**: RSI (14-period), MACD
- **Volatility**: Bollinger Bands
- **Volume**: Volume Moving Average, Volume Ratio

### Market Sentiment Score
Proprietary algorithm combining multiple indicators to generate a sentiment score (-1 to +1):
- Market indices performance
- Volatility measures (VIX)
- Safe haven assets (Gold, Treasuries)
- Currency strength

## 🔐 Security & Privacy

### Data Security
- No sensitive data storage
- API keys handled securely via environment variables
- HTTPS enforced in production deployments

### Privacy
- No user data collection
- No tracking or analytics by default
- Open source and transparent

## 🧪 Testing

### Run Tests
```bash
pytest tests/
```

### Test Coverage
```bash
pytest --cov=dashboard
```

### Load Testing
```bash
python tests/load_test.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📋 API Reference

### Core Classes

#### `EconomicDataFetcher`
Main class for data retrieval:
```python
fetcher = EconomicDataFetcher(fred_api_key="your_key")
data = fetcher.fetch_economic_indicator("sp500", period="1y")
```

#### `EconomicDashboard`
Dashboard controller class:
```python
dashboard = EconomicDashboard()
fig = dashboard.create_price_chart(data, "S&P 500", "green")
```

### Available Indicators
| Indicator | Symbol | Source | Frequency |
|-----------|--------|--------|-----------|
| S&P 500 | ^GSPC | Yahoo | Daily |
| NASDAQ | ^IXIC | Yahoo | Daily |
| Oil (WTI) | CL=F | Yahoo | Daily |
| USD Index | DX-Y.NYB | Yahoo | Daily |
| Inflation | CPIAUCSL | FRED | Monthly |
| Unemployment | UNRATE | FRED | Monthly |
| Fed Funds Rate | FEDFUNDS | FRED | Monthly |
| US Debt/GDP | GFDEGDQ188S | FRED | Quarterly |

## 🐛 Troubleshooting

### Common Issues

#### 1. No data loading
- Check internet connection
- Verify API keys (for FRED data)
- Check Yahoo Finance symbol accuracy

#### 2. Slow performance
- Reduce time period range
- Limit number of indicators
- Clear cache and refresh

#### 3. Deployment issues
- Check Python version compatibility
- Verify all dependencies installed
- Check port availability

### Debug Mode
Run with verbose logging:
```bash
streamlit run dashboard.py --logger.level=debug
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing framework
- [Plotly](https://plotly.com/) for interactive visualizations
- [Yahoo Finance](https://finance.yahoo.com/) for market data
- [FRED](https://fred.stlouisfed.org/) for economic data
- [yfinance](https://github.com/ranaroussi/yfinance) for the Python API

## 📞 Support

- 📧 Email: support@yourdomain.com
- 💬 Discord: [Join our community](https://discord.gg/your-invite)
- 📖 Documentation: [Full docs](https://yourdomain.com/docs)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/economic-dashboard/issues)

## 🗓️ Roadmap

### Version 2.0 (Planned)
- [ ] Machine learning predictions
- [ ] Custom alert system
- [ ] Portfolio tracking integration
- [ ] Mobile app version

### Version 1.5 (In Development)
- [ ] Additional international markets
- [ ] Cryptocurrency integration
- [ ] Advanced charting tools
- [ ] Export to PDF reports

---

**Made with ❤️ and Python**

*Last updated: January 2025*
