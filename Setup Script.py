#!/usr/bin/env python3
"""
Setup script and usage examples for the Economic Dashboard
This script helps users set up the dashboard and provides usage examples
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import argparse

def check_python_version():
    """Check if Python version is compatible"""
    min_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < min_version:
        print(f"❌ Python {min_version[0]}.{min_version[1]}+ required. You have {current_version[0]}.{current_version[1]}")
        return False
    
    print(f"✅ Python version {current_version[0]}.{current_version[1]} is compatible")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def setup_environment():
    """Set up environment variables and configuration"""
    print("🔧 Setting up environment...")
    
    # Create .env file template if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_template = """# Economic Dashboard Environment Variables

# FRED API Key (get from https://fred.stlouisfed.org/docs/api/api_key.html)
FRED_API_KEY=your_fred_api_key_here

# Optional API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWS_API_KEY=your_news_api_key

# Dashboard Settings
UI_THEME=light
DEFAULT_PERIOD=1y
ENABLE_CACHE=true
CACHE_DURATION_MINUTES=15

# Data Settings
PRIMARY_DATA_SOURCE=yahoo
DATA_QUALITY_CHECKS=true
"""
        
        with open(env_file, "w") as f:
            f.write(env_template)
        
        print(f"✅ Created .env template file: {env_file}")
    
    # Create cache directory
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    print(f"✅ Created cache directory: {cache_dir}")
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    print(f"✅ Created logs directory: {logs_dir}")

def create_sample_config():
    """Create a sample configuration file"""
    config_file = Path("config.json")
    
    if config_file.exists():
        print("⚠️  Configuration file already exists, skipping...")
        return
    
    sample_config = {
        "api": {
            "fred_api_key": "",
            "alpha_vantage_api_key": "",
            "news_api_key": ""
        },
        "ui": {
            "theme": "light",
            "default_period": "1y",
            "default_indicators": ["sp500", "nasdaq", "oil", "inflation"],
            "auto_refresh": False,
            "refresh_interval_minutes": 5
        },
        "cache": {
            "enable_cache": True,
            "cache_duration_minutes": 15,
            "max_cache_size_mb": 100
        },
        "charts": {
            "color_scheme": "plotly",
            "chart_height": 400,
            "show_volume": True,
            "technical_indicators": ["sma", "rsi", "macd"]
        },
        "data": {
            "primary_source": "yahoo",
            "backup_source": "mock",
            "data_quality_checks": True
        }
    }
    
    with open(config_file, "w") as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"✅ Created sample configuration file: {config_file}")

def run_tests():
    """Run the test suite"""
    print("🧪 Running tests...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pytest", "test_dashboard.py", "-v"])
        print("✅ All tests passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Some tests failed: {e}")
        return False
    except FileNotFoundError:
        print("⚠️  pytest not found, skipping tests")
        return True

def launch_dashboard():
    """Launch the dashboard"""
    print("🚀 Launching Economic Dashboard...")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def show_usage_examples():
    """Show usage examples"""
    examples = """
📊 Economic Dashboard Usage Examples

1. Basic Usage:
   python dashboard.py
   # Launches the dashboard on http://localhost:8501

2. Custom Configuration:
   export FRED_API_KEY="your_api_key"
   export UI_THEME="dark"
   python dashboard.py

3. Docker Usage:
   docker-compose up -d
   # Launches dashboard in container

4. Production Deployment:
   # Heroku
   git push heroku main
   
   # Google Cloud
   gcloud app deploy
   
   # AWS EC2
   python dashboard.py --server.port=80

5. API Integration Examples:

   # Using the data fetcher programmatically
   from dashboard import EconomicDataFetcher
   
   fetcher = EconomicDataFetcher()
   sp500_data = fetcher.fetch_sp500("1y")
   print(sp500_data.tail())

6. Custom Indicators:

   # Add custom indicators to config.json
   {
     "custom_indicators": {
       "tesla": {
         "name": "Tesla Stock",
         "symbol": "TSLA",
         "source": "yahoo"
       }
     }
   }

7. Advanced Visualization:

   from visualizations import AdvancedVisualizations
   
   viz = AdvancedVisualizations()
   correlation_chart = viz.create_correlation_heatmap(data)

8. Performance Monitoring:

   # Enable debug mode
   export LOG_LEVEL=DEBUG
   python dashboard.py

9. Data Export:

   # Export data to CSV
   data.to_csv('economic_data.csv')
   
   # Generate PDF report
   # (requires additional setup)

10. Alert System Setup:

    # Configure alerts in config.json
    {
      "alerts": {
        "sp500_threshold": 5.0,
        "inflation_threshold": 4.0,
        "email_notifications": true
      }
    }
"""
    
    print(examples)

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    requirements_ok = True
    
    # Check Python version
    if not check_python_version():
        requirements_ok = False
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 2:
            print(f"⚠️  Low memory detected: {memory_gb:.1f}GB (recommended: 2GB+)")
        else:
            print(f"✅ Memory: {memory_gb:.1f}GB")
    except ImportError:
        print("⚠️  Could not check memory (psutil not installed)")
    
    # Check disk space
    try:
        import shutil
        free_gb = shutil.disk_usage(".").free / (1024**3)
        if free_gb < 1:
            print(f"⚠️  Low disk space: {free_gb:.1f}GB (recommended: 1GB+)")
            requirements_ok = False
        else:
            print(f"✅ Free disk space: {free_gb:.1f}GB")
    except Exception:
        print("⚠️  Could not check disk space")
    
    # Check internet connection
    try:
        import urllib.request
        urllib.request.urlopen('https://finance.yahoo.com', timeout=5)
        print("✅ Internet connection available")
    except Exception:
        print("⚠️  Internet connection required for live data")
    
    return requirements_ok

def setup_fred_api():
    """Interactive setup for FRED API"""
    print("\n🏦 FRED API Setup")
    print("The Federal Reserve Economic Data (FRED) API provides access to economic indicators.")
    print("Visit: https://fred.stlouisfed.org/docs/api/api_key.html")
    
    api_key = input("Enter your FRED API key (or press Enter to skip): ").strip()
    
    if api_key:
        # Update .env file
        env_file = Path(".env")
        if env_file.exists():
            content = env_file.read_text()
            content = content.replace("FRED_API_KEY=your_fred_api_key_here", f"FRED_API_KEY={api_key}")
            env_file.write_text(content)
            print("✅ FRED API key saved to .env file")
        else:
            with open(env_file, "w") as f:
                f.write(f"FRED_API_KEY={api_key}\n")
            print("✅ Created .env file with FRED API key")
    else:
        print("⚠️  Skipped FRED API setup - will use mock data for economic indicators")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Economic Dashboard Setup")
    parser.add_argument("--quick", action="store_true", help="Quick setup without interactive prompts")
    parser.add_argument("--examples", action="store_true", help="Show usage examples")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--launch", action="store_true", help="Launch dashboard after setup")
    
    args = parser.parse_args()
    
    if args.examples:
        show_usage_examples()
        return
    
    print("🏗️  Economic Dashboard Setup")
    print("=" * 50)
    
    # Check system requirements
    if not check_system_requirements():
        print("\n❌ System requirements not met. Please address the issues above.")
        return
    
    # Setup steps
    setup_steps = [
        ("📦 Installing packages", install_requirements),
        ("🔧 Setting up environment", setup_environment),
        ("⚙️  Creating configuration", create_sample_config),
    ]
    
    for step_name, step_func in setup_steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"❌ Setup failed at step: {step_name}")
            return
    
    # Interactive API setup
    if not args.quick:
        setup_fred_api()
    
    # Run tests if requested
    if args.test:
        if not run_tests():
            print("⚠️  Some tests failed, but setup is complete")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file to add your API keys")
    print("2. Customize config.json if needed") 
    print("3. Run: python dashboard.py")
    print("4. Open: http://localhost:8501")
    
    # Launch if requested
    if args.launch:
        print("\n🚀 Launching dashboard...")
        launch_dashboard()

if __name__ == "__main__":
    main()

# Additional utility functions for advanced users

class DashboardUtils:
    """Utility class for dashboard operations"""
    
    @staticmethod
    def export_data(data_dict, format="csv", filename="economic_data"):
        """Export dashboard data to various formats"""
        if format.lower() == "csv":
            for indicator, df in data_dict.items():
                df.to_csv(f"{filename}_{indicator}.csv")
                print(f"Exported {indicator} data to {filename}_{indicator}.csv")
        
        elif format.lower() == "json":
            # Convert DataFrames to JSON-serializable format
            export_data = {}
            for indicator, df in data_dict.items():
                export_data[indicator] = {
                    'data': df.reset_index().to_dict('records'),
                    'columns': df.columns.tolist(),
                    'index_name': df.index.name
                }
            
            with open(f"{filename}.json", "w") as f:
                json.dump(export_data, f, indent=2, default=str)
            print(f"Exported all data to {filename}.json")
    
    @staticmethod
    def generate_report(data_dict, template="basic"):
        """Generate automated reports"""
        report = f"""
# Economic Dashboard Report
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
"""
        
        for indicator, df in data_dict.items():
            if not df.empty:
                if 'Close' in df.columns:
                    current = df['Close'].iloc[-1]
                    change = df['Close'].pct_change().iloc[-1] * 100
                else:
                    current = df.iloc[-1, 0]
                    change = df.iloc[:, 0].pct_change().iloc[-1] * 100
                
                report += f"\n### {indicator.upper()}\n"
                report += f"Current Value: {current:.2f}\n"
                report += f"Daily Change: {change:+.2f}%\n"
        
        with open("economic_report.md", "w") as f:
            f.write(report)
        
        print("Report generated: economic_report.md")
    
    @staticmethod
    def backup_config():
        """Backup current configuration"""
        import shutil
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        files_to_backup = ["config.json", ".env"]
        
        for file in files_to_backup:
            if os.path.exists(file):
                backup_name = f"{file}.backup_{timestamp}"
                shutil.copy2(file, backup_name)
                print(f"Backed up {file} to {backup_name}")

# Quick start function
def quick_start():
    """Quick start function for immediate use"""
    print("🚀 Economic Dashboard Quick Start")
    
    if not check_python_version():
        return
    
    print("Installing minimal requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "streamlit", "pandas", "numpy", "plotly", "yfinance"
        ])
        print("✅ Minimal setup complete!")
        print("Starting dashboard...")
        subprocess.run([sys.executable, "-c", """
import streamlit as st
st.title('Economic Dashboard - Quick Start')
st.success('Dashboard is running! Use the full setup for complete features.')
st.info('Run setup.py for full installation with all features.')
"""])
    except Exception as e:
        print(f"Error in quick start: {e}")

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "quick":
    quick_start()
elif __name__ == "__main__":
    main()