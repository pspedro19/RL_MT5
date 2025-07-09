# RL_SP500 - Reinforcement Learning for S&P 500 Trading

A comprehensive reinforcement learning framework for algorithmic trading on the S&P 500 index, featuring advanced data capture, validation, and machine learning pipelines.

## 🚀 Features

- **Multi-source Data Capture**: Real-time and historical data from MT5 and other sources
- **Advanced Data Validation**: Comprehensive quality checks and market hours filtering
- **Reinforcement Learning Pipeline**: Complete RL framework for trading strategy development
- **Granular Traceability**: Detailed data lineage and quality scoring
- **Market Hours Compliance**: Automatic filtering of non-market hours data
- **Quality Assurance**: Multi-level validation and reporting systems
- **Aggregated Quality Report**: Merges capture and validation metrics

## 📁 Project Structure

```
RL_SP500/
├── pipelines/
│   ├── 01_capture/          # Data capture pipeline
│   │   ├── config/          # Configuration files
│   │   ├── data/           # Data connectors and processors
│   │   ├── features/       # Technical indicators
│   │   ├── utils/          # Utility functions
│   │   ├── main.py         # Main capture script
│   │   └── run.py          # Standalone capture entry
│   ├── 02_validate/        # Data validation pipeline
│   │   ├── analysis/       # Data analysis tools
│   │   ├── preprocessing/  # Data cleaning and normalization
│   │   ├── validation/     # Validation modules
│   │   ├── main.py         # Main validation script
│   │   └── run.py          # Standalone validation entry
│   └── requirements.txt    # Python dependencies
├── data/                   # Data storage
│   ├── validated/         # Validated datasets
│   └── ...               # Raw and processed data
├── logs/                  # Execution logs
├── run.py                 # Main execution script
└── README.md             # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- MetaTrader 5 (for live data capture)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/pspedro19/RL_MT5.git
   cd RL_MT5
   ```

2. **Install dependencies**
   ```bash
   pip install -r pipelines/requirements.txt
   ```

3. **Configure MT5 connection** (if using live data)
   - Update MT5 credentials in `pipelines/01_capture/config/settings.py`
   - Ensure MT5 terminal is running

4. **Obtain datasets**
   - Large CSV and Parquet files are stored via [Git LFS](https://git-lfs.com/).
   - Download them with `git lfs pull` or reproduce them using the capture pipeline:

   ```bash
   cd pipelines/01_capture
   python main.py --instrument US500 --start_date 2020-01-01 --end_date 2025-01-01
   ```

## 🚀 Quick Start

### 1. Data Capture

Capture historical or real-time data:

```bash
cd pipelines/01_capture
python main.py --instrument US500 --start_date 2020-01-01 --end_date 2025-01-01
```

### 2. Data Validation

Validate and clean captured data:

```bash
cd pipelines/02_validate
python main.py --input ../data/us500_m5_hpc_2020_2025.parquet
```

### 3. Run Complete Pipeline

Execute the full pipeline from the root directory:

```bash
python run.py --pipeline full --instrument US500
```

### Script Entry Points

The repository contains several run scripts:

- **`run.py` (root)** – orchestrates the full workflow (capture + validation).
  Use this as the default entry point.
- **`pipelines/01_capture/run.py`** – runs only the capture phase for quick
  data downloads or debugging.
- **`pipelines/02_validate/run.py`** – validates an existing dataset without
  executing the capture step.

## 📊 Data Quality Features

### Market Hours Filtering
- Automatically filters data to market hours only
- Supports multiple instruments (US500, USDCOP, etc.)
- Configurable market calendars

### Granular Traceability
- Tracks data origin (native, aggregated, imputed)
- Quality scoring for each data point
- Detailed lineage tracking

### Quality Validation
- Multi-level quality checks
- Gap detection and reporting
- Statistical analysis and reporting

## 🔧 Configuration

### Pipeline Configuration
- `pipelines/01_capture/config/constants.py`: Data sources and constants
- `pipelines/01_capture/config/settings.py`: MT5 and general settings
- `pipelines/02_validate/config.py`: Validation parameters

### Data Sources
The system supports multiple data sources:
- **MT5 Live**: Real-time data from MetaTrader 5
- **MT5 Historical**: Historical data from MT5
- **External APIs**: Additional data sources
- **Aggregated Data**: Multi-source data combination

## 📈 Usage Examples

### Capture US500 Data
```python
from pipelines.capture.main import capture_pipeline

# Capture 5-minute data for US500
result = capture_pipeline(
    instrument='US500',
    timeframe='M5',
    start_date='2020-01-01',
    end_date='2025-01-01'
)
```

### Validate Dataset
```python
from pipelines.validate.main import validate_pipeline

# Validate captured data
validation = validate_pipeline(
    input_file='data/us500_m5_hpc_2020_2025.parquet',
    output_dir='data/validated/'
)
```

### Apply Market Hours Filter
```python
from pipelines.capture.utils.market_hours_filter import apply_market_hours_filter

# Filter data to market hours only
filtered_data, analysis = apply_market_hours_filter(df, 'US500')
```

## 📋 Requirements

### Core Dependencies
- `pandas>=1.5.0`
- `numpy>=1.21.0`
- `MetaTrader5>=5.0.0`
- `polars>=0.19.0`
- `pyarrow>=10.0.0`

### Optional Dependencies
- `ray>=2.0.0` (for distributed processing)
- `plotly>=5.0.0` (for visualization)
- `scikit-learn>=1.1.0` (for ML features)

## 🧪 Running Tests

After installing the required dependencies, you can run the included test suites
to verify that the environment is correctly configured:

```bash
python test_pipeline.py
python test_enhanced_quality.py
```

Both scripts exit with a non-zero status if mandatory files are missing or any
core functions fail during execution.

## 🔍 Monitoring and Logging

### Log Files
- `logs/capture_run.log`: Data capture execution logs
- `logs/pipeline.log`: Pipeline execution logs
- `logs/validation.log`: Validation process logs

### Reports
- Quality analysis reports in `data/validated/`
- Execution reports with timestamps
- Gap detection and statistical summaries
- Final aggregated quality report in `data/validated/final_quality_report.json`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation in the `docs/` folder
- Review the execution logs for troubleshooting

## 🔄 Version History

- **v3.0**: Enhanced quality tracking and market hours filtering
- **v2.0**: Added validation pipeline and granular traceability
- **v1.0**: Initial data capture pipeline

---
**Note**: This project is designed for educational and research purposes. Always test thoroughly before using in live trading environments. 