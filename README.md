# Fartcoin Alpha Framework

Quantitative analysis framework for detecting market maker manipulation patterns
in Fartcoin using spot/perps divergence signals.

## Setup
```bash
pip install -r requirements.txt
```

## Usage
1. Set your CoinMarketCap API key: `export CMC_API_KEY=your_key`
2. Run data collection: `python data_collector.py`
3. Run signal analysis: `python signal_engine.py`
4. Run backtest: `python backtest.py`
