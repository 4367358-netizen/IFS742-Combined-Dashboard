# George Market Forecast Lab — five-model comparison

Streamlit dashboard comparing **Random Forest, XGBoost, SVR, LSTM and ANN** on saved historical one-month-ahead price predictions for potatoes, brown onions and tomatoes.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data sources
- Existing four-model `combined_predictions.csv` and `combined_metrics.csv` from the previous dashboard package.
- ANN full-precision prediction CSV embedded as a base64 data-download in `GeorgeMarketANNDashboard.html`, cross-checked against `IFS742_ANN_Report.docx`.
- Historical monthly prices and original LSTM run metadata are retained from the previous package.

## Interpretation
- All five models have **four common held-out test months (May–August 2026)** per commodity.
- ANN uses **24 historical months**, starting September 2024, yielding **13 training targets**. The other model datasets use 25 months starting August 2024, yielding 14 training targets.
- ANN is trained on the original training period only and was not refitted on validation data; Random Forest was reportedly refitted on training plus validation. Selection methods differ.
- Last-month baseline is shared and appears once.
- This is a **saved-results evaluation dashboard**, not a live or future-price prediction service.
- Only four test months per commodity; do not interpret tiny differences as statistically meaningful.
