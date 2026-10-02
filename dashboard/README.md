# Buyer Segmentation & Investment Profiling Dashboard

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads `clients_segmented.csv` and `model_summary.json` from the same
folder (both produced by `pipeline.py` in the project root — copy fresh
versions here if you re-run the pipeline).

## Dashboard modules
1. **Buyer Segmentation Overview** — cluster distribution, elbow/silhouette validation
2. **Investor Behavior Dashboard** — spend, price, financing patterns by segment
3. **Geographic Buyer Analysis** — country/region breakdowns, choropleth map
4. **Segment Insights Panel** — descriptive stats per cluster, raw data explorer, CSV export

Filters (sidebar): country, region, acquisition purpose, client type, segment.
