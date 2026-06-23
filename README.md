# Credit Risk Intelligence

A production-style credit default prediction system built on a multi-schema PostgreSQL data warehouse integrating four public government datasets. Demonstrates end-to-end ML pipeline development including data ingestion, EDA, feature engineering, modeling, drift monitoring, and fairness auditing.

---

## Results

| Metric | Value |
|---|---|
| AUROC | 0.9699 |
| AUPRC | 0.5971 (random baseline: 0.0155) |
| Recall @ threshold 0.49 | 82% of defaults detected |
| Data drift (Evidently AI) | 0 / 18 features drifted |
| Max geographic FPR gap | 12x (NY 9.1% vs MT 0.76%) |
| Fairness mitigation | Threshold 0.58 → near-zero FPR parity |

---

## Architecture

4 Government Data Sources → PostgreSQL (4 schemas) → Feature Engineering → LightGBM → MLflow → Evidently AI → Fairness Audit

### Data Sources
| Schema | Source | Rows | Description |
|---|---|---|---|
| `sba` | SBA 7(a) FOIA | 373,981 | Loan-level data with default outcomes |
| `fred` | Federal Reserve (FRED) | 11,734 | GDP growth, fed funds rate, BAA credit spread |
| `bls` | Bureau of Labor Statistics | 940 | National monthly unemployment rate |
| `fdic` | FDIC BankFind | 27,836 | Lender asset size and status |

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_ingest.ipynb` | Downloads all 4 datasets, loads into PostgreSQL across separate schemas |
| `02_eda.ipynb` | Full EDA on 373K loans — default rate analysis, segment disparities, vintage effects |
| `03_features.ipynb` | Cross-schema SQL joins, feature engineering, missingness analysis, Parquet export |
| `04_model.ipynb` | LightGBM classifier with MLflow experiment tracking, AUROC/AUPRC evaluation |
| `05_monitoring.ipynb` | Evidently AI data drift detection simulating production monitoring workflow |
| `06_fairness.ipynb` | FPR disparity audit across state, industry, and business age segments with threshold mitigation |

---

## Key Findings

**Modeling**
- `term_years` is the dominant feature (5x gain over next feature) — defaulted loans average 83-month terms vs 139 months for non-defaults
- Macro features (fed funds rate, BAA credit spread, unemployment) all contribute meaningfully — cross-schema joins add predictive value beyond loan-level data alone
- COVID-era loans (FY2020, FY2022) show elevated default rates — vintage effect captured by `approvalfy`

**Fairness**
- Largest disparity is geographic: NY FPR 9.1% vs MT 0.76% (12x gap) — reflects structural default rate differences, not model bias
- New businesses flagged at 26% higher FPR than established businesses
- Threshold adjustment (0.49 → 0.58 for new businesses) achieves near-zero FPR parity at +0.8pp FNR cost
- Finding consistent with DSA5900 capstone: fairness improvements do not require accuracy sacrifice

---

## Tech Stack

- **Database:** PostgreSQL 18 with 4 schemas (sba, fred, bls, fdic)
- **ML:** LightGBM 4.6, scikit-learn
- **Tracking:** MLflow 3.14 (SQLite backend)
- **Monitoring:** Evidently AI 0.7
- **Data:** pandas, SQLAlchemy, psycopg2
- **Visualization:** matplotlib, seaborn

---

## Reproduce

```bash
# 1. Clone and create environment
git clone https://github.com/enny223/credit-risk-intelligence.git
cd credit-risk-intelligence
conda create -n credit-risk python=3.11 -y
conda activate credit-risk
pip install -r requirements.txt

# 2. Set up PostgreSQL
# Create database and schemas (requires PostgreSQL 14+)
psql -U postgres -c "CREATE DATABASE credit_risk_db;"
psql -U postgres -d credit_risk_db -c "CREATE SCHEMA sba; CREATE SCHEMA fred; CREATE SCHEMA bls; CREATE SCHEMA fdic;"

# 3. Add credentials
echo "DB_PASSWORD=your_password" > .env

# 4. Run notebooks in order
# 01_ingest → 02_eda → 03_features → 04_model → 05_monitoring → 06_fairness
```

---

## Author

Enoch Oseibonsu — M.S. Data Science & Analytics, University of Oklahoma (2026)  
U.S. Army Veteran (E-5/Sergeant) | Oklahoma City, OK  
[LinkedIn](https://linkedin.com/in/enoch-oseibonsu-6b1baa202) · [GitHub](https://github.com/enny223)
