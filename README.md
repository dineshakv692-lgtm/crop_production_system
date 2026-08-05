 AI Crop Production Prediction System

<div align="center">

![Crop Production Banner](https://img.shields.io/badge/AI-Crop%20Intelligence-10b981?style=for-the-badge&logo=leaflet&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An intelligent agricultural forecasting platform powered by Ensemble Machine Learning**  
*Predict crop yields, analyze production trends, and optimize farming decisions across India*

[Live Demo](#running-locally) · [Features](#features) · [Installation](#installation) · [Project Structure](#project-structure)

</div>

- Overview

The **AI Crop Production Prediction System** is a full-stack machine learning web application that leverages a **Super-Enhanced ExtraTreesRegressor model** (500 estimators, R² = 99.85%) trained on 23 years of Indian agricultural data (1997–2020). It enables farmers, agri-economists, researchers, and government planners to:

-  **Predict** crop production in Metric Tonnes for any combination of crop, state, season, and farm inputs
-  **Explore** historical data with interactive Plotly visualizations
-  **Compare** performance across crops and Indian states
-  **Understand** model internals through feature importance and performance metrics

---
 Features

 Home & Overview
- Animated glassmorphism dashboard with AI-generated hero banner
- Key platform stats: 19,689 records, 55 crops, 30 states
- Top 10 producing states horizontal bar chart

 Crop Yield Predictor
- **14-feature AI pipeline** for precise production forecasting
- **4 Quick Scenario Presets**: Punjab Rice, UP Wheat, MH Sugarcane, Kerala Coconut
- Yield Benchmark Rating (High / Moderate / Low output)
- Agronomic Insights with fertilizer density recommendations
- **Download CSV** prediction summary reports

Analytics & Data Insights
| Tab | Content |
|---|---|
| State & Crop Distribution | Donut + Bar charts for top producers |
|  Seasonal Trends | Season-wise & year-wise production lines |
|  Inputs vs Production | Correlation heatmap of key agricultural parameters |
|  Raw Data Explorer | Filterable, searchable dataset with 19,689 records |

Crop & State Comparison
- Side-by-side comparison of any two crops or states
- Quick stats: Total production, avg area, rainfall, fertilizer usage

 Model Performance
- Live feature importance bar chart (14 engineered features)
- Training metrics: R², MAE, RMSE vs baseline comparison

---

 Model Architecture

| Property | Value |
|---|---|
| **Algorithm** | ExtraTreesRegressor |
| **Estimators** | 500 |
| **R² Score** | **99.85%** |
| **MAE** | 520,381 tonnes |
| **MAE Improvement** | 35% reduction vs baseline |
| **Training Data** | 19,689 records (1997–2020) |
| **Target Variable** | Crop Production (Metric Tonnes) |

Engineered Features (14 Total)

| Feature | Description |
|---|---|
| `Crop_Enc` | Label-encoded crop type |
| `Crop_Year` | Year of cultivation |
| `Season_Enc` | Encoded season (Kharif, Rabi, etc.) |
| `State_Enc` | Encoded Indian state |
| `Area` | Cultivated area in hectares |
| `Annual_Rainfall` | Annual rainfall in mm |
| `Fertilizer` | Total fertilizer usage (kg) |
| `Pesticide` | Total pesticide usage (kg) |
| `Fert_Per_Area` | Fertilizer per hectare (engineered) |
| `Pest_Per_Area` | Pesticide per hectare (engineered) |
| `Rainfall_Per_Area` | Rainfall per hectare (engineered) |
| `Crop_Avg_Yield` | Historical avg yield for the crop (engineered) |
| `State_Avg_Yield` | Historical avg yield for the state (engineered) |
| `Expected_Yield_Prod` | `Crop_Avg_Yield × Area` (engineered) |

---

Project Structure

```
Crop Production Prediction System/
│
├── app.py                         Main Streamlit application
├── app1.py                        Alternate entry point
│
├── crop_yield_processed.csv       Processed dataset (19,689 records)
├── crop_production_model.pkl      Trained ExtraTrees model (500 estimators)
├── feature_meta.pkl              Feature metadata & yield maps
│
├── crop_encoder.pkl               LabelEncoder – Crop (55 classes)
├── season_encoder.pkl             LabelEncoder – Season (6 classes)
├── state_encoder.pkl              LabelEncoder – State (30 classes)
│
├── hero_banner.jpg                AI-generated hero banner graphic
├── farm_analytics.jpg             AI-generated analytics section banner
│
├── 04_Model_Training.ipynb        Model training & evaluation notebook
├── app.ipynb                     Data exploration notebook
│
└── README.md                      This file
```

---

 Installation

Prerequisites
- Python 3.8+ (Anaconda recommended)
- pip or conda package manager

1. Clone the Repository
```bash
git clone https://github.com/yourusername/crop-production-prediction.git
cd crop-production-prediction
```

 2. Install Dependencies
```bash
pip install streamlit pandas numpy scikit-learn joblib plotly pillow
```

Or with conda:
```bash
conda install streamlit pandas numpy scikit-learn joblib plotly pillow -c conda-forge
```
 3. Run the Application
```bash
streamlit run app.py
```

The app will launch at **http://localhost:8501** 

---

Running Locally

```bash
# Clone repo
git clone https://github.com/yourusername/crop-production-prediction.git
cd crop-production-prediction

# Install requirements
pip install -r requirements.txt

# Launch Streamlit dashboard
streamlit run app.py --server.port 8501
```

> **Note**: Ensure `crop_production_model.pkl`, `feature_meta.pkl`, and all encoder `.pkl` files are in the same directory as `app.py`.

---

 Requirements

Create a `requirements.txt` with:

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.3.0
joblib>=1.3.0
plotly>=5.15.0
pillow>=10.0.0
```

---

 Dataset

| Property | Value |
|---|---|
| **Source** | Indian Agricultural Statistics (1997–2020) |
| **Records** | 19,689 rows |
| **Crops** | 55 unique varieties |
| **States** | 30 Indian states |
| **Seasons** | 6 (Kharif, Rabi, Zaid, Whole Year, Autumn, Summer) |
| **Features** | 8 raw → 14 engineered |
| **Target** | Production (Metric Tonnes) |

---

 How Prediction Works

```
User Input (Crop, State, Season, Area, Rainfall, Fertilizer, Pesticide)
         ↓
Label Encoding (crop_encoder, season_encoder, state_encoder)
         ↓
Feature Engineering:
  • Fert_Per_Area = Fertilizer / Area
  • Pest_Per_Area = Pesticide / Area
  • Rainfall_Per_Area = Rainfall / Area
  • Crop_Avg_Yield = from feature_meta.pkl
  • State_Avg_Yield = from feature_meta.pkl
  • Expected_Yield_Prod = Crop_Avg_Yield × Area
         ↓
14-Feature Input Vector → ExtraTreesRegressor (500 trees)
         ↓
Predicted Production (Metric Tonnes) + Yield per Hectare
```

---
 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

 Author

Built with  using **Python**, **Streamlit**, **Scikit-Learn**, and **Plotly**.

> *"Empowering agriculture with the precision of AI."*

---

<div align="center">

 **Star this repo** if you found it useful!

![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B?style=flat-square&logo=streamlit)
![ML Powered](https://img.shields.io/badge/ML-ExtraTrees%20Ensemble-10b981?style=flat-square)
![India](https://img.shields.io/badge/Dataset-Indian%20Agriculture-FF9933?style=flat-square)

</div>
