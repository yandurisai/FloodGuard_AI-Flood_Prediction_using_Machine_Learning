# 🌊 Rising Waters: A Machine Learning Approach To Flood Prediction

A complete end-to-end Machine Learning project for flood prediction with a Flask web dashboard.

---

## 📁 Project Structure

```
flood_prediction/
├── data/
│   ├── generate_dataset.py     # Synthetic dataset generator
│   └── flood_data.csv          # Generated training data
├── models/
│   ├── best_model.pkl          # Saved best ML model
│   ├── scaler.pkl              # Feature scaler
│   ├── feature_names.json      # Feature list
│   └── metrics.json            # Evaluation results
├── static/
│   └── plots/                  # EDA & evaluation plots
├── templates/
│   ├── index.html              # Prediction UI
│   └── dashboard.html          # ML Dashboard
├── train_model.py              # Full ML training pipeline
├── app.py                      # Flask web application
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```
This will:
- Generate a 5,000-sample synthetic flood dataset
- Perform EDA and save plots
- Train 5 ML classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM)
- Run 5-fold cross-validation
- Run Bootstrap confidence interval estimation
- Save the best model and all evaluation artifacts

### 3. Run the Web App
```bash
python app.py
```
Visit: http://localhost:5000

---

## 🧠 Machine Learning Pipeline

### Features Used
| Feature | Description |
|--------|-------------|
| `rainfall_mm` | Current day rainfall in mm |
| `rainfall_3day_avg` | 3-day rolling average rainfall |
| `rainfall_7day_avg` | 7-day rolling average rainfall |
| `river_level_m` | River water level in meters |
| `temperature_c` | Air temperature |
| `humidity_pct` | Relative humidity % |
| `wind_speed_kmh` | Wind speed |
| `soil_moisture_pct` | Soil saturation % |
| `elevation_m` | Terrain elevation |
| `distance_to_river_km` | Proximity to river |
| `drainage_quality` | 0=Poor, 1=Medium, 2=Good |
| `rainfall_river_interaction` | Engineered: rainfall × river level |
| `low_elevation_near_river` | Engineered: binary risk flag |
| `high_risk_conditions` | Engineered: binary risk flag |

### Models Trained
- **Logistic Regression** — baseline classifier
- **Decision Tree** — interpretable tree-based model
- **Random Forest** — ensemble of trees (usually best)
- **Gradient Boosting** — sequential boosting
- **SVM** — support vector classifier

### Evaluation
- Accuracy, Precision, Recall, F1, ROC-AUC
- 5-fold Stratified Cross-Validation
- Bootstrap (n=100) 95% Confidence Interval for AUC
- Confusion Matrix, ROC Curves, Feature Importance plots

---

## 🌐 Web Application

### Prediction Page (`/`)
- Input environmental parameters
- Instant flood risk prediction with probability %
- Risk levels: Low / Moderate / High / Critical
- Actionable recommendations
- Quick scenario presets (Safe / Moderate / Flood)

### Dashboard (`/dashboard`)
- Model comparison table
- All EDA and evaluation plots
- Class distribution, correlation heatmap
- ROC curves, confusion matrix, feature importance
- Bootstrap AUC confidence interval

---

## 📊 Use Cases

1. **Early Warning Systems** — Real-time flood risk alerts for residents
2. **Disaster Response Planning** — Resource allocation for emergency services
3. **Infrastructure Resilience** — Urban planning and flood barrier design

---

## 🛠️ Tech Stack
- **Python** — Core language
- **Scikit-learn** — ML models & evaluation
- **Pandas / NumPy** — Data analysis
- **Matplotlib / Seaborn** — Visualization
- **Flask** — Web framework
- **Bootstrap 5** — Responsive UI
