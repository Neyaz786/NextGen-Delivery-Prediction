# 🚚 Predictive Delivery Optimizer

A production-ready machine learning prototype for **NexGen Logistics** that predicts delivery delays using Random Forest models for both classification and regression tasks.

## 🧠 Overview

This project provides a comprehensive ML system for predicting delivery delays:

- **Classification Model**: Predicts whether an order will be delayed (binary classification)
- **Regression Model**: Predicts the number of delay days (regression)
- **Interactive Dashboard**: Streamlit-based dashboard with 4 pages for insights, predictions, and analysis
- **EDA Notebook**: Comprehensive exploratory data analysis with 5 key visualizations

## 📁 Project Structure

```
Predictive-Delivery-Optimizer/
│
├── src/
│   ├── data_prep.py          # Data preparation and merging
│   ├── features.py            # Feature engineering and preprocessing
│   └── modeling.py            # Model training with RandomizedSearchCV
│
├── processed/
│   ├── merged.csv             # Processed and merged dataset
│   ├── merged_processed.csv   # Feature-engineered dataset
│   ├── data_stats.json        # Dataset statistics
│   └── preprocessor.joblib    # Fitted preprocessing pipeline
│
├── models/
│   ├── classifier.joblib      # Trained Random Forest Classifier
│   ├── regressor.joblib       # Trained Random Forest Regressor
│   ├── classifier_metadata.json
│   └── regressor_metadata.json
│
├── notebooks/
│   └── data_prep.ipynb        # EDA notebook with 5 visualizations
│
├── data/
│   ├── orders.csv
│   ├── delivery_performance.csv
│   └── routes_distance.csv
│
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Run the data preparation script to merge datasets and create target variables:

```bash
python src/data_prep.py
```

This will:
- Merge three CSV files on `Order_ID`
- Drop data leakage columns
- Create target variables (`delay_days`, `is_delayed`)
- Save processed data to `processed/merged.csv`
- Generate statistics in `processed/data_stats.json`

### 3. Feature Engineering

Run the feature engineering script to create derived features and preprocessing pipeline:

```bash
python src/features.py
```

This will:
- Create derived features (value_per_km, fuel_efficiency)
- Apply ordinal encoding for Priority and Weather_Impact
- Build and fit preprocessing pipeline
- Save preprocessor to `processed/preprocessor.joblib`

### 4. Train Models

Train the Random Forest classifier and regressor with hyperparameter tuning:

```bash
python src/modeling.py
```

This will:
- Load processed data and preprocessor
- Perform 80/20 train-test split
- Train Random Forest Classifier with RandomizedSearchCV
- Train Random Forest Regressor with RandomizedSearchCV
- Save models and metadata to `models/` directory

**Note**: Model training uses RandomizedSearchCV with a medium-large parameter space (75 iterations) and may take several minutes.

### 5. Launch Dashboard

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard will open in your browser with 4 pages:
1. **Data Insights**: KPIs, filters, and visualizations
2. **ML Prediction**: Single order prediction interface
3. **Business Insights**: Carrier rankings and recommendations
4. **Model Performance**: Model metrics and feature importance

## 📊 EDA Notebook

Open and run the Jupyter notebook for exploratory data analysis:

```bash
jupyter notebook notebooks/data_prep.ipynb
```

The notebook contains 5 key visualizations:
1. Distribution of delay_days
2. Correlation heatmap between numeric features
3. Delay rate by carrier
4. Delay vs weather impact
5. Priority level vs delay_days

## 🎯 Features

### Data Preparation
- Automatic merging of three data sources
- Data leakage prevention (drops post-delivery columns)
- Target variable creation (delay_days, is_delayed)
- Statistics generation

### Feature Engineering
- Two derived features (value_per_km, fuel_efficiency)
- Ordinal encoding for Priority and Weather_Impact
- Comprehensive preprocessing pipeline:
  - Numeric features: Median imputation + Standard scaling
  - Ordinal features: Custom mapping
  - Nominal features: One-hot encoding

### Model Training
- **RandomForestClassifier**: Predicts delayed/on-time
- **RandomForestRegressor**: Predicts delay days
- **RandomizedSearchCV**: Hyperparameter tuning with medium-large parameter space
  - n_estimators: 50-500
  - max_depth: 5-50
  - min_samples_split: 2-20
  - min_samples_leaf: 1-10
  - max_features: ['sqrt', 'log2', None]
  - bootstrap: [True, False]
- 80/20 train-test split
- Comprehensive metrics evaluation
- Model metadata saving

### Dashboard
- **Interactive visualizations** with Plotly
- **Real-time predictions** for single orders
- **Business insights** and recommendations
- **Model performance** tracking

## 📈 Model Metrics

The models are evaluated using:

**Classifier Metrics**:
- Accuracy
- Precision
- Recall
- F1-Score

**Regressor Metrics**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

## ⚙️ Configuration

- All scripts use `random_state=42` for reproducibility
- Train-test split: 80/20
- Cross-validation: 5-fold CV for hyperparameter tuning
- RandomizedSearchCV iterations: 75 (medium-large parameter space)

## 📝 Notes

- **No MLflow or external tracking**: All models and metadata are saved locally
- **Standalone scripts**: Each module can be run independently
- **Type hints and docstrings**: All code includes documentation
- **Logging**: Comprehensive logging for traceability
- **PEP8 compliant**: Code follows Python style guidelines

## 🔧 Troubleshooting

### Models not found error
Ensure you've run the model training script (`src/modeling.py`) before launching the dashboard.

### Preprocessor not found error
Ensure you've run the feature engineering script (`src/features.py`) before training models.

### Data not found error
Ensure CSV files are present in the `data/` directory and run `src/data_prep.py` first.

## 👤 Author

**NexGen Logistics** - Machine Learning Prototype

## 📄 License

This project is a prototype for internal use at NexGen Logistics.

