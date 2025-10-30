"""
Streamlit Dashboard for Predictive Delivery Optimizer.

Four-page dashboard:
1. Data Insights - KPIs, filters, and visualizations
2. ML Prediction - Single order prediction
3. Business Insights - Carrier rankings and recommendations
4. Model Performance - Model metrics and feature importance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import joblib
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Page configuration
st.set_page_config(
    page_title="Predictive Delivery Optimizer",
    page_icon="🚚",
    layout="wide"
)

# Define paths
project_root = Path(__file__).parent
data_dir = project_root / "data"
processed_dir = project_root / "processed"
models_dir = project_root / "models"


@st.cache_data
def load_data():
    """Load processed data."""
    # Always load and merge from original data to ensure all columns exist
    orders_df = pd.read_csv(data_dir / "orders.csv")
    delivery_df = pd.read_csv(data_dir / "delivery_performance.csv")
    routes_df = pd.read_csv(data_dir / "routes_distance.csv")
    
    df = pd.merge(orders_df, delivery_df, on="Order_ID", how="inner")
    df = pd.merge(df, routes_df, on="Order_ID", how="inner")
    
    # Create targets BEFORE dropping leakage columns
    if "Actual_Delivery_Days" in df.columns and "Promised_Delivery_Days" in df.columns:
        if "delay_days" not in df.columns:
            df["delay_days"] = df["Actual_Delivery_Days"] - df["Promised_Delivery_Days"]
        if "is_delayed" not in df.columns:
            df["is_delayed"] = (df["delay_days"] > 0).astype(int)
    
    # Drop leakage columns AFTER creating targets
    leakage_columns = [
        "Route", "Origin", "Destination",
        "Order_Date", "Special_Handling", "Delivery_Status", "Quality_Issue",
        "Actual_Delivery_Days", "Customer_Rating", "Traffic_Delay_Minutes"
        # Note: Promised_Delivery_Days kept - available before delivery (promise at order time)
        # Note: Customer_Segment kept - available before delivery (customer information)
    ]
    columns_to_drop = [col for col in leakage_columns if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    
    return df


@st.cache_data
def load_processed_data():
    """Load processed data with features."""
    processed_path = processed_dir / "merged_processed.csv"
    if processed_path.exists():
        df = pd.read_csv(processed_path)
        # Drop any leakage columns that might still be in the processed data
        # Note: Promised_Delivery_Days and Customer_Segment are kept (available before delivery)
        leakage_columns = [
            "Actual_Delivery_Days", "Customer_Rating", "Traffic_Delay_Minutes",
            "Route", "Origin", "Destination",
            "Order_Date", "Special_Handling", 
            "Delivery_Status", "Quality_Issue"
        ]
        columns_to_drop = [col for col in leakage_columns if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
        return df
    return None


@st.cache_resource
def load_models():
    """Load trained models and preprocessor."""
    preprocessor_path = processed_dir / "preprocessor.joblib"
    classifier_path = models_dir / "classifier.joblib"
    regressor_path = models_dir / "regressor.joblib"
    
    preprocessor = None
    classifier = None
    regressor = None
    
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)
    if classifier_path.exists():
        classifier = joblib.load(classifier_path)
    if regressor_path.exists():
        regressor = joblib.load(regressor_path)
    
    return preprocessor, classifier, regressor


def ensure_preprocessor_columns(X: pd.DataFrame, preprocessor) -> pd.DataFrame:
    """Align X columns to what the preprocessor expects by adding any missing
    columns with safe defaults. This avoids errors when old preprocessors were
    fit with columns that are now removed (e.g., leakage columns).

    Numeric missing cols -> 0.0, Ordinal missing cols -> 0, Nominal -> 'Unknown'.
    """
    if preprocessor is None or not hasattr(preprocessor, "transformers_"):
        return X

    required_numeric = []
    required_ordinal = []
    required_nominal = []

    for name, transformer, cols in preprocessor.transformers_:
        if cols is None:
            continue
        if name == 'num':
            required_numeric.extend(cols)
        elif name == 'ord':
            required_ordinal.extend(cols)
        elif name == 'nom':
            required_nominal.extend(cols)

    # Add missing numeric
    for col in required_numeric:
        if col not in X.columns:
            X[col] = 0.0
    # Add missing ordinal
    for col in required_ordinal:
        if col not in X.columns:
            X[col] = 0
    # Add missing nominal
    for col in required_nominal:
        if col not in X.columns:
            X[col] = 'Unknown'

    # Ensure column order contains all required columns (others can remain)
    required_all = required_numeric + required_ordinal + required_nominal
    # Keep existing cols but place required first to be safe
    ordered_cols = [c for c in required_all if c in X.columns] + [c for c in X.columns if c not in required_all]
    return X[ordered_cols]


def get_preprocessor_feature_names(preprocessor) -> list:
    """Return expanded feature names from a fitted ColumnTransformer preprocessor.
    Handles numeric/ordinal passthrough names and OneHotEncoder expanded names.
    """
    if preprocessor is None or not hasattr(preprocessor, "transformers_"):
        return []

    feature_names: list[str] = []

    for name, transformer, cols in preprocessor.transformers_:
        if cols is None:
            continue
        if name in ['num', 'ord']:
            feature_names.extend(list(cols))
        elif name == 'nom':
            # 'transformer' is a Pipeline with an OneHotEncoder step named 'onehot'
            try:
                onehot = transformer.named_steps.get('onehot')
                if hasattr(onehot, 'get_feature_names_out'):
                    ohe_names = onehot.get_feature_names_out(cols)
                    feature_names.extend(ohe_names.tolist())
                else:
                    # Fallback if API differs
                    feature_names.extend([f"{name}_{i}" for i in range(len(cols))])
            except Exception:
                feature_names.extend([f"{name}_{i}" for i in range(len(cols))])

    return feature_names

@st.cache_resource
def load_metadata():
    """Load model metadata."""
    classifier_meta_path = models_dir / "classifier_metadata.json"
    regressor_meta_path = models_dir / "regressor_metadata.json"
    
    classifier_meta = {}
    regressor_meta = {}
    
    if classifier_meta_path.exists():
        with open(classifier_meta_path, 'r') as f:
            classifier_meta = json.load(f)
    
    if regressor_meta_path.exists():
        with open(regressor_meta_path, 'r') as f:
            regressor_meta = json.load(f)
    
    return classifier_meta, regressor_meta


# Page 1: Data Insights
def page_data_insights():
    """Display data insights page."""
    st.header("📊 Data Insights")
    
    df = load_data()
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_orders = len(df)
    delay_rate = df["is_delayed"].mean() * 100
    mean_delay = df["delay_days"].mean()
    total_features = len([c for c in df.columns if c not in ["delay_days", "is_delayed", "Order_ID"]])
    
    with col1:
        st.metric("Total Orders", f"{total_orders:,}")
    with col2:
        st.metric("Delay Rate", f"{delay_rate:.2f}%")
    with col3:
        st.metric("Mean Delay", f"{mean_delay:.2f} days")
    with col4:
        st.metric("Features", total_features)
    
    st.divider()
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        carriers = ["All"] + sorted(df["Carrier"].unique().tolist())
        selected_carrier = st.selectbox("Filter by Carrier", carriers)
    
    with col2:
        product_types = ["All"] + sorted(df["Product_Category"].unique().tolist())
        selected_product = st.selectbox("Filter by Product Type", product_types)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_carrier != "All":
        filtered_df = filtered_df[filtered_df["Carrier"] == selected_carrier]
    if selected_product != "All":
        filtered_df = filtered_df[filtered_df["Product_Category"] == selected_product]
    
    st.divider()
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Delay Rate by Carrier")
        carrier_delays = filtered_df.groupby("Carrier").agg({
            "is_delayed": "mean"
        }).reset_index()
        carrier_delays["Delay Rate (%)"] = carrier_delays["is_delayed"] * 100
        
        fig = px.bar(
            carrier_delays,
            x="Carrier",
            y="Delay Rate (%)",
            title="Delay Rate by Carrier",
            color="Delay Rate (%)",
            color_continuous_scale="Reds"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Delay by Product Type")
        product_delays = filtered_df.groupby("Product_Category").agg({
            "delay_days": "mean"
        }).reset_index()
        
        fig = px.bar(
            product_delays,
            x="Product_Category",
            y="delay_days",
            title="Average Delay by Product Type",
            labels={"delay_days": "Avg Delay (days)", "Product_Category": "Product Type"},
            color="delay_days",
            color_continuous_scale="Oranges"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Delay vs Weather Impact
    st.subheader("Delay vs Weather Impact")
    weather_delays = filtered_df.groupby("Weather_Impact").agg({
        "delay_days": "mean",
        "is_delayed": "mean"
    }).reset_index()
    weather_delays["Delay Rate (%)"] = weather_delays["is_delayed"] * 100
    
    fig = px.scatter(
        weather_delays,
        x="delay_days",
        y="Delay Rate (%)",
        size="delay_days",
        hover_name="Weather_Impact",
        title="Delay Analysis by Weather Impact",
        labels={"delay_days": "Average Delay (days)", "Delay Rate (%)": "Delay Rate (%)"}
    )
    st.plotly_chart(fig, use_container_width=True)


# Page 2: ML Prediction
def page_ml_prediction():
    """Display ML prediction page."""
    st.header("🤖 ML Prediction")
    
    preprocessor, classifier, regressor = load_models()
    
    if classifier is None or regressor is None:
        st.error("Models not found. Please train models first by running src/modeling.py")
        return
    
    st.subheader("Single Order Prediction")
    st.markdown("Enter order details to predict delivery delay:")
    
    # Get feature information
    processed_df = load_processed_data()
    if processed_df is None:
        st.error("Processed data not found. Please run feature engineering first.")
        return
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            priority = st.selectbox("Priority", ["Economy", "Standard", "Express", "Unknown"])
            product_category = st.selectbox("Product Category", sorted(processed_df["Product_Category"].unique()))
            order_value = st.number_input("Order Value (INR)", min_value=0.0, value=1000.0)
            carrier = st.selectbox("Carrier", sorted(processed_df["Carrier"].unique()))
        
        with col2:
            distance_km = st.number_input("Distance (KM)", min_value=0.0, value=500.0)
            fuel_consumption = st.number_input("Fuel Consumption (L)", min_value=0.0, value=50.0)
            toll_charges = st.number_input("Toll Charges (INR)", min_value=0.0, value=400.0)
            delivery_cost = st.number_input("Delivery Cost (INR)", min_value=0.0, value=500.0)
        
        with col3:
            weather_impact = st.selectbox("Weather Impact", ["None", "Unknown", "Fog", "Light_Rain", "Heavy_Rain"])
        
        submitted = st.form_submit_button("Predict Delivery")
    
    if submitted:
        # Prepare input data (only features available BEFORE delivery)
        # Removed leakage columns: Actual_Delivery_Days, Customer_Rating, Traffic_Delay_Minutes
        # Kept: Promised_Delivery_Days (promise at order time), Customer_Segment (customer info)
        input_data = {
            "Priority": priority,
            "Product_Category": product_category,
            "Order_Value_INR": order_value,
            "Carrier": carrier,
            "Distance_KM": distance_km,
            "Fuel_Consumption_L": fuel_consumption,
            "Toll_Charges_INR": toll_charges,
            "Weather_Impact": weather_impact,
            "Delivery_Cost_INR": delivery_cost
        }
        
        # Add Promised_Delivery_Days if available in processed data (get default value)
        if processed_df is not None and "Promised_Delivery_Days" in processed_df.columns:
            # Use mean or median as default, or get from form if we add it
            promised_days_default = int(processed_df["Promised_Delivery_Days"].median() if len(processed_df) > 0 else 5)
        else:
            promised_days_default = 5
        
        # Add Customer_Segment if available in processed data
        if processed_df is not None and "Customer_Segment" in processed_df.columns:
            # Get unique segments, use first as default or add to form
            segments = processed_df["Customer_Segment"].unique().tolist()
            if len(segments) > 0:
                customer_segment_default = segments[0]
            else:
                customer_segment_default = "Individual"
        else:
            customer_segment_default = "Individual"
        
        # Add these features if they're expected by the model
        if processed_df is not None:
            if "Promised_Delivery_Days" in processed_df.columns and "Promised_Delivery_Days" not in input_data:
                input_data["Promised_Delivery_Days"] = promised_days_default
            if "Customer_Segment" in processed_df.columns and "Customer_Segment" not in input_data:
                input_data["Customer_Segment"] = customer_segment_default
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Apply feature engineering
        try:
            from features import create_derived_features, apply_ordinal_encoding
        except ImportError:
            import features as feat_module
            create_derived_features = feat_module.create_derived_features
            apply_ordinal_encoding = feat_module.apply_ordinal_encoding
        
        input_df = create_derived_features(input_df)
        input_df = apply_ordinal_encoding(input_df)
        
        # Ensure no leakage columns are present
        # Note: Promised_Delivery_Days and Customer_Segment are kept (available before delivery)
        leakage_columns = [
            "Actual_Delivery_Days", "Customer_Rating", "Traffic_Delay_Minutes",
            "Delivery_Status"
        ]
        for col in leakage_columns:
            if col in input_df.columns:
                input_df = input_df.drop(columns=[col])
                st.warning(f"Removed leakage column: {col}")
        
        # Get feature columns (same as training) - should match modeling.py
        target_cols = ["delay_days", "is_delayed", "Order_ID"]
        feature_cols = [col for col in input_df.columns if col not in target_cols]
        
        # Expected features after feature engineering:
        # Numeric: Order_Value_INR, Delivery_Cost_INR, Distance_KM, Fuel_Consumption_L,
        #          Toll_Charges_INR, Promised_Delivery_Days, value_per_km, fuel_efficiency
        # Ordinal (encoded): Priority, Weather_Impact
        # Nominal: Product_Category, Carrier, Customer_Segment
        
        X_input = input_df[feature_cols]
        
        # Transform and predict (align columns to preprocessor expectations)
        if preprocessor is not None:
            X_aligned = ensure_preprocessor_columns(X_input.copy(), preprocessor)
            X_transformed = preprocessor.transform(X_aligned)
        else:
            X_transformed = X_input.values
        
        # Predictions
        is_delayed_pred = classifier.predict(X_transformed)[0]
        delay_days_pred = regressor.predict(X_transformed)[0]
        
        # Display results
        st.success("Prediction Complete!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if is_delayed_pred == 1:
                st.error(f"**Predicted Status: DELAYED**")
            else:
                st.success(f"**Predicted Status: ON-TIME**")
        
        with col2:
            st.metric("Expected Delay Days", f"{delay_days_pred:.2f} days")


# Page 3: Business Insights
def page_business_insights():
    """Display business insights page."""
    st.header("💼 Business Insights")
    
    df = load_data()
    
    st.subheader("Carrier Reliability Ranking")
    
    # Calculate carrier metrics
    carrier_stats = df.groupby("Carrier").agg({
        "is_delayed": ["mean", "count"],
        "delay_days": "mean",
        "Delivery_Cost_INR": "mean"
    }).reset_index()
    
    carrier_stats.columns = ["Carrier", "Delay_Rate", "Total_Orders", "Avg_Delay_Days", "Avg_Cost"]
    carrier_stats["Reliability_Score"] = 100 - (carrier_stats["Delay_Rate"] * 100)
    carrier_stats = carrier_stats.sort_values("Reliability_Score", ascending=False)
    
    st.dataframe(
        carrier_stats[["Carrier", "Reliability_Score", "Avg_Delay_Days", "Delay_Rate", "Total_Orders", "Avg_Cost"]].round(2),
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()
    
    st.subheader("📋 Data-Driven Recommendations")
    
    # Generate recommendations
    high_delay_carriers = carrier_stats[carrier_stats["Delay_Rate"] > 0.5]["Carrier"].tolist()
    high_delay_products = df.groupby("Product_Category")["is_delayed"].mean().sort_values(ascending=False).head(3)
    
    recommendations = []
    
    if high_delay_carriers:
        recommendations.append(
            f"⚠️ **Focus on High-Delay Carriers**: {', '.join(high_delay_carriers)} have delay rates >50%. "
            "Consider renegotiating contracts or providing additional support."
        )
    
    if len(high_delay_products) > 0:
        products_str = ", ".join(high_delay_products.index.tolist())
        recommendations.append(
            f"📦 **Product Type Optimization**: {products_str} show higher delay rates. "
            "Review packaging, handling procedures, or carrier assignment for these product types."
        )
    
    weather_impact = df.groupby("Weather_Impact")["is_delayed"].mean().sort_values(ascending=False)
    if len(weather_impact) > 0:
        worst_weather = weather_impact.index[0]
        recommendations.append(
            f"🌧️ **Weather Mitigation**: {worst_weather} conditions show highest delays. "
            "Develop contingency plans for weather-affected routes."
        )
    
    avg_distance = df["Distance_KM"].mean()
    long_distance_delays = df[df["Distance_KM"] > avg_distance]["is_delayed"].mean()
    if long_distance_delays > df["is_delayed"].mean():
        recommendations.append(
            f"🛣️ **Route Optimization**: Long-distance routes show higher delays. "
            "Consider breaking long routes into segments or using express carriers."
        )
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")


# Page 4: Model Performance
def page_model_performance():
    """Display model performance page."""
    st.header("📈 Model Performance")
    
    classifier_meta, regressor_meta = load_metadata()
    
    if not classifier_meta and not regressor_meta:
        st.error("Model metadata not found. Please train models first.")
        return
    
    # Classification Metrics
    if classifier_meta:
        st.subheader("Classification Model Metrics")
        
        metrics = classifier_meta.get("metrics", {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
        
        # Feature importance
        st.subheader("Classification Model - Feature Importance")
        
        preprocessor, classifier, _ = load_models()
        if classifier is not None:
            feature_importance = classifier.feature_importances_
            
            # Build readable feature names from the preprocessor
            feature_names = get_preprocessor_feature_names(preprocessor)
            if len(feature_names) != len(feature_importance):
                # Fallback if mismatch
                feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": feature_importance
            }).sort_values("Importance", ascending=False).head(15)
            
            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 Feature Importances",
                labels={"Importance": "Importance Score"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Regression Metrics
    if regressor_meta:
        st.subheader("Regression Model Metrics")
        
        metrics = regressor_meta.get("metrics", {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
        with col2:
            st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
        with col3:
            st.metric("R²", f"{metrics.get('r2', 0):.4f}")
        
        # Feature importance
        st.subheader("Regression Model - Feature Importance")
        
        preprocessor, _, regressor = load_models()
        if regressor is not None:
            feature_importance = regressor.feature_importances_

            feature_names = get_preprocessor_feature_names(preprocessor)
            if len(feature_names) != len(feature_importance):
                feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": feature_importance
            }).sort_values("Importance", ascending=False).head(15)
            
            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 Feature Importances",
                labels={"Importance": "Importance Score"}
            )
            st.plotly_chart(fig, use_container_width=True)


# Main app
def main():
    """Main application."""
    st.title("🚚 Predictive Delivery Optimizer")
    st.markdown("**NexGen Logistics** - Machine Learning Prototype for Delivery Delay Prediction")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Data Insights", "ML Prediction", "Business Insights", "Model Performance"]
    )
    
    # Route to selected page
    if page == "Data Insights":
        page_data_insights()
    elif page == "ML Prediction":
        page_ml_prediction()
    elif page == "Business Insights":
        page_business_insights()
    elif page == "Model Performance":
        page_model_performance()


if __name__ == "__main__":
    main()

