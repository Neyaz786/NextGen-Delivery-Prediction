# Features Used in the Pipeline

## Summary
After removing data leakage columns, the following features are used for model training and prediction.

---

## Removed (Data Leakage)
The following columns were removed as they cause data leakage:

1. **Actual_Delivery_Days** - Post-delivery data, directly used in target calculation
2. **Customer_Rating** - Post-delivery rating, highly correlated with delays
3. **Traffic_Delay_Minutes** - Post-delivery data
4. **Promised_Delivery_Days** - Used in target calculation (delay_days = Actual - Promised)
5. **Route** - Post-delivery route information
6. **Origin** - Post-delivery information
7. **Destination** - Post-delivery information
8. **Customer_Segment** - Potentially post-delivery
9. **Order_Date** - Post-delivery information
10. **Special_Handling** - Post-delivery information
11. **Delivery_Status** - Post-delivery information
12. **Quality_Issue** - Post-delivery information

---

## Features Used in Pipeline

### Numeric Features
These features are processed with Median Imputation + Standard Scaling:

1. **Order_Value_INR** - Order value in Indian Rupees
2. **Delivery_Cost_INR** - Delivery cost in Indian Rupees
3. **Distance_KM** - Distance in kilometers
4. **Fuel_Consumption_L** - Fuel consumption in liters
5. **Toll_Charges_INR** - Toll charges in Indian Rupees
6. **value_per_km** - Derived feature: Order_Value_INR / Distance_KM
7. **fuel_efficiency** - Derived feature: Distance_KM / Fuel_Consumption_L

### Ordinal Features
These features are encoded with custom ordinal mapping:

1. **Priority** 
   - Economy (0) < Standard (1) < Express (2) < Unknown (3)
   
2. **Weather_Impact**
   - Unknown/None (0) < Fog (1) < Light_Rain (2) < Heavy_Rain (3)

### Nominal/Categorical Features
These features are processed with One-Hot Encoding:

1. **Product_Category** - Product category type
2. **Carrier** - Delivery carrier company

---

## Features Expected in app.py (User Input Form)

The Streamlit dashboard prediction form collects the following features (available BEFORE delivery):

1. **Priority** - Dropdown: Economy, Standard, Express, Unknown
2. **Product_Category** - Dropdown: Based on available categories
3. **Order_Value_INR** - Number input
4. **Carrier** - Dropdown: Based on available carriers
5. **Distance_KM** - Number input
6. **Fuel_Consumption_L** - Number input
7. **Toll_Charges_INR** - Number input
8. **Weather_Impact** - Dropdown: None, Unknown, Fog, Light_Rain, Heavy_Rain
9. **Delivery_Cost_INR** - Number input

**Derived features (created internally):**
- `value_per_km` = Order_Value_INR / Distance_KM
- `fuel_efficiency` = Distance_KM / Fuel_Consumption_L

---

## Total Feature Count

- **Total features before preprocessing:** 9 user inputs + 2 derived = 11 features
- **After one-hot encoding:** Approximately 15-20 features (depending on unique values in Product_Category and Carrier)
- **All features are available BEFORE delivery** - no data leakage

---

## Model Input Pipeline

1. User provides 9 input features
2. Derived features are created (value_per_km, fuel_efficiency)
3. Ordinal encoding applied to Priority and Weather_Impact
4. Numeric features: Median imputation + Standard scaling
5. Categorical features: One-hot encoding
6. Final feature vector passed to Random Forest models

