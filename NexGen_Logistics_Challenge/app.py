"""
Project Name: NexGen Logistics – Predictive Delivery Optimizer

Description:
This Streamlit application analyzes end-to-end logistics data to identify
delivery delays, operational inefficiencies, and cost drivers. It integrates
multiple datasets such as orders, delivery performance, routing, and cost
breakdowns to generate actionable insights. A machine learning based delay
risk predictor helps estimate the probability of delivery delays based on
route distance, traffic conditions, and weather impact.

Key Features:
- Data ingestion and cleaning from multiple operational datasets
- Merged analytics view across orders, delivery, routing, and costs
- Delay metrics and performance KPIs
- Interactive visual analysis for logistics optimization
- Predictive delivery delay risk estimator using machine learning

Tech Stack:
- Python
- Streamlit
- Pandas
- Matplotlib
- Scikit-learn

Intended Audience:
Operations managers, supply chain analysts, and logistics planners seeking
data-driven insights for improving delivery reliability and cost efficiency.

Author: Tanisha Priya

Last Updated: December 2025
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Configure the Streamlit page
st.set_page_config(
    page_title="NexGen Logistics Challenge",
    layout="wide"
)

st.title("NexGen Logistics – Predictive Delivery Optimizer")

st.write(
    "This application analyzes logistics performance data to identify delivery delays, "
    "operational inefficiencies, and improvement opportunities using data-driven insights."
)

# Load all datasets
@st.cache_data
def load_data():
    orders = pd.read_csv("data/orders.csv")
    delivery = pd.read_csv("data/delivery_performance.csv")
    routes = pd.read_csv("data/routes_distance.csv")
    vehicles = pd.read_csv("data/vehicle_fleet.csv")
    cost = pd.read_csv("data/cost_breakdown.csv")
    feedback = pd.read_csv("data/customer_feedback.csv")
    return orders, delivery, routes, vehicles, cost, feedback

orders, delivery, routes, vehicles, cost, feedback = load_data()

# Standardize column names for consistency
orders.columns = orders.columns.str.lower()
delivery.columns = delivery.columns.str.lower()
routes.columns = routes.columns.str.lower()
cost.columns = cost.columns.str.lower()

# Display dataset previews
st.subheader("Dataset Preview")

tab1, tab2, tab3 = st.tabs(["Orders", "Delivery Performance", "Routes"])

with tab1:
    st.write("Orders data")
    st.dataframe(orders.head())

with tab2:
    st.write("Delivery performance data")
    st.dataframe(delivery.head())

with tab3:
    st.write("Routes and distance data")
    st.dataframe(routes.head())

st.success("All datasets loaded successfully")

# Handle missing values
delivery.fillna(0, inplace=True)
routes.fillna(0, inplace=True)
cost.fillna(0, inplace=True)

st.success("Missing values handled successfully")

# Merge datasets using order_id
df = delivery.merge(orders, on="order_id", how="left")
df = df.merge(routes, on="order_id", how="left")
df = df.merge(cost, on="order_id", how="left")

st.subheader("Merged Dataset Preview")
st.dataframe(df.head())

# Create delivery delay metrics
df["delay_days"] = df["actual_delivery_days"] - df["promised_delivery_days"]
df["is_delayed"] = (df["delay_days"] > 0).astype(int)

st.subheader("Delay Metrics Preview")
st.dataframe(
    df[
        [
            "order_id",
            "promised_delivery_days",
            "actual_delivery_days",
            "delay_days",
            "is_delayed",
        ]
    ].head()
)

# Display key performance indicators
st.subheader("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Orders", len(df))
col2.metric("Delayed Orders (%)", round(df["is_delayed"].mean() * 100, 2))
col3.metric("Average Delay (days)", round(df["delay_days"].mean(), 2))
col4.metric("Average Delivery Cost (INR)", round(df["delivery_cost_inr"].mean(), 2))

# Visual analytics section
st.subheader("Operational Insights and Visual Analysis")

st.write("Delay percentage by warehouse")
warehouse_delay = df.groupby("origin")["is_delayed"].mean().sort_values(ascending=False)

fig, ax = plt.subplots()
warehouse_delay.plot(kind="bar", ax=ax)
ax.set_xlabel("Warehouse")
ax.set_ylabel("Delayed Orders (%)")
ax.set_title("Delay Rate by Warehouse")
st.pyplot(fig)

st.write("Delay trend over time")
df["order_date"] = pd.to_datetime(df["order_date"])
delay_trend = df.groupby(df["order_date"].dt.date)["is_delayed"].mean()

fig, ax = plt.subplots()
delay_trend.plot(ax=ax)
ax.set_xlabel("Order Date")
ax.set_ylabel("Delayed Orders (%)")
ax.set_title("Delivery Delay Trend Over Time")
st.pyplot(fig)

st.write("Relationship between route distance and delivery delay")
fig, ax = plt.subplots()
ax.scatter(df["distance_km"], df["delay_days"])
ax.set_xlabel("Distance (KM)")
ax.set_ylabel("Delay (Days)")
ax.set_title("Distance vs Delivery Delay")
st.pyplot(fig)

st.write("Average cost distribution per order")
cost_columns = [
    "fuel_cost",
    "labor_cost",
    "vehicle_maintenance",
    "insurance",
    "packaging_cost",
    "technology_platform_fee",
    "other_overhead",
]

avg_costs = df[cost_columns].mean()
fig, ax = plt.subplots()
avg_costs.plot(kind="pie", autopct="%1.1f%%", ax=ax)
ax.set_ylabel("")
ax.set_title("Cost Breakdown")
st.pyplot(fig)

# Convert weather conditions into numeric severity for modeling
weather_mapping = {
    "Clear": 0,
    "Light_Rain": 1,
    "Moderate_Rain": 2,
    "Heavy_Rain": 3,
    "Storm": 4
}

df["weather_impact_numeric"] = df["weather_impact"].map(weather_mapping).fillna(0)

# Delay prediction model
st.subheader("Delivery Delay Risk Predictor")

predictor_df = df[
    [
        "distance_km",
        "traffic_delay_minutes",
        "weather_impact_numeric",
        "is_delayed",
    ]
].dropna()

X = predictor_df[
    [
        "distance_km",
        "traffic_delay_minutes",
        "weather_impact_numeric",
    ]
]
y = predictor_df["is_delayed"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.write("Enter order and route details to estimate delay risk")

distance_input = st.slider("Route distance (KM)", 10, 3000, 500)
traffic_input = st.slider("Expected traffic delay (minutes)", 0, 300, 30)

weather_label = st.selectbox(
    "Weather condition",
    ["Clear", "Light_Rain", "Moderate_Rain", "Heavy_Rain", "Storm"],
)

weather_numeric = weather_mapping.get(weather_label, 0)

input_data = pd.DataFrame(
    [[distance_input, traffic_input, weather_numeric]],
    columns=[
        "distance_km",
        "traffic_delay_minutes",
        "weather_impact_numeric",
    ],
)

risk_probability = model.predict_proba(input_data)[0][1] * 100

st.metric("Predicted Delay Risk Percentage", round(risk_probability, 2))

if risk_probability > 70:
    st.error(
        "High delay risk detected. Consider using express delivery, assigning a priority vehicle, "
        "or selecting an alternate route."
    )
elif risk_probability > 40:
    st.warning(
        "Moderate delay risk detected. Adding buffer time and monitoring traffic is recommended."
    )
else:
    st.success(
        "Low delay risk detected. Standard delivery planning should be sufficient."
    )

st.subheader("Export Processed Data")

st.write(
    "You can download the processed and merged dataset used in this analysis "
    "for further offline review or reporting."
)

csv_data = df.to_csv(index=False)

st.download_button(
    label="Download processed dataset as CSV",
    data=csv_data,
    file_name="nexgen_processed_logistics_data.csv",
    mime="text/csv"
)