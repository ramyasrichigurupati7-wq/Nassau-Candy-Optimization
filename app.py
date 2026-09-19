import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Nassau Candy Optimization",
    page_icon="🍬",
    layout="wide"
)


# ==========================================
# LOAD DATA AND MODEL
# ==========================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        "cleaned_nassau_candy.csv"
    )

    return data


@st.cache_resource
def load_model():

    model = joblib.load(
        "shipping_model.pkl"
    )

    return model


df = load_data()
model = load_model()


# ==========================================
# TITLE
# ==========================================

st.title(
    "🍬 Nassau Candy Distributor"
)

st.subheader(
    "Factory Reallocation & Shipping Optimization Recommendation System"
)

st.write(
    "This dashboard predicts shipping time and "
    "simulates factory allocation scenarios."
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Order Selection")


# Product

product_list = sorted(
    df["Product Name"].dropna().unique()
)

selected_product = st.sidebar.selectbox(
    "Select Product",
    product_list
)


# Region

region_list = sorted(
    df["Region"].dropna().unique()
)

selected_region = st.sidebar.selectbox(
    "Select Region",
    region_list
)


# Ship Mode

ship_mode_list = sorted(
    df["Ship Mode"].dropna().unique()
)

selected_ship_mode = st.sidebar.selectbox(
    "Select Ship Mode",
    ship_mode_list
)


# Units

selected_units = st.sidebar.number_input(
    "Number of Units",
    min_value=1,
    value=10
)


# Sales

selected_sales = st.sidebar.number_input(
    "Sales Value",
    min_value=0.0,
    value=100.0
)


# Cost

selected_cost = st.sidebar.number_input(
    "Product Cost",
    min_value=0.0,
    value=50.0
)


# Optimization priority

priority = st.sidebar.slider(
    "Shipping Priority",
    min_value=0,
    max_value=100,
    value=70
)


# ==========================================
# CURRENT DATA
# ==========================================

filtered_data = df[
    (df["Product Name"] == selected_product)
    &
    (df["Region"] == selected_region)
    &
    (df["Ship Mode"] == selected_ship_mode)
]


# ==========================================
# DASHBOARD KPIs
# ==========================================

st.header("📊 Current Performance")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Orders",
        f"{len(df):,}"
    )


with col2:

    st.metric(
        "Total Sales",
        f"${df['Sales'].sum():,.2f}"
    )


with col3:

    st.metric(
        "Total Units",
        f"{df['Units'].sum():,}"
    )


with col4:

    st.metric(
        "Average Shipping Days",
        f"{df['Shipping Days'].mean():.2f}"
    )


# ==========================================
# PREDICTION
# ==========================================

st.header("🚚 Shipping Time Prediction")


input_data = pd.DataFrame({

    "Product Name": [
        selected_product
    ],

    "Region": [
        selected_region
    ],

    "Ship Mode": [
        selected_ship_mode
    ],

    "Units": [
        selected_units
    ],

    "Sales": [
        selected_sales
    ],

    "Cost": [
        selected_cost
    ]
})


predicted_days = model.predict(
    input_data
)[0]


predicted_days = max(
    0,
    predicted_days
)


st.metric(
    "Predicted Shipping Days",
    f"{predicted_days:.2f} days"
)


# ==========================================
# FACTORY SIMULATION
# ==========================================

st.header(
    "🏭 Factory Reallocation Simulation"
)

st.info(
    "Factory locations are simulated because "
    "the supplied dataset does not contain actual "
    "factory-location information."
)


factory_factors = {

    "Factory A": 1.00,

    "Factory B": 0.90,

    "Factory C": 0.80,

    "Factory D": 0.70

}


factory_results = []


for factory, factor in factory_factors.items():

    estimated_days = (
        predicted_days * factor
    )

    improvement = 0

    if predicted_days > 0:

        improvement = (
            (predicted_days - estimated_days)
            / predicted_days
        ) * 100

    factory_results.append({

        "Factory": factory,

        "Estimated Shipping Days":
            round(estimated_days, 2),

        "Improvement %":
            round(improvement, 2)

    })


factory_df = pd.DataFrame(
    factory_results
)


# ==========================================
# FACTORY TABLE
# ==========================================

st.dataframe(
    factory_df,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# RECOMMENDED FACTORY
# ==========================================

recommended_factory = factory_df.loc[
    factory_df[
        "Estimated Shipping Days"
    ].idxmin(),
    "Factory"
]


recommended_days = factory_df.loc[
    factory_df["Factory"] == recommended_factory,
    "Estimated Shipping Days"
].iloc[0]


improvement = factory_df.loc[
    factory_df["Factory"] == recommended_factory,
    "Improvement %"
].iloc[0]


st.header(
    "🎯 Recommended Factory Scenario"
)


st.success(
    f"Recommended simulated factory: "
    f"{recommended_factory}"
)


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Estimated Shipping Days",
        f"{recommended_days:.2f}"
    )


with col2:

    st.metric(
        "Estimated Improvement",
        f"{improvement:.2f}%"
    )


# ==========================================
# FACTORY CHART
# ==========================================

st.header(
    "📈 Factory Shipping Comparison"
)


chart_data = factory_df.set_index(
    "Factory"
)[
    "Estimated Shipping Days"
]


st.bar_chart(
    chart_data
)


# ==========================================
# PROFIT INFORMATION
# ==========================================

st.header(
    "💰 Order Profit Information"
)


estimated_profit = (
    selected_sales - selected_cost
)


profit_margin = 0

if selected_sales > 0:

    profit_margin = (
        estimated_profit
        / selected_sales
    ) * 100


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Estimated Gross Profit",
        f"${estimated_profit:,.2f}"
    )


with col2:

    st.metric(
        "Profit Margin",
        f"{profit_margin:.2f}%"
    )


# ==========================================
# DATASET SUMMARY
# ==========================================

st.header(
    "📋 Dataset Summary"
)


summary_col1, summary_col2 = st.columns(2)


with summary_col1:

    st.write(
        "**Number of Records:**",
        len(df)
    )

    st.write(
        "**Number of Products:**",
        df["Product Name"].nunique()
    )

    st.write(
        "**Number of Regions:**",
        df["Region"].nunique()
    )


with summary_col2:

    st.write(
        "**Total Sales:**",
        f"${df['Sales'].sum():,.2f}"
    )

    st.write(
        "**Total Cost:**",
        f"${df['Cost'].sum():,.2f}"
    )

    st.write(
        "**Total Gross Profit:**",
        f"${df['Gross Profit'].sum():,.2f}"
    )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Nassau Candy Distributor — "
    "Factory Reallocation & Shipping Optimization "
    "Recommendation System"
)