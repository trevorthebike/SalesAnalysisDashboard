"""
Sales Optimization Streamlit App (Rossmann-style Forecasting)

This application:
- Loads a trained XGBoost sales prediction model
- Uses historical retail data to forecast demand
- Optimizes product price to maximize revenue
- Visualizes predicted sales, revenue, and optimal pricing trends

Author: Your Name
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# -----------------------------------
# DATA + MODEL LOADING
# -----------------------------------
def load_data():
    model = joblib.load("models/sales_model.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")

    df = pd.read_csv("data/retail_sales.csv")

    df["date"] = pd.to_datetime(df["date"])
    df["store_id"] = df["store_id"].astype("category")
    df["item_id"] = df["item_id"].astype("category")

    return df, model, feature_cols


# -----------------------------------
# APP INITIALIZATION
# -----------------------------------
def setup_app(df):

    st.markdown("## Sales Analysis of Rossmann Dataset Using XGBoost Regressor")

    top_items = df.groupby("item_id")["sales"].sum().nlargest(3).index
    top_stores = df.groupby("store_id")["sales"].sum().nlargest(5).index

    item_id = st.selectbox("Item", sorted(top_items))
    prediction_week = st.selectbox("Forecast Week", range(1, 53))

    promo = 0
    year = 2023

    start_date = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(
        (prediction_week - 1) * 7, unit="D"
    )

    history = df[
        (df["item_id"] == item_id) & (df["date"] < start_date)
    ].sort_values("date")

    if len(history) < 28:
        st.error("Not enough history")
        st.stop()

    price_mean = history["price"].mean()

    price_range = np.linspace(df["price"].min(), df["price"].max(), 40)

    return top_stores, history, start_date, promo, price_mean, price_range


# -----------------------------------
# FEATURE ENGINEERING
# -----------------------------------
def create_features(store_id, current_date, sales_history, price, promo):

    return pd.DataFrame([{
        "store_id": store_id,
        "price": price,
        "promo": promo,
        "weekday": current_date.weekday(),
        "month": current_date.month,
        "lag_7": sales_history[-7],
        "lag_28": sales_history[-28],
        "ewm_sales_7": pd.Series(sales_history).ewm(span=7).mean().iloc[-1],
        "log_price": np.log1p(price),
        "price_x_promo": price * promo
    }]).astype({"store_id": "category"})


# -----------------------------------
# MODEL PREDICTION
# -----------------------------------
def predict_sales(model, feature_cols, features):
    X = features.reindex(columns=feature_cols)
    return model.predict(X)[0]


# -----------------------------------
# PRICE OPTIMIZATION
# -----------------------------------
def find_optimal_price(model, feature_cols, base_input, price_range):

    best_price = None
    best_revenue = -np.inf
    price_mean_local = np.mean(price_range)

    for price in price_range:

        temp = base_input.copy()
        temp["price"] = price
        temp["log_price"] = np.log1p(price)
        temp["price_x_promo"] = price * temp["promo"].iloc[0]

        pred_sales = predict_sales(model, feature_cols, temp)

        elasticity = np.exp(-2.5 * (price / price_mean_local))

        revenue = pred_sales * price * elasticity

        if revenue > best_revenue:
            best_revenue = revenue
            best_price = price

    return best_price


# -----------------------------------
# FORECASTING ENGINE
# -----------------------------------
def find_price(model, feature_cols, top_stores, history, start_date,
               promo, price_mean, price_range):

    results = []

    for store_id in top_stores:

        store_history = history[
            history["store_id"] == store_id
        ].sort_values("date")

        sales_history = store_history["sales"].tolist()

        if len(sales_history) < 28:
            continue

        for day in range(7):

            current_date = start_date + pd.Timedelta(days=day)

            base_input = create_features(
                store_id, current_date, sales_history, price_mean, promo
            )

            optimal_price = find_optimal_price(
                model, feature_cols, base_input, price_range
            )

            final_input = create_features(
                store_id, current_date, sales_history, optimal_price, promo
            )

            pred_sales = predict_sales(model, feature_cols, final_input)

            revenue = pred_sales * optimal_price

            sales_history.append(pred_sales)

            results.append({
                "store_id": store_id,
                "day": day + 1,
                "optimal_price": optimal_price,
                "pred_sales": float(pred_sales),
                "revenue": float(revenue),
            })

    return pd.DataFrame(results)


# -----------------------------------
# VISUALIZATION
# -----------------------------------
def display(results_df):

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Optimal Price",
                f"${results_df['optimal_price'].mean():.2f}")

    col2.metric("Average Daily Sales",
                f"{results_df['pred_sales'].mean():.0f}")

    col3.metric("Average Daily Revenue",
                f"${results_df['revenue'].mean():.2f}")

    st.divider()

    st.markdown("## Performance Trends")

    price_chart = results_df.pivot(index="day", columns="store_id", values="optimal_price")
    sales_chart = results_df.pivot(index="day", columns="store_id", values="pred_sales")
    revenue_chart = results_df.pivot(index="day", columns="store_id", values="revenue")

    tab1, tab2, tab3 = st.tabs([
        "Price Optimization",
        "Sales Forecast",
        "Revenue Forecast",
    ])

    with tab1:
        st.line_chart(price_chart)

    with tab2:
        st.line_chart(sales_chart)

    with tab3:
        st.line_chart(revenue_chart)


# -----------------------------------
# MAIN APP
# -----------------------------------
def main():

    df, model, feature_cols = load_data()

    top_stores, history, start_date, promo, price_mean, price_range = setup_app(df)

    results_df = find_price(
        model, feature_cols, top_stores, history,
        start_date, promo, price_mean, price_range
    )

    display(results_df)


if __name__ == "__main__":
    main()