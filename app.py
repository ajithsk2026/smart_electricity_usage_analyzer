import sqlite3
from io import BytesIO

import pandas as pd
import streamlit as st
import plotly.express as px

from database import create_database, DB_PATH, validate_meter_reading
from analysis import (
    calculate_daily_usage,
    calculate_monthly_usage,
    calculate_electricity_bill,
    detect_unusual_usage,
    analyze_monthly_trend,
    calculate_cesc_bill
)
from prediction import (
    predict_next_month,
    evaluate_prediction_model
)


# Page settings
st.set_page_config(
    page_title="Smart Electricity Analyzer",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Smart Electricity Usage Analyzer")
st.caption("track,analyze,predict and manage your electricity consumption")
st.write("Monitor, analyze and predict your electricity consumption.")

# Create database
create_database()


# Load readings from database
def load_readings():
    connection = sqlite3.connect(DB_PATH)

    readings = pd.read_sql_query(
        """
        SELECT reading_date, meter_reading
        FROM meter_readings
        ORDER BY reading_date
        """,
        connection
    )

    connection.close()
    return readings


# Add new meter reading
st.header("Add Electricity Meter Reading")

with st.form("reading_form"):

    reading_date = st.date_input("Reading Date")

    meter_reading = st.number_input(
        "Current Meter Reading (kWh)",
        min_value=0.0,
        step=1.0
    )

    submitted = st.form_submit_button("Save Reading")

    if submitted:
        is_valid, error_message = validate_meter_reading(
            reading_date.isoformat(),
            meter_reading
        )

        if not is_valid:
            st.error(error_message)

        else:
            connection = sqlite3.connect(DB_PATH)
            cursor = connection.cursor()

            try:
                cursor.execute(
                    """
                    INSERT INTO meter_readings
                    (reading_date, meter_reading)
                    VALUES (?, ?)
                    """,
                    (reading_date.isoformat(), meter_reading)
                )

                connection.commit()
                st.success("Meter reading saved successfully!")

            except sqlite3.IntegrityError:
                st.error("A reading for this date already exists.")

            finally:
                connection.close()


# Display analysis
readings = load_readings()

if len(readings) >= 2:

    analyzed, warning_message = detect_unusual_usage(readings)

    # Unusual usage detection
    if warning_message:
        st.info(warning_message)

    else:
        unusual_days = analyzed[analyzed["unusual_usage"]]

        if not unusual_days.empty:
            st.warning("Unusually high electricity usage detected!")

            st.dataframe(
                unusual_days[["reading_date", "daily_usage"]],
                width="stretch"
            )

        else:
            st.success("No unusually high electricity usage detected.")

    # Remove first row because it has no daily usage
    valid_usage = analyzed.dropna(subset=["daily_usage"])

    st.header("Electricity Consumption Analysis")

    total_units = valid_usage["daily_usage"].sum()

    # Custom electricity rate
    rate_per_unit = st.number_input(
        "Electricity Rate per Unit (₹)",
        min_value=0.0,
        value=5.0,
        step=0.5
    )

    estimated_bill = calculate_electricity_bill(
        total_units,
        rate_per_unit
    )

    col1, col2 = st.columns(2)

    col1.metric("Total Units Consumed", f"{total_units:.2f} kWh")
    col2.metric("Estimated Bill", f"₹{estimated_bill:.2f}")

    # Daily usage chart
    st.subheader("Daily Electricity Usage")

    daily_chart = px.bar(
        valid_usage,
        x="reading_date",
        y="daily_usage",
        title="Daily Electricity Consumption"
    )

    st.plotly_chart(daily_chart, width="stretch")

    # Monthly usage chart
    monthly_usage = calculate_monthly_usage(readings)

    st.subheader("Monthly Electricity Usage")

    monthly_chart = px.bar(
        x=monthly_usage.index.astype(str),
        y=monthly_usage.values,
        labels={
            "x": "Month",
            "y": "Units (kWh)"
        },
        title="Monthly Electricity Consumption"
    )

    st.plotly_chart(monthly_chart, width="stretch")

    # Monthly electricity bill report
    st.subheader("Monthly Electricity Bill Report")

    monthly_bill = monthly_usage * rate_per_unit

    monthly_report = pd.DataFrame({
        "Month": monthly_usage.index.astype(str),
        "Units Consumed (kWh)": monthly_usage.values,
        "Rate per Unit (₹)": rate_per_unit,
        "Estimated Bill (₹)": monthly_bill.values
    })

    st.dataframe(monthly_report, width="stretch")
    # Month-to-Month Electricity Comparison
    st.subheader("Month-to-Month Usage Comparison")

    if len(monthly_usage) >= 2:

        previous_month = monthly_usage.iloc[-2]
        current_month = monthly_usage.iloc[-1]

        col1, col2 = st.columns(2)

        col1.metric(
            "Previous Month",
            f"{previous_month:.2f} kWh"
        )

        col2.metric(
            "Current Month",
            f"{current_month:.2f} kWh",
            delta=f"{current_month - previous_month:.2f} kWh"
        )

        if previous_month > 0:
            percentage_change = (
                (current_month - previous_month)
                / previous_month
            ) * 100

            st.write(
                f"Consumption changed by {percentage_change:.2f}%"
            )

        else:
            st.info("Percentage comparison is unavailable.")

    else:
        st.info("At least two months of data are needed for comparison.")

    # Next month prediction
    # Monthly Consumption Trend Analysis
    st.subheader("Monthly Consumption Trend Analysis")

    trend_message = analyze_monthly_trend(monthly_usage)

    st.info(trend_message)

    trend_chart = px.line(
        x=monthly_usage.index.astype(str),
        y=monthly_usage.values,
        markers=True,
        labels={
            "x": "Month",
            "y": "Units (kWh)"
        },
        title="Monthly Electricity Consumption Trend"
    )

    st.plotly_chart(trend_chart, width="stretch")
    # Electricity Usage Insights
    st.subheader("Electricity Usage Insights")

    if len(monthly_usage) >= 2:

        highest_month = monthly_usage.idxmax()
        lowest_month = monthly_usage.idxmin()

        highest_usage = monthly_usage.max()
        lowest_usage = monthly_usage.min()

        st.write("Highest Consumption Month:", highest_month)
        st.write("Highest Usage:", round(highest_usage, 2), "kWh")

        st.write("Lowest Consumption Month:", lowest_month)
        st.write("Lowest Usage:", round(lowest_usage, 2), "kWh")

        if monthly_usage.iloc[0] != 0:
            percentage_change = (
                (monthly_usage.iloc[-1] - monthly_usage.iloc[0])
                / monthly_usage.iloc[0]
            ) * 100

            st.write(
                "Usage Change:",
                round(abs(percentage_change), 2),
                "%"
            )

            if percentage_change > 0:
                st.warning("Electricity usage increased.")
            elif percentage_change < 0:
                st.success("Electricity usage decreased.")
            else:
                st.info("Electricity usage remained stable.")

    else:
        st.info("Add readings from at least two months to view insights.")
    # CESC Electricity Bill Estimator
    st.subheader("CESC Electricity Bill Estimator")

    if len(monthly_usage) >= 1:

        latest_units = float(monthly_usage.iloc[-1])

        st.write("Latest Month Usage:", round(latest_units, 2), "kWh")

        fixed_charge = st.number_input(
            "Monthly Fixed Charge (₹)",
            min_value=0.0,
            value=150.0
        )

        rate = st.number_input(
            "Rate Per Unit (₹)",
            min_value=0.0,
            value=5.80
        )

        fuel_adjustment = st.number_input(
            "Fuel Adjustment (₹)",
            min_value=0.0,
            value=0.0
        )

        duty_rate = st.number_input(
            "Electricity Duty (%)",
            min_value=0.0,
            value=0.0
        )

        bill_details = calculate_cesc_bill(
            latest_units,
            fixed_charge,
            rate,
            fuel_adjustment,
            duty_rate
        )

        st.write("Energy Charge: ₹", bill_details["energy_charge"])
        st.write("Fixed Charge: ₹", bill_details["fixed_charge"])
        st.write("Fuel Adjustment: ₹", bill_details["fuel_adjustment"])
        st.write("Electricity Duty: ₹", bill_details["duty"])

        st.success(
            f"Estimated Electricity Bill: ₹{bill_details['total_bill']}"
        )
    st.subheader("Next Month Prediction")

    prediction = predict_next_month(
        monthly_usage.values.tolist()
    )

    if prediction is not None:
        st.success(
            f"Predicted consumption: {prediction} kWh"
        )

    else:
        st.info(
            "At least 3 months of monthly usage data "
            "are needed for prediction."
        )

    # Excel report download
    # Prediction Model Evaluation
    evaluation = evaluate_prediction_model(monthly_usage)

    st.subheader("Prediction Model Evaluation")

    if evaluation is None:
        st.info("Add at least 4 months of data to evaluate prediction accuracy.")

    else:
        st.write("Actual Usage:", evaluation["actual_usage"], "kWh")
        st.write("Predicted Usage:", evaluation["predicted_usage"], "kWh")

        st.metric(
            "Estimated Model Accuracy",
            f'{evaluation["accuracy"]}%'
        )
    # Historical vs Predicted Usage Chart
    chart_prediction = predict_next_month(monthly_usage)

    if chart_prediction is not None and len(monthly_usage) >= 3:

        chart_months = list(monthly_usage.index.astype(str)) + ["Next Month"]
        chart_values = list(monthly_usage.values) + [chart_prediction]

        prediction_chart = px.line(
            x=chart_months,
            y=chart_values,
            markers=True,
            labels={
                "x": "Month",
                "y": "Units (kWh)"
            },
            title="Historical and Predicted Electricity Usage"
        )

        st.plotly_chart(prediction_chart, width="stretch")

    else:
        st.info("Add at least 3 months of usage data to display the prediction chart.")
    st.subheader("Download Electricity Report")

    excel_data = BytesIO()

    with pd.ExcelWriter(excel_data, engine="openpyxl") as writer:
        analyzed.to_excel(
            writer,
            index=False,
            sheet_name="Daily Usage"
        )

        monthly_report.to_excel(
            writer,
            index=False,
            sheet_name="Monthly Report"
        )

    st.download_button(
        label="Download Excel Report",
        data=excel_data.getvalue(),
        file_name="electricity_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # CSV report download
    st.subheader("Download CSV Report")

    csv_data = analyzed.to_csv(index=False)

    st.download_button(
        label="Download CSV Report",
        data=csv_data,
        file_name="electricity_report.csv",
        mime="text/csv"
    )

    # Reading history
    st.subheader("Meter Reading History")

    st.dataframe(analyzed, width="stretch")

else:
    st.info(
        "Add at least two meter readings "
        "to see electricity consumption analysis."
    )

    if not readings.empty:
        st.subheader("Saved Meter Readings")
        st.dataframe(readings, width="stretch")

    # Electricity Usage Alerts
st.subheader("Daily and Monthly Usage Alerts")

daily_limit = st.number_input(
    "Daily Usage Limit (kWh)",
    min_value=1.0,
    value=15.0
)

monthly_limit = st.number_input(
    "Monthly Usage Limit (kWh)",
    min_value=1.0,
    value=300.0
)

if len(readings) >= 2:

    daily_readings = calculate_daily_usage(readings)
    latest_daily_usage = daily_readings["daily_usage"].dropna().iloc[-1]

    st.write("Latest Daily Usage:", round(latest_daily_usage, 2), "kWh")

    if latest_daily_usage > daily_limit:
        st.warning("Alert: Daily electricity usage exceeded your limit!")
    else:
        st.success("Daily electricity usage is within your limit.")

    latest_month_usage = float(monthly_usage.iloc[-1])

    st.write("Latest Month Usage:", round(latest_month_usage, 2), "kWh")

    if latest_month_usage > monthly_limit:
        st.warning("Alert: Monthly electricity usage exceeded your limit!")
    else:
        st.success("Monthly electricity usage is within your limit.")

else:
    st.info("Add at least two meter readings to activate usage alerts.")        