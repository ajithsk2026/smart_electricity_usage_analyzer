import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import date, datetime
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Electricity Usage Analyzer",
    page_icon="⚡",
    layout="wide"
)

DB_PATH = os.path.join("data", "electricity.db")

# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def get_connection():
    return sqlite3.connect(DB_PATH)

# --------------------------------------------------
# LOAD EXISTING METER READINGS
# --------------------------------------------------

def get_readings():
    try:
        with get_connection() as conn:
            data = pd.read_sql_query(
                """
                SELECT id, reading_date, meter_reading
                FROM meter_readings
                ORDER BY reading_date ASC
                """,
                conn
            )

        data["reading_date"] = pd.to_datetime(
            data["reading_date"]
        )

        data["meter_reading"] = pd.to_numeric(
            data["meter_reading"],
            errors="coerce"
        )

        data = data.dropna(
            subset=["reading_date", "meter_reading"]
        )

        return data.reset_index(drop=True)

    except Exception as error:
        st.error(f"Database loading error: {error}")
        return pd.DataFrame(
            columns=["id", "reading_date", "meter_reading"]
        )

# --------------------------------------------------
# CALCULATE DAILY ELECTRICITY USAGE
# --------------------------------------------------

def calculate_daily_usage(data):
    result = data.copy()

    if result.empty:
        result["previous_reading"] = pd.Series(dtype=float)
        result["Daily Units"] = pd.Series(dtype=float)
        return result

    result["reading_date"] = pd.to_datetime(
        result["reading_date"]
    )

    result = result.sort_values("reading_date")

    result["previous_reading"] = (
        result["meter_reading"].shift(1)
    )

    result["Daily Units"] = (
        result["meter_reading"] -
        result["previous_reading"]
    )

    return result.reset_index(drop=True)

# --------------------------------------------------
# CALCULATE MONTHLY ELECTRICITY USAGE
# --------------------------------------------------

def calculate_monthly_usage(daily_data):
    valid = daily_data[
        daily_data["previous_reading"].notna()
    ].copy()

    if valid.empty:
        return pd.DataFrame(
            columns=["Month", "Units", "Estimated Bill"]
        )

    valid["Month"] = (
        valid["reading_date"].dt.to_period("M").astype(str)
    )

    monthly = (
        valid.groupby("Month")["Daily Units"]
        .sum()
        .reset_index()
    )

    monthly.rename(
        columns={"Daily Units": "Units"},
        inplace=True
    )

    monthly["Estimated Bill"] = (
        monthly["Units"] * 4.97 + 150
    )

    return monthly
# ============================================================
# PART 2 OF 6
# ANALYSIS, TRENDS AND LINEAR REGRESSION
# ============================================================

# ------------------------------------------------------------
# ELECTRICITY BILL CALCULATION
# ------------------------------------------------------------

def calculate_electricity_bill(units):
    units = max(0, float(units))
    return units * 4.97 + 150


# ------------------------------------------------------------
# UNUSUAL ELECTRICITY USAGE DETECTION
# MEDIAN AND MAD METHOD
# ------------------------------------------------------------

def find_unusual_usage(daily_data):

    valid = daily_data[
        daily_data["previous_reading"].notna()
    ].copy()

    columns = [
        "Date",
        "Daily Units",
        "Typical Usage",
        "Extra Units",
        "Reason"
    ]

    if len(valid) < 3:
        return pd.DataFrame(columns=columns)

    values = valid["Daily Units"].astype(float)

    median_usage = float(values.median())

    mad = float(
        np.median(np.abs(values - median_usage))
    )

    threshold = median_usage + (3 * 1.4826 * mad)

    threshold = max(
        threshold,
        median_usage * 1.5
    )

    unusual = valid[
        valid["Daily Units"] > threshold
    ].copy()

    if unusual.empty:
        return pd.DataFrame(columns=columns)

    unusual["Typical Usage"] = median_usage

    unusual["Extra Units"] = (
        unusual["Daily Units"] - median_usage
    )

    unusual["Reason"] = (
        "Possible increase from longer appliance usage"
    )

    unusual["Date"] = (
        unusual["reading_date"].dt.strftime("%d-%m-%Y")
    )

    return unusual[columns].reset_index(drop=True)


# ------------------------------------------------------------
# MONTHLY INCREASING / DECREASING TREND
# ------------------------------------------------------------

def get_monthly_trend(monthly):

    if len(monthly) < 2:
        return "Insufficient data", 0.0

    previous = float(monthly.iloc[-2]["Units"])
    current = float(monthly.iloc[-1]["Units"])

    if previous == 0:
        change = 0.0
    else:
        change = (
            (current - previous) / previous
        ) * 100

    if current > previous:
        return "Increasing", change

    elif current < previous:
        return "Decreasing", change

    return "Stable", 0.0


# ------------------------------------------------------------
# DAILY INCREASING / DECREASING TREND
# ------------------------------------------------------------

def get_daily_trend(daily_data):

    valid = daily_data[
        daily_data["previous_reading"].notna()
    ]

    if len(valid) < 2:
        return "Insufficient data", 0.0

    previous = float(valid.iloc[-2]["Daily Units"])
    current = float(valid.iloc[-1]["Daily Units"])

    difference = current - previous

    if current > previous:
        return "Increasing", difference

    elif current < previous:
        return "Decreasing", difference

    return "Stable", 0.0


# ------------------------------------------------------------
# LINEAR REGRESSION - NEXT MONTH PREDICTION
# ------------------------------------------------------------

def predict_next_month(monthly):

    if len(monthly) < 3:
        return None

    values = monthly["Units"].astype(float).values

    X = np.arange(len(values)).reshape(-1, 1)
    y = values

    model = LinearRegression()
    model.fit(X, y)

    next_month_number = np.array(
        [[len(values)]]
    )

    prediction = model.predict(
        next_month_number
    )[0]

    return max(0.0, float(prediction))


# ------------------------------------------------------------
# DAILY AVERAGE USAGE
# ------------------------------------------------------------

def calculate_average_daily_usage(daily_data):

    valid = daily_data[
        daily_data["previous_reading"].notna()
    ]

    if valid.empty:
        return 0.0

    return float(valid["Daily Units"].mean())


# ------------------------------------------------------------
# HIGHEST AND LOWEST CONSUMPTION
# ------------------------------------------------------------

def get_consumption_insights(monthly):

    if monthly.empty:
        return None, None

    highest = monthly.loc[
        monthly["Units"].idxmax()
    ]

    lowest = monthly.loc[
        monthly["Units"].idxmin()
    ]

    return highest, lowest
# ============================================================
# PART 3 OF 6
# DESIGN, DATA LOADING AND TOP NAVIGATION
# ============================================================

from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# LOAD APPLICATION DATA
# ------------------------------------------------------------

df = get_readings()

daily = calculate_daily_usage(df)

monthly = calculate_monthly_usage(daily)

unusual = find_unusual_usage(daily)


# ------------------------------------------------------------
# PROFESSIONAL NAVY BLUE AND WHITE DESIGN
# ------------------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F4F7FB;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

.main-header {
    background: linear-gradient(120deg, #142D4E, #28558A);
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
}

.main-header h1 {
    color: white;
    font-size: 30px;
    margin-bottom: 5px;
}

.main-header p {
    color: #DCE7F5;
    margin: 0;
}

.metric-card {
    background: white;
    border: 1px solid #E1E8F0;
    border-radius: 12px;
    padding: 17px;
    box-shadow: 0 2px 8px rgba(20,45,78,0.05);
}

.section-title {
    color: #142D4E;
    font-weight: 700;
    font-size: 22px;
    margin-top: 12px;
    margin-bottom: 12px;
}

.info-card {
    background: white;
    border-left: 5px solid #28558A;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
}

.footer {
    background: #142D4E;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    color: white;
    margin-top: 30px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E1E8F0;
    padding: 15px;
    border-radius: 12px;
}

div.stButton > button {
    border-radius: 8px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>⚡ Smart Electricity Usage Analyzer</h1>
    <p>Track • Analyze • Understand • Save Energy</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# TOP HORIZONTAL NAVIGATION
# ------------------------------------------------------------

pages = [
    "Overview",
    "Daily Usage",
    "Monthly Usage",
    "Unusual Usage",
    "Consumption Trends",
    "Prediction",
    "Reports",
    "Manage Data"
]

selected_page = st.radio(
    "Application Navigation",
    pages,
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()


# ------------------------------------------------------------
# COMMON DATA SUMMARY
# ------------------------------------------------------------

valid_daily = daily[
    daily["previous_reading"].notna()
].copy()

total_units = (
    float(valid_daily["Daily Units"].sum())
    if not valid_daily.empty else 0.0
)

average_daily = calculate_average_daily_usage(daily)

latest_reading = (
    float(df.iloc[-1]["meter_reading"])
    if not df.empty else 0.0
)

estimated_bill = calculate_electricity_bill(total_units)


# ------------------------------------------------------------
# COMMON FOOTER
# ------------------------------------------------------------

def show_footer():

    st.markdown("""
    <div class="footer">
        <b>Smart Electricity Usage Analyzer</b><br>
        Track • Analyze • Understand • Save Energy
    </div>
    """, unsafe_allow_html=True)
# ============================================================
# PART 4 OF 6
# OVERVIEW AND DAILY USAGE
# ============================================================


# ============================================================
# PAGE 1 - OVERVIEW
# ============================================================

if selected_page == "Overview":

    st.markdown(
        '<div class="section-title">Electricity Overview</div>',
        unsafe_allow_html=True
    )

    # FOUR SUMMARY CARDS
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total Electricity Usage",
            f"{total_units:.2f} kWh"
        )

    with c2:
        st.metric(
            "Average Daily Usage",
            f"{average_daily:.2f} units"
        )

    with c3:
        st.metric(
            "Latest Meter Reading",
            f"{latest_reading:.0f}"
        )

    with c4:
        st.metric(
            "Estimated Electricity Bill",
            f"₹{estimated_bill:.2f}"
        )

    st.divider()

    # DAILY AND MONTHLY CHARTS
    left, right = st.columns(2)

    with left:
        st.markdown("### Daily Electricity Consumption")

        if not valid_daily.empty:

            chart_data = valid_daily.tail(7)

            fig = px.line(
                chart_data,
                x="reading_date",
                y="Daily Units",
                markers=True,
                title="Last 7 Recorded Usage Intervals"
            )

            fig.update_traces(
                line=dict(color="#2878B5", width=3),
                marker=dict(size=8)
            )

            fig.update_layout(
                xaxis_title="Reading Date",
                yaxis_title="Units Consumed",
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Add more readings to display the daily chart.")

    with right:
        st.markdown("### Monthly Electricity Consumption")

        if not monthly.empty:

            chart_data = monthly.tail(6)

            fig = px.bar(
                chart_data,
                x="Month",
                y="Units",
                text_auto=".2f",
                title="Last 6 Recorded Months"
            )

            fig.update_traces(
                marker_color="#28558A"
            )

            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Units Consumed",
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Monthly chart will appear when data is available.")

    st.divider()

    # KEY MONTHLY INSIGHTS
    st.markdown("### Key Consumption Insights")

    a, b, c = st.columns(3)

    highest, lowest = get_consumption_insights(monthly)

    with a:
        if highest is not None:
            st.success(
                f"Highest Month: {highest['Month']}\n\n"
                f"{float(highest['Units']):.2f} units"
            )
        else:
            st.info("Highest month: Waiting for data.")

    with b:
        if lowest is not None:
            st.info(
                f"Lowest Month: {lowest['Month']}\n\n"
                f"{float(lowest['Units']):.2f} units"
            )
        else:
            st.info("Lowest month: Waiting for data.")

    with c:
        trend, change = get_monthly_trend(monthly)

        if trend == "Increasing":
            st.warning(
                f"Monthly Trend: Increasing\n\n"
                f"Usage increased by {change:.1f}%."
            )
        elif trend == "Decreasing":
            st.success(
                f"Monthly Trend: Decreasing\n\n"
                f"Usage decreased by {abs(change):.1f}%."
            )
        elif trend == "Stable":
            st.info("Monthly Trend: Stable")
        else:
            st.info("Need at least two months of data.")

    # DAILY TREND
    st.markdown("### Latest Daily Usage Trend")

    daily_trend, daily_change = get_daily_trend(daily)

    if daily_trend == "Increasing":
        st.warning(
            f"Daily consumption increased by {daily_change:.2f} units "
            "compared with the previous recorded interval."
        )
    elif daily_trend == "Decreasing":
        st.success(
            f"Daily consumption decreased by {abs(daily_change):.2f} units "
            "compared with the previous recorded interval."
        )
    elif daily_trend == "Stable":
        st.info("Latest daily consumption is stable.")
    else:
        st.info("Add more readings to compare daily consumption.")

    st.divider()

    # UNUSUAL USAGE SUMMARY
    st.markdown("### Unusual Usage Alert")

    if not unusual.empty:

        latest_alert = unusual.iloc[-1]

        st.warning(
            f"Unusual usage detected on {latest_alert['Date']}: "
            f"{float(latest_alert['Daily Units']):.2f} units."
        )

        st.caption(
            f"Typical usage: {float(latest_alert['Typical Usage']):.2f} units | "
            f"Extra usage: {float(latest_alert['Extra Units']):.2f} units"
        )

        st.caption(
            "Possible reasons include longer operation of a water motor, "
            "heater, air conditioner or other appliances."
        )

    else:
        st.success("No unusual usage detected in the available data.")

    st.divider()

    # NEXT MONTH PREDICTION
    st.markdown("### Next Month Prediction")

    if len(monthly) >= 3:

        predicted_units = predict_next_month(monthly)

        if predicted_units is not None:

            predicted_units = max(0, float(predicted_units))

            predicted_bill = calculate_electricity_bill(
                predicted_units
            )

            p1, p2 = st.columns(2)

            with p1:
                st.metric(
                    "Predicted Units",
                    f"{predicted_units:.2f} kWh"
                )

            with p2:
                st.metric(
                    "Predicted Bill",
                    f"₹{predicted_bill:.2f}"
                )

            st.caption(
                "Calculated using Linear Regression from monthly usage."
            )

    else:
        st.info(
            "At least three months of recorded consumption are needed "
            "for Linear Regression prediction."
        )

    # ENERGY SAVING TIP
    st.markdown("### Energy Saving Tip")

    if average_daily > 0 and not valid_daily.empty:

        latest_usage = float(
            valid_daily.iloc[-1]["Daily Units"]
        )

        if latest_usage > average_daily:
            st.info(
                "Your latest usage is above your average. Check whether "
                "high-power appliances such as motors, heaters or ACs "
                "were used for longer periods."
            )
        else:
            st.success(
                "Your latest usage is within or below your average. "
                "Continue monitoring and switching off unused appliances."
            )

    else:
        st.info("More readings are needed for personalized usage tips.")

    show_footer()


# ============================================================
# PAGE 2 - DAILY USAGE
# ============================================================

elif selected_page == "Daily Usage":

    st.markdown(
        '<div class="section-title">Daily Electricity Usage Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Understand how much electricity was consumed between "
        "successive meter readings."
    )

    if not valid_daily.empty:

        # DAILY SUMMARY
        highest_day = valid_daily.loc[
            valid_daily["Daily Units"].idxmax()
        ]

        lowest_day = valid_daily.loc[
            valid_daily["Daily Units"].idxmin()
        ]

        d1, d2, d3 = st.columns(3)

        with d1:
            st.metric(
                "Total Recorded Units",
                f"{total_units:.2f}"
            )

        with d2:
            st.metric(
                "Average Daily Usage",
                f"{average_daily:.2f}"
            )

        with d3:
            st.metric(
                "Highest Recorded Usage",
                f"{float(highest_day['Daily Units']):.2f}"
            )

        st.divider()

        # DAILY LINE CHART
        st.markdown("### Daily Consumption Chart")

        fig = px.line(
            valid_daily,
            x="reading_date",
            y="Daily Units",
            markers=True,
            title="Electricity Units Consumed Between Readings"
        )

        fig.update_traces(
            line=dict(color="#2878B5", width=3),
            marker=dict(size=8)
        )

        fig.update_layout(
            xaxis_title="Reading Date",
            yaxis_title="Daily Units",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # HIGHEST AND LOWEST DAYS
        st.markdown("### Daily Usage Insights")

        x, y = st.columns(2)

        with x:
            st.warning(
                f"Highest usage: "
                f"{float(highest_day['Daily Units']):.2f} units "
                f"on {highest_day['reading_date'].strftime('%d-%m-%Y')}."
            )

        with y:
            st.success(
                f"Lowest usage: "
                f"{float(lowest_day['Daily Units']):.2f} units "
                f"on {lowest_day['reading_date'].strftime('%d-%m-%Y')}."
            )

        # DAILY INCREASE / DECREASE
        trend, difference = get_daily_trend(daily)

        st.markdown("### Increasing / Decreasing Usage")

        if trend == "Increasing":
            st.warning(
                f"Usage increased by {difference:.2f} units "
                "compared with the previous recorded interval."
            )

        elif trend == "Decreasing":
            st.success(
                f"Usage decreased by {abs(difference):.2f} units "
                "compared with the previous recorded interval."
            )

        elif trend == "Stable":
            st.info("Usage remained stable.")

        else:
            st.info("More daily readings are needed for comparison.")

        # DAILY DATA TABLE
        st.markdown("### Daily Usage Records")

        display_daily = valid_daily[
            ["reading_date", "previous_reading",
             "meter_reading", "Daily Units"]
        ].copy()

        display_daily.columns = [
            "Date",
            "Previous Meter Reading",
            "Current Meter Reading",
            "Units Consumed"
        ]

        display_daily["Date"] = (
            pd.to_datetime(display_daily["Date"])
            .dt.strftime("%d-%m-%Y")
        )

        st.dataframe(
            display_daily.sort_values(
                "Date", ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "Add at least two meter readings to calculate daily usage."
        )

    show_footer()
elif selected_page == "Monthly Usage":
    st.markdown("## Monthly Electricity Usage")

    if monthly.empty:
        st.info("Add at least two meter readings to view monthly usage.")
    else:
        col1, col2, col3 = st.columns(3)

        average_monthly = monthly["Units"].mean()
        highest_month = monthly.loc[monthly["Units"].idxmax()]
        latest_month = monthly.iloc[-1]

        col1.metric("Average Monthly Usage", f"{average_monthly:.2f} units")
        col2.metric("Highest Usage Month", f"{highest_month['Units']:.2f} units")
        col3.metric("Latest Month Bill", f"₹{latest_month['Estimated Bill']:.2f}")

        st.markdown("### Monthly Consumption Chart")
        fig = px.bar(
            monthly,
            x="Month",
            y="Units",
            text="Units",
            title="Month-wise Electricity Consumption",
            labels={"Month": "Month", "Units": "Electricity Units"},
            color="Units",
            color_continuous_scale=["#A9C7E8", "#28558A"]
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Monthly Usage Details")
        display_monthly = monthly.copy()
        display_monthly["Units"] = display_monthly["Units"].round(2)
        display_monthly["Estimated Bill"] = display_monthly["Estimated Bill"].round(2)
        st.dataframe(display_monthly, use_container_width=True, hide_index=True)

        st.markdown("### Month-to-Month Comparison")
        if len(monthly) >= 2:
            previous = float(monthly.iloc[-2]["Units"])
            current = float(monthly.iloc[-1]["Units"])
            difference = current - previous

            if difference > 0:
                st.warning(
                    f"Usage increased by {difference:.2f} units "
                    "compared with the previous month."
                )
            elif difference < 0:
                st.success(
                    f"Usage decreased by {abs(difference):.2f} units "
                    "compared with the previous month."
                )
            else:
                st.info("Usage remained the same as the previous month.")
        else:
            st.info("Add more monthly readings to compare months.")

        st.caption(
            "Bill is a project estimate using ₹4.97 per unit + ₹150 fixed charge. "
            "It may differ from your actual electricity bill."
        )

    show_footer()


elif selected_page == "Unusual Usage":
    st.markdown("## Unusual Electricity Usage")
    st.write(
        "This section identifies days when electricity consumption is "
        "much higher than your usual usage."
    )

    if len(daily[daily["previous_reading"].notna()]) < 3:
        st.info("At least three daily usage records are needed to identify unusual usage.")
    else:
        if unusual.empty:
            st.success("No unusual usage was detected in the available readings.")
        else:
            st.warning(f"{len(unusual)} unusual usage record(s) detected.")

            st.markdown("### Unusual Usage Details")
            st.dataframe(unusual, use_container_width=True, hide_index=True)

            st.markdown("### Actual Usage Compared with Typical Usage")
            chart_data = unusual[["Date", "Daily Units", "Typical Usage"]].melt(
                id_vars="Date",
                var_name="Usage Type",
                value_name="Units"
            )
            fig = px.bar(
                chart_data,
                x="Date",
                y="Units",
                color="Usage Type",
                barmode="group",
                title="Unusual Days: Actual vs Typical Usage",
                color_discrete_map={
                    "Daily Units": "#28558A",
                    "Typical Usage": "#A9C7E8"
                }
            )
            st.plotly_chart(fig, use_container_width=True)

            for _, item in unusual.iterrows():
                st.markdown(
                    f"**{item['Date']} — {item['Daily Units']:.2f} units used**"
                )
                st.write(
                    f"Your usage was approximately {item['Extra Units']:.2f} "
                    f"units above the typical usage of {item['Typical Usage']:.2f} units."
                )

    st.markdown("### Possible Appliance-Related Reasons")
    st.write(
        "A sudden increase may happen when appliances are used for longer hours "
        "or when several appliances operate together. These are examples, not "
        "confirmed causes of your meter's usage."
    )

    appliance_examples = [
        {
            "Appliance": "Water Motor",
            "Typical Power": "500–1000 W",
            "How it uses electricity": (
                "Pumps water to an overhead tank. Longer pumping time or "
                "frequent switching can increase electricity consumption."
            ),
            "Saving Tip": "Check for water leakage and avoid unnecessary pumping."
        },
        {
            "Appliance": "Television",
            "Typical Power": "50–150 W",
            "How it uses electricity": (
                "Uses electricity while displaying programmes. Long viewing "
                "hours increase its total energy use."
            ),
            "Saving Tip": "Switch it off when nobody is watching."
        },
        {
            "Appliance": "Electric Heater",
            "Typical Power": "1000–2000 W",
            "How it uses electricity": (
                "Converts electricity into heat. Extended use can consume "
                "a relatively large amount of energy."
            ),
            "Saving Tip": "Use only when required and switch it off afterwards."
        },
        {
            "Appliance": "Air Conditioner",
            "Typical Power": "800–2000 W",
            "How it uses electricity": (
                "Cools indoor air. Longer operating hours and heavy cooling "
                "demand can increase consumption."
            ),
            "Saving Tip": "Use a moderate temperature and clean the filters."
        },
        {
            "Appliance": "Refrigerator",
            "Typical Power": "100–300 W",
            "How it uses electricity": (
                "Keeps food cold and operates throughout the day in cycles. "
                "Frequent door opening can increase cooling demand."
            ),
            "Saving Tip": "Keep the door closed and allow proper ventilation."
        }
    ]

    for appliance in appliance_examples:
        with st.expander(appliance["Appliance"]):
            st.write(f"**Typical power:** {appliance['Typical Power']}")
            st.write(f"**How it uses electricity:** {appliance['How it uses electricity']}")
            st.write(f"**Saving tip:** {appliance['Saving Tip']}")

    st.info(
        "Important: Meter readings show total electricity consumption. "
        "They cannot identify exactly which appliance caused an increase. "
        "The appliance examples above are only possible explanations."
    )

    show_footer()


elif selected_page == "Consumption Trends":
    st.markdown("## Electricity Consumption Trends")

    if monthly.empty:
        st.info("Add more meter readings to view consumption trends.")
    else:
        trend, change = get_monthly_trend(monthly)

        col1, col2 = st.columns(2)
        col1.metric("Monthly Trend", trend)
        col2.metric("Change from Previous Month", f"{change:.2f}%")

        st.markdown("### Monthly Consumption Trend")
        fig = px.line(
            monthly,
            x="Month",
            y="Units",
            markers=True,
            title="Increasing or Decreasing Electricity Usage",
            labels={"Month": "Month", "Units": "Electricity Units"}
        )
        fig.update_traces(line=dict(color="#28558A", width=3))
        st.plotly_chart(fig, use_container_width=True)

        if trend == "Increasing":
            st.warning(
                f"Your electricity usage is increasing. The latest month "
                f"shows a {change:.2f}% increase compared with the previous month."
            )
            st.write(
                "Check whether appliances are being used for longer hours "
                "and try to reduce unnecessary consumption."
            )
        elif trend == "Decreasing":
            st.success(
                f"Your electricity usage is decreasing. The latest month "
                f"shows a {abs(change):.2f}% reduction compared with the previous month."
            )
            st.write("Continue following energy-saving habits.")
        elif trend == "Stable":
            st.info("Your electricity usage is stable compared with the previous month.")
        else:
            st.info("At least two months of usage data are needed to identify a trend.")

        st.markdown("### Daily Usage Trend")
        valid_daily_trend = daily[daily["previous_reading"].notna()].copy()

        if not valid_daily_trend.empty:
            fig_daily = px.line(
                valid_daily_trend,
                x="reading_date",
                y="Daily Units",
                markers=True,
                title="Daily Electricity Consumption"
            )
            fig_daily.update_layout(
                xaxis_title="Date",
                yaxis_title="Daily Units"
            )
            st.plotly_chart(fig_daily, use_container_width=True)
        else:
            st.info("More readings are needed to display daily trends.")

    show_footer()


elif selected_page == "Prediction":
    st.markdown("## Next Month Electricity Prediction")

    st.write(
        "This section uses Linear Regression to estimate next month's "
        "electricity consumption from your previous monthly usage."
    )

    if len(monthly) < 3:
        st.warning(
            "At least 3 months of monthly usage data are required "
            "to generate a prediction."
        )
    else:
        predicted_units = predict_next_month(monthly)
        predicted_bill = calculate_electricity_bill(predicted_units)

        col1, col2 = st.columns(2)
        col1.metric("Predicted Next Month Units", f"{predicted_units:.2f} units")
        col2.metric("Estimated Next Month Bill", f"₹{predicted_bill:.2f}")

        st.markdown("### Historical Usage and Next Month Prediction")

        history = monthly[["Month", "Units"]].copy()
        prediction_row = pd.DataFrame({
            "Month": ["Next Month"],
            "Units": [predicted_units]
        })
        prediction_chart = pd.concat(
            [history, prediction_row],
            ignore_index=True
        )
        prediction_chart["Type"] = (
            ["Historical"] * len(history) + ["Prediction"]
        )

        fig = px.bar(
            prediction_chart,
            x="Month",
            y="Units",
            color="Type",
            title="Historical and Predicted Electricity Usage",
            color_discrete_map={
                "Historical": "#28558A",
                "Prediction": "#E6A23C"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### How the Prediction Works")
        st.write(
            "1. The application collects monthly electricity usage.\n"
            "2. It uses the month sequence as the input feature.\n"
            "3. Linear Regression learns the usage trend.\n"
            "4. The trained model estimates the next month's usage."
        )

        st.caption(
            "This is an estimate based on past readings. Actual usage and "
            "electricity bills may change depending on weather, appliances, "
            "occupancy and daily habits."
        )

    show_footer()
elif selected_page == "Reports":
    st.markdown("## Electricity Usage Reports")
    st.write("Download your electricity consumption data for future reference.")

    daily_report = daily.copy()
    monthly_report = monthly.copy()

    if not daily_report.empty:
        daily_report["reading_date"] = (
            daily_report["reading_date"].dt.strftime("%d-%m-%Y")
        )

    st.markdown("### Daily Usage Report")
    st.dataframe(daily_report, use_container_width=True, hide_index=True)

    st.markdown("### Monthly Usage Report")
    st.dataframe(monthly_report, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        csv_data = daily_report.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Daily Report (CSV)",
            data=csv_data,
            file_name="electricity_daily_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        monthly_csv = monthly_report.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Monthly Report (CSV)",
            data=monthly_csv,
            file_name="electricity_monthly_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("### Excel Report")

    from io import BytesIO

    try:
        excel_buffer = BytesIO()

        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            daily_report.to_excel(
                writer,
                sheet_name="Daily Usage",
                index=False
            )
            monthly_report.to_excel(
                writer,
                sheet_name="Monthly Usage",
                index=False
            )

        st.download_button(
            label="Download Complete Excel Report",
            data=excel_buffer.getvalue(),
            file_name="smart_electricity_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    except Exception as error:
        st.error(f"Excel report could not be created: {error}")
        st.info("Check that openpyxl is installed.")

    show_footer()


elif selected_page == "Manage Data":
    st.markdown("## Manage Meter Readings")
    st.write(
        "Add new readings or edit and delete individual stored readings. "
        "Your existing readings will be preserved."
    )

    st.markdown("### Add New Reading")

    with st.form("add_reading_form"):
        new_date = st.date_input(
            "Reading Date",
            value=date.today(),
            key="new_reading_date"
        )

        new_reading = st.number_input(
            "Current Meter Reading",
            min_value=0.0,
            step=1.0,
            format="%.2f",
            key="new_meter_reading"
        )

        add_submitted = st.form_submit_button(
            "Add Reading",
            use_container_width=True
        )

    if add_submitted:
        current_data = get_readings()
        duplicate = (
            current_data["reading_date"].dt.date == new_date
        ).any() if not current_data.empty else False

        if duplicate:
            st.error("A reading already exists for this date.")
        else:
            candidate = pd.concat(
                [
                    current_data[["reading_date", "meter_reading"]],
                    pd.DataFrame({
                        "reading_date": [pd.Timestamp(new_date)],
                        "meter_reading": [float(new_reading)]
                    })
                ],
                ignore_index=True
            ).sort_values("reading_date")

            if (
                len(candidate) > 1
                and (candidate["meter_reading"].diff().dropna() < 0).any()
            ):
                st.error(
                    "Invalid meter reading. The reading cannot be lower "
                    "than the previous chronological meter reading."
                )
            else:
                try:
                    with get_connection() as conn:
                        conn.execute(
                            """
                            INSERT INTO meter_readings
                            (reading_date, meter_reading)
                            VALUES (?, ?)
                            """,
                            (new_date.strftime("%Y-%m-%d"), float(new_reading))
                        )
                    st.success("New meter reading added successfully.")
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not add reading: {error}")

    st.divider()
    st.markdown("### Stored Readings")

    current_data = get_readings()

    if current_data.empty:
        st.info("No meter readings are currently stored.")
    else:
        st.caption(
            "Each stored reading has its own Edit and Delete controls."
        )

        for _, reading in current_data.sort_values(
            "reading_date",
            ascending=False
        ).iterrows():

            reading_id = int(reading["id"])
            reading_date = reading["reading_date"].date()
            meter_value = float(reading["meter_reading"])

            st.markdown(
                f"**Reading: {reading_date.strftime('%d-%m-%Y')}** "
                f"— {meter_value:.2f} units on the meter"
            )

            edit_col, delete_col = st.columns([3, 1])

            with edit_col:
                with st.expander(
                    f"Edit Reading {reading_id}",
                    expanded=False
                ):
                    with st.form(f"edit_form_{reading_id}"):
                        edited_date = st.date_input(
                            "Reading Date",
                            value=reading_date,
                            key=f"edit_date_{reading_id}"
                        )

                        edited_value = st.number_input(
                            "Meter Reading",
                            min_value=0.0,
                            value=meter_value,
                            step=1.0,
                            format="%.2f",
                            key=f"edit_value_{reading_id}"
                        )

                        edit_submitted = st.form_submit_button(
                            "Save Changes",
                            use_container_width=True
                        )

                    if edit_submitted:
                        other_data = current_data[
                            current_data["id"] != reading_id
                        ].copy()

                        duplicate_date = (
                            other_data["reading_date"].dt.date == edited_date
                        ).any()

                        if duplicate_date:
                            st.error(
                                "Another reading already exists for this date."
                            )
                        else:
                            candidate = pd.concat(
                                [
                                    other_data[
                                        ["reading_date", "meter_reading"]
                                    ],
                                    pd.DataFrame({
                                        "reading_date": [
                                            pd.Timestamp(edited_date)
                                        ],
                                        "meter_reading": [
                                            float(edited_value)
                                        ]
                                    })
                                ],
                                ignore_index=True
                            ).sort_values("reading_date")

                            if (
                                len(candidate) > 1
                                and (
                                    candidate["meter_reading"]
                                    .diff()
                                    .dropna() < 0
                                ).any()
                            ):
                                st.error(
                                    "Invalid change. Meter readings must "
                                    "not decrease in date order."
                                )
                            else:
                                try:
                                    with get_connection() as conn:
                                        conn.execute(
                                            """
                                            UPDATE meter_readings
                                            SET reading_date = ?,
                                                meter_reading = ?
                                            WHERE id = ?
                                            """,
                                            (
                                                edited_date.strftime("%Y-%m-%d"),
                                                float(edited_value),
                                                reading_id
                                            )
                                        )
                                    st.success("Reading updated successfully.")
                                    st.rerun()
                                except Exception as error:
                                    st.error(
                                        f"Could not update reading: {error}"
                                    )

            with delete_col:
                if st.button(
                    "Delete",
                    key=f"delete_button_{reading_id}",
                    use_container_width=True
                ):
                    st.session_state[
                        f"confirm_delete_{reading_id}"
                    ] = True

                if st.session_state.get(
                    f"confirm_delete_{reading_id}",
                    False
                ):
                    st.warning("Are you sure you want to delete this reading?")

                    confirm_col, cancel_col = st.columns(2)

                    with confirm_col:
                        if st.button(
                            "Yes",
                            key=f"confirm_yes_{reading_id}"
                        ):
                            try:
                                with get_connection() as conn:
                                    conn.execute(
                                        "DELETE FROM meter_readings WHERE id = ?",
                                        (reading_id,)
                                    )

                                st.session_state[
                                    f"confirm_delete_{reading_id}"
                                ] = False

                                st.success("Reading deleted successfully.")
                                st.rerun()

                            except Exception as error:
                                st.error(f"Could not delete reading: {error}")

                    with cancel_col:
                        if st.button(
                            "Cancel",
                            key=f"confirm_cancel_{reading_id}"
                        ):
                            st.session_state[
                                f"confirm_delete_{reading_id}"
                            ] = False
                            st.rerun()

            st.divider()

    show_footer()