import pandas as pd


def calculate_daily_usage(readings):
    """Calculate electricity consumption between meter readings."""

    readings = readings.sort_values("reading_date").copy()

    readings["daily_usage"] = (
        readings["meter_reading"].diff()
    )

    return readings


def calculate_monthly_usage(readings):
    """Calculate total electricity consumption per month."""

    readings = calculate_daily_usage(readings)

    readings["month"] = pd.to_datetime(
        readings["reading_date"]
    ).dt.to_period("M")

    monthly_usage = readings.groupby("month")[
        "daily_usage"
    ].sum()

    return monthly_usage


def calculate_electricity_bill(units, rate_per_unit=5.0):
    """Estimate electricity bill using a simple demo rate."""

    if units < 0:
        raise ValueError("Units cannot be negative.")

    bill = units * rate_per_unit

    return round(bill, 2)

def calculate_cesc_bill(units, fixed_charge=150, rate_per_unit=5.80,
                        fuel_adjustment=0, duty_rate=0):

    if units < 0:
        raise ValueError("Units cannot be negative.")

    energy_charge = units * rate_per_unit
    total_before_duty = energy_charge + fixed_charge + fuel_adjustment
    duty = total_before_duty * (duty_rate / 100)

    total_bill = total_before_duty + duty

    return {
        "units": units,
        "energy_charge": round(energy_charge, 2),
        "fixed_charge": round(fixed_charge, 2),
        "fuel_adjustment": round(fuel_adjustment, 2),
        "duty": round(duty, 2),
        "total_bill": round(total_bill, 2)
    }
def detect_unusual_usage(readings):
    """Identify unusually high daily electricity consumption."""

    readings = calculate_daily_usage(readings)

    usage = readings["daily_usage"].dropna()

    if len(usage) < 4:
        return readings, "Add at least 4 daily usage records for anomaly detection."

    average = usage.mean()
    standard_deviation = usage.std()

    threshold = average + (2 * standard_deviation)

    readings["unusual_usage"] = (
        readings["daily_usage"] > threshold
    )

    return readings, None

def analyze_monthly_trend(monthly_usage):

    if len(monthly_usage) < 2:
        return "Not enough data"

    first_month = monthly_usage.iloc[0]
    last_month = monthly_usage.iloc[-1]

    if last_month > first_month:
        return "Electricity usage is increasing"

    elif last_month < first_month:
        return "Electricity usage is decreasing"

    else:
        return "Electricity usage is stable"