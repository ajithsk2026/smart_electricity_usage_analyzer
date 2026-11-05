import pandas as pd
from sklearn.linear_model import LinearRegression


def predict_next_month(monthly_usage):
    """Predict next month's electricity consumption."""

    if len(monthly_usage) < 3:
        return None

    data = pd.DataFrame({
        "month_number": range(len(monthly_usage)),
        "usage": monthly_usage
    })

    X = data[["month_number"]]
    y = data["usage"]

    model = LinearRegression()
    model.fit(X, y)

    next_month = [[len(monthly_usage)]]

    prediction = model.predict(next_month)[0]

    return round(max(0, prediction), 2)

def evaluate_prediction_model(monthly_usage):
    """Evaluate prediction using the last month's actual usage."""

    if len(monthly_usage) < 4:
        return None

    train_data = monthly_usage.iloc[:-1]
    actual_usage = float(monthly_usage.iloc[-1])

    X_train = [[i] for i in range(len(train_data))]
    y_train = train_data.values

    model = LinearRegression()
    model.fit(X_train, y_train)

    predicted_usage = model.predict([[len(train_data)]])[0]

    error = abs(actual_usage - predicted_usage)

    if actual_usage != 0:
        accuracy = max(0, 100 - (error / actual_usage * 100))
    else:
        accuracy = 0

    return {
        "actual_usage": round(actual_usage, 2),
        "predicted_usage": round(max(0, predicted_usage), 2),
        "accuracy": round(accuracy, 2)
    }