"""Per-horizon evaluation and a no-change forecasting baseline."""
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


def inverse_prices(values, scaler):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Expected nonempty two-dimensional (samples, horizons) array")
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def metric_summary(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Actual and predicted shapes must match and be nonempty")
    mae = float(mean_absolute_error(actual.ravel(), predicted.ravel()))
    rmse = float(np.sqrt(mean_squared_error(actual.ravel(), predicted.ravel())))
    valid = np.abs(actual) > 1e-8
    mape = float(np.mean(np.abs((actual[valid] - predicted[valid]) / actual[valid])) * 100) if valid.any() else None
    return {"MAE": mae, "RMSE": rmse, "MAPE_percent": mape}


def evaluate_predictions(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.ndim != 2 or actual.shape != predicted.shape:
        raise ValueError("Expected equal (samples, horizons) arrays")
    metrics = metric_summary(actual, predicted)
    metrics["per_horizon"] = {
        f"day_{i+1}": metric_summary(actual[:, i], predicted[:, i])
        for i in range(actual.shape[1])
    }
    return metrics


def naive_last_close_baseline(X, y_scaler, target_idx):
    """Repeated last observed close; assumes X Close feature uses y_scaler."""
    if X.ndim != 3:
        raise ValueError("Expected three-dimensional windows")
    return y_scaler.inverse_transform(X[:, -1, target_idx].reshape(-1, 1)).ravel()


def evaluate_model(model, X, y_scaled, y_scaler, device="cpu", target_idx=None):
    if len(X) == 0 or len(X) != len(y_scaled):
        raise ValueError("Empty test set or mismatched samples")
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    actual = inverse_prices(y_scaled, y_scaler)
    predicted = inverse_prices(pred_scaled, y_scaler)
    metrics = evaluate_predictions(actual, predicted)
    if target_idx is not None:
        last_close = naive_last_close_baseline(X, y_scaler, target_idx)
        baseline = np.repeat(last_close[:, None], actual.shape[1], axis=1)
        metrics["naive_baseline"] = evaluate_predictions(actual, baseline)
    return metrics, actual, predicted
