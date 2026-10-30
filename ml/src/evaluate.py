import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, X, y_scaled, y_scaler, device="cpu"):
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
    actual = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
    predicted = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    nonzero = np.abs(actual) > 1e-8
    mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE_percent": mape}, actual, predicted