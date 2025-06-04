import numpy as np


def print_model_evaluation(metrics):
    """
    Print model evaluation results in a user-friendly format
    """
    print(f"\n{'='*50}")
    print(f"🎯 MODEL PERFORMANCE EVALUATION")
    print(f"{'='*50}")
    
    print(f"\n📊 GOODNESS OF FIT:")
    print(f"   R² (R-squared): {metrics['R_squared']:.4f}")
    if metrics['R_squared'] >= 0.9:
        print(f"   📈 EXCELLENT! Model explains {metrics['R_squared']*100:.1f}% of variance")
    elif metrics['R_squared'] >= 0.7:
        print(f"   📊 GOOD! Model explains {metrics['R_squared']*100:.1f}% of variance")
    elif metrics['R_squared'] >= 0.5:
        print(f"   📉 MODERATE! Model explains {metrics['R_squared']*100:.1f}% of variance")
    else:
        print(f"   ❌ POOR! Model explains only {metrics['R_squared']*100:.1f}% of variance")
    
    print(f"\n💰 ERROR METRICS:")
    print(f"   RMSE: €{metrics['RMSE']:.2f}")
    print(f"   MSE: {metrics['MSE']:.2f}")
    print(f"   MAE:  €{metrics['MAE']:.2f}")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    
    print(f"\n🔍 WHAT THIS MEANS:")
    print(f"   • On average, predictions are off by €{metrics['MAE']:.0f}")
    print(f"   • Typical prediction error: €{metrics['RMSE']:.0f}")
    print(f"   • Average percentage error: {metrics['MAPE']:.1f}%")
    
    # Interpretation based on MAPE
    if metrics['MAPE'] <= 10:
        print(f"   ✅ EXCELLENT prediction accuracy!")
    elif metrics['MAPE'] <= 20:
        print(f"   ✅ GOOD prediction accuracy!")
    elif metrics['MAPE'] <= 30:
        print(f"   ⚠️  MODERATE prediction accuracy")
    else:
        print(f"   ❌ POOR prediction accuracy")

def evaluate_model(x, y, a, b):
    """
    Comprehensive model evaluation using multiple regression metrics
    
    Args:
        x: Original feature data (kilometers)
        y: Original target data (prices)
        a: Intercept parameter
        b: Slope parameter
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    # Make predictions using the fitted model
    y_pred = a + b * x
    
    # Calculate various regression metrics
    metrics = {}
    
    # 1. R-squared (Coefficient of Determination)
    # Measures how much variance in y is explained by the model
    ss_res = np.sum((y - y_pred) ** 2)  # Sum of squares of residuals
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    metrics['R_squared'] = r_squared
    
    # 2. Mean Squared Error (MSE)
    # Average of squared differences between actual and predicted values
    mse = np.mean((y - y_pred) ** 2)
    metrics['MSE'] = mse
    
    # 3. Root Mean Squared Error (RMSE)
    # Square root of MSE, in same units as target variable
    rmse = np.sqrt(mse)
    metrics['RMSE'] = rmse
    
    # 4. Mean Absolute Error (MAE)
    # Average absolute difference between actual and predicted values
    mae = np.mean(np.abs(y - y_pred))
    metrics['MAE'] = mae
    
    # 5. Mean Absolute Percentage Error (MAPE)
    # Percentage-based error metric
    # Avoid division by zero by adding small epsilon
    epsilon = 1e-8
    mape = np.mean(np.abs((y - y_pred) / (y + epsilon))) * 100
    metrics['MAPE'] = mape
    
    return metrics, y_pred