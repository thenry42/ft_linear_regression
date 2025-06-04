import numpy as np
import matplotlib.pyplot as plt


def plot_data(x, y):
    """ Plot the data points """
    plt.scatter(x, y)
    plt.xlabel('km')
    plt.ylabel('price')
    plt.show()


def plot_regression(x, y, theta_zero_norm, theta_one_norm):
    """ Plot the data points with the linear regression line """
    # Convert normalized parameters back to original scale
    x_mean, x_std = np.mean(x), np.std(x)
    y_mean, y_std = np.mean(y), np.std(y)
    
    # Convert back to original scale
    theta_one_orig = theta_one_norm * y_std / x_std
    theta_zero_orig = y_mean - theta_one_orig * x_mean + theta_zero_norm * y_std
    
    # Create regression line
    x_line = np.linspace(min(x), max(x), 100)
    y_line = theta_zero_orig + theta_one_orig * x_line
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6, label='Data points')
    plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'Linear regression\ny = {theta_zero_orig:.2f} + {theta_one_orig:.4f}x')
    plt.xlabel('Kilometers')
    plt.ylabel('Price')
    plt.title('Linear Regression: Car Price vs Kilometers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_predictions_vs_actual(x, y, y_pred):
    """
    Create visualizations to assess model performance
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Predictions vs Actual values
    ax1.scatter(y, y_pred, alpha=0.6)
    ax1.plot([min(y), max(y)], [min(y), max(y)], 'r--', linewidth=2, label='Perfect predictions')
    ax1.set_xlabel('Actual Price (€)')
    ax1.set_ylabel('Predicted Price (€)')
    ax1.set_title('Predictions vs Actual Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals (errors) vs Kilometers
    residuals = y - y_pred
    ax2.scatter(x, residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Kilometers')
    ax2.set_ylabel('Residuals (Actual - Predicted)')
    ax2.set_title('Residuals vs Kilometers')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_prediction(x, y, a, b, prediction_km, predicted_price):
        # Plot the data with prediction point highlighted
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, alpha=0.6, label='Data points')
        
        # Create regression line
        x_line = np.linspace(min(x), max(x), 100)
        y_line = a + b * x_line
        plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'Linear regression\ny = {a:.2f} + {b:.6f}x')
        
        # Highlight the prediction point
        plt.scatter([prediction_km], [predicted_price], color='green', s=100, 
                   edgecolors='black', linewidth=2, zorder=5, 
                   label=f'Prediction: {predicted_price:.2f} € at {prediction_km} km')
        
        plt.xlabel('Kilometers')
        plt.ylabel('Price (€)')
        plt.title('Linear Regression with Prediction Point')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()