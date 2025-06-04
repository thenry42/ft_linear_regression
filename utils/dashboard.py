import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure matplotlib for headless environments
if os.environ.get('DISPLAY') is None:
    plt.switch_backend('Agg')


def create_comprehensive_dashboard(x, y, a, b, cost_history, metrics, y_pred):
    """
    Create a focused dashboard with essential visualizations and metrics
    
    Args:
        x: Original feature values (km)
        y: Original target values (price)
        a: Intercept parameter
        b: Slope parameter
        cost_history: List of cost values during training
        metrics: Dictionary containing model performance metrics
        y_pred: Predicted values
    """
    # Create the dashboard with 2x2 grid
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('Car Price Prediction - Linear Regression Dashboard', 
                 fontsize=20, fontweight='bold', y=0.95)

    # ========== 1. MAIN REGRESSION PLOT ==========
    ax1 = fig.add_subplot(gs[0, :])
    
    # Scatter plot with regression line
    ax1.scatter(x, y, alpha=0.6, s=60, color='steelblue', edgecolor='white', linewidth=0.5)
    
    # Add regression line
    x_line = np.linspace(min(x), max(x), 100)
    y_line = a + b * x_line
    ax1.plot(x_line, y_line, 'red', linewidth=2, label=f'y = {a:.0f} + {b:.6f}x')
    
    ax1.set_xlabel('Kilometers', fontsize=12)
    ax1.set_ylabel('Price (€)', fontsize=12)
    ax1.set_title('Car Price vs Kilometers', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Format axes
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x/1000:.0f}K'))

    # Add key metrics as text
    metrics_text = f'R² = {metrics["R_squared"]:.3f}  |  RMSE = €{metrics["RMSE"]:.0f}  |  MAE = €{metrics["MAE"]:.0f}'
    ax1.text(0.5, 0.02, metrics_text, transform=ax1.transAxes, 
             ha='center', va='bottom', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    # ========== 2. COST FUNCTION CONVERGENCE ==========
    ax2 = fig.add_subplot(gs[1, 0])
    
    iterations = range(1, len(cost_history) + 1)
    ax2.plot(iterations, cost_history, 'darkgreen', linewidth=2)
    ax2.set_xlabel('Iterations', fontsize=12)
    ax2.set_ylabel('Cost (MSE)', fontsize=12)
    ax2.set_title('Cost Function Convergence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add convergence info
    ax2.text(0.02, 0.98, f'Converged in {len(cost_history)} iterations', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

    # ========== 3. PREDICTIONS vs ACTUAL ==========
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Perfect prediction line
    min_val, max_val = min(min(y), min(y_pred)), max(max(y), max(y_pred))
    ax3.plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--', 
             linewidth=2, label='Perfect Predictions', alpha=0.8)
    
    # Actual vs predicted scatter
    ax3.scatter(y, y_pred, alpha=0.6, s=60, color='darkgreen', edgecolor='white', linewidth=0.5)
    
    ax3.set_xlabel('Actual Price (€)', fontsize=12)
    ax3.set_ylabel('Predicted Price (€)', fontsize=12)
    ax3.set_title('Predictions vs Actual Values', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Format axes
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x/1000:.0f}K'))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x/1000:.0f}K'))

    # Use subplots_adjust instead of tight_layout to avoid warnings
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.90, hspace=0.3, wspace=0.3)
    
    # Display the dashboard
    plt.show()

    # Print concise summary
    print_summary(metrics, a, b, len(cost_history), len(x))


def print_summary(metrics, a, b, iterations, n_samples):
    """Print a concise summary of the model"""
    print("\n" + "="*60)
    print("LINEAR REGRESSION SUMMARY")
    print("="*60)
    
    print(f"Dataset: {n_samples} samples")
    print(f"Model: price = {a:.2f} + {b:.6f} × km")
    print(f"Training: Converged in {iterations} iterations")
    
    print(f"\nPerformance Metrics:")
    print(f"  R² Score:  {metrics['R_squared']:.3f}")
    print(f"  RMSE:      €{metrics['RMSE']:.0f}")
    print(f"  MAE:       €{metrics['MAE']:.0f}")
    print(f"  MAPE:      {metrics['MAPE']:.1f}%")
    
    print(f"\nKey Insights:")
    print(f"  • Model explains {metrics['R_squared']*100:.1f}% of price variation")
    print(f"  • Every 10,000 km reduces value by €{abs(b*10000):.0f}")
    print(f"  • Typical prediction error: ±€{metrics['RMSE']:.0f}")
    
    print("="*60)
