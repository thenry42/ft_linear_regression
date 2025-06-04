import pandas as pd
import argparse
from utils.convert import normalize_features, convert_theta_to_ab
from utils.gradient import gradient_descent
from utils.accuracy import evaluate_model
from utils.dashboard import create_comprehensive_dashboard
from utils.plot import plot_data, plot_regression, plot_prediction


DATA_FILE = 'data.csv'
LEARNING_RATE = 0.01
CONVERGENCE_THRESHOLD = 0.0001
MAX_ITERATIONS = 10000


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Linear Regression with Gradient Descent')
    parser.add_argument('--prediction', '-p', type=int, default=0,
                        help='Enable price prediction given a km value (default: 0)')
    args = parser.parse_args()
    
    try:
        # Load the data
        df = pd.read_csv(DATA_FILE)
        x = df['km'].values
        y = df['price'].values
    except Exception as e:
        print(f"Error: {e}")
        return 1

    # Normalize the features to prevent overflow
    x_normalized = normalize_features(x)
    y_normalized = normalize_features(y)

    # Run the gradient descent algorithm
    theta_zero, theta_one, cost_history = gradient_descent(
        x_normalized, y_normalized, LEARNING_RATE, CONVERGENCE_THRESHOLD, MAX_ITERATIONS
    )

    # Convert normalized parameters to original scale (a and b)
    a, b = convert_theta_to_ab(x, y, theta_zero, theta_one)
    print(f"\n=== CONVERSION TO ORIGINAL SCALE ===")
    print(f"a (intercept): {a:.2f}")
    print(f"b (slope): {b:.6f}")
    print(f"Equation: price = {a:.2f} + {b:.6f} * km")

    # Evaluate the model performance
    metrics, y_pred = evaluate_model(x, y, a, b)

    # Plot the data
    plot_data(x, y)
    plot_regression(x, y, theta_zero, theta_one)
    
    # Display dashboard with all computed values
    create_comprehensive_dashboard(x, y, a, b, cost_history, metrics, y_pred)

    # Predict price for a given km value
    if args.prediction:
        print(f"\n=== PREDICTION FOR {args.prediction} KM ===")
        predicted_price = a + b * args.prediction
        print(f"Predicted price: {predicted_price:.2f} €")
        
        # Plot the predicted price for the given km value
        plot_prediction(x, y, a, b, args.prediction, predicted_price)



if __name__ == "__main__":
    main()
