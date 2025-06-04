import numpy as np


def cost_function(theta_zero, theta_one, x, y):
    """ Mean Squared Error (MSE) """
    m = len(x)
    cost = np.sum((theta_zero + theta_one * x - y) ** 2)
    return cost / m


def gradient_descent(x, y, learning_rate, convergence_threshold, max_iterations):
    """ Gradient Descent """
    m = len(x)
    cost_history = []
    theta_zero = 0  # init at 0 like said in the subject
    theta_one = 0  # init at 0 like said in the subject

    for i in range(max_iterations):
        # Calculate the prediction
        h = theta_zero + theta_one * x

        # ========== DERIVATION HAPPENS HERE ==========
        # Cost function: J = (1/m) * Σ(h - y)²
        # Where h = theta_zero + theta_one * x
        
        # Derivative with respect to theta_zero:
        # ∂J/∂theta_zero = (1/m) * Σ(2 * (h - y) * 1) = (2/m) * Σ(h - y)
        # We ignore the constant 2 (absorbed into learning rate)
        theta_zero_gradient = np.sum(h - y) / m
        
        # Derivative with respect to theta_one:
        # ∂J/∂theta_one = (1/m) * Σ(2 * (h - y) * x) = (2/m) * Σ((h - y) * x)
        # We ignore the constant 2 (absorbed into learning rate)
        theta_one_gradient = np.sum((h - y) * x) / m
        # ============================================
        
        # Update the parameters using gradient descent rule:
        # theta = theta - learning_rate * gradient
        new_theta_zero = theta_zero - learning_rate * theta_zero_gradient
        new_theta_one = theta_one - learning_rate * theta_one_gradient
        
        # Calculate the cost
        cost = cost_function(new_theta_zero, new_theta_one, x, y)

        # Check for convergence (optional)
        if cost_history and abs(cost_history[-1] - cost) < convergence_threshold:
            print(f"Converged after {i+1} iterations")
            break

        # Update the parameters
        theta_zero = new_theta_zero
        theta_one = new_theta_one

        # Add the cost to the cost history
        cost_history.append(cost)

    return theta_zero, theta_one, cost_history