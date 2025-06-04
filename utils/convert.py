import numpy as np


def convert_theta_to_ab(x, y, theta_zero_norm, theta_one_norm):
    """ Convert normalized theta parameters to original scale a and b """
    # Get original data statistics
    x_mean, x_std = np.mean(x), np.std(x)
    y_mean, y_std = np.mean(y), np.std(y)
    
    print(f"\n=== STEP BY STEP CONVERSION ===")
    print(f"Original data stats:")
    print(f"  x_mean (km): {x_mean:.1f}")
    print(f"  x_std (km): {x_std:.1f}")
    print(f"  y_mean (price): {y_mean:.1f}")
    print(f"  y_std (price): {y_std:.1f}")
    
    print(f"\nNormalized parameters:")
    print(f"  theta_zero: {theta_zero_norm}")
    print(f"  theta_one: {theta_one_norm}")
    
    # Step 1: Convert slope (b)
    b = theta_one_norm * y_std / x_std
    print(f"\nStep 1 - Convert slope:")
    print(f"  b = theta_one * (y_std / x_std)")
    print(f"  b = {theta_one_norm:.6f} * ({y_std:.1f} / {x_std:.1f})")
    print(f"  b = {b:.6f}")
    
    # Step 2: Convert intercept (a)
    a = y_mean - b * x_mean + theta_zero_norm * y_std
    print(f"\nStep 2 - Convert intercept:")
    print(f"  a = y_mean - b * x_mean + theta_zero * y_std")
    print(f"  a = {y_mean:.1f} - {b:.6f} * {x_mean:.1f} + {theta_zero_norm} * {y_std:.1f}")
    print(f"  a = {a:.2f}")
    
    return a, b


def normalize_features(x):
    """ Normalize features to prevent overflow """
    return (x - np.mean(x)) / np.std(x)