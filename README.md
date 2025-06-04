# ft_linear_regression 📈

A comprehensive implementation of **Linear Regression from scratch** using **Gradient Descent**. This project predicts car prices based on their mileage (kilometers driven) without using any machine learning libraries.

## 🎯 Project Overview

This project implements the fundamental machine learning algorithm **Linear Regression** to find the relationship between:
- **Input (x)**: Kilometers driven by a car
- **Output (y)**: Price of the car

We use **Gradient Descent** to find the best line that fits through our data points, allowing us to predict car prices for any given mileage.

## 📊 What is Linear Regression?

Linear regression finds the "best fit" straight line through data points. The line is represented by the equation:

```
price = a + b × kilometers
```

Where:
- **a** (intercept): The base price when kilometers = 0
- **b** (slope): How much the price decreases per kilometer driven

## 🧮 The Mathematics Behind It

### 1. The Hypothesis Function
Our prediction model is:
```
h(x) = θ₀ + θ₁ × x
```
- **θ₀** (theta_zero): The intercept parameter
- **θ₁** (theta_one): The slope parameter
- **x**: Input feature (kilometers)

### 2. Cost Function (Mean Squared Error)
We measure how "wrong" our predictions are using:
```
J(θ₀, θ₁) = (1/m) × Σ(hᵢ - yᵢ)²
```
- **m**: Number of data points
- **hᵢ**: Our prediction for data point i
- **yᵢ**: Actual value for data point i

### 3. Gradient Descent
To minimize the cost, we calculate gradients (derivatives) and update parameters:

**For θ₀ (intercept):**
```
∂J/∂θ₀ = (1/m) × Σ(h - y)
θ₀ = θ₀ - α × ∂J/∂θ₀
```

**For θ₁ (slope):**
```
∂J/∂θ₁ = (1/m) × Σ((h - y) × x)
θ₁ = θ₁ - α × ∂J/∂θ₁
```

Where **α** (alpha) is the learning rate.

## 🔧 How the Algorithm Works

### Step 1: Data Preparation
```python
# Read the car data
df = pd.read_csv('data.csv')
x = df['km'].values    # Kilometers (input)
y = df['price'].values # Prices (output)
```

### Step 2: Feature Normalization
**Why normalize?** Raw kilometer values (like 240,000) are much larger than normalized values (like 0.5). This can cause:
- **Overflow errors** in calculations
- **Slow convergence** in gradient descent
- **Numerical instability**

```python
def normalize_features(x):
    return (x - np.mean(x)) / np.std(x)
```

This transforms our data to have:
- **Mean = 0**
- **Standard deviation = 1**

### Step 3: Gradient Descent Algorithm
```python
def gradient_descent(x, y):
    θ₀ = 0  # Start with zero intercept
    θ₁ = 0  # Start with zero slope
    
    for each iteration:
        # Make predictions
        h = θ₀ + θ₁ × x
        
        # Calculate gradients
        θ₀_gradient = (1/m) × Σ(h - y)
        θ₁_gradient = (1/m) × Σ((h - y) × x)
        
        # Update parameters
        θ₀ = θ₀ - learning_rate × θ₀_gradient
        θ₁ = θ₁ - learning_rate × θ₁_gradient
```

### Step 4: Converting Back to Original Scale
Since we normalized our data, we need to convert our θ parameters back to work with original kilometer/price values:

```python
# Convert slope
b = θ₁ × (y_std / x_std)

# Convert intercept  
a = y_mean - b × x_mean + θ₀ × y_std
```

Now we can use: `price = a + b × kilometers`

## 📁 Project Structure

```
ft_linear_regression/
├── main.py               # Main program execution
├── data.csv              # Training data (km, price)
├── requirements.txt      # Python dependencies
├── Makefile              # Build and run automation
├── utils/                # Utility modules
│   ├── accuracy.py       # Model evaluation metrics
│   ├── convert.py        # Parameter conversion functions
│   ├── dashboard.py      # Interactive visualization dashboard
│   ├── gradient.py       # Gradient descent implementation
│   └── plot.py           # Visualization functions
└── README.md             # This file
```

## 🚀 How to Run

### Prerequisites
```bash
# Install dependencies
make install
```

### Run the Program
```bash
# Train model and show visualizations
make run

# Predict price for a specific mileage
make predict KM=100000
```

### Command Line Interface
```bash
# View all options
python main.py --help

# Predict a price for a specific mileage
python main.py --prediction 100000
```

### What You'll See

1. **Data visualization**:
   - Scatter plot of original data
   - Regression line visualization
   - Cost convergence plot
   - Predictions vs actual values

2. **Console output** showing:
   - Normalized parameters (θ₀, θ₁)
   - Converted parameters (a, b) 
   - Step-by-step conversion process
   - Final equation
   - Performance metrics (R², RMSE, MAE, etc.)

3. **Comprehensive dashboard** with:
   - Main regression plot
   - Cost convergence visualization
   - Prediction accuracy visualization
   - Performance metrics summary

## 📈 Understanding the Output

### Example Output:
```
=== STEP BY STEP CONVERSION ===
Original data stats:
  x_mean (km): 100185.1
  x_std (km): 52029.4
  y_mean (price): 5081.5
  y_std (price): 2189.9

Normalized parameters:
  theta_zero: -0.023456
  theta_one: -0.789123

Step 1 - Convert slope:
  b = theta_one * (y_std / x_std)
  b = -0.789123 * (2189.9 / 52029.4)
  b = -0.033218

Step 2 - Convert intercept:
  a = y_mean - b * x_mean + theta_zero * y_std
  a = 5081.5 - -0.033218 * 100185.1 + -0.023456 * 2189.9
  a = 8406.95

=== CONVERSION TO ORIGINAL SCALE ===
a (intercept): 8406.95
b (slope): -0.033218
Equation: price = 8406.95 + -0.033218 * km

LINEAR REGRESSION SUMMARY
============================================================
Dataset: 25 samples
Model: price = 8406.95 + -0.033218 × km
Training: Converged in 583 iterations

Performance Metrics:
  R² Score:  0.742
  RMSE:      €1112
  MAE:       €867
  MAPE:      20.3%

Key Insights:
  • Model explains 74.2% of price variation
  • Every 10,000 km reduces value by €332
  • Typical prediction error: ±€1112
============================================================
```

### What This Means:
- **Base price** (a): €8,406.95 (theoretical price at 0 km)
- **Depreciation rate** (b): €0.033 per kilometer
- A car loses about **3.3 cents per kilometer** driven
- The model explains about **74%** of price variation
- A car with **100,000 km** is predicted to cost **€5,085**

## 🔍 Key Features

### 1. Modular Code Structure
The code is organized into specialized modules:
- `gradient.py`: Core gradient descent algorithm
- `convert.py`: Parameter conversion utilities
- `accuracy.py`: Performance evaluation metrics
- `plot.py`: Data visualization functions
- `dashboard.py`: Comprehensive visualization dashboard

### 2. Comprehensive Model Evaluation
- **R² Score**: Measures explanatory power (0-1)
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

### 3. Interactive Visualization
- **Scatter plots**: Original data visualization
- **Regression line**: Model fit visualization
- **Cost history**: Convergence monitoring
- **Prediction plots**: Visual assessment of accuracy

### 4. Command-Line Interface
- **Makefile support**: Easy installation and running
- **Prediction mode**: Quick price prediction for any mileage
- **Argument parsing**: Flexible command-line options

## 🎓 Educational Value

This implementation teaches:

1. **Core ML Concepts**:
   - Supervised learning
   - Cost functions
   - Optimization algorithms

2. **Mathematical Understanding**:
   - Derivatives and gradients
   - Linear algebra basics
   - Statistical normalization

3. **Programming Skills**:
   - Modular code organization
   - Data visualization
   - Command-line interfaces

4. **Real-world Application**:
   - Predicting car prices
   - Understanding depreciation
   - Market analysis
