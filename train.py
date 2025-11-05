import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import numpy as np

def load_coffee_data():
    """Load coffee analysis data from the URL."""
    url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
    df = pd.read_csv(url)
    return df

def roast_category(roast_value):
    """
    Maps roast values to numerical categories.
    Missing values (NaN) will remain as NaN.
    """
    if pd.isna(roast_value):
        return np.nan
    
    # Convert to string and normalize case
    roast_str = str(roast_value).strip().lower()
    
    # Define mapping based on roast intensity
    roast_mapping = {
        'light': 1,
        'medium-light': 2,
        'medium': 3,
        'medium-dark': 4,
        'dark': 5,
        'french': 6,  # Very dark roast
        'italian': 6, # Very dark roast
    }
    
    # Try exact match first
    if roast_str in roast_mapping:
        return roast_mapping[roast_str]
    
    # Try partial matching for compound roast names
    for roast_type, value in roast_mapping.items():
        if roast_type in roast_str:
            return value
    
    # If no match found, return a default value (medium roast)
    return 3

def train_linear_regression_model(df):
    """
    Exercise 1: Train a linear regression model to predict rating based on 100g_USD.
    """
    print("Training Linear Regression Model (Exercise 1)...")
    
    # Prepare data - remove rows with missing values in required columns
    model_data = df[['100g_USD', 'rating']].dropna()
    
    X = model_data[['100g_USD']]
    y = model_data['rating']
    
    # Train the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Save the model
    with open('model_1.pickle', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Linear Regression model trained on {len(model_data)} samples")
    print(f"Model coefficient: {model.coef_[0]:.4f}")
    print(f"Model intercept: {model.intercept_:.4f}")
    print("Model saved as 'model_1.pickle'\n")
    
    return model

def train_decision_tree_model(df):
    """
    Exercise 2: Train a Decision Tree Regressor to predict rating based on 100g_USD and roast.
    """
    print("Training Decision Tree Regressor Model (Exercise 2)...")
    
    # Create roast_cat column using the roast_category function
    df['roast_cat'] = df['roast'].apply(roast_category)
    
    # Prepare data - remove rows with missing values in required columns
    model_data = df[['100g_USD', 'roast_cat', 'rating']].dropna()
    
    X = model_data[['100g_USD', 'roast_cat']]
    y = model_data['rating']
    
    # Train the model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)
    
    # Save the model
    with open('model_2.pickle', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Decision Tree model trained on {len(model_data)} samples")
    print(f"Feature importances: 100g_USD: {model.feature_importances_[0]:.4f}, roast_cat: {model.feature_importances_[1]:.4f}")
    print("Model saved as 'model_2.pickle'\n")
    
    return model

def main():
    """Main function to execute both exercises."""
    print("Loading coffee analysis data...")
    df = load_coffee_data()
    print(f"Loaded {len(df)} rows of data\n")
    
    # Exercise 1: Linear Regression
    model_1 = train_linear_regression_model(df)
    
    # Exercise 2: Decision Tree Regressor
    model_2 = train_decision_tree_model(df)
    
    print("Both models have been trained and saved successfully!")
    
    # Display some sample roast category mappings
    print("\nSample roast category mappings:")
    unique_roasts = df['roast'].dropna().unique()[:10]  # Show first 10 unique roasts
    for roast in unique_roasts:
        print(f"  '{roast}' -> {roast_category(roast)}")

if __name__ == "__main__":
    main()