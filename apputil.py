import pandas as pd
import pickle
import numpy as np

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
    
    # If no match found, return None to indicate unknown roast
    return None

def predict_rating(df_X):
    """
    Predicts rating values for a DataFrame with columns '100g_USD' and 'roast'.
    
    Parameters:
    -----------
    df_X : pandas.DataFrame
        DataFrame with columns '100g_USD' (numerical) and 'roast' (text)
    
    Returns:
    --------
    numpy.array
        Array of predicted rating values
    """
    # Load the models
    try:
        with open('model_1.pickle', 'rb') as f:
            model_1 = pickle.load(f)  # Linear regression (100g_USD only)
        
        with open('model_2.pickle', 'rb') as f:
            model_2 = pickle.load(f)  # Decision tree (100g_USD + roast_cat)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file not found: {e}. Make sure to run train.py first.")
    
    # Validate input DataFrame
    if not isinstance(df_X, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    required_columns = ['100g_USD', 'roast']
    missing_columns = [col for col in required_columns if col not in df_X.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Create a copy to avoid modifying the original DataFrame
    df_work = df_X.copy()
    
    # Convert roast values to numerical categories
    df_work['roast_cat'] = df_work['roast'].apply(roast_category)
    
    # Initialize predictions array
    predictions = np.zeros(len(df_work))
    
    # For each row, decide which model to use
    for i, row in df_work.iterrows():
        price = row['100g_USD']
        roast_cat = row['roast_cat']
        
        # Check if price is valid
        if pd.isna(price):
            predictions[i] = np.nan
            continue
        
        # If roast category is recognized (not None/NaN), use model_2
        if roast_cat is not None and not pd.isna(roast_cat):
            prediction = model_2.predict([[price, roast_cat]])[0]
        else:
            # If roast is unrecognized or missing, use model_1 (price only)
            prediction = model_1.predict([[price]])[0]
        
        predictions[i] = prediction
    
    return predictions
