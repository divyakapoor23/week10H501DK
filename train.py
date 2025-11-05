import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
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

def train_tfidf_text_model(df):
    """
    Bonus Exercise 4: Train a linear regression model to predict rating based on TF-IDF vectorized desc_3 text.
    """
    print("Training TF-IDF Text Model (Bonus Exercise 4)...")
    
    # Prepare data - remove rows with missing values in required columns
    model_data = df[['desc_3', 'rating']].dropna()
    
    # Fill any remaining NaN values in desc_3 with empty strings
    model_data = model_data.copy()
    model_data['desc_3'] = model_data['desc_3'].fillna('')
    
    # Convert desc_3 to string type to handle any non-string values
    model_data['desc_3'] = model_data['desc_3'].astype(str)
    
    X_text = model_data['desc_3']
    y = model_data['rating']
    
    # Create TF-IDF vectorizer
    # max_features limits vocabulary size, min_df ignores terms that appear in less than 2 documents
    # max_df ignores terms that appear in more than 95% of documents (too common)
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,  # Limit to top 1000 features
        min_df=2,           # Ignore terms that appear in less than 2 documents
        max_df=0.95,        # Ignore terms that appear in more than 95% of documents
        stop_words='english',  # Remove common English stop words
        lowercase=True,     # Convert to lowercase
        ngram_range=(1, 2)  # Use both unigrams and bigrams
    )
    
    # Vectorize the text data
    X_tfidf = tfidf_vectorizer.fit_transform(X_text)
    
    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_tfidf, y)
    
    # Save both the model and the vectorizer
    model_data_to_save = {
        'model': model,
        'vectorizer': tfidf_vectorizer
    }
    
    with open('model_3.pickle', 'wb') as f:
        pickle.dump(model_data_to_save, f)
    
    print(f"TF-IDF Text model trained on {len(model_data)} samples")
    print(f"TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
    print(f"Model coefficient range: [{model.coef_.min():.4f}, {model.coef_.max():.4f}]")
    print(f"Model intercept: {model.intercept_:.4f}")
    print("Model and vectorizer saved as 'model_3.pickle'\n")
    
    return model, tfidf_vectorizer

def main():
    """Main function to execute all exercises."""
    print("Loading coffee analysis data...")
    df = load_coffee_data()
    print(f"Loaded {len(df)} rows of data\n")
    
    # Exercise 1: Linear Regression
    model_1 = train_linear_regression_model(df)
    
    # Exercise 2: Decision Tree Regressor
    model_2 = train_decision_tree_model(df)
    
    # Bonus Exercise 4: TF-IDF Text Model
    model_3, vectorizer_3 = train_tfidf_text_model(df)
    
    print("All models have been trained and saved successfully!")
    
    # Display some sample roast category mappings
    print("\nSample roast category mappings:")
    unique_roasts = df['roast'].dropna().unique()[:10]  # Show first 10 unique roasts
    for roast in unique_roasts:
        print(f"  '{roast}' -> {roast_category(roast)}")
        
    # Display some sample text features
    print(f"\nSample TF-IDF features (top 10):")
    feature_names = vectorizer_3.get_feature_names_out()
    for i, feature in enumerate(feature_names[:10]):
        print(f"  '{feature}'")

if __name__ == "__main__":
    main()