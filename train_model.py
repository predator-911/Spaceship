import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV # type: ignore
from sklearn.ensemble import RandomForestClassifier# type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler# type: ignore
from sklearn.impute import SimpleImputer# type: ignore
from sklearn.pipeline import Pipeline# type: ignore
import joblib# type: ignore

# Load the train dataset
train_df = pd.read_csv('train.csv')

def preprocess_data(df):
    # Handle missing values for numerical columns
    numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    imputer_num = SimpleImputer(strategy='median')
    df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])
    
    # Handle missing values for categorical columns
    categorical_cols = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP']
    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Convert categorical columns to numerical
    df['CryoSleep'] = LabelEncoder().fit_transform(df['CryoSleep'])
    df['VIP'] = LabelEncoder().fit_transform(df['VIP'])
    
    # Convert categorical columns to dummy variables
    df = pd.get_dummies(df, columns=['HomePlanet', 'Destination'], drop_first=True)
    
    return df

# Preprocess the dataset
train_df = preprocess_data(train_df)

# Define features and target
X = train_df.drop(['Transported', 'PassengerId', 'Cabin', 'Name'], axis=1)
y = train_df['Transported']

# Create a pipeline with StandardScaler and RandomForestClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define parameter grid for RandomForestClassifier
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X, y)

# Print best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Get the best model
best_model = grid_search.best_estimator_

# Save the model
joblib.dump(best_model, 'best_model.pkl')
print("Model saved!")
