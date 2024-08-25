import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load the trained model and get feature names
model = joblib.load('spaceship_titanic_model.pkl')
feature_names = ['CryoSleep', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa',
                  'VRDeck', 'HomePlanet_Europa', 'HomePlanet_Mars',
                  'Destination_PSO J318.5-22', 'Destination_TRAPPIST-1e']

def preprocess_data(df, feature_names):
    # Handle missing values for numerical columns
    numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    imputer_num = SimpleImputer(strategy='median')
    df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])
    
    # Handle missing values for categorical columns
    categorical_cols = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP']
    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Convert categorical variables to numerical
    df['CryoSleep'] = LabelEncoder().fit_transform(df['CryoSleep'])
    df['VIP'] = LabelEncoder().fit_transform(df['VIP'])
    
    # Convert categorical columns to dummy variables
    df = pd.get_dummies(df, columns=['HomePlanet', 'Destination'], drop_first=True)
    
    # Ensure all expected columns are present and in the correct order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default value 0
    
    # Reorder columns to match the feature names used during training
    df = df[feature_names]
    
    return df

# Streamlit app
st.title('Spaceship Titanic Survival Prediction')

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Preprocess the uploaded data
    df_processed = preprocess_data(df, feature_names)
    
    # Drop irrelevant columns for prediction
    df_processed = df_processed.drop(['PassengerId', 'Cabin', 'Name'], axis=1, errors='ignore')
    
    # Make predictions
    predictions = model.predict(df_processed)
    
    # Create a DataFrame with the results
    result = pd.DataFrame({'PassengerId': df['PassengerId'], 'Transported': predictions})
    
    # Display the results
    st.write(result)
    
    # Option to download results
    st.download_button(
        label="Download Predictions",
        data=result.to_csv(index=False),
        file_name='submission.csv',
        mime='text/csv'
    )
else:
    st.write("Please upload a CSV file for prediction.")

# Manual input form
with st.form(key='manual_input_form'):
    st.subheader('Enter passenger information manually')
    
    home_planet = st.selectbox('HomePlanet', ['Earth', 'Europa', 'Mars'])
    cryo_sleep = st.selectbox('CryoSleep', ['True', 'False'])
    age = st.number_input('Age', min_value=0)
    vip = st.selectbox('VIP', ['True', 'False'])
    room_service = st.number_input('RoomService', min_value=0)
    food_court = st.number_input('FoodCourt', min_value=0)
    shopping_mall = st.number_input('ShoppingMall', min_value=0)
    spa = st.number_input('Spa', min_value=0)
    vr_deck = st.number_input('VRDeck', min_value=0)
    destination = st.selectbox('Destination', ['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'])

    submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Create a DataFrame for the manual input
        manual_data = pd.DataFrame({
            'HomePlanet': [home_planet],
            'CryoSleep': [cryo_sleep],
            'Age': [age],
            'VIP': [vip],
            'RoomService': [room_service],
            'FoodCourt': [food_court],
            'ShoppingMall': [shopping_mall],
            'Spa': [spa],
            'VRDeck': [vr_deck],
            'Destination': [destination]
        })
        
        # Preprocess the manual input data
        manual_data_processed = preprocess_data(manual_data, feature_names)
        
        # Drop irrelevant columns for prediction
        manual_data_processed = manual_data_processed.drop(['PassengerId', 'Cabin', 'Name'], axis=1, errors='ignore')
        
        # Make prediction
        prediction = model.predict(manual_data_processed)
        
        # Display the result
        st.write(f'Prediction: {"Transported" if prediction[0] else "Not Transported"}')
