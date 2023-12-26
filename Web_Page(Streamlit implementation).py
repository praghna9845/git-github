import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
# streamlit run "C:\Users\Hp EliteBook 1030 G2\Desktop\Web pages\apps.py"

# Load the pre-trained model
model = pickle.load(open('C:/Users/Hp EliteBook 1030 G2/Desktop/Web pages/random_forest_model.pkl', 'rb'))

# Load the dataset
dataset = pd.read_csv(r"C:/Users/Hp EliteBook 1030 G2/Desktop/Copy of crop_production.csv")

# Extract features from the dataset
dataset_x = dataset.drop('Production', axis=1)

# Perform one-hot encoding on categorical features
train_dummy_crop = pd.get_dummies(dataset_x)

# Fit and transform the data using StandardScaler
scaler_crop = StandardScaler()
scaler_crop.fit(train_dummy_crop)

# Streamlit UI
def main():
    st.title("Crop Yield Prediction App")

    # User input form
    # st.header("Crop Yield Prediction")
    state_name = st.text_input("State Name:", 'Andaman and Nicobar Islands')
    district_name = st.text_input("District Name:", 'NICOBARS')
    crop_year = st.number_input("Crop Year:", 2000)
    season = st.text_input("Season:", 'Kharif')
    crop = st.text_input("Crop:", 'Arecanut')
    area = st.number_input("Area:", 15)

    # Button to trigger prediction
    if st.button("Predict Yield"):
        # Create a DataFrame with user input
        new_data = pd.DataFrame({
            'State_Name': [state_name],
            'District_Name': [district_name],
            'Crop_Year': [crop_year],
            'Season': [season],
            'Crop': [crop],
            'Area': [area]
        })

        # Perform one-hot encoding on the user input
        new_df = pd.get_dummies(new_data)
        new_df = new_df.reindex(columns=train_dummy_crop.columns, fill_value=0)

        # Scale the user input using the trained scaler
        scaled_input = scaler_crop.transform(new_df)

        # Make prediction using the trained model
        prediction = model.predict(scaled_input)

        # Display the result
        st.header("Prediction:")
        st.write(f'The predicted Yield for the given input is: {prediction[0]/area}')

if __name__ == "__main__":
    main()