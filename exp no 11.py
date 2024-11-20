# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the pre-trained model (train your model first or load a pre-trained model)
# You need to train the model first and save it to a file using pickle
# Example: model = RandomForestClassifier().fit(X_train, y_train)
# pickle.dump(model, open('loan_default_model.pkl', 'wb'))
# In this case, we are assuming the model is already trained and saved.

# Load the trained model
model = pickle.load(open('loan_default_model.pkl', 'rb'))

# Function to get user input and make a prediction
def predict_loan_default():
    # Get user input
    print("Please enter the following details:")

    age = int(input("Age: "))
    income = float(input("Income: "))
    loan_amount = float(input("Loan Amount: "))
    credit_history = input("Credit History (Good/Poor): ")

    # Map credit history to numeric value
    if credit_history.lower() == 'good':
        credit_history = 1
    elif credit_history.lower() == 'poor':
        credit_history = 0
    else:
        print("Invalid credit history value. Please enter 'Good' or 'Poor'.")
        return

    # Create a DataFrame with the user input
    user_data = pd.DataFrame([[age, income, loan_amount, credit_history]], columns=['Age', 'Income', 'Loan_Amount', 'Credit_History'])

    # Feature scaling using the same scaler used during training
    scaler = StandardScaler()
    user_data_scaled = scaler.fit_transform(user_data)

    # Predict whether the user will default on the loan
    prediction = model.predict(user_data_scaled)
    
    if prediction[0] == 0:
        print("The prediction is: The user will NOT default on the loan.")
    else:
        print("The prediction is: The user WILL default on the loan.")

# Call the function to get user input and make prediction
predict_loan_default()
