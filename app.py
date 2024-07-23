import streamlit as st
import pickle
import numpy as np

# Load models
with open('decision_tree_model.pkl', 'rb') as file:
    decision_tree_model = pickle.load(file)

with open('random_forest_model.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)

with open('gradient_boosting_model.pkl', 'rb') as file:
    gradient_boosting_model = pickle.load(file)

# Function to make prediction
def make_prediction(model, input_data):
    prediction = model.predict(np.array(input_data).reshape(1, -1))
    return prediction[0]

# Streamlit UI
st.title("Online Payment Fraud Detection")

# User input
transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"])
amount = st.number_input("Transaction Amount")
oldbalanceOrg = st.number_input("Old Balance of Origin")
newbalanceOrig = st.number_input("New Balance of Origin")

# Mapping transaction type to numerical value
transaction_type_mapping = {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5}
transaction_type_num = transaction_type_mapping[transaction_type]

# Model selection
model_choice = st.selectbox("Choose Model", ["Decision Tree", "Random Forest", "Gradient Boosting"])

# Prediction button
if st.button("Predict"):
    input_data = [transaction_type_num, amount, oldbalanceOrg, newbalanceOrig]

    if model_choice == "Decision Tree":
        prediction = make_prediction(decision_tree_model, input_data)
    elif model_choice == "Random Forest":
        prediction = make_prediction(random_forest_model, input_data)
    else:
        prediction = make_prediction(gradient_boosting_model, input_data)
    
    st.write("Prediction: ", "Fraud" if prediction == 'Fraud' else "Not Fraud")

# To run this app, use the command: streamlit run app.py
