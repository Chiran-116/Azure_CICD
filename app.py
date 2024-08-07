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

# Sidebar
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox("Choose Model", ["Decision Tree", "Random Forest", "Gradient Boosting"])

st.sidebar.title("Project Information")
st.sidebar.info("""
This application predicts whether an online payment transaction is fraudulent based on the transaction type, 
amount, and balance information. 
Choose a model from the dropdown menu to make a prediction.
""")

# Main UI
st.title("🔍 Online Payment Fraud Detection")
st.subheader("Predict whether a transaction is fraudulent based on its details.")

st.markdown("""
Please provide the details of the transaction below and choose a model to make a prediction.
""")

# User input
st.markdown("### Transaction Details")
col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"])
    amount = st.number_input("Transaction Amount", min_value=0.0, format="%0.2f")
    
with col2:
    oldbalanceOrg = st.number_input("Old Balance of Origin", min_value=0.0, format="%0.2f")
    newbalanceOrig = st.number_input("New Balance of Origin", min_value=0.0, format="%0.2f")

# Mapping transaction type to numerical value
transaction_type_mapping = {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5}
transaction_type_num = transaction_type_mapping[transaction_type]

# Prediction button
if st.button("Predict"):
    input_data = [transaction_type_num, amount, oldbalanceOrg, newbalanceOrig]

    if model_choice == "Decision Tree":
        prediction = make_prediction(decision_tree_model, input_data)
    elif model_choice == "Random Forest":
        prediction = make_prediction(random_forest_model, input_data)
    else:
        prediction = make_prediction(gradient_boosting_model, input_data)
    
    st.markdown("### Prediction Result")
    if prediction == 'Fraud':
        st.error("🚨 The transaction is predicted to be **Fraudulent**.")
    else:
        st.success("✅ The transaction is predicted to be **Not Fraudulent**.")

    st.markdown("""
    ### Prediction Details
    - **Transaction Type:** {}
    - **Transaction Amount:** ${:.2f}
    - **Old Balance of Origin:** ${:.2f}
    - **New Balance of Origin:** ${:.2f}
    """.format(transaction_type, amount, oldbalanceOrg, newbalanceOrig))