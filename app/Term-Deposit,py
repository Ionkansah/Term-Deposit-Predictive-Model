import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

# Load the dataset
data = pd.read_csv('data/bank-additional-full.csv', sep=';')

# Title and Introduction
st.title("Term Deposit Subscription Prediction by Isaac Opoku")
st.markdown("""
### Introduction
This project predicts the likelihood of a client subscribing to a term deposit based on key features. 
The analysis explores the data, builds predictive models, and provides information for decision.
""")

# EDA Section
st.subheader("Exploratory Data Analysis")
if st.checkbox("Show Dataset"):
    st.write(data.head())

# Display correlation heatmap
st.subheader("Correlation Heatmap")
numeric_data = data.select_dtypes(include=['float64', 'int64'])
corr = numeric_data.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Predictive Modeling Section
st.subheader("Predictive Modeling Results")
st.markdown("""
- **Best Parameters:** `{'max_depth': 20, 'n_estimators': 200}`
- **Confusion Matrix:**
""")
conf_matrix = np.array([[7070, 233], [484, 451]])
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No Subscription", "Subscription"], 
            yticklabels=["No Subscription", "Subscription"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.markdown("""
- **Classification Report:**
""")
st.text("""
              precision    recall  f1-score   support

           0       0.94      0.97      0.95      7303
           1       0.66      0.48      0.56       935

    accuracy                           0.91      8238
   macro avg       0.80      0.73      0.75      8238
weighted avg       0.90      0.91      0.91      8238
""")

st.markdown(f"- **ROC AUC Score:** `{0.7252241222382422}`")

# Insights Section
st.subheader("Key Insights")
st.markdown("""
1. **Feature Importance:** The model highlights key predictors influencing term deposit subscription.
2. **Performance:** While the model achieves high accuracy (91%), there is room for improvement in recall for subscription prediction.
3. **Recommendations:**
    - There is the need to focus marketing efforts on customer groups identified as likely subscribers.
    - Addressing of potential biases in the dataset to improve prediction for minority classes.
""")

# Interactive Prediction
st.subheader("Make a Prediction")
st.markdown("Enter client details to predict the likelihood of term deposit subscription.")
# Example of user input form
age = st.number_input("Age", min_value=18, max_value=100, value=35)
balance = st.number_input("Account Balance", min_value=0, value=1000)
duration = st.number_input("Last Contact Duration (seconds)", min_value=0, value=100)
campaign = st.number_input("Number of Contacts During Campaign", min_value=1, value=1)
pred_button = st.button("Predict")

# Mock prediction result (replace with model prediction)
if pred_button:
    prediction = "Yes" if duration > 200 else "No"
    st.write(f"Predicted Term Deposit Subscription: **{prediction}**")

