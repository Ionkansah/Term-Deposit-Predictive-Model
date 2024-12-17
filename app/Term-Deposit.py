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

### Overview of the Dataset and Problem Statement
The dataset contains information about **bank marketing campaigns** conducted by a banking institution.  
The goal is to **predict whether a client will subscribe to a term deposit** based on client features, previous contacts, and campaign-related data.

The dataset contains:
- **Numeric Features**: Age, balance, duration, campaign count, etc.
- **Categorical Features**: Job, marital status, education, contact type, etc.
- **Target Variable**: 'y' (1: Subscription, 0: No Subscription)

The problem is a **binary classification** task that determines the likelihood of subscription based on the given features. Accurately predicting this outcome enables targeted marketing strategies, saving time and resources.
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

st.subheader("Interpretation of Results")
st.markdown("""
## Model Performance Metrics
The following metrics were computed for the model's performance:

1. **Precision**: Indicates how many of the predicted "term deposit subscriptions" were correct.  
   - Precision for class '1' (Subscription): **66%**  
     *This means that 66% of predicted positive cases are actual subscriptions.*

2. **Recall**: Measures how well the model identifies actual subscriptions.  
   - Recall for class '1' (Subscription): **48%**  
     *The model correctly identified 48% of actual subscriptions.*  
     A lower recall suggests that some actual subscriptions are missed.

3. **F1-Score**: The harmonic mean of precision and recall.  
   - F1-Score for class '1' (Subscription): **56%**  
     *This combines precision and recall into one balanced measure.*

4. **Accuracy**: **91%**  
   *The overall model accuracy is high, but this is heavily influenced by the majority class ('No Subscription').*

5. **Macro Average**: The average precision, recall, and F1-score for both classes (unweighted).  
   - **Macro Precision**: 80%  
   - **Macro Recall**: 73%  
   *This provides an average performance across classes.*

6. **Weighted Average**: The average precision, recall, and F1-score weighted by class size.  
   - **Weighted Precision**: 90%  
   - **Weighted Recall**: 91%  
   *This emphasizes performance on the dominant class while accounting for imbalances.*

7. **ROC-AUC Score**: **0.73**  
   *The ROC-AUC score reflects the model's ability to distinguish between classes. A score of 0.73 indicates moderate performance.*

---

### Key Observations:
- The model performs well on the majority class ('No Subscription') but struggles with the minority class ('Subscription').
- Improving recall for subscriptions could enhance the model's ability to capture more true positives.
- Despite this, the overall accuracy and AUC indicate that the model is reasonably reliable.
""")


from sklearn.metrics import classification_report

# Mock classification report data
report_data = {
    "Class": ["No Subscription (0)", "Subscription (1)", "Accuracy", "Macro Avg", "Weighted Avg"],
    "Precision": [0.94, 0.66, np.nan, 0.80, 0.90],
    "Recall": [0.97, 0.48, np.nan, 0.73, 0.91],
    "F1-Score": [0.95, 0.56, np.nan, 0.75, 0.91],
    "Support": [7303, 935, 8238, 8238, 8238]
}

# Convert to DataFrame
report_df = pd.DataFrame(report_data)

# Style and display the DataFrame in Streamlit
st.subheader("Classification Report")
st.dataframe(report_df.style.format({
    "Precision": lambda x: f"{x:.2f}" if not pd.isna(x) else "-",
    "Recall": lambda x: f"{x:.2f}" if not pd.isna(x) else "-",
    "F1-Score": lambda x: f"{x:.2f}" if not pd.isna(x) else "-",
    "Support": lambda x: f"{int(x)}" if not pd.isna(x) else "-"
}))

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
st.subheader("Make a Prediction. Try it Now!")
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

