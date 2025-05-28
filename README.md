# DataScienceInternshipProject
AttriSense Predicting employee attrition with explainable AI models.  Summarizr Automated text summarization for news and articles.  HealthPredict Disease diagnosis prediction using medical data.  LoanGuard Loan default prediction to identify high-risk applicants.
# DataPulse  
*Hands-on Data Science Projects Across HR, NLP, Healthcare, and Finance*

---

## Overview  
This repository contains four key projects developed during the Data Science Internship aiming to build predictive and NLP models in different domains:

- **Employee Attrition Prediction:** Classification models to predict if employees will leave and provide HR retention strategies.  
- **Text Summarization:** Generate concise summaries from long articles using extractive and abstractive methods.  
- **Disease Diagnosis Prediction:** Predict the likelihood of diseases such as diabetes and heart disease for early detection.  
- **Loan Default Prediction:** Identify high-risk loan applicants to reduce defaults using financial data.

---

## Project Structure


Each task folder contains:  
- Data preprocessing scripts and notebooks  
- Model training and evaluation code  
- Reports detailing methodology, insights, and challenges  


Task 1: Employee Attrition Prediction 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap

# Load dataset
df = pd.read_csv('IBM_HR_Attrition.csv')

# Basic EDA
print(df['Attrition'].value_counts())

# Preprocessing (example)
df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})
X = df.drop('Attrition', axis=1).select_dtypes(include=['int64', 'float64'])
y = df['Attrition']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

# Explainability with SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test)

Task 2: Text Summarization (Using HuggingFace Pretrained Model)
from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization")

text = """Your long article or text here."""

summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
print(summary[0]['summary_text'])
ask 3: Disease Diagnosis Prediction (Using PIMA Diabetes Dataset & Gradient Boosting)




- Video presentation of the project

---

## How to Use

1. Clone this repository:  https://github.com/yourusername/DataPulse.git
2. Navigate to the desired task folder.  
3. Follow instructions in the notebooks/scripts to run the analyses.  
4. Check the reports folder for detailed insights and findings.  

---

## Tools & Libraries Used  
- Python (Pandas, NumPy, scikit-learn)  
- SHAP, LIME for explainability  
- spaCy, HuggingFace Transformers for NLP  
- LightGBM, XGBoost, SVM for modeling  
- Matplotlib, Seaborn for visualization  

---


---

## License  
This project is for educational purposes only.
Task 1: Employee Attrition Prediction (Basic Random Forest Example)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap

# Load dataset
df = pd.read_csv('IBM_HR_Attrition.csv')

# Basic EDA
print(df['Attrition'].value_counts())

# Preprocessing (example)
df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})
X = df.drop('Attrition', axis=1).select_dtypes(include=['int64', 'float64'])
y = df['Attrition']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

# Explainability with SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test)

Task 2: Text Summarization (Using HuggingFace Pretrained Model)
from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization")

text = """Your long article or text here."""

summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
print(summary[0]['summary_text'])

Task 3: Disease Diagnosis Prediction (Using PIMA Diabetes Dataset & Gradient boosting)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

# Load dataset
df = pd.read_csv('pima_diabetes.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("F1 Score:", f1_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_prob))
Task 4: Loan Default Prediction (Handling Imbalance with SMOTE and LightGBM)
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv('loan_data.csv')

# Preprocessing example
X = df.drop('loan_default', axis=1)
y = df['loan_default']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle imbalance
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Train model
model = LGBMClassifier()
model.fit(X_train_res, y_train_res)

# Predict
y_pred = model.predict(X_test)

# Metrics
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

 


