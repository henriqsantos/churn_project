# Telecommunications Churn Prediction with Machine Learning

This project aims to predict **Churn** (contract cancellation) for a telecommunications company using Machine Learning algorithms. The goal is to identify customers at the highest risk of leaving so that the retention team can act proactively.

---

##  Data Source

The data used in this project is publicly available and was sourced from the **IBM Sample Data Sets** (commonly found on Kaggle as *Telco Customer Churn*). The dataset contains information on 7,043 customers in California, covering demographic characteristics, contracted services, and financial history.

---

##  Reasoning & Methodology

The project was structured following the standard Data Science project lifecycle (CRISP-DM):

### 1. Exploratory Data Analysis (EDA)

Before touching any modelling code, the focus was on understanding the business behaviour. Three major risk factors were identified:

- **Tenure:** Newer customers (0–5 months) show the highest churn rate.
- **Contract Type:** Month-to-month contracts concentrate the vast majority of cancellations.
- **Payment Method:** Customers paying via Electronic Check show a drastically higher propensity to churn.

### 2. Data Pre-processing

To prepare the data for the mathematical models:

- The target variable (`Churn`) was converted to binary (`0` for retained customers, `1` for churned customers).
- **One-Hot Encoding** (`pd.get_dummies`) was applied to convert categorical text variables (such as payment methods and contract types) into numerical 0/1 columns.
- The most relevant features were selected, and the dataset was split into **80% Training** (for the model to learn from) and **20% Testing** (for unbiased performance evaluation).

### 3. Baseline Model (v1.0)

An initial **Logistic Regression** model was trained.

- **Result:** The model achieved **76% overall accuracy**.
- **Diagnosis:** Despite solid accuracy on stable customers, the model produced a low *Recall* score (42%) for the Churn class (1), failing to identify more than half of the customers who actually cancelled their contracts.

### 4. Feature Engineering & Optimisation
* We incorporated the client’s financial context by adding the `MonthlyCharges` variable (invoice amount).
* We balanced the class weights (`class_weight=“balanced”`) in the logistic regression to force the model to focus on cancellation attempts.

### 5. Results of the Optimised Model (Version 2.0) 
Following the optimisations, the model showed a significant improvement for the business:
* **Churn Recall (Class 1):** Jumped from **42%** to **79%** (a 37 percentage point increase in the ability to detect cancellations).
* **Overall Accuracy:** Adjusted from **76%** to **73%**, an extremely healthy and expected trade-off for the success of the retention strategy.

##  Technologies & Tools

| Category | Tools |
|---|---|
| **Language** | Python 3 |
| **Data Manipulation** | Pandas, NumPy |
| **Data Visualisation** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-Learn (modelling, data splitting, and metrics) |
| **Version Control** | Git & GitHub |
| **Development Environment** | Jupyter Notebook / VS Code |