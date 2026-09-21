
# 💳 Credit Card Fraud Detection

A machine learning practice project for detecting fraudulent credit card transactions using classification algorithms and techniques for handling highly imbalanced datasets.

The project focuses on **Exploratory Data Analysis (EDA), preprocessing, model comparison, cross-validation, hyperparameter tuning, and model evaluation using PR-AUC**.

---

## 📌 Project Overview

Credit card fraud detection is a binary classification problem where the goal is to identify whether a transaction is:

* `0` → Non-Fraud
* `1` → Fraud

The dataset is highly imbalanced, meaning fraudulent transactions represent only a very small portion of all transactions.

Because of this imbalance, **accuracy alone is not a reliable evaluation metric**. This project therefore uses **Average Precision / PR-AUC** along with classification reports and confusion matrices.

---

## 📂 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── dataset/
│   └── creditcard.csv
│
├── notebook/
│   └── credit_card_fraud_detection.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📊 Dataset

The project uses the **Credit Card Fraud Detection** dataset from Kaggle.

Dataset reference:

**Credit Card Fraud Detection — Machine Learning Group, ULB**

The dataset contains:

* **284,807 transactions**
* **30 input features**
* **1 target variable (`Class`)**
* `0` → Non-fraud
* `1` → Fraud

The dataset contains:

```text
Non-Fraud: 284,315
Fraud:         492
```

This makes the dataset highly imbalanced, with approximately **1 fraudulent transaction for every 577 non-fraudulent transactions**.

### Dataset Source

Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

> The dataset is not included in this GitHub repository because of its large size. Download `creditcard.csv` from Kaggle and place it inside the `dataset/` folder.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 🔍 Project Workflow

### 1. Data Loading

The dataset is loaded using Pandas.

```python
df = pd.read_csv("creditcard.csv")
```

### 2. Exploratory Data Analysis

The following aspects are explored:

* Dataset shape
* Data types
* Missing values
* Duplicate rows
* Class distribution
* Descriptive statistics
* Transaction amount distribution
* Transaction time distribution
* Amount distribution by class
* Correlation heatmap
* Correlation with the target variable

---

## ⚙️ Data Preprocessing

The following preprocessing steps are performed:

### Log Transformation

A new feature is created using `log1p` transformation:

```python
df["Amount_log1p"] = np.log1p(df["Amount"])
```

The original `Amount` feature is then removed because the transformed version is used.

### Feature / Target Separation

```python
X = df.drop(columns=["Class", "Amount"])
y = df["Class"]
```

### Train-Test Split

The dataset is divided into training and testing sets using a stratified split:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

Stratification is used to preserve the class distribution in both datasets.

---

## 🤖 Models Used

### Baseline Model

A Decision Tree Classifier is used as the baseline model.

```text
Decision Tree Classifier
```

### Models Compared

The project compares:

1. Decision Tree Classifier
2. HistGradientBoosting Classifier
3. Random Forest Classifier

Class balancing is incorporated into the models to account for the highly imbalanced target variable.

---

## 🔄 Cross-Validation

A **5-fold Stratified Cross-Validation** strategy is used.

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The models are compared using **Average Precision (PR-AUC)**.

PR-AUC is used because the dataset contains a very small proportion of fraud cases.

---

## 🎯 Hyperparameter Tuning

After model comparison, Random Forest is selected for further optimization.

`RandomizedSearchCV` is used to search for better hyperparameters.

The parameters explored include:

* `n_estimators`
* `max_depth`
* `min_samples_leaf`
* `max_features`

The optimization uses:

```python
scoring="average_precision"
```

with 5-fold cross-validation.

---

## 🌲 Final Model

The final model is a:

**Random Forest Classifier**

with the tuned parameters used in the notebook.

The model uses:

```python
class_weight="balanced_subsample"
```

to account for class imbalance.

---

## 📈 Model Evaluation

The final model is evaluated on both training and testing data using:

* Accuracy
* Classification Report
* Confusion Matrix
* Average Precision / PR-AUC

### Why not rely only on accuracy?

For highly imbalanced datasets, a model can achieve high accuracy by mostly predicting the majority class.

Therefore, this project focuses more on **PR-AUC and fraud-class performance** rather than accuracy alone.

---

## 🔮 Predictive System

A simple prediction function is created in the notebook:

```python
def predict_class(input_features):
    ...
```

The function accepts transaction features and predicts whether the transaction is:

```text
Fraud Transaction detected ❌
```

or

```text
Non-Fraud Transaction ✅
```

---

## 📚 Key Concepts Practiced

This project helped practice:

* Exploratory Data Analysis
* Data preprocessing
* Feature transformation
* Train-test splitting
* Stratified sampling
* Classification
* Decision Trees
* Random Forest
* Gradient Boosting
* Class imbalance
* Cross-validation
* RandomizedSearchCV
* Hyperparameter tuning
* Confusion Matrix
* Classification Report
* Precision
* PR-AUC / Average Precision
* Predictive systems

---

## 🚀 Future Improvements

Possible improvements for this project include:

* Probability threshold tuning
* Precision-Recall curve visualization
* ROC-AUC comparison
* Feature importance analysis
* SHAP explanations
* Compare additional boosting models
* Handling class imbalance using SMOTE
* Build a Streamlit interface
* Create a REST API using FastAPI
* Dockerize the application
* Deploy the model to the cloud

---

## 🎯 Purpose of This Project

This project is part of my **Machine Learning practice projects**.

The goal is to build small machine learning projects, understand the complete ML workflow, experiment with different algorithms, and gradually improve my practical machine learning skills.

---

## 👨‍💻 Author

**Manmadh**

B.Tech Computer Science Engineering Student
Machine Learning & Software Development Enthusiast

---



⭐ If you find this repository useful, feel free to explore the individual projects.
>>>>>>> 10baada6548affd81c31c4b87106d196dafa60ed
