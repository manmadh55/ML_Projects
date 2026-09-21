<<<<<<< HEAD
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

## ⭐ Acknowledgement

Dataset:

**Credit Card Fraud Detection Dataset**

Source: Kaggle — Machine Learning Group, ULB

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
=======
# 🤖 Machine Learning Projects

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=24&pause=1000&color=36BCF7&center=true&vCenter=true&width=700&lines=Learning+Machine_Learning+by+Building+%F0%9F%9A%80;One+Project+at+a+Time+%F0%9F%A7%A0;Learn+%E2%86%92+Build+%E2%86%92+Experiment+%E2%86%92+Improve" alt="Typing SVG" />
</p>

<p align="center">
  <b>A hands-on Machine Learning learning journey 🧠</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/GitHub-Projects-181717?style=for-the-badge&logo=github&logoColor=white"/>
</p>

---

## 🌱 About This Repository

Welcome to my **Machine Learning Projects** repository!

I am currently learning Machine Learning **practically by building projects**, experimenting with different datasets, algorithms, preprocessing techniques, and evaluation methods.

Instead of only learning ML theoretically, I am following a simple approach:

```text
        📚 Learn
           ↓
        🛠️ Build
           ↓
      🧪 Experiment
           ↓
       📊 Evaluate
           ↓
        🔧 Improve
           ↓
       📝 Document
           ↓
      🚀 Push to GitHub
```

This repository is my **learning playground** where I document that journey.

---

# 🎯 What I'm Learning

Through these projects, I am gradually learning:

* 📊 Data Analysis
* 🔍 Exploratory Data Analysis
* 🧹 Data Cleaning
* ⚙️ Data Preprocessing
* 🧩 Feature Engineering
* 📏 Feature Scaling
* 🤖 Machine Learning Algorithms
* 📈 Model Evaluation
* 🔄 Cross-Validation
* 🎛️ Hyperparameter Tuning
* 🧠 Feature Selection
* 🌲 Ensemble Learning
* 🔵 Unsupervised Learning
* 🚀 Model Deployment

---

# 🗺️ My ML Roadmap

```text
🐍 Python
   ↓
🔢 NumPy & Pandas
   ↓
📊 Data Visualization
   ↓
🔍 Exploratory Data Analysis
   ↓
🧹 Data Preprocessing
   ↓
🛠️ Feature Engineering
   ↓
📏 Feature Scaling
   ↓
🤖 Supervised Learning
   ↓
📈 Model Evaluation
   ↓
🔄 Cross-Validation
   ↓
⚙️ Hyperparameter Tuning
   ↓
🎯 Feature Selection
   ↓
🌲 Ensemble Learning
   ↓
🔵 Unsupervised Learning
   ↓
🧪 Practical ML Projects
   ↓
🚀 Model Deployment
   ↓
☁️ Cloud & MLOps
```

> 🚧 This roadmap is continuously evolving as I learn.

---

# 📂 Projects

| #  | Project                                                          | Type                        | Main Concepts                                      | Status       |
| -- | ---------------------------------------------------------------- | --------------------------- | -------------------------------------------------- | ------------ |
| 01 | [🎵 Sonar Signal Classification](./Sonar_Signal_Classification/) | Classification              | EDA, preprocessing, scaling, binary classification | 🟢 Completed |
| 02 | [🎗️ Breast Cancer Diagnosis](./Breast_Cancer_Diagnosis/)        | Classification              | EDA, preprocessing, PCA, SVM, GridSearchCV         | 🟢 Completed |
| 03 | 📧 Spam Email Classification                                     | NLP / Classification        | Text preprocessing, TF-IDF, Naive Bayes            | 🔵 Planned   |
| 04 | 👥 Customer Churn Prediction                                     | Classification              | Categorical data, pipelines, evaluation            | 🔵 Planned   |
| 05 | 💳 Fraud Detection                                               | Classification              | Imbalanced data, SMOTE, precision, recall          | 🔵 Planned   |
| 06 | 🛍️ Customer Segmentation                                        | Clustering                  | K-Means, PCA, unsupervised learning                | 🔵 Planned   |
| 07 | 🍷 Wine Quality Prediction                                       | Classification / Regression | Feature selection, ensemble learning               | 🔵 Planned   |
| 08 | 🎬 Movie Recommendation System                                   | Recommendation              | Similarity, recommendation techniques              | 🔵 Planned   |

> 🟢 **Completed** = Project implemented and pushed
> 🟡 **In Progress** = Currently building
> 🔵 **Planned** = Future project idea

---

# 🧠 Concepts I'm Practicing

### 📊 Data Analysis

```text
NumPy
Pandas
Data Cleaning
EDA
Data Visualization
Correlation Analysis
```

### ⚙️ Data Preprocessing

```text
Missing Values
Categorical Encoding
Feature Scaling
Train-Test Split
Feature Engineering
Feature Selection
```

### 🤖 Supervised Learning

```text
Linear Regression
Logistic Regression
K-Nearest Neighbors
Support Vector Machines
Decision Trees
Random Forest
Gradient Boosting
Ensemble Learning
```

### 📈 Model Evaluation

```text
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
ROC-AUC
MAE
MSE
RMSE
R² Score
```

### ⚙️ Model Optimization

```text
Cross-Validation
GridSearchCV
RandomizedSearchCV
Hyperparameter Tuning
Pipelines
ColumnTransformer
```

### 🔵 Unsupervised Learning

```text
K-Means Clustering
PCA
Cluster Evaluation
```

---

# 🔬 How I Build Each Project

Most projects follow a workflow similar to:

```text
01. 🎯 Understand the Problem
             ↓
02. 📂 Understand the Dataset
             ↓
03. 🔍 Perform EDA
             ↓
04. 🧹 Clean & Preprocess Data
             ↓
05. 🛠️ Feature Engineering
             ↓
06. ✂️ Train-Test Split
             ↓
07. 🤖 Train Models
             ↓
08. 📊 Evaluate Models
             ↓
09. 🔄 Cross-Validation
             ↓
10. ⚙️ Hyperparameter Tuning
             ↓
11. 🏆 Select Final Model
             ↓
12. 🔮 Make Predictions
             ↓
13. 📝 Document Results
```

The workflow may change depending on the problem and dataset.

---

# 🛠️ Tech Stack

<p align="center">

<img src="https://skillicons.dev/icons?i=python,vscode,git,github" />

</p>

### Python Libraries

* 🐼 **Pandas**
* 🔢 **NumPy**
* 📈 **Matplotlib**
* 🎨 **Seaborn**
* 🤖 **Scikit-learn**
* 📓 **Jupyter Notebook**

Additional libraries will be added to individual projects whenever required.

---

# 📁 Repository Structure

Each project is kept **independent** with its own environment and dependencies.

```text
ML-Projects/
│
├── 📂 Sonar_Signal_Classification/
│   ├── sonar_data.csv
│   ├── sonar_signal_classification.ipynb
│   ├── .venv/
│   ├── requirements.txt
│   └── README.md
│
├── 📂 Breast_Cancer_Diagnosis/
│   ├── breast_cancer_dataset.csv
│   ├── breast_cancer_diagnosis.ipynb
│   ├── .venv/
│   ├── requirements.txt
│   └── README.md
│
├── 📂 Future_Projects/
│
└── 📄 README.md
```

### 🔒 Project Environments

Each ML project has its own virtual environment:

```text
Sonar_Signal_Classification/
        └── .venv/

Breast_Cancer_Diagnosis/
        └── .venv/
```

This keeps project dependencies isolated and avoids package/version conflicts.

> ⚠️ `.venv/` folders are kept locally and should **not** be pushed to GitHub.



So for each project, I try to:

```text
Learn → Implement → Experiment → Compare → Improve

---

# 👨‍💻 About Me

## G. Manmadh

🎓 **B.Tech Computer Science Engineering Student**

I enjoy solving difficult problems and consistently learning new technologies.

### Interested in

```text
💻 Software Development
🤖 Machine Learning
🧠 Artificial Intelligence
📊 Data & Problem Solving
```

<p align="center">
  <b>Building my skills one project at a time 🚀</b>
</p>

---

# ⭐ Repository Status

<p align="center">

### 🚧 ACTIVE LEARNING REPOSITORY 🚧

This repository will continue to evolve as I learn, experiment, and build more Machine Learning projects.

</p>

```text
        ┌─────────────────────────────┐
        │     KEEP LEARNING 📚       │
        │            ↓                │
        │      KEEP BUILDING 🛠️      │
        │            ↓                │
        │     KEEP EXPERIMENTING 🧪  │
        │            ↓                │
        │       KEEP IMPROVING 🚀    │
        └─────────────────────────────┘
```

<p align="center">
  <b>⭐ Learn. Build. Experiment. Repeat. ⭐</b>
</p>


⭐ If you find this repository useful, feel free to explore the individual projects.
>>>>>>> 10baada6548affd81c31c4b87106d196dafa60ed
