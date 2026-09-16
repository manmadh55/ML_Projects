# 🎗️ Breast Cancer Diagnosis using Machine Learning

<p align="center">
  <b>Machine Learning Classification Project</b>
  <br>
  Predicting Breast Tumor Diagnosis using PCA, SVM & Hyperparameter Optimization
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn">
  <img src="https://img.shields.io/badge/Model-SVM-green">
  <img src="https://img.shields.io/badge/Accuracy-96.49%25-brightgreen">
</p>

---

## 📌 About the Project

**Breast Cancer Diagnosis** is a machine learning classification project built to predict whether a breast tumor is **Benign** or **Malignant** based on diagnostic measurements.

The project follows a complete machine learning workflow:

```text
📂 Dataset
    ↓
🔍 Data Understanding
    ↓
📊 Exploratory Data Analysis
    ↓
🧹 Data Preprocessing
    ↓
📏 Feature Scaling
    ↓
📉 PCA Dimensionality Reduction
    ↓
🤖 Model Training
    ↓
⚙️ Hyperparameter Optimization
    ↓
📈 Model Evaluation
    ↓
🔮 Predictive System
```

The main optimized model uses a **StandardScaler → PCA → SVM** pipeline.

> ⚠️ **Disclaimer:** This project is developed for educational and machine-learning practice purposes. It is **not a medical diagnostic system** and should not be used for clinical decision-making.

---

# 🎯 Objective

The main objective of this project is to develop a classification model capable of distinguishing between:

| Class                | Meaning             |
| -------------------- | ------------------- |
| 🟢 **Benign (0)**    | Non-cancerous tumor |
| 🔴 **Malignant (1)** | Cancerous tumor     |

The project also focuses on understanding how **feature scaling, dimensionality reduction, cross-validation, and hyperparameter tuning** affect model performance.

---

# 📊 Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

**Dataset:** 569 samples with 30 numerical diagnostic features after removing the `id` column.

### Dataset Distribution

```text
Benign       : 357 samples  (62.74%)
Malignant    : 212 samples  (37.26%)
```

The dataset contains features describing characteristics of cell nuclei, including:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Concave Points
* Symmetry
* Fractal Dimension

For several measurements, the dataset provides:

```text
Mean values
Standard Error values
Worst values
```

---

# 🔎 Exploratory Data Analysis

The project performs several EDA steps to understand the dataset.

### ✔ Dataset Inspection

* Dataset shape
* Feature information
* Data types
* Statistical analysis
* Missing-value detection
* Duplicate detection
* Class distribution

### ✔ Important Findings

* Dataset contains **569 observations**
* 30 numerical input features are used
* No missing values were found
* No duplicate rows were found
* The dataset contains a moderate class imbalance
* Several features show strong correlation
* Feature distributions differ noticeably between benign and malignant cases

### 📊 Class Distribution

```text
Benign      ███████████████████████████████  62.74%

Malignant   ██████████████████               37.26%
```

Because this is a medical classification problem, **recall is treated as an important evaluation metric**, particularly for the malignant class.

---

# 🧹 Data Preprocessing

The following preprocessing steps were performed:

### 1. Remove unnecessary column

The `id` column was removed because it does not represent a diagnostic feature.

### 2. Separate features and target

```python
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]
```

### 3. Encode target

```text
B → 0
M → 1
```

### 4. Train-Test Split

The dataset was split using:

```text
80% → Training
20% → Testing
```

with stratification to preserve the class distribution.

```text
Total Dataset : 569
Training      : 455
Testing       : 114
```

---

# 🤖 Baseline Model

## Logistic Regression

A Logistic Regression model was first trained as a baseline.

Feature scaling was performed using:

```python
StandardScaler()
```

### Baseline Results

| Dataset  |   Accuracy |
| -------- | ---------: |
| Training | **98.68%** |
| Testing  | **96.49%** |

The baseline model provided a strong starting point before applying dimensionality reduction and SVM optimization.

---

# 📉 PCA — Principal Component Analysis

Because the dataset contains **30 input features** with significant correlations, PCA was explored for dimensionality reduction.

PCA transforms the original feature space into a smaller number of principal components while preserving important information.

The optimized pipeline selected:

```text
30 Features
     ↓
StandardScaler
     ↓
PCA
     ↓
10 Principal Components
```

This reduces the dimensionality from:

```text
30 → 10
```

---

# ⚙️ Model Optimization

The optimized model uses an `sklearn` Pipeline:

```text
StandardScaler
      ↓
PCA
      ↓
SVM (RBF Kernel)
```

### Hyperparameters Tuned

GridSearchCV was used to search over:

### PCA

```text
n_components:
10, 15, 20, 25, 30
```

### SVM

```text
C:
0.1, 1, 10, 100, 1000

gamma:
0.001, 0.01, 0.1, 1, scale
```

### Cross-Validation

A **5-fold Stratified Cross-Validation** strategy was used.

The optimization metric was:

```text
Recall
```

This was selected because missing a malignant case is particularly important in the context of this classification problem.

---

# 🏆 Optimized Model

The best configuration obtained from GridSearchCV was:

```text
Model              : SVM
Kernel             : RBF
C                  : 10
Gamma              : scale
PCA Components     : 10
Cross-Validation   : 5-Fold Stratified
Scoring            : Recall
```

### Best Cross-Validation Recall

```text
96.47%
```

---

# 📈 Final Model Performance

## Training Performance

```text
Training Accuracy: 98.90%
```

### Training Classification Report

| Class     | Precision | Recall | F1-Score |
| --------- | --------: | -----: | -------: |
| Benign    |      0.98 |   1.00 |     0.99 |
| Malignant |      1.00 |   0.97 |     0.99 |

---

## 🧪 Test Performance

```text
Test Accuracy: 96.49%
```

### Test Classification Report

| Class       | Precision |   Recall | F1-Score |
| ----------- | --------: | -------: | -------: |
| Benign      |      0.96 | **0.99** |     0.97 |
| Malignant   |      0.97 | **0.93** |     0.95 |
| **Overall** |  **0.97** | **0.96** | **0.96** |

### Key Metrics

```text
Accuracy   : 96.49%
Macro F1   : 96%
Malignant Recall : 93%
```

---

# 🧩 Confusion Matrix

The project generates confusion matrices for both training and test predictions to understand classification errors.

The test confusion matrix helps identify:

```text
True Negatives
False Positives
False Negatives
True Positives
```

Particular attention is given to **false negatives**, where a malignant tumor is incorrectly classified as benign.

---

# 🔮 Predictive System

A reusable prediction function was created:

```python
def predict_cancer(input_features):
    input_df = pd.DataFrame(
        [input_features],
        columns=X_train.columns
    )

    prediction = best_pipeline.predict(input_df)

    if prediction[0] == 1:
        print("Diagnosis - Malignant 🔴")
    else:
        print("Diagnosis - Benign 🟢")
```

The system can take a new set of diagnostic features and return a predicted class.

### Example Predictions

```text
Sample 1 → Benign 🟢

Sample 2 → Malignant 🔴

Sample 3 → Malignant 🔴
```

---

# 🛠️ Tech Stack

### Programming Language

🐍 **Python**

### Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

The project's current `requirements.txt` contains NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Jupyter, and IPykernel.

---

# 📁 Project Structure

```text
Breast_Cancer_Diagnosis/
│
├── 📂 dataset/
│   └── breast_cancer_dataset.csv
│
├── 📂 notebooks/
│   └── breast_cancer_diagnosis.ipynb
│
├── 📂 models/
│   └── ...
│
├── 📂 src/
│   └── ...
│
├── 📄 requirements.txt
├── 📄 README.md
│
└── 📂 .venv/
```

> `.venv/` should **not** be uploaded to GitHub. Add it to `.gitignore`.

---

# 🚀 How to Run

## 1️⃣ Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
```

## 2️⃣ Navigate to the Project

```bash
cd Breast_Cancer_Diagnosis
```

## 3️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

## 4️⃣ Activate Virtual Environment

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
.venv\Scripts\activate
```

## 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 6️⃣ Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/breast_cancer_diagnosis.ipynb
```

---

# 🧠 What I Learned

Through this project, I practiced:

* Data loading and inspection
* Exploratory Data Analysis
* Handling categorical target variables
* Train-test splitting
* Feature scaling
* Logistic Regression
* Support Vector Machines
* PCA
* Machine Learning Pipelines
* Stratified K-Fold Cross-Validation
* GridSearchCV
* Hyperparameter tuning
* Classification reports
* Confusion matrices
* Recall-focused model optimization
* Building a reusable prediction function

---

# 🔭 Future Improvements

The project can be extended with:

* [ ] Feature selection techniques
* [ ] Compare more classification algorithms
* [ ] RandomizedSearchCV
* [ ] ROC-AUC analysis
* [ ] Precision-Recall curve
* [ ] SHAP / model explainability
* [ ] Save trained model using `joblib`
* [ ] Build a Streamlit interface
* [ ] Create a FastAPI prediction API
* [ ] Dockerize the application
* [ ] Cloud deployment

---

# 📚 Learning Journey

This project is part of my **hands-on Machine Learning practice**, where I am building small projects to understand ML concepts by implementing them from scratch and experimenting with different techniques.

Current learning progression:

```text
Python
  ↓
NumPy & Pandas
  ↓
Data Visualization
  ↓
Exploratory Data Analysis
  ↓
Data Preprocessing
  ↓
Classification
  ↓
Model Evaluation
  ↓
Cross-Validation
  ↓
Hyperparameter Tuning
  ↓
PCA
  ↓
Machine Learning Pipelines
  ↓
Model Deployment
```

---

# 👨‍💻 Author

### G. Manmadh

**B.Tech Computer Science Engineering Student**

Interested in:

```text
💻 Software Development
🤖 Machine Learning
🧠 Artificial Intelligence
📊 Data Science
```

GitHub: **[@manmadh55](https://github.com/manmadh55)**

---

<p align="center">
  ⭐ If you found this project useful, consider giving the repository a star!
</p>

<p align="center">
  <i>Built with Python • Scikit-learn • Curiosity 🚀</i>
</p>
