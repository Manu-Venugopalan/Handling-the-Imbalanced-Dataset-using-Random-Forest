# 🧠 Handling the Imbalanced Dataset using Random Forest

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen) ![License](https://img.shields.io/badge/License-MIT-green.svg)

This project demonstrates how to handle **imbalanced datasets** using **Random Forest classifiers** with class weights. We use a real-world **Credit Card Fraud Detection dataset** from Kaggle to experiment with different classification techniques and compare the effectiveness of weighting strategies.

> 📂 Dataset Source: [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 📌 Problem Statement

Credit card fraud is a rare but critical issue. In the dataset, only **0.17%** of transactions are fraudulent. This imbalance presents a serious challenge for traditional machine learning models, which tend to favor the majority class (non-fraudulent transactions).

---

## 🚧 Challenges with Imbalanced Datasets

- **Model Bias**: Algorithms tend to predict the majority class.
- **Poor Recall**: Fails to detect minority class (fraud cases).
- **Misleading Accuracy**: High overall accuracy but poor detection of minority class.

---

## 🧪 Experiment Workflow

### 🔶 1. **Data Loading & Exploration**
- Load `creditcard.csv` dataset
- Analyze class distribution and identify imbalance
- Split data into training and testing sets (80/20 split)

### 🔶 2. **Baseline Model: Logistic Regression**
- Fit standard Logistic Regression
- Evaluate performance using:
  - Confusion Matrix
  - Precision, Recall, F1-score

### 🔶 3. **Decision Tree Classifier (Weighted)**
- Applied `class_weight="balanced"` to counter imbalance
- Trained on same split and compared against baseline

### 🔶 4. **Random Forest Classifier**
- Applied `class_weight="balanced_subsample"`
- Fitted on training set and evaluated on test set

---

## 🌲 How Random Forest Helps with Imbalance

Random Forest can internally handle imbalanced datasets by adjusting class weights during training:

- `class_weight="balanced_subsample"` tells each decision tree in the forest to balance classes based on the bootstrap sample it receives.
- This reduces the bias toward the majority class and improves recall for the minority class.
- The ensemble nature of Random Forest makes it more stable and robust compared to single decision trees.

---

## 🔧 Parameters Used

| Model               | Parameter                     | Value               |
|--------------------|-------------------------------|---------------------|
| Logistic Regression| None                          | Default             |
| Decision Tree      | `class_weight`                | `balanced`          |
| Random Forest      | `class_weight`                | `balanced_subsample`|
| Train-Test Split   | `test_size`                   | `0.2`               |
| Random State       | `random_state`                | `101`               |

---

## 📊 Evaluation Metrics

- **Confusion Matrix**: Visual representation of classification results
- **Classification Report**: Includes Precision, Recall, F1-Score
- Focused on improving **Recall** for the minority class (fraudulent transactions)

---

## 📁 Repository Structure

```bash
📦 Handling-the-Imbalanced-dataset/
├── imbalance_handling.ipynb      # Jupyter notebook with full implementation
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/Handling-the-Imbalanced-dataset.git
cd Handling-the-Imbalanced-dataset
pip install -r requirements.txt
```

---

## 🧾 Requirements

```txt
matplotlib
numpy
pandas
scikit-learn
seaborn
```

---

## 📈 Future Improvements

* Explore SMOTE or ADASYN for synthetic oversampling
* Try ensemble techniques like XGBoost or LightGBM with class weighting
* Deploy model as an API for real-time fraud detection

---


