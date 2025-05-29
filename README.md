# 💳 Credit Risk Assessment Using LASSO Logistic Regression

This project builds a predictive model to assess customer credit risk using logistic regression with LASSO regularization. It identifies the most important predictors of default and calculates an optimal threshold for classification, helping financial institutions mitigate risk and improve lending strategies.

---

## 🔍 Problem Statement

Financial institutions face the constant challenge of lending to the right customers while avoiding defaults. This project aims to:

- Predict whether a customer will default on credit based on financial and demographic data.  
- Select the most relevant features automatically using LASSO regularization.  
- Optimize decision thresholds for better precision and recall trade-offs.

---

## 📦 Dataset

The dataset includes:
- Customer demographics
- Financial behaviors
- Credit history
- Binary target variable `TARGET`:  
  - `1` = default  
  - `0` = no default

---

## 🛠️ Tools & Technologies

- **R**
- `glmnet` – LASSO logistic regression  
- `pROC` – ROC curve analysis  
- `doParallel` – parallel model training  
- Tableau – interactive dashboards  

---

## ⚙️ How It Works

### 1. Data Preprocessing
- Removed rows with missing values
- Converted features to matrix format for `glmnet`
- Handled class imbalance

### 2. Feature Selection with LASSO
- Applied logistic regression with LASSO regularization (`alpha = 1`)
- Automatically selected meaningful predictors by penalizing less informative ones

### 3. Model Training
- Used cross-validation (`cv.glmnet`) to tune lambda and prevent overfitting
- Refit logistic model using selected features

### 4. Threshold Optimization
- ROC curve used to determine the optimal threshold  
- Final threshold: `0.47` for default classification

---

## 📈 Results

| Metric           | Value       |
|------------------|-------------|
| Accuracy         | 91%         |
| Optimal Threshold| 0.47        |

**Confusion Matrix:**

|                | Predicted 0 | Predicted 1 |
|----------------|-------------|-------------|
| Actual 0       | 56,511      | 26          |
| Actual 1       | 4,941       | 24          |

- **True Negatives (TN):** 56,511  
- **False Positives (FP):** 26  
- **False Negatives (FN):** 4,941  
- **True Positives (TP):** 24

---

## 📊 Output & Artifacts

- 📈 ROC Curve
- 📉 Feature Importance Plot  
- 📊 Tableau Dashboard:  
  - Accuracy: `91.84%`  
  - Default Rate: `8.07%`  
  - Visuals for predicted risk distribution and threshold tuning

---

## ✅ Business Value

- 🏦 Helps banks and NBFCs minimize non-performing assets (NPAs)
- 🔍 Provides interpretable insights into risk-driving factors
- 💰 Enhances credit decisions using data-driven classification

---

## 🚀 Future Work

- 🔄 Include behavioral data like payment timelines or transaction frequency
- 🤖 Try advanced models like Random Forests or XGBoost
- 📉 Apply SMOTE or other resampling techniques to address class imbalance
- 🌐 Deploy the model using Shiny or Streamlit for live prediction

---

## 👨‍💻 Author

**Nithin Sai Adru**  
📧 nithin.adru@email.ucr.edu  

---
