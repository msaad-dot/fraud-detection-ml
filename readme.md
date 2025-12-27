# 💳 Fraud Detection — Machine Learning Pipeline

## 📌 Project Overview
This project tackles a **highly imbalanced credit card fraud detection** problem,
where the goal is to maximize fraud detection while controlling false positive alerts
that negatively impact customer experience.

Fraud detection is treated as a **cost-sensitive classification problem**, where
false negatives (missed fraud) typically incur significantly higher cost than
false positives.

---

## 📊 Dataset
The dataset contains anonymized credit card transactions:
- Features **V1–V28** are **PCA-transformed**
- **Time** and **Amount** represent transaction time and value
- Target variable **Class**:
  - `0` → Normal transaction  
  - `1` → Fraudulent transaction  

Fraud cases represent approximately **0.17%** of the dataset, making
traditional accuracy-based evaluation misleading.

> The dataset is excluded from version control and should be placed locally under the `data/` directory.  
> Dataset source: Kaggle — Credit Card Fraud Dataset.

---

## 🗂️ Project Structure
> Trained models and preprocessing artifacts are persisted locally for
> reproducibility but are excluded from version control.

```text
fraud-detection-ml/
├── data/       (local only, excluded from version control)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_cost_evaluation.ipynb
│
├── images/
│   └── pr_curve_comparison.png
│
├── requirements.txt
└── README.md
```

---

## 🔁 Machine Learning Pipeline

```text
Raw Dataset (creditcard.csv)
        ↓
01_eda — Exploratory Data Analysis
        ↓
02_preprocessing
  • Stratified train/test split
  • Feature scaling (Time, Amount)
  • Numeric type enforcement
  • Artifact persistence
        ↓
03_modeling
  • Logistic Regression (baseline)
  • Random Forest
  • XGBoost
        ↓
04_evaluation
  • PR-AUC / ROC-AUC
  • Precision–Recall analysis
  • Baseline threshold selection
        ↓
05_model_comparison
  • Cross-model comparison
  • Model-specific threshold tuning
        ↓
06_cost_evaluation
  • Expected financial loss analysis
  • Cost-based threshold optimization
  • Final model selection
```

---

## ⚙️ Modeling Strategy
- Severe class imbalance handled using **class-weighted training**
- **PR-AUC** prioritized over accuracy due to extreme imbalance
- **Probability-based evaluation** used instead of hard predictions
- **Threshold tuning** aligned with operational and business risk
- Preprocessing, modeling, and evaluation are fully **decoupled**
  to resemble real-world ML pipelines
- Different probability distributions across models required
  **model-specific threshold selection**

---

## 📊 Final Model Comparison

| Model | PR-AUC | Threshold | Precision | Recall | False Positives |
|------|--------|-----------|-----------|--------|-----------------|
| Logistic Regression | 0.716 | 0.70 | 0.12 | 0.91 | 644 |
| Random Forest | 0.854 | 0.35 | 0.94 | 0.81 | 5 |
| **XGBoost** | **0.861** | **0.50** | **0.67** | **0.86** | **41** |

---

## 💰 Cost-Based Model Selection
Beyond statistical performance, models were evaluated using a
**business-oriented cost framework**, where:

- False negatives represent missed fraud losses
- False positives represent operational and customer experience costs

A cost-sensitive threshold analysis demonstrated that **XGBoost**
achieves the **lowest expected financial loss** while maintaining
strong fraud recall and manageable alert volume.

➡️ **XGBoost was selected as the final production candidate based on
expected business impact, not metric maximization alone.**

> **Key takeaway:** In real-world fraud detection systems, the optimal model
is defined by business trade-offs rather than accuracy or recall in isolation.

---

## 📈 Precision–Recall Curve Comparison
![Precision–Recall Curve](images/pr_curve_comparison.png)

---

## 🛠️ Tech Stack
- Python
- scikit-learn
- XGBoost
- NumPy / Pandas
- Matplotlib

---

## 👤 Author
**Mohamed Saad**
