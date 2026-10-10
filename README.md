<div align="center">
<img src="banner.jpg" alt="Telco Churn Prediction Banner" width="100%">

  # 📡 Telco Customer Churn Prediction

**Predicting which telecom customers are about to leave, using XGBoost and Logistic Regression**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-189AB4)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rxr20/machinelearning_telco_churn/blob/main/Telco-Churn-Predictor.ipynb)

</div>

---

This project uses machine learning to predict whether a tech customer is likely to leave the company (churn) based on their subscribed services, contract details and billing information.

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 |
|---|:-:|:-:|:-:|:-:|
| Baseline (always "No churn") | 0.73 | 0.00 | 0.00 | 0.00 |
| Logistic Regression (CV) | 0.76 | 0.53 | **0.79** | **0.64** |
| ⭐ **XGBoost + tuned threshold (test set)** | **0.80** | **0.65** | 0.57 | 0.61 |

> XGBoost beats the baseline by about **+7 points of accuracy** and correctly flags more than half of the churners, with about 2 out of 3 churn predictions being right. Across runs it stays at **0.80–0.81 accuracy** and **0.60–0.61 F1**.

## 🗂️ Dataset

This project uses the [IBM Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), containing **7,043 customer records** and 21 columns covering customer demographics, subscribed services, contracts, and billing information. The target variable, `Churn`, indicates whether a customer left the company (~26.5% did).

The notebook loads the CSV directly from a URL, so there's nothing to download.

## ⚙️ How it works

1. **Data exploration**: target balance, missing values, and charts for charges, tenure, contract and payment method
2. **Data prep**: fix the blank `TotalCharges` values, merge the "No internet/phone service" categories, and add new features (`n_services`, `avg_charge`, `charge_diff`, `is_new`, `auto_pay`, `fiber_no_support`)
3. **Pipeline**: a `ColumnTransformer` with imputing + scaling for numeric columns and one-hot encoding for categorical ones, applied after the train/test split (80/20, stratified)
4. **Baseline**: `DummyClassifier`
5. **Models**: XGBoost and Logistic Regression, both tuned with `RandomizedSearchCV` and 5-fold stratified cross-validation
6. **Threshold tuning**: the decision threshold is picked on out-of-fold predictions, never on the test set
7. **Final evaluation**: a single run on the held-out test set, with a confusion matrix

## 📓 About the notebook (`.ipynb`)

`Telco-Churn-Predictor.ipynb` is a **Jupyter Notebook**. It mixes code, explanations and outputs (tables and charts) in one file. You can open it several ways:

| Option | How |
|---|---|
| 👀 **Just read it** | Click `Telco-Churn-Predictor.ipynb` above. GitHub shows it with all outputs already rendered. If it fails to load, paste the link into [nbviewer.org](https://nbviewer.org/). |
| ☁️ **Run it online** | Click the **Open in Colab** badge at the top, then *Runtime → Run all*. No install needed. |
| 💻 **Run it locally** | See below. |

```bash
pip install numpy pandas matplotlib scikit-learn xgboost jupyter
jupyter notebook Telco-Churn-Predictor.ipynb
```

You can also open it in **VS Code** with the *Jupyter* extension. Run the cells from top to bottom; the hyperparameter search takes a few minutes.

## 🛠️ Tech stack

`Python` · `pandas` · `NumPy` · `Matplotlib` · `scikit-learn` · `XGBoost` · `Jupyter`

---
