# Medical Insurance Cost Prediction

## Overview

This project focuses on predicting individual medical insurance costs using supervised regression models. The dataset is the Kaggle medical insurance dataset and includes demographic and lifestyle variables such as age, BMI, smoking status, region, sex, and number of children.

The objective is to achieve strong predictive performance while prioritizing robustness, interpretability, and proper validation.

---

## Data Preparation

The target variable, insurance charges, exhibits strong right skewness. A logarithmic transformation is applied to stabilize variance and improve model performance.

Categorical variables are encoded using One-Hot Encoding. All preprocessing steps are applied after the train-test split to prevent data leakage. One category is dropped for the region variable to avoid multicollinearity.

---

## Models Evaluated

The following regression models are implemented and compared:

- Linear Regression  
- Ridge Regression  
- Polynomial Ridge Regression (degree 2)  
- Random Forest Regressor  

Models are evaluated using R², Mean Squared Error, and 10-fold cross-validation.

---

## Final Model

**Polynomial Ridge Regression (degree 2)** is selected as the final model.

This model provides the best trade-off between performance and stability. It consistently achieves the highest cross-validated R² while maintaining low variance across folds. The polynomial features capture non-linear relationships in the data, while Ridge regularization controls overfitting and preserves interpretability.

The Random Forest model performs well on a single test split but shows higher variability during cross-validation. Simpler linear models, although strong, fail to capture certain interaction effects.

---

## Key Results

- Test R² above 0.85 on the log-transformed target  
- Cross-validated R² at 0.82  
- Stable and well-distributed prediction errors  

---

## Tools Used

- Python  
- pandas, numpy  
- matplotlib  
- scikit-learn  

---

## Notes

This project emphasizes sound machine learning practices, including strict leakage prevention, systematic cross-validation, and model selection based on generalization rather than isolated test performance.
