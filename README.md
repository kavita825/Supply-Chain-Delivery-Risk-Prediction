# Supply Chain Delivery Risk Prediction

A machine learning project that predicts whether an order is at risk of **late delivery** *before* it is dispatched, so that teams can act early instead of reacting after the delay has happened.


## Overview

Late deliveries hurt customer trust, raise operational cost, and are usually detected only after the delivery promise is already broken. This project treats late-delivery prediction as a **binary classification** problem:

- `0` – low / no risk
- `1` – order is at risk of late delivery

Three classifiers are trained on the same data and compared:

1. Logistic Regression (interpretable baseline)
2. Random Forest (bagging ensemble)
3. Gradient Boosting (boosting ensemble)

## Dataset

A prototype dataset of **100 orders** with **13 columns**, stored as a CSV file.

| Property | Value |
|---|---|
| Records | 100 |
| Columns | 13 |
| Target | `Late_Delivery_Risk` (binary) |
| Class balance | 57 no-risk / 43 at-risk |
| Missing values | None |
| Train / test split | 80 / 20 (stratified, fixed random seed) |

**Features**

- *Categorical:* shipping mode, customer segment, product category, region
- *Numerical:* shipping distance, processing time, shipping time, scheduled delivery days, order value, previous delays, warehouse workload

## Methodology

1. Load the CSV into a pandas DataFrame and explore it.
2. Preprocess with a scikit-learn `ColumnTransformer`:
   - One-Hot Encoding for categorical features
   - `StandardScaler` for numerical features
3. Wrap preprocessing and classifier together in a single `Pipeline`, so the same transformations apply to training data, test data, and any new order.
4. Train all three models on the same stratified 80/20 split.
5. Evaluate using accuracy, precision, recall, and F1-score.
6. Predict risk for a new, unseen order.

## Results

Evaluated on the same 20 held-out test orders:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 80.0% | 1.00 | 0.56 | 0.71 |
| **Random Forest** | **85.0%** | 1.00 | 0.67 | **0.80** |
| Gradient Boosting | 75.0% | 0.75 | 0.67 | 0.71 |

**Random Forest** gave the best overall balance. Because the dataset is small, these results are indicative rather than conclusive.

## Tech Stack

- Python
- Jupyter Notebook / Google Colab
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn

**Hardware:** Intel i5 (or equivalent), 8 GB RAM.

## Project Structure

```
.
├── COA_CODE.ipynb        # Notebook: EDA, preprocessing, training, evaluation
├── <dataset>.csv         # 100-order dataset (rename to match your file)
├── final_.docx           # Project report
└── README.md
```

## How to Run

1. Clone or download this repository.
2. Install the dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. Keep the dataset CSV in the same folder as the notebook.
4. Open the notebook and run all cells:

   ```bash
   jupyter notebook COA_CODE.ipynb
   ```

   Or upload both files to Google Colab and use **Runtime → Run all**.

## Future Scope

- Integrate real-time traffic, weather, and GPS data
- Retrain on a larger, production-scale order history
- Deploy predictions inside delivery management tools
- Tune hyperparameters with cross-validation and grid/random search
- Add feature-importance or SHAP-based explanations

## References

- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.*
- Breiman, L. (2001). *Random Forests.*
- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.*
- Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow.*
