# California Housing Price Prediction

## Overview

This project develops a baseline machine learning model for predicting residential property prices in California using the California Housing Dataset. The objective is to establish a reliable benchmark model through comprehensive data preprocessing, exploratory data analysis, and performance evaluation.

The solution employs a Linear Regression model integrated within a Scikit-Learn pipeline to ensure reproducibility, prevent data leakage, and provide a foundation for future model improvements.

---

## Problem Statement

Accurate housing price estimation is a critical component of the real estate industry, enabling informed decision-making for buyers, sellers, investors, and policymakers.

This project aims to predict median house values using demographic, socioeconomic, and geographic attributes available within the California Housing Dataset.

---

## Dataset

**Source:** Scikit-Learn California Housing Dataset

**Dataset Characteristics**

| Attribute | Value |
|------------|---------|
| Total Records | 20,640 |
| Features | 8 Numerical + 1 Categorical |
| Target Variable | Median House Value |

### Features

- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Median Income
- Ocean Proximity

---

## Exploratory Data Analysis

A detailed exploratory analysis was conducted to understand the dataset characteristics and identify potential relationships between features and the target variable.

### Key Findings

- Median Income demonstrated the strongest positive correlation with housing prices.
- Housing prices exhibit a capped distribution near \$500,000.
- Geographic location significantly influences property values.
- Several features required preprocessing to ensure compatibility with machine learning algorithms.

---

## Data Preprocessing

To maintain model integrity and prevent data leakage, a structured preprocessing pipeline was implemented.

### Missing Value Treatment

- Missing values in the `total_bedrooms` feature were imputed using median values derived from the training dataset.

### Feature Scaling

- Numerical features were standardized using `StandardScaler`.

### Categorical Encoding

- The `ocean_proximity` feature was transformed using One-Hot Encoding.

### Data Split

- Training Set: 80%
- Testing Set: 20%
- Random State: 42

---

## Model Development

### Baseline Model

A Linear Regression model was selected as the initial benchmark due to its interpretability and computational efficiency.

The model was trained using Scikit-Learn's Pipeline architecture, combining preprocessing and model training into a single workflow.

---

## Performance Evaluation

The model was evaluated using industry-standard regression metrics.

| Metric | Score |
|----------|----------|
| Mean Absolute Error (MAE) | \$50,670.49 |
| Root Mean Squared Error (RMSE) | \$70,059.19 |
| R² Score | 0.6254 |

### Interpretation

- The model explains approximately **62.5%** of the variance in housing prices.
- Average prediction error is approximately **\$50,670**.
- Results establish a solid baseline for future model enhancements.

---

## Residual Analysis

Residual analysis was performed to evaluate:

- Linearity assumptions
- Error distribution
- Variance consistency
- Prediction reliability

The analysis indicates that while the model captures general trends, it struggles to model complex nonlinear relationships present in housing markets.

---

## Feature Importance Insights

The most influential predictors identified by the model include:

| Feature | Impact |
|----------|----------|
| Median Income | Strong Positive |
| Ocean Proximity (Island) | Positive |
| Latitude | Negative |
| Longitude | Negative |
| Population | Negative |

These findings align with real-world housing market behavior where income levels and location are primary determinants of property value.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---

## Project Structure

```text
california-housing-price-prediction/
│
├── data/
├── notebooks/
├── reports/
│   └── california_housing_report.pdf
├── src/
├── README.md
├── requirements.txt
└── housing_price_prediction.ipynb
```

---

## Future Enhancements

The current implementation serves as a baseline model. Future improvements may include:

- Random Forest Regression
- XGBoost Regression
- Gradient Boosting Models
- Hyperparameter Optimization
- Advanced Feature Engineering
- Geospatial Feature Extraction
- Model Deployment using Flask or FastAPI

---

## Results

The project successfully demonstrates an end-to-end machine learning workflow, including data preparation, model development, evaluation, and interpretation. The baseline Linear Regression model provides meaningful predictive capability and establishes a benchmark for more sophisticated approaches.<img width="3600" height="3000" alt="correlation_matrix" src="https://github.com/user-attachments/assets/207b7296-d42d-4ee9-8d22-5de168d9582f" />


<img width="4500" height="1800" alt="residuals_analysis" src="https://github.com/user-attachments/assets/3b526679-d24a-4e29-afd5-b69645c9666f" />
<img width="3000" height="1800" alt="target_distribution" src="https://github.com/user-attachments/assets/4620e9e7-ac6a-4611-9697-69ee0ca9ca0f" />
<img width="3000" height="1800" alt="actual_vs_predicted" src="https://github.com/user-attachments/assets/e8e8922e-c9ba-4c49-862e-4572536210e8" />

## Author

**Sreepathi Praveena**

B.Tech Student | Artificial Intelligence & Machine Learning Enthusiast

### Repository Highlights

✔ End-to-End Machine Learning Workflow  
✔ Production-Style Preprocessing Pipeline  
✔ Comprehensive Model Evaluation  
✔ Statistical Interpretation of Results  
✔ Scalable Foundation for Advanced Models
