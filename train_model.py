import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    print("Loading California Housing dataset...")
    # Load dataset
    data_path = "housing.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    
    # ------------------ EDA and Visualizations ------------------
    print("Performing Exploratory Data Analysis...")
    os.makedirs("plots", exist_ok=True)
    
    # Set style for visuals
    sns.set_theme(style="whitegrid")
    
    # 1. Target Variable Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["median_house_value"], kde=True, color="#1a365d", bins=50)
    plt.title("Distribution of Median House Value", fontsize=16, pad=15, color="#1a365d", weight="bold")
    plt.xlabel("Median House Value ($)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    dist_path = os.path.join("plots", "target_distribution.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"Target distribution plot saved to {dist_path}")
    
    # 2. Correlation Matrix Heatmap
    plt.figure(figsize=(12, 10))
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numerical_cols].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        cmap="coolwarm", 
        fmt=".2f", 
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Numerical Features Correlation Heatmap", fontsize=16, pad=20, color="#1a365d", weight="bold")
    plt.tight_layout()
    corr_path = os.path.join("plots", "correlation_matrix.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"Correlation matrix heatmap saved to {corr_path}")
    
    # ------------------ Data Preprocessing ------------------
    print("Preparing data for modeling...")
    # Separate features and target
    X = df.drop(columns=["median_house_value"])
    y = df["median_house_value"]
    
    # Identify numerical and categorical features
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"Numerical features: {num_features}")
    print(f"Categorical features: {cat_features}")
    
    # Define preprocessing pipeline
    # Impute missing values with median for numerical columns and scale
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # One-hot encode categorical features
    cat_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Combine preprocessors
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ])
    
    # ------------------ Model Pipeline ------------------
    # Create complete machine learning pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Train model
    print("Training Linear Regression model...")
    pipeline.fit(X_train, y_train)
    print("Model training complete.")
    
    # ------------------ Evaluation ------------------
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("MODEL PERFORMANCE METRICS (TEST SET)")
    print("="*40)
    print(f"Mean Absolute Error (MAE):     ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"R² (Coefficient of Det.):     {r2:.4f}")
    print("="*40 + "\n")
    
    # ------------------ Post-training Visualizations ------------------
    print("Generating evaluation plots...")
    # 3. Actual vs Predicted Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color="#2b6cb0", edgecolors="w", linewidth=0.5)
    # Target line (y = x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#e53e3e", linestyle="--", linewidth=2, label="Perfect Prediction")
    plt.title("Actual vs. Predicted Median House Value", fontsize=16, pad=15, color="#1a365d", weight="bold")
    plt.xlabel("Actual Value ($)", fontsize=12)
    plt.ylabel("Predicted Value ($)", fontsize=12)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    pred_path = os.path.join("plots", "actual_vs_predicted.png")
    plt.savefig(pred_path, dpi=300)
    plt.close()
    print(f"Actual vs. Predicted plot saved to {pred_path}")
    
    # 4. Residuals Plot
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residuals vs. Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.3, color="#4a5568", edgecolors="w", linewidth=0.5)
    axes[0].axhline(y=0, color="#e53e3e", linestyle="--", linewidth=2)
    axes[0].set_title("Residuals vs. Predicted Values", fontsize=14, pad=12, color="#1a365d", weight="bold")
    axes[0].set_xlabel("Predicted Value ($)", fontsize=11)
    axes[0].set_ylabel("Residual ($)", fontsize=11)
    
    # Distribution of Residuals
    sns.histplot(residuals, kde=True, color="#4a5568", ax=axes[1], bins=50)
    axes[1].axvline(x=0, color="#e53e3e", linestyle="--", linewidth=2)
    axes[1].set_title("Distribution of Residuals", fontsize=14, pad=12, color="#1a365d", weight="bold")
    axes[1].set_xlabel("Residual ($)", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    
    plt.tight_layout()
    resid_path = os.path.join("plots", "residuals_analysis.png")
    plt.savefig(resid_path, dpi=300)
    plt.close()
    print(f"Residuals analysis plot saved to {resid_path}")
    
    # ------------------ Save Model ------------------
    model_filename = "model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Trained model pipeline saved successfully as '{model_filename}'")
    
    # ------------------ Print Coefficients for Report ------------------
    # Retrieve features after One-Hot Encoding
    ohe_categories = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].categories_
    encoded_cat_cols = []
    for i, col in enumerate(cat_features):
        encoded_cat_cols.extend([f"{col}_{cat}" for cat in ohe_categories[i]])
        
    all_features = num_features + encoded_cat_cols
    coefficients = pipeline.named_steps["regressor"].coef_
    intercept = pipeline.named_steps["regressor"].intercept_
    
    print("\nModel Coefficients:")
    print(f"Intercept: {intercept:,.2f}")
    coef_df = pd.DataFrame({
        "Feature": all_features,
        "Coefficient": coefficients
    }).sort_values(by="Coefficient", key=abs, ascending=False)
    print(coef_df.to_string(index=False))
    
    # Save metrics to a text file for report generation to read
    with open("metrics.txt", "w") as f:
        f.write(f"MAE:{mae}\n")
        f.write(f"RMSE:{rmse}\n")
        f.write(f"R2:{r2}\n")
        f.write(f"Intercept:{intercept}\n")
        for idx, row in coef_df.iterrows():
            f.write(f"COEF:{row['Feature']}:{row['Coefficient']}\n")
            
if __name__ == "__main__":
    main()
