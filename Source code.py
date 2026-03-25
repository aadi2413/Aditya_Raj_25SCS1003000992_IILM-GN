# ==============================================================
# Comparative Study of ML Algorithms for Concrete Strength Prediction
# Full Colab-ready source code (single block). Replace filenames if needed.
# ==============================================================

# ---------- 1) Imports ----------
import warnings
warnings.filterwarnings("ignore")

import os
from google.colab import files            # for Colab upload
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Make plots look decent
plt.rcParams['figure.figsize'] = (8,5)
sns.set(style="whitegrid")

# ---------- 2) Upload and load data (Colab) ----------
print("Step 1: Upload your dataset (concrete_data.csv). Use the file upload dialog.")
uploaded = files.upload()   # run and choose your concrete_data.csv

# Get uploaded filename
if len(uploaded) == 0:
    raise SystemExit("No file uploaded. Upload concrete_data.csv and re-run this cell.")

file_name = list(uploaded.keys())[0]
print(f"Uploaded file detected: {file_name}")

# Load CSV
df = pd.read_csv(file_name)
print("\nStep 2: Dataset loaded. Preview:")
display(df.head())

# ---------- 3) Basic EDA ----------
print("\n===== Basic Info =====")
print(df.info())

print("\n===== Descriptive Statistics =====")
display(df.describe().T)

print("\n===== Column names =====")
print(df.columns.tolist())

print("\n===== Missing values per column =====")
print(df.isnull().sum())

# ---------- 4) Target check and feature/target split ----------
TARGET = "Strength"   # confirmed name from your dataset
if TARGET not in df.columns:
    raise KeyError(f"The expected target column '{TARGET}' was not found in the dataset. Columns: {df.columns.tolist()}")

X = df.drop(TARGET, axis=1)
y = df[TARGET]

print(f"\nFeatures used ({len(X.columns)}): {list(X.columns)}")
print(f"Target variable: {TARGET}")

# ---------- 5) Handle missing values ----------
# Strategy: if very few missing -> drop rows; else simple imputation (median).
missing_counts = df.isnull().sum()
total_rows = len(df)
cols_with_missing = missing_counts[missing_counts > 0]

if len(cols_with_missing) == 0:
    print("\nNo missing values detected.")
else:
    print("\nMissing values detected:")
    print(cols_with_missing)
    pct_missing = (cols_with_missing / total_rows) * 100
    print("\nPercent missing per column:")
    print(pct_missing)
    # Simple policy:
    # If a column has <5% missing, drop rows. Else, impute median.
    cols_drop_rows = pct_missing[pct_missing < 5].index.tolist()
    cols_impute = pct_missing[pct_missing >= 5].index.tolist()
    if cols_drop_rows:
        print(f"Dropping rows with missing in columns: {cols_drop_rows}")
        df = df.dropna(subset=cols_drop_rows)
    if cols_impute:
        print(f"Imputing median for columns: {cols_impute}")
        for c in cols_impute:
            df[c].fillna(df[c].median(), inplace=True)
    # Recompute X,y after handling
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    print("Missing handling done. Remaining missing counts:")
    print(df.isnull().sum())

# ---------- 6) Exploratory plots & correlation ----------
os.makedirs("outputs", exist_ok=True)

# Histogram of target
plt.figure()
plt.hist(y, bins=25)
plt.title("Distribution of Concrete Compressive Strength")
plt.xlabel("Strength")
plt.ylabel("Frequency")
plt.tight_layout()
hist_path = "outputs/hist_strength.png"
plt.savefig(hist_path)
plt.show()

# Pairwise correlation heatmap (numeric columns)
plt.figure(figsize=(9,7))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix")
plt.tight_layout()
corr_path = "outputs/correlation_matrix.png"
plt.savefig(corr_path)
plt.show()

# ---------- 7) Train-test split & scaling ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "outputs/scaler.joblib")

# ---------- 8) Models to train ----------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
}

results = []

# ---------- 9) Train, predict and evaluate ----------
for name, model in models.items():
    print(f"\nTraining & evaluating: {name}")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    # 5-fold CV (R2)
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="r2")
    cv_mean = cv_scores.mean()

    print(f"R2 (test): {r2:.4f}")
    print(f"MSE (test): {mse:.4f}")
    print(f"RMSE (test): {rmse:.4f}")
    print(f"MAE (test): {mae:.4f}")
    print(f"R2 CV (5-fold mean): {cv_mean:.4f}")

    # Save model
    model_path = f"outputs/{name.replace(' ', '_')}.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    results.append({
        "Model": name,
        "R2": r2,
        "R2_CV": cv_mean,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Model_Path": model_path
    })

results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False).reset_index(drop=True)
print("\n===== Summary Results =====")
display(results_df)

# Save results table
results_df.to_csv("outputs/model_comparison_results.csv", index=False)

# ---------- 10) Visual comparison charts ----------
# R2 line
plt.figure()
plt.plot(results_df["Model"], results_df["R2"], marker="o")
plt.title("Model Comparison - R2 Score")
plt.xlabel("Model")
plt.ylabel("R2 Score")
plt.grid(True)
plt.tight_layout()
r2_path = "outputs/r2_comparison.png"
plt.savefig(r2_path)
plt.show()

# MSE bar chart
plt.figure()
sns.barplot(x="Model", y="MSE", data=results_df)
plt.title("Model Comparison - MSE")
plt.ylabel("MSE")
plt.tight_layout()
mse_path = "outputs/mse_comparison.png"
plt.savefig(mse_path)
plt.show()

# Feature importance (from Random Forest)
if "Random Forest" in models:
    rf = models["Random Forest"]
    try:
        importances = rf.feature_importances_
        feat_labels = X.columns
        fi_df = pd.DataFrame({"feature": feat_labels, "importance": importances}).sort_values("importance", ascending=False)
        # Pie chart
        plt.figure(figsize=(6,6))
        plt.pie(fi_df["importance"], labels=fi_df["feature"], autopct="%1.1f%%", startangle=140)
        plt.title("Feature Importance (Random Forest)")
        plt.tight_layout()
        pie_path = "outputs/feature_importance_pie.png"
        plt.savefig(pie_path)
        plt.show()
        # Bar chart for better readability
        plt.figure()
        sns.barplot(x="importance", y="feature", data=fi_df)
        plt.title("Feature Importance (Random Forest)")
        plt.tight_layout()
        fi_bar_path = "outputs/feature_importance_bar.png"
        plt.savefig(fi_bar_path)
        plt.show()
    except Exception as e:
        print("Could not compute feature importances:", e)

# Actual vs Predicted scatter for best model
best_model_name = results_df.loc[0, "Model"]
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

plt.figure()
plt.scatter(y_test, y_pred_best, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # ideal line
plt.xlabel("Actual Strength")
plt.ylabel("Predicted Strength")
plt.title(f"Actual vs Predicted - {best_model_name}")
plt.tight_layout()
avp_path = f"outputs/actual_vs_predicted_{best_model_name.replace(' ', '_')}.png"
plt.savefig(avp_path)
plt.show()

# Residuals histogram
residuals = y_test - y_pred_best
plt.figure()
sns.histplot(residuals, kde=True)
plt.title(f"Residuals Distribution - {best_model_name}")
plt.xlabel("Residual (Actual - Predicted)")
plt.tight_layout()
resid_path = f"outputs/residuals_{best_model_name.replace(' ', '_')}.png"
plt.savefig(resid_path)
plt.show()

# ---------- 11) Print/save final notes and file list ----------
print("\nAll outputs saved in the 'outputs' folder.")
print("Files generated:")
for root, dirs, files in os.walk("outputs"):
    for f in files:
        print(" -", os.path.join(root, f))

# Provide download links in Colab environment (optional)
try:
    from google.colab import files as colab_files
    # compress outputs to zip for easy download
    import shutil
    shutil.make_archive("concrete_strength_outputs", 'zip', "outputs")
    print("\nCreated 'concrete_strength_outputs.zip' for download.")
    # colab_files.download("concrete_strength_outputs.zip")  # Uncomment to auto-download
except Exception as e:
    print("Could not create zip for download automatically:", e)

# ---------- 12) OPTIONAL: Print code summary for report (small snippet) ----------
print("\nCode execution complete. Summary:")
print(f" - Best model: {best_model_name}")
print(f" - R2 (test) for best model: {results_df.loc[0,'R2']:.4f}")
print(f" - MSE (test) for best model: {results_df.loc[0,'MSE']:.4f}")

# Save final trained best model path alias
final_model_path = results_df.loc[0, "Model_Path"]
print(f"\nBest model file: {final_model_path}")

# ============================================================== End of script