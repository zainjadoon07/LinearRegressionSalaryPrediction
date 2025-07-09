import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Step 1: Load the dataset
# -----------------------------
df = pd.read_csv('ai_job_dataset.csv')  # Replace with your CSV path

# -----------------------------
# Step 2: Define input features and target
# -----------------------------
response = df['salary_usd'].to_numpy()  # Target variable
predictor = df[['company_location', 'years_experience', 'job_title']].to_numpy()

# -----------------------------
# Step 3: Encode categorical features
# -----------------------------
le_location = LabelEncoder()
le_title = LabelEncoder()

predictor[:, 0] = le_location.fit_transform(predictor[:, 0])   # company_location
predictor[:, 2] = le_title.fit_transform(predictor[:, 2])      # job_title

# -----------------------------
# Step 4: Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    predictor, response, test_size=0.2, random_state=42
)

# -----------------------------
# Step 5: Train Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Step 6: Make Predictions
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# Step 7: Evaluate Model
# -----------------------------
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"✅ RMSE: ${rmse:,.2f}")
print(f"📈 R² Score: {r2:.4f}")

# -----------------------------
# Step 8: Visualization
# -----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.7, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Predicted vs. Actual Salaries')
plt.xlabel('Actual Salary (USD)')
plt.ylabel('Predicted Salary (USD)')
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Step 9: Custom Test Prediction
# -----------------------------
# Example input:
# Company Location: 'United States'
# Years Experience: 5
# Job Title: 'Machine Learning Engineer'

try:
    loc_encoded = le_location.transform(['United States'])[0]
    title_encoded = le_title.transform(['Machine Learning Engineer'])[0]
    custom_input = np.array([[loc_encoded, 5, title_encoded]])

    predicted_salary = model.predict(custom_input)
    print(f"🧪 Predicted salary: ${predicted_salary[0]:,.2f}")
except Exception as e:
    print("⚠️ Error in test prediction. Possibly due to unseen label.")
    print(e)
