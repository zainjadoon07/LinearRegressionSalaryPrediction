# 💼 AI Job Salary Prediction (Linear Regression Model)

This project uses **Linear Regression** to predict the salary (in USD) of AI-related jobs based on features like:

- 🌍 Company Location  
- 🧠 Job Title  
- 📈 Years of Experience

The dataset is cleaned and preprocessed, categorical data is encoded using `LabelEncoder`, and the model is evaluated using **Root Mean Squared Error (RMSE)** and **R² Score**. It also includes a visualization of predicted vs. actual salaries.

---

## 🧠 Project Overview

- ✅ Language: Python 3
- ✅ Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`
- ✅ Model: `LinearRegression` from `sklearn`
- ✅ Task: Salary Prediction (Regression)
- ✅ Dataset: `ai_job_dataset.csv` (custom AI jobs dataset)

---

## 📂 Files Structure

```
salary-predictor/
│
├── salary_prediction.py       # 🔢 Full Python script for training & testing
├── ai_job_dataset.csv         # 📊 Input dataset
├── README.md                  # 📘 Project overview and instructions
└── .gitignore                 # (Optional) Ignore Python cache files, etc.
```

---

## 🗂️ Dataset Description (`ai_job_dataset.csv`)

| Column Name        | Description                             |
|--------------------|-----------------------------------------|
| `job_title`        | Title of the AI/ML job                  |
| `company_location` | Country where the job is located        |
| `years_experience` | Number of years of experience required  |
| `salary_usd`       | Annual salary in US dollars             |
| `required_skills`  | (Optional) Skills listed in the job     |

> Note: `required_skills` is not used in this version but can be added later using NLP or keyword flags.

---

## ⚙️ How to Run

### 🔹 Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/salary-prediction-linear-regression.git
cd salary-prediction-linear-regression
```

### 🔹 Step 2: Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib
```

### 🔹 Step 3: Run the Project

```bash
python salary_prediction.py
```

You’ll see:

- ✅ RMSE: Root Mean Squared Error  
- 📈 R² Score  
- 🧪 Prediction on a test job input  
- 📊 A plot comparing actual vs. predicted salaries

---

## 📉 Model Evaluation

- **RMSE** gives the average error between predicted and actual salaries  
- **R² Score** tells how well the model explains the variance in salaries

> The lower the RMSE and the higher the R², the better.

---

## 📊 Sample Output

```
✅ RMSE: $32,541.12
📈 R² Score: 0.6543
🧪 Predicted salary: $126,540.42
```

> Visual: Red line in the plot shows perfect prediction. Dots closer to the line mean better accuracy.

---

## 🔮 Predict Custom Salary (Example)

Inside the script, you can change this section to predict your own salary:

```python
loc_encoded = le_location.transform(['United States'])[0]
title_encoded = le_title.transform(['Machine Learning Engineer'])[0]
custom_input = np.array([[loc_encoded, 5, title_encoded]])
predicted_salary = model.predict(custom_input)
```

---

## 🧹 Future Improvements

- ✅ Use OneHotEncoding for better regression accuracy  
- ✅ Include `required_skills` via NLP or skill flags  
- ✅ Add Ridge/Lasso Regression  
- ✅ Build a Web UI using Streamlit  
- ✅ Save/load models with `joblib` or `pickle`

---

## 🛡️ License

MIT License © 2025 — Zain Jadoon

---

## 💬 Contact

📧 For inquiries, collaboration, or feedback:  
**Zain Jadoon**  
[GitHub](https://github.com/zainjadoon07) | [LinkedIn](https://linkedin.com/in/zainjadoon07)
