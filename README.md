# 💼 Employee Salary Prediction

A Machine Learning web app to predict whether an individual’s income is above or below \$50K/year using demographic and job-related inputs.

**Live Demo**: https://employee-salary-prediction-6yc6gnecxkdrr5vmdi46kz.streamlit.app/  
**Source Code**: https://github.com/lp-0406/Employee-Salary-Prediction



## 🧾 Dataset & Features

- **Dataset**: UCI Adult Census Income (~48,000 records)
- **Input variables** (11+): age, workclass, education-num, marital-status, occupation, relationship, race, gender, capital‑gain, capital‑loss, hours‑per‑week (and possibly native‑country)
- **Target**: Binary label (`<=50K` or `>50K`)



## 🧪 Data Preprocessing

- Handle missing entries (`?`) by treating them as unknown categories
- Use **LabelEncoder** for categorical features, saved as `label_encoders.pkl`
- Normalize or scale numeric features if needed
- Train-test split (e.g. 80–20)
- Optional: feature selection or balancing


## 🤖 Machine Learning Model

- **Algorithm**: Gradient Boosting Classifier
- Ensemble of decision trees built in stages using gradient descent optimization → robust and high-performing :contentReference[oaicite:20]{index=20}
- **Accuracy**: ~88–90%, benchmark level for this dataset :contentReference[oaicite:21]{index=21}
- **Confidence Scores**: Uses `predict_proba()` to show probability of predicted class


## ⚙️ Application Features

- **Single prediction** mode: Enter feature values via UI and receive predicted salary class + confidence
- **Batch prediction**: Upload CSV of multiple records and see predictions at once
- Interactive, real‑time feedback with clear UI using Streamlit



## 📦 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/lp-0406/Employee-Salary-Prediction.git
cd Employee-Salary-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
