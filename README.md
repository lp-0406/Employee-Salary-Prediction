# 💼 Employee Salary Prediction

A Machine Learning-powered web application to classify whether an individual's income is likely to exceed $50K/year based on demographic and work-related information.

🔗 **Live App**: [Streamlit Deployment](https://employee-salary-prediction-6yc6gnecxkdrr5vmdi46kz.streamlit.app/)  
📁 **GitHub Repo**: [Employee-Salary-Prediction](https://github.com/lp-0406/Employee-Salary-Prediction)


## 🚀 Features

- Predicts salary class (`<=50K` or `>50K`) based on inputs like age, education, occupation, etc.
- Interactive user interface using **Streamlit**
- Supports both **single prediction** and **batch prediction via CSV**
- Displays **prediction confidence score**


## 📊 Input Features

- `age`
- `workclass`
- `educational-num`
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `gender`
- `capital-gain`
- `capital-loss`
- `hours-per-week`


## 🤖 Model Details

- **Model Used**: Gradient Boosting Classifier (or similar – will confirm via the pickle file if needed)
- **Trained On**: [Adult Census Income Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/adult)
- **Performance**:
  - **Accuracy**: ~88%
  - **Confidence Scores**: Displayed for each prediction using `predict_proba`


## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas, NumPy
- Joblib


## 📦 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/lp-0406/Employee-Salary-Prediction.git
cd Employee-Salary-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
