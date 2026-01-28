# Smart Hiring Prediction System using Machine Learning

dataset link:
https://www.kaggle.com/datasets/roiiith/ai-fair-recrutment-dataset/data

## 🎯 Problem Statement

Build a machine learning classification model that predicts hiring decisions based on candidate attributes while evaluating and minimizing bias in AI-driven recruitment systems.

## Objectives

- Predict hiring outcomes using interview performance, skill scores, education, experience, and job role  
- Perform fairness analysis across demographic features (gender, education level, location)  
- Develop a transparent, data-driven, and bias-aware recruitment support tool  
- Provide statistical validation of model performance and fairness claims

## ✨ Key Features

- Real-time hiring prediction with probability scores  
- Interactive web interface built with Flask  
- Advanced feature engineering (Total Skill Score, Experience Ratio)  
- Comprehensive fairness & bias analysis  
- Strong model performance: **94.6% AUC** (Gradient Boosting)  
- Hypothesis testing & statistical validation

## 📊 Model Performance

| Metric      | Value   | Interpretation                          |
|-------------|---------|-----------------------------------------|
| AUC Score   | 0.946   | Excellent discriminative power          |
| Accuracy    | ~91%    | High overall correctness                |
| Precision   | 92%     | Low false positives                     |
| Recall      | 89%     | Good capture of true positives          |
| F1-Score    | 90.5%   | Balanced precision & recall             |

**Top 3 Features by Importance:**
1. Interview Score (~42%)
2. Total Skill Score (~28%)
3. Education Level (~15%)

## ⚖️ Fairness & Bias Analysis

- **Gender**: No significant bias (p-value = 0.12)  
- **Education Level**: Equitable treatment across levels (p > 0.05)  
- **Location**: Urban/rural hiring rates fair (p = 0.08)  
- **Hypothesis Testing**: Model significantly outperforms baseline (p < 0.001)

## 🛠 Tech Stack

| Category            | Technology                          |
|---------------------|-------------------------------------|
| Language            | Python 3.11                         |
| ML Framework        | Scikit-learn (GradientBoosting)     |
| Web Framework       | Flask                               |
| Data Processing     | Pandas, NumPy                       |
| Visualization       | Matplotlib, Seaborn                 |
| Notebook            | Jupyter                             |
| Version Control     | Git / GitHub                        |
## 📸 Project Screenshots

### 🔹 Home Page
![Home Page](https://github.com/SANA956792/Smart-Hiring-Prediction-System-using-Machine-Learning/blob/main/image/Screenshot%202026-01-28%20151046.png)

### 🔹 Result Page
![Result Page](https://github.com/SANA956792/Smart-Hiring-Prediction-System-using-Machine-Learning/blob/main/image/Screenshot%202026-01-28%20161329.png)

## 🚀 Quick Start (Local Setup)

```bash
# 1. Clone the repo
git clone https://github.com/SANA956792/Smart-Hiring-Prediction-System-using-Machine-Learning-in-the-box-below.git
cd Smart-Hiring-Prediction-System-using-Machine-Learning-in-the-box-below

# 2. Create & activate virtual environment
python -m venv venv
.\venv\Scripts\activate          # Windows
# or
source venv/bin/activate         # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
