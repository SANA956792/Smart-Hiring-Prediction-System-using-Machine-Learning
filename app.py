

from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
try:
    with open('hiring.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# MUST match training column names & order exactly!
FEATURE_ORDER = [
    'Gender',
    'Age',
    'Education_Level',
    'Certifications_Count',
    'Previous_Companies',
    'Interview_Score',
    'Location',
    'Job_Role_Applied',
    'Expected_Salary',
    'Experience_Level',       # ← added here
    'Total_Skill_Score'
]

# Encoding mappings — must match what LabelEncoder / manual mapping used in notebook
GENDER_MAP = {
    'Female': 0,
    'Male':   1,
    'Other':  2
}

EDUCATION_MAP = {
    'Bachelors': 0,
    'Diploma':   1,
    'Masters':   2,
    'PhD':       3
}

LOCATION_MAP = {
    'Rural':      0,
    'Semi-Urban': 1,
    'Urban':      2
}

JOB_ROLE_MAP = {
    'Data Analyst':     0,
    'HR Executive':     1,
    'ML Engineer':      2,
    'Manager':          3,
    'Software Engineer': 4
}

EXPERIENCE_LEVEL_MAP = {
    'Junior': 0,
    'Mid':    1,
    'Senior': 2
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('result.html',
                              prediction_text="Model failed to load. Server error.",
                              probability=None)

    try:
        form = request.form

        # Get & map categorical fields
        gender_str     = form.get('Gender', '')
        education_str  = form.get('Education_Level', '')
        location_str   = form.get('Location', '')
        job_str        = form.get('Job_Role_Applied', '')
        exp_level_str  = form.get('Experience_Level', '')

        gender     = GENDER_MAP.get(gender_str, -1)
        education  = EDUCATION_MAP.get(education_str, -1)
        location   = LOCATION_MAP.get(location_str, -1)
        job_role   = JOB_ROLE_MAP.get(job_str, -1)
        exp_level  = EXPERIENCE_LEVEL_MAP.get(exp_level_str, -1)

        if -1 in [gender, education, location, job_role, exp_level]:
            return render_template('result.html',
                                  prediction_text="Please select all required fields correctly",
                                  probability=None)

        # Build feature dictionary (raw values — no scaling)
        features = {
            'Gender':               gender,
            'Age':                  int(form.get('Age', 0)),
            'Education_Level':      education,
            'Certifications_Count': int(form.get('Certifications_Count', 0)),
            'Previous_Companies':   int(form.get('Previous_Companies', 0)),
            'Interview_Score':      int(form.get('Interview_Score', 0)),
            'Location':             location,
            'Job_Role_Applied':     job_role,
            'Expected_Salary':      int(form.get('Expected_Salary', 0)),
            'Experience_Level':     exp_level,              # ← now included
            'Total_Skill_Score':    int(form.get('Total_Skill_Score', 0))
        }

        # Create DataFrame in training order
        df_input = pd.DataFrame([features])[FEATURE_ORDER]

        # Make prediction
        pred = model.predict(df_input)[0]
        prob = None
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(df_input)[0][1]

        result_text = "Hired ✓" if pred == 1 else "Not Selected"
        prob_text   = f"{prob:.1%}" if prob is not None else "N/A"

        return render_template('result.html',
                              prediction_text=result_text,
                              probability=prob_text)

    except Exception as e:
        return render_template('result.html',
                              prediction_text=f"Error during prediction: {str(e)}",
                              probability=None)


if __name__ == '__main__':
    app.run(debug=True, port=5000)