# 🏥 MediPredict AI

A clinical-grade, multi-disease prediction web application built with Python and Streamlit, powered by Gradient Boosting machine learning models trained on 5,000 records per disease.

---

## 🔬 Diseases Covered

| Disease | Algorithm | Accuracy | Features |
|---|---|---|---|
| ❤️ Heart Disease | Gradient Boosting | 97.2% | 13 clinical indicators |
| 🩸 Diabetes | Gradient Boosting | 95.8% | 8 blood markers |
| 🫀 Liver Disease | Gradient Boosting | 97.1% | 10 enzyme & protein levels |
| 🫁 Lung Cancer | Gradient Boosting + SMOTE | 96.7% | 15 symptom features |

---

## ✨ Features

- 🎯 Real-time disease risk prediction with probability scores
- 📊 Visual probability bars showing risk vs safe confidence
- 🌙 Dark clinical UI with smooth animations
- 🧪 SMOTE applied on lung cancer model to fix class imbalance
- 📋 Grouped input panels organized by clinical category
- ⚡ Models cached on startup for fast inference

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit + Custom CSS |
| ML Models | Scikit-learn GradientBoostingClassifier |
| Balancing | imbalanced-learn SMOTE |
| Preprocessing | StandardScaler |
| Serialization | Joblib |
| Language | Python 3.11 |

---

## 📁 Project Structure
```
Project1/
│
├── app.py                    # Streamlit web application
├── train_models.py           # Model training script (run once)
├── requirements.txt          # Python dependencies
│
├── heart_5000.csv            # Heart disease dataset
├── diabetes_5000.csv         # Diabetes dataset
├── liver_5000.csv            # Liver disease dataset
├── lung_5000.csv             # Lung cancer dataset
│
├── heart_model.pkl           # Trained heart model
├── heart_scaler.pkl          # Heart feature scaler
├── heart_cols.pkl            # Heart column order
├── diabetes_model.pkl        # Trained diabetes model
├── diabetes_scaler.pkl       # Diabetes feature scaler
├── diabetes_cols.pkl         # Diabetes column order
├── liver_model.pkl           # Trained liver model
├── liver_scaler.pkl          # Liver feature scaler
├── liver_cols.pkl            # Liver column order
├── lung_model.pkl            # Trained lung cancer model
├── lung_scaler.pkl           # Lung feature scaler
├── lung_cols.pkl             # Lung column order
│
└── docs/
    ├── Heart_Disease_ML_Report.docx
    ├── Liver_Patient_ML_Report.docx
    ├── Lung_Cancer_ML_Report.docx
    ├── heart_disease_prediction.ipynb
    ├── liver_patient_prediction.ipynb
    ├── lung_cancer_prediction.ipynb
    └── diabetes_explained.ipynb
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- pip

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/medipredict-ai.git
cd medipredict-ai
```

### 2. Create a virtual environment
```bash
# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# Mac / Linux
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the models (run once)
```bash
python train_models.py
```
This generates all `.pkl` files from the CSV datasets.

### 5. Launch the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📊 Model Details

### Algorithm — Gradient Boosting Classifier
All four models use `sklearn.ensemble.GradientBoostingClassifier` with tuned hyperparameters:

| Parameter | Heart | Diabetes | Liver | Lung |
|---|---|---|---|---|
| n_estimators | 300 | 300 | 300 | 200 |
| max_depth | 5 | 5 | 5 | 4 |
| learning_rate | 0.1 | 0.1 | 0.1 | 0.1 |
| subsample | — | — | — | 0.85 |

### Class Imbalance — SMOTE
The lung cancer dataset has a **2.27:1 class imbalance**. SMOTE is applied exclusively on the training set to oversample the minority class, while the test set retains the real-world distribution for honest evaluation.

| Metric | Value |
|---|---|
| Test Accuracy | 96.7% |
| CV Score | 97.8% ± 0.26% |
| Train-Test Gap | 2.81% ✅ |

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.
It is not a substitute for professional medical advice, diagnosis, or treatment.
Always consult a qualified healthcare professional.

---

## 👤 Author

**Your Name**
[GitHub](https://github.com/Kartik252004) · [LinkedIn](https://www.linkedin.com/in/kartik-salunkhe-9a518226a/)
```

---

**Short GitHub repo description** (goes in the About box):
```
An AI-powered machine learning project designed to predict diseases including heart, diabetes, lung, and liver conditions. It leverages data-driven models to assist in early diagnosis, improve healthcare outcomes, and support preventive medical decision-making.
```

**GitHub topics to add:**
```
machine-learning  streamlit  python  scikit-learn  gradient-boosting  
healthcare-ai  disease-prediction  smote  data-science  clinical-ml
