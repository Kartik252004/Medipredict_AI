import pandas as pd, numpy as np, joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

print("=" * 55)
print("  MediPredict AI — Training All Models")
print("=" * 55)

# ── HEART ─────────────────────────────────────────────────
print("\n[1/4] Heart Disease...")
df = pd.read_csv("heart_5000.csv")
print("  Columns:", list(df.columns))
print("  Shape:", df.shape)

# Auto-detect target (last column or 'target')
target = "target" if "target" in df.columns else df.columns[-1]
X, y = df.drop(columns=[target]), df[target]

# Remap if target uses 1/2 instead of 0/1
if sorted(y.unique().tolist()) == [1, 2]:
    y = y.map({1: 1, 2: 0})
    print("  Remapped target: 1->1, 2->0")

print("  Target distribution:", dict(y.value_counts()))
HEART_COLS = list(X.columns)  # Save column order

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
m = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42)
m.fit(sc.fit_transform(Xtr), ytr)
acc = accuracy_score(yte, m.predict(sc.transform(Xte)))
print(f"  Accuracy: {acc*100:.1f}%")
print(classification_report(yte, m.predict(sc.transform(Xte))))
joblib.dump(m, "heart_model.pkl")
joblib.dump(sc, "heart_scaler.pkl")
joblib.dump(HEART_COLS, "heart_cols.pkl")  # Save column order!
print("  ✓ Saved heart_model.pkl, heart_scaler.pkl, heart_cols.pkl")

# ── DIABETES ──────────────────────────────────────────────
print("\n[2/4] Diabetes...")
df = pd.read_csv("diabetes_5000.csv")
print("  Columns:", list(df.columns))

target = "Outcome" if "Outcome" in df.columns else df.columns[-1]
X, y = df.drop(columns=[target]), df[target]

if sorted(y.unique().tolist()) == [1, 2]:
    y = y.map({1: 1, 2: 0})
    print("  Remapped target: 1->1, 2->0")

print("  Target distribution:", dict(y.value_counts()))
DIAB_COLS = list(X.columns)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
m = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42)
m.fit(sc.fit_transform(Xtr), ytr)
acc = accuracy_score(yte, m.predict(sc.transform(Xte)))
print(f"  Accuracy: {acc*100:.1f}%")
print(classification_report(yte, m.predict(sc.transform(Xte))))
joblib.dump(m, "diabetes_model.pkl")
joblib.dump(sc, "diabetes_scaler.pkl")
joblib.dump(DIAB_COLS, "diabetes_cols.pkl")
print("  ✓ Saved diabetes_model.pkl, diabetes_scaler.pkl, diabetes_cols.pkl")

# ── LIVER ─────────────────────────────────────────────────
print("\n[3/4] Liver Disease...")
df = pd.read_csv("liver_5000.csv")
print("  Columns:", list(df.columns))

target = "Dataset" if "Dataset" in df.columns else df.columns[-1]
X, y = df.drop(columns=[target]), df[target]

# CRITICAL FIX: Indian Liver dataset uses 1=patient, 2=no disease
if sorted(y.unique().tolist()) == [1, 2]:
    y = y.map({1: 1, 2: 0})
    print("  Remapped target: 1(patient)->1, 2(no disease)->0")
elif sorted(y.unique().tolist()) == [0, 1]:
    print("  Target already 0/1")

print("  Target distribution:", dict(y.value_counts()))

# Fix Gender column if it's M/F strings
if "Gender" in X.columns and X["Gender"].dtype == object:
    X["Gender"] = X["Gender"].map({"Male": 1, "Female": 0, "M": 1, "F": 0})
    print("  Encoded Gender: Male->1, Female->0")

LIVER_COLS = list(X.columns)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
m = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42)
m.fit(sc.fit_transform(Xtr), ytr)
acc = accuracy_score(yte, m.predict(sc.transform(Xte)))
print(f"  Accuracy: {acc*100:.1f}%")
print(classification_report(yte, m.predict(sc.transform(Xte))))
joblib.dump(m, "liver_model.pkl")
joblib.dump(sc, "liver_scaler.pkl")
joblib.dump(LIVER_COLS, "liver_cols.pkl")
print("  ✓ Saved liver_model.pkl, liver_scaler.pkl, liver_cols.pkl")

# ── LUNG (SMOTE) ──────────────────────────────────────────
print("\n[4/4] Lung Cancer (SMOTE)...")
df = pd.read_csv("lung_5000.csv")
print("  Columns:", list(df.columns))

target = "LUNG_CANCER" if "LUNG_CANCER" in df.columns else df.columns[-1]
X, y = df.drop(columns=[target]), df[target]

# Lung dataset often uses YES/NO strings
if y.dtype == object:
    y = y.map({"YES": 1, "NO": 0, "Yes": 1, "No": 0, "1": 1, "0": 0})
    print("  Encoded target: YES->1, NO->0")

if sorted(y.unique().tolist()) == [1, 2]:
    y = y.map({1: 1, 2: 0})
    print("  Remapped target: 1->1, 2->0")

# Lung features often use 1/2 encoding — remap all binary cols to 0/1
for col in X.columns:
    if sorted(X[col].unique().tolist()) == [1, 2]:
        X[col] = X[col].map({1: 0, 2: 1})
        print(f"  Remapped {col}: 1->0, 2->1")

# Encode GENDER if string
if "GENDER" in X.columns and X["GENDER"].dtype == object:
    X["GENDER"] = X["GENDER"].map({"M": 1, "F": 0, "Male": 1, "Female": 0})

print("  Target distribution:", dict(y.value_counts()))
LUNG_COLS = list(X.columns)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
Xtr_sm, ytr_sm = SMOTE(random_state=42).fit_resample(Xtr, ytr)
sc = StandardScaler()
m = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.85, random_state=42)
m.fit(sc.fit_transform(Xtr_sm), ytr_sm)
acc = accuracy_score(yte, m.predict(sc.transform(Xte)))
print(f"  Accuracy: {acc*100:.1f}%")
print(classification_report(yte, m.predict(sc.transform(Xte))))
joblib.dump(m, "lung_model.pkl")
joblib.dump(sc, "lung_scaler.pkl")
joblib.dump(LUNG_COLS, "lung_cols.pkl")
print("  ✓ Saved lung_model.pkl, lung_scaler.pkl, lung_cols.pkl")

print("\n" + "=" * 55)
print("  ALL MODELS TRAINED — run: streamlit run app.py")
print("=" * 55)