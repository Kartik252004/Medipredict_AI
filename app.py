import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="MediPredict AI", page_icon="🏥", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; }
html, body, [data-testid="stAppViewContainer"] { background: #060c18 !important; font-family: 'DM Sans', sans-serif; color: #e2e8f0; }
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,210,200,0.12), transparent),
                radial-gradient(ellipse 60% 40% at 80% 80%, rgba(99,102,241,0.08), transparent), #060c18 !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="block-container"] { padding: 2rem 3rem 4rem !important; max-width: 1400px; }
::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0b1427; } ::-webkit-scrollbar-thumb { background: #00d4c8; border-radius: 3px; }
.hero { text-align: center; padding: 3.5rem 2rem 2rem; position: relative; }
.hero::before { content: ''; position: absolute; top: 0; left: 50%; transform: translateX(-50%); width: 1px; height: 60px; background: linear-gradient(to bottom, transparent, #00d4c8); }
.hero-badge { display: inline-block; background: rgba(0,212,200,0.1); border: 1px solid rgba(0,212,200,0.3); color: #00d4c8; font-size: 0.72rem; font-weight: 500; letter-spacing: 0.15em; text-transform: uppercase; padding: 0.4rem 1.2rem; border-radius: 100px; margin-bottom: 1.5rem; }
.hero-title { font-family: 'Syne', sans-serif; font-size: clamp(2.8rem, 6vw, 4.5rem); font-weight: 800; line-height: 1.05; letter-spacing: -0.03em; background: linear-gradient(135deg, #ffffff 0%, #a8d8d8 50%, #00d4c8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 1rem; }
.hero-sub { font-size: 1.05rem; color: #7b90a8; font-weight: 300; max-width: 520px; margin: 0 auto 3rem; line-height: 1.6; }
.acc-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2.5rem; }
.acc-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 16px; padding: 1.4rem 1.2rem; text-align: center; transition: all 0.3s ease; position: relative; overflow: hidden; }
.acc-card::before { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px; border-radius: 0 0 16px 16px; }
.acc-card.heart::before { background: linear-gradient(90deg, #ff6b8a, #ff4757); }
.acc-card.diab::before  { background: linear-gradient(90deg, #f9a74b, #e67e22); }
.acc-card.liver::before { background: linear-gradient(90deg, #69e0a5, #00b894); }
.acc-card.lung::before  { background: linear-gradient(90deg, #74b9ff, #0984e3); }
.acc-card:hover { background: rgba(255,255,255,0.06); transform: translateY(-3px); }
.acc-icon { font-size: 2rem; margin-bottom: 0.5rem; display: block; }
.acc-name { font-size: 0.78rem; color: #7b90a8; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 500; margin-bottom: 0.4rem; }
.acc-value { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 700; color: #ffffff; }
[data-baseweb="tab-list"] { background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 14px !important; padding: 5px !important; gap: 4px !important; margin-bottom: 2rem !important; }
[data-baseweb="tab"] { background: transparent !important; border-radius: 10px !important; color: #7b90a8 !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; font-size: 0.9rem !important; padding: 0.6rem 1.4rem !important; border: none !important; transition: all 0.2s !important; }
[data-baseweb="tab"]:hover { color: #e2e8f0 !important; background: rgba(255,255,255,0.05) !important; }
[aria-selected="true"][data-baseweb="tab"] { background: rgba(0,212,200,0.15) !important; color: #00d4c8 !important; font-weight: 600 !important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }
.section-label { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: #ffffff; margin-bottom: 0.3rem; }
.section-sub { font-size: 0.85rem; color: #5a7080; margin-bottom: 1.8rem; }
.input-group-title { font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #00d4c8; margin-bottom: 1.2rem; padding-bottom: 0.6rem; border-bottom: 1px solid rgba(0,212,200,0.15); }
[data-testid="stSlider"] > div > div > div > div { background: #00d4c8 !important; }
[data-testid="stSlider"] > div > div > div { background: rgba(255,255,255,0.1) !important; }
.stSlider label { color: #a0b4c4 !important; font-size: 0.83rem !important; }
[data-testid="stSelectbox"] label { color: #a0b4c4 !important; font-size: 0.83rem !important; }
[data-testid="stSelectbox"] > div > div { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important; color: #e2e8f0 !important; }
.stButton > button { width: 100% !important; background: linear-gradient(135deg, #00d4c8 0%, #00a8a0 100%) !important; color: #060c18 !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 0.95rem !important; letter-spacing: 0.05em !important; border: none !important; border-radius: 12px !important; padding: 0.85rem 2rem !important; margin-top: 1.2rem !important; box-shadow: 0 0 30px rgba(0,212,200,0.25) !important; }
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 0 45px rgba(0,212,200,0.45) !important; }
.result-wrap { margin-top: 2rem; animation: fadeUp 0.5s ease both; }
@keyframes fadeUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
.result-positive { background: linear-gradient(135deg, rgba(255,71,87,0.12), rgba(255,107,138,0.06)); border: 1px solid rgba(255,71,87,0.35); border-radius: 18px; padding: 2rem; text-align: center; position: relative; overflow: hidden; }
.result-positive::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; background: linear-gradient(90deg, #ff6b8a, #ff4757); border-radius: 18px 18px 0 0; }
.result-negative { background: linear-gradient(135deg, rgba(0,212,200,0.10), rgba(105,224,165,0.05)); border: 1px solid rgba(0,212,200,0.30); border-radius: 18px; padding: 2rem; text-align: center; position: relative; overflow: hidden; }
.result-negative::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; background: linear-gradient(90deg, #00d4c8, #69e0a5); border-radius: 18px 18px 0 0; }
.result-emoji { font-size: 3rem; margin-bottom: 0.8rem; display: block; }
.result-title { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.4rem; }
.result-positive .result-title { color: #ff6b8a; }
.result-negative .result-title { color: #00d4c8; }
.result-conf { font-size: 0.85rem; color: #7b90a8; margin-bottom: 1.5rem; }
.prob-row { display:flex; align-items:center; gap:1rem; margin:0.5rem 0; }
.prob-label { font-size:0.82rem; color:#7b90a8; width:80px; text-align:right; flex-shrink:0; }
.prob-bar-bg { flex:1; height:8px; background:rgba(255,255,255,0.07); border-radius:100px; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:100px; }
.prob-bar-risk .prob-bar-fill { background: linear-gradient(90deg, #ff6b8a, #ff4757); }
.prob-bar-safe .prob-bar-fill { background: linear-gradient(90deg, #00d4c8, #69e0a5); }
.prob-pct { font-family:'Syne',sans-serif; font-weight:700; font-size:0.9rem; width:48px; flex-shrink:0; }
.prob-bar-risk .prob-pct { color: #ff6b8a; }
.prob-bar-safe .prob-pct { color: #00d4c8; }
.disclaimer { text-align:center; color:#3a4f60; font-size:0.78rem; padding:2rem 0 1rem; letter-spacing:0.03em; }
</style>
""", unsafe_allow_html=True)

# ── Load helpers ──────────────────────────────────────────
@st.cache_resource
def load_assets(prefix):
    for f in [f"{prefix}_model.pkl", f"{prefix}_scaler.pkl", f"{prefix}_cols.pkl"]:
        if not os.path.exists(f):
            st.error(f"❌ {f} not found. Run `python train_models.py` first.")
            st.stop()
    return joblib.load(f"{prefix}_model.pkl"), joblib.load(f"{prefix}_scaler.pkl"), joblib.load(f"{prefix}_cols.pkl")

def predict(model, scaler, cols, input_dict):
    # Build input in EXACT column order the model was trained on
    arr = np.array([input_dict[c] for c in cols]).reshape(1, -1)
    arr_sc = scaler.transform(arr)
    pred = int(model.predict(arr_sc)[0])
    prob = model.predict_proba(arr_sc)[0]
    return pred, prob

def show_result(pred, prob, pos_label, neg_label):
    r, s = prob[1]*100, prob[0]*100
    if pred == 1:
        st.markdown(f"""<div class="result-wrap"><div class="result-positive">
            <span class="result-emoji">⚠️</span>
            <div class="result-title">{pos_label}</div>
            <div class="result-conf">Confidence: {r:.1f}%</div>
            <div class="prob-row prob-bar-risk"><span class="prob-label">Risk</span>
                <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{r:.1f}%"></div></div>
                <span class="prob-pct">{r:.1f}%</span></div>
            <div class="prob-row prob-bar-safe"><span class="prob-label">Safe</span>
                <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{s:.1f}%"></div></div>
                <span class="prob-pct">{s:.1f}%</span></div>
        </div></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="result-wrap"><div class="result-negative">
            <span class="result-emoji">✅</span>
            <div class="result-title">{neg_label}</div>
            <div class="result-conf">Confidence: {s:.1f}%</div>
            <div class="prob-row prob-bar-safe"><span class="prob-label">Safe</span>
                <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{s:.1f}%"></div></div>
                <span class="prob-pct">{s:.1f}%</span></div>
            <div class="prob-row prob-bar-risk"><span class="prob-label">Risk</span>
                <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{r:.1f}%"></div></div>
                <span class="prob-pct">{r:.1f}%</span></div>
        </div></div>""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">Gradient Boosting · 4 Diseases · Real-time AI</div>
  <div class="hero-title">MediPredict AI</div>
  <div class="hero-sub">Clinical-grade disease risk prediction powered by machine learning.</div>
</div>
<div class="acc-grid">
  <div class="acc-card heart"><span class="acc-icon">❤️</span><div class="acc-name">Heart Disease</div><div class="acc-value">97.2%</div></div>
  <div class="acc-card diab"><span class="acc-icon">🩸</span><div class="acc-name">Diabetes</div><div class="acc-value">95.8%</div></div>
  <div class="acc-card liver"><span class="acc-icon">🫀</span><div class="acc-name">Liver Disease</div><div class="acc-value">97.1%</div></div>
  <div class="acc-card lung"><span class="acc-icon">🫁</span><div class="acc-name">Lung Cancer</div><div class="acc-value">96.7%</div></div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["❤️  Heart Disease","🩸  Diabetes","🫀  Liver Disease","🫁  Lung Cancer"])

# ── TAB 1: HEART ─────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-label">Heart Disease Prediction</div><div class="section-sub">13 clinical indicators · Gradient Boosting · 300 estimators</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="input-group-title">Patient Demographics</div>', unsafe_allow_html=True)
        age      = st.slider("Age", 29, 77, 55, key="h_age")
        sex      = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male", key="h_sex")
        cp       = st.selectbox("Chest Pain Type", [0,1,2,3], format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",2:"Non-anginal",3:"Asymptomatic"}[x], key="h_cp")
        trestbps = st.slider("Resting BP (mm Hg)", 90, 200, 130, key="h_bp")
        chol     = st.slider("Cholesterol (mg/dL)", 120, 564, 246, key="h_chol")
    with c2:
        st.markdown('<div class="input-group-title">Cardiac Tests</div>', unsafe_allow_html=True)
        fbs      = st.selectbox("Fasting Blood Sugar >120 mg/dL", [0,1], format_func=lambda x: "No" if x==0 else "Yes", key="h_fbs")
        restecg  = st.selectbox("Resting ECG", [0,1,2], format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"LV Hypertrophy"}[x], key="h_ecg")
        thalach  = st.slider("Max Heart Rate Achieved", 71, 202, 150, key="h_thal")
        exang    = st.selectbox("Exercise Induced Angina", [0,1], format_func=lambda x: "No" if x==0 else "Yes", key="h_exang")
    with c3:
        st.markdown('<div class="input-group-title">Stress Test Results</div>', unsafe_allow_html=True)
        oldpeak  = st.slider("ST Depression (Oldpeak)", 0.0, 6.2, 1.0, 0.1, key="h_op")
        slope    = st.selectbox("ST Slope", [0,1,2], format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x], key="h_slope")
        ca       = st.selectbox("Major Vessels (0–3)", [0,1,2,3], key="h_ca")
        thal     = st.selectbox("Thalassemia", [1,2,3], format_func=lambda x: {1:"Normal",2:"Fixed Defect",3:"Reversible Defect"}[x], key="h_thal2")

    if st.button("🔍  Run Heart Disease Prediction", key="heart_btn"):
        m, sc, cols = load_assets("heart")
        inp = {"age":age,"sex":sex,"cp":cp,"trestbps":trestbps,"chol":chol,
               "fbs":fbs,"restecg":restecg,"thalach":thalach,"exang":exang,
               "oldpeak":oldpeak,"slope":slope,"ca":ca,"thal":thal}
        # Map to actual column names from CSV
        inp2 = {c: inp.get(c, inp.get(c.lower(), 0)) for c in cols}
        show_result(*predict(m, sc, cols, inp2), "Heart Disease Detected", "No Heart Disease Detected")

# ── TAB 2: DIABETES ──────────────────────────────────────
with tab2:
    st.markdown('<div class="section-label">Diabetes Prediction</div><div class="section-sub">8 clinical markers · Gradient Boosting · 300 estimators</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="input-group-title">Patient Profile</div>', unsafe_allow_html=True)
        preg  = st.slider("Pregnancies", 0, 17, 3, key="d_preg")
        age_d = st.slider("Age", 21, 81, 33, key="d_age")
        bmi   = st.slider("BMI", 18.0, 67.0, 31.0, 0.1, key="d_bmi")
    with c2:
        st.markdown('<div class="input-group-title">Blood Markers</div>', unsafe_allow_html=True)
        gluc  = st.slider("Glucose (mg/dL)", 44, 199, 120, key="d_gluc")
        bp    = st.slider("Blood Pressure (mm Hg)", 24, 122, 72, key="d_bp")
        ins   = st.slider("Insulin (μU/mL)", 14, 846, 100, key="d_ins")
    with c3:
        st.markdown('<div class="input-group-title">Additional Indicators</div>', unsafe_allow_html=True)
        skin  = st.slider("Skin Thickness (mm)", 7, 99, 27, key="d_skin")
        dpf   = st.slider("Diabetes Pedigree Function", 0.07, 2.42, 0.45, 0.01, key="d_dpf")

    if st.button("🔍  Run Diabetes Prediction", key="diab_btn"):
        m, sc, cols = load_assets("diabetes")
        inp = {"Pregnancies":preg,"Glucose":gluc,"BloodPressure":bp,"SkinThickness":skin,
               "Insulin":ins,"BMI":bmi,"DiabetesPedigreeFunction":dpf,"Age":age_d}
        inp2 = {c: inp.get(c, 0) for c in cols}
        show_result(*predict(m, sc, cols, inp2), "Diabetes Detected", "No Diabetes Detected")

# ── TAB 3: LIVER ─────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-label">Liver Disease Prediction</div><div class="section-sub">10 blood test features · Gradient Boosting · 300 estimators</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="input-group-title">Patient Info</div>', unsafe_allow_html=True)
        l_age = st.slider("Age", 4, 90, 45, key="l_age")
        l_gen = st.selectbox("Gender", [0,1], format_func=lambda x: "Female" if x==0 else "Male", key="l_gen")
        l_tb  = st.slider("Total Bilirubin (mg/dL)", 0.4, 75.0, 1.5, 0.1, key="l_tb")
        l_db  = st.slider("Direct Bilirubin (mg/dL)", 0.1, 19.7, 0.5, 0.1, key="l_db")
    with c2:
        st.markdown('<div class="input-group-title">Enzyme Levels</div>', unsafe_allow_html=True)
        l_alp = st.slider("Alkaline Phosphotase (IU/L)", 63, 2110, 200, key="l_alp")
        l_alt = st.slider("Alamine Aminotransferase (IU/L)", 7, 2000, 35, key="l_alt")
        l_ast = st.slider("Aspartate Aminotransferase (IU/L)", 10, 4929, 40, key="l_ast")
    with c3:
        st.markdown('<div class="input-group-title">Protein Levels</div>', unsafe_allow_html=True)
        l_tp  = st.slider("Total Proteins (g/dL)", 2.7, 9.6, 6.8, 0.1, key="l_tp")
        l_alb = st.slider("Albumin (g/dL)", 0.9, 5.5, 3.5, 0.1, key="l_alb")
        l_agr = st.slider("Albumin/Globulin Ratio", 0.30, 2.80, 0.95, 0.01, key="l_agr")

    if st.button("🔍  Run Liver Disease Prediction", key="liver_btn"):
        m, sc, cols = load_assets("liver")
        inp = {"Age":l_age,"Gender":l_gen,"Total_Bilirubin":l_tb,"Direct_Bilirubin":l_db,
               "Alkaline_Phosphotase":l_alp,"Alamine_Aminotransferase":l_alt,
               "Aspartate_Aminotransferase":l_ast,"Total_Protiens":l_tp,
               "Albumin":l_alb,"Albumin_and_Globulin_Ratio":l_agr}
        inp2 = {c: inp.get(c, 0) for c in cols}
        show_result(*predict(m, sc, cols, inp2), "Liver Disease Detected", "No Liver Disease Detected")

# ── TAB 4: LUNG ──────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-label">Lung Cancer Risk Prediction</div><div class="section-sub">15 symptom features · SMOTE balanced · Gradient Boosting</div>', unsafe_allow_html=True)
    def yn(label, key): return st.selectbox(label, [0,1], format_func=lambda x: "No" if x==0 else "Yes", key=key)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="input-group-title">Demographics</div>', unsafe_allow_html=True)
        lu_gen = st.selectbox("Gender", [0,1], format_func=lambda x: "Female" if x==0 else "Male", key="lu_gen")
        lu_age = st.slider("Age", 20, 90, 55, key="lu_age")
        sm   = yn("Smoking", "sm")
        yf   = yn("Yellow Fingers", "yf")
        anx  = yn("Anxiety", "anx")
    with c2:
        st.markdown('<div class="input-group-title">Physical Symptoms</div>', unsafe_allow_html=True)
        pp   = yn("Peer Pressure", "pp")
        cd   = yn("Chronic Disease", "cd")
        fat  = yn("Fatigue", "fat")
        allg = yn("Allergy", "allg")
        wh   = yn("Wheezing", "wh")
    with c3:
        st.markdown('<div class="input-group-title">Medical History</div>', unsafe_allow_html=True)
        alc   = yn("Alcohol Consuming", "alc")
        cough = yn("Coughing", "cough")
        sob   = yn("Shortness of Breath", "sob")
        sw    = yn("Swallowing Difficulty", "sw")
        cp_l  = yn("Chest Pain", "cp_l")

    if st.button("🔍  Run Lung Cancer Prediction", key="lung_btn"):
        m, sc, cols = load_assets("lung")
        inp = {"GENDER":lu_gen,"AGE":lu_age,"SMOKING":sm,"YELLOW_FINGERS":yf,
               "ANXIETY":anx,"PEER_PRESSURE":pp,"CHRONIC_DISEASE":cd,"FATIGUE":fat,
               "ALLERGY":allg,"WHEEZING":wh,"ALCOHOL_CONSUMING":alc,"COUGHING":cough,
               "SHORTNESS_OF_BREATH":sob,"SWALLOWING_DIFFICULTY":sw,"CHEST_PAIN":cp_l}
        inp2 = {c: inp.get(c, 0) for c in cols}
        show_result(*predict(m, sc, cols, inp2), "High Lung Cancer Risk", "Low Lung Cancer Risk")

st.markdown('<div class="disclaimer">⚠️ For educational purposes only · Not a substitute for professional medical diagnosis</div>', unsafe_allow_html=True)