import os
import streamlit as st
import joblib
import pandas as pd

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Internet Access Predictor Among Ugandan HouseHolds",
    page_icon="🌐",
    layout="centered",
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f0f4f8; }

    .card {
        background: white;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    }

    .section-title {
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #6b7280;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e5e7eb;
    }

    /* ── Input labels ── */
    .stNumberInput label,
    .stSelectbox label {
        color: #1e293b !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }

    /* ── Input boxes ── */
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput input {
        color: #1e293b !important;
        background-color: #f8fafc !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }

    /* ── Dropdown options ── */
    [data-baseweb="menu"] li {
        color: #1e293b !important;
    }

    /* ── Metric label (Model Confidence) ── */
    [data-testid="stMetricLabel"] p {
        color: #1e293b !important;
        font-size: 15px !important;
        font-weight: 700 !important;
    }

    /* ── Metric value (the percentage number) ── */
    [data-testid="stMetricValue"] {
        color: #3b82f6 !important;
        font-size: 36px !important;
        font-weight: 800 !important;
    }

    /* ── Has Internet / No Internet text rows ── */
    .conf-row {
        display: flex;
        justify-content: space-between;
        margin-top: 12px;
    }
    .conf-item {
        flex: 1;
        text-align: center;
        padding: 12px 16px;
        border-radius: 10px;
        font-size: 15px;
        font-weight: 700;
    }
    .conf-yes {
        background: #d1fae5;
        color: #065f46;
        margin-right: 8px;
    }
    .conf-no {
        background: #fee2e2;
        color: #991b1b;
        margin-left: 8px;
    }
    .conf-item span {
        display: block;
        font-size: 12px;
        font-weight: 500;
        margin-bottom: 4px;
        opacity: 0.75;
    }

    .result-yes {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border-left: 6px solid #10b981;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .result-no {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border-left: 6px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .result-label {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .result-sub {
        font-size: 15px;
        color: #374151;
    }

    #MainMenu, footer { visibility: hidden; }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white;
        font-size: 17px;
        font-weight: 700;
        padding: 14px;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    div.stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ── Load model ──────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
model     = joblib.load(os.path.join(BASE_DIR, "lightgbm_internet_model.pkl"))
features  = joblib.load(os.path.join(BASE_DIR, "features.pkl"))
threshold = joblib.load(os.path.join(BASE_DIR, "threshold.pkl"))

# ── Header ──────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 12px 0 4px 0;'>
    <h1 style='margin:0; font-size:28px; color:#1e293b;'>Internet Access Predictor Among Ugandan HouseHolds</h1>
    <p style='color:#64748b; font-size:15px; margin-top:4px;'>Uganda National and housing Census 2014 </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

yn = lambda x: "No" if x == 0 else "Yes"

# ── Section 1: Demographics ─────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Demographics</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
with col2:
    sex = st.selectbox("Sex", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
with col3:
    rururb = st.selectbox("Area type", [1, 2], format_func=lambda x: "Urban" if x == 1 else "Rural")

col4, col5 = st.columns(2)
with col4:
    grade = st.number_input("Education Grade", min_value=0, max_value=99, value=7)
with col5:
    literacy = st.selectbox("Literacy", [1, 2, 3, 4],
                   format_func=lambda x: {1: "Reads & Writes", 2: "Reads Only",
                                          3: "Cannot Read", 4: "N/A"}[x])

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 2: Household Assets ─────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Household Assets</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    phone = st.selectbox("Owns Phone", [0, 1], format_func=yn)
with col2:
    computer = st.selectbox("Owns Computer", [0, 1], format_func=yn)
with col3:
    radio = st.selectbox("Owns Radio", [0, 1], format_func=yn)
with col4:
    television = st.selectbox("Owns Television", [0, 1], format_func=yn)

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 3: Economic Indicators ──────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Economic Indicators</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    meals = st.selectbox("Meals per day", [1, 2, 3, 4, 5])
with col2:
    bank_account = st.selectbox("Has Bank Account", [0, 1], format_func=yn)
with col3:
    livelihood = st.number_input("Livelihood source code", min_value=0, value=10)
with col4:
    energysource = st.number_input("Energy source code", min_value=0, value=1)

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict Button ───────────────────────────────────────────
predict = st.button("Predict")

if predict:
    input_data = pd.DataFrame([[
        age, sex, grade, literacy, phone, computer,
        radio, television, meals, livelihood, rururb, energysource, bank_account
    ]], columns=features)

    proba = model.predict_proba(input_data)[0][1]
    pred  = int(proba >= threshold)

    st.markdown("<br>", unsafe_allow_html=True)

    if pred == 1:
        st.markdown("""
        <div class="result-yes">
            <div class="result-label">Has Internet Access</div>
            <div class="result-sub">This individual is likely to have internet access.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-no">
            <div class="result-label">No Internet Access</div>
            <div class="result-sub">This individual is unlikely to have internet access.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confidence card ──────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Confidence Score</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric(label="Model Confidence", value=f"{proba:.1%}")

    st.progress(float(proba))

    # Replaced st.markdown bold/code with styled HTML blocks
    st.markdown(f"""
    <div class="conf-row">
        <div class="conf-item conf-yes">
            <span>Has Internet</span>
            {proba:.1%}
        </div>
        <div class="conf-item conf-no">
            <span>No Internet</span>
            {1 - proba:.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
