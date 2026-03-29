import os
import streamlit as st
import joblib
import pandas as pd

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Internet Access Predictor",
    page_icon="🌐",
    layout="centered",
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background-color: #f0f4f8; }

    /* Card container */
    .card {
        background: white;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    }

    /* Section headers */
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

    /* Result box */
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

    /* Confidence bar label */
    .conf-label {
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 4px;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* Button */
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
    <span style='font-size:48px;'>🌐</span>
    <h1 style='margin:0; font-size:28px; color:#1e293b;'>Internet Access Predictor</h1>
    <p style='color:#64748b; font-size:15px; margin-top:4px;'>Uganda National Census — LightGBM Model</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 1: Demographics ─────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">👤 Demographics</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
with col2:
    sex = st.selectbox("Sex", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
with col3:
    rururb = st.selectbox("Area", [1, 2], format_func=lambda x: "Urban" if x == 1 else "Rural")

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
st.markdown('<div class="section-title">🏠 Household Assets</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
yn = lambda x: "No" if x == 0 else "Yes"
with col1:
    phone = st.selectbox("📱 Phone", [0, 1], format_func=yn)
with col2:
    computer = st.selectbox("💻 Computer", [0, 1], format_func=yn)
with col3:
    radio = st.selectbox("📻 Radio", [0, 1], format_func=yn)
with col4:
    television = st.selectbox("📺 Television", [0, 1], format_func=yn)

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 3: Economic Indicators ──────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">💰 Economic Indicators</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    meals = st.selectbox("Meals/day", [1, 2, 3, 4, 5])
with col2:
    bank_account = st.selectbox("🏦 Bank Account", [0, 1], format_func=yn)
with col3:
    livelihood = st.number_input("Livelihood Code", min_value=0, value=10)
with col4:
    energysource = st.number_input("Energy Source", min_value=0, value=1)

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict Button ───────────────────────────────────────────
predict = st.button("🔍 Predict Internet Access")

if predict:
    input_data = pd.DataFrame([[
        age, sex, grade, literacy, phone, computer,
        radio, television, meals, livelihood, rururb, energysource, bank_account
    ]], columns=features)

    proba = model.predict_proba(input_data)[0][1]
    pred  = int(proba >= threshold)

    st.markdown("<br>", unsafe_allow_html=True)

    if pred == 1:
        st.markdown(f"""
        <div class="result-yes">
            <div class="result-label">✅ Has Internet Access</div>
            <div class="result-sub">This individual is likely to have internet access.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-no">
            <div class="result-label">❌ No Internet Access</div>
            <div class="result-sub">This individual is unlikely to have internet access.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Confidence meter
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Confidence Score</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric(label="Model Confidence", value=f"{proba:.1%}")

    st.progress(float(proba))

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Has Internet:** `{proba:.1%}`")
    with col_b:
        st.markdown(f"**No Internet:** `{1-proba:.1%}`")

    st.markdown('</div>', unsafe_allow_html=True)
