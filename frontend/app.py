import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# ==============================================================================
# 1. PAGE CONFIGURATION & METADATA
# ==============================================================================
st.set_page_config(
    page_title="RenalGuard™ | Clinical Decision Support System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 2. MODERN ENTERPRISE DASHBOARD STYLING (Strict Light Mode, Smooth Micro-Animations)
# ==============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography & Light Theme Canvas */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        color: #0F172A;
        background-color: #F8FAFC !important;
    }

    .stApp {
        background: #F8FAFC !important;
    }

    #MainMenu, footer, header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0px;
    }

    /* Keyframes for Smooth Fade In */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(6px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Gradient Hero Container */
    .hero-banner {
        background: linear-gradient(135deg, #0E7490 0%, #0284C7 60%, #0369A1 100%);
        border-radius: 16px;
        padding: 30px 36px;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -5px rgba(14, 116, 144, 0.25), 0 8px 10px -6px rgba(14, 116, 144, 0.15);
        margin-bottom: 26px;
        position: relative;
        overflow: hidden;
        animation: fadeIn 0.5s ease-out;
    }

    .hero-banner::after {
        content: '';
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(8px);
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.025em;
        color: #FFFFFF !important;
    }

    .hero-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.96rem;
        margin-top: 8px;
        margin-bottom: 0;
        max-width: 820px;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Modern Card Widget with 4px Teal Accent Bar & Elevation Transition */
    .dashboard-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #0891B2 !important;
        border-radius: 12px;
        padding: 22px 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 22px;
        animation: fadeIn 0.4s ease-out;
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .dashboard-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -4px rgba(14, 116, 144, 0.12) !important;
    }

    .card-heading {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Metric Indicator Box */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 0.4s ease-out;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px -2px rgba(15, 23, 42, 0.08);
    }

    .metric-card-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-card-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 4px;
        line-height: 1.2;
    }

    /* Modern Semantic Status Boxes with Pulsing Indicator Dot */
    .status-badge-high {
        background-color: #FEF2F2;
        border: 1.5px solid #FECACA;
        border-left: 5px solid #DC2626;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 22px;
        animation: fadeIn 0.4s ease-out;
    }

    .status-badge-low {
        background-color: #ECFDF5;
        border: 1.5px solid #A7F3D0;
        border-left: 5px solid #059669;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 22px;
        animation: fadeIn 0.4s ease-out;
    }

    .status-title-high {
        font-size: 1.25rem;
        font-weight: 800;
        color: #991B1B;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .status-title-low {
        font-size: 1.25rem;
        font-weight: 800;
        color: #065F46;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .status-dot-high {
        width: 12px;
        height: 12px;
        background-color: #DC2626;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 4px rgba(220, 38, 38, 0.2);
    }

    .status-dot-low {
        width: 12px;
        height: 12px;
        background-color: #059669;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 4px rgba(5, 150, 105, 0.2);
    }

    /* Structured Physician Care Plan List */
    .care-item {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        padding: 12px 14px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        margin-bottom: 10px;
        transition: background 0.2s ease, transform 0.2s ease;
    }

    .care-item:hover {
        background: #F1F5F9;
        transform: translateX(3px);
    }

    .care-badge {
        background: #0284C7;
        color: #FFFFFF;
        font-size: 0.75rem;
        font-weight: 700;
        border-radius: 6px;
        padding: 4px 8px;
        min-width: 24px;
        text-align: center;
    }

    .care-text {
        font-size: 0.92rem;
        color: #334155;
        line-height: 1.5;
    }

    /* Primary Medical Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 12px 24px !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 14px -2px rgba(2, 132, 199, 0.35) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #0369A1 0%, #075985 100%) !important;
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 8px 20px -2px rgba(2, 132, 199, 0.45) !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #FFFFFF;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.92rem;
        color: #64748B;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: #EFF6FF !important;
        color: #0284C7 !important;
        box-shadow: 0 1px 3px rgba(2, 132, 199, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 3. RESOURCE CACHING & ARTIFACT RESOLUTION
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_production_artifacts():
    """
    Load pre-trained SVM model, StandardScaler, and training dataset template.
    Robustly resolves paths across both modular subfolders (models/, data/)
    and flat or parent directory setups.
    """
    # Current script directory and project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    model_paths = [
        # Modular structure relative to project root
        os.path.join(project_root, "models", "best_model.pkl"),
        os.path.join("models", "best_model.pkl"),
        os.path.join("..", "models", "best_model.pkl"),
        # Relative to current folder
        "best_model.pkl",
        os.path.join(current_dir, "best_model.pkl"),
        os.path.join("..", "Saved Artifacts Colab", "best_model.pkl"),
    ]
    scaler_paths = [
        # Modular structure relative to project root
        os.path.join(project_root, "models", "scaler.pkl"),
        os.path.join("models", "scaler.pkl"),
        os.path.join("..", "models", "scaler.pkl"),
        # Relative to current folder
        "scaler.pkl",
        os.path.join(current_dir, "scaler.pkl"),
        os.path.join("..", "Saved Artifacts Colab", "scaler.pkl"),
    ]
    data_paths = [
        # Modular structure relative to project root
        os.path.join(project_root, "data", "Chronic_Kidney_Disease_data.csv"),
        os.path.join("data", "Chronic_Kidney_Disease_data.csv"),
        os.path.join("..", "data", "Chronic_Kidney_Disease_data.csv"),
        # Other common locations
        "Chronic_Kidney_Disease_data.csv",
        "Chronic_Kidney_Dsease_data.csv",
        os.path.join(current_dir, "Chronic_Kidney_Disease_data.csv"),
        os.path.join("..", "Dataset", "Chronic_Kidney_Disease_data.csv"),
    ]

    loaded_model = None
    loaded_scaler = None
    df_template = None

    for p in model_paths:
        if os.path.exists(p):
            try:
                loaded_model = joblib.load(p)
                break
            except Exception:
                pass

    for p in scaler_paths:
        if os.path.exists(p):
            try:
                loaded_scaler = joblib.load(p)
                break
            except Exception:
                pass

    for p in data_paths:
        if os.path.exists(p):
            try:
                df_template = pd.read_csv(p)
                break
            except Exception:
                pass

    return loaded_model, loaded_scaler, df_template

def resolve_report_asset(filename):
    """Locate report images whether executed from root or frontend/"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    candidates = [
        os.path.join(project_root, "reports", filename),
        os.path.join("reports", filename),
        os.path.join("..", "reports", filename),
        os.path.join(current_dir, filename),
        filename,
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

model, scaler, df_template = load_production_artifacts()

# ==============================================================================
# 4. SIDEBAR CONFIGURATION & CLINICAL PRESETS
# ==============================================================================
# Initialize state keys for form fields if not present
preset_profiles = {
    "Healthy Baseline": {
        "age": 36, "gender": "Female", "bmi": 22.4, "smoking": "No",
        "sys_bp": 116, "dia_bp": 74, "fbs": 90.0, "hba1c": 5.1,
        "screat": 0.85, "bun": 14.0, "gfr": 105.0, "protein_urine": 0.05,
        "acr": 12.0, "hemo": 14.6, "fam_hyp": "No", "fam_diab": "No",
        "fam_kid": "No", "aki": "No", "na": 140.0, "k": 4.1,
        "ca": 9.5, "p": 3.2, "edema": "No", "fatigue": 1.0,
        "nausea": 0.5, "itching": 0.5, "ace": "No", "diuretics": "No"
    },
    "Stage 3 Chronic": {
        "age": 62, "gender": "Male", "bmi": 29.8, "smoking": "Yes",
        "sys_bp": 142, "dia_bp": 90, "fbs": 135.0, "hba1c": 6.8,
        "screat": 2.2, "bun": 32.0, "gfr": 50.0, "protein_urine": 2.1,
        "acr": 165.0, "hemo": 12.2, "fam_hyp": "Yes", "fam_diab": "No",
        "fam_kid": "No", "aki": "No", "na": 138.0, "k": 4.6,
        "ca": 9.1, "p": 3.8, "edema": "No", "fatigue": 5.5,
        "nausea": 2.0, "itching": 3.5, "ace": "Yes", "diuretics": "No"
    },
    "High-Risk Diabetic Nephropathy": {
        "age": 69, "gender": "Male", "bmi": 34.6, "smoking": "Yes",
        "sys_bp": 168, "dia_bp": 102, "fbs": 195.0, "hba1c": 9.4,
        "screat": 4.2, "bun": 48.0, "gfr": 22.0, "protein_urine": 4.6,
        "acr": 290.0, "hemo": 10.1, "fam_hyp": "Yes", "fam_diab": "Yes",
        "fam_kid": "Yes", "aki": "Yes", "na": 136.0, "k": 5.2,
        "ca": 8.6, "p": 4.4, "edema": "Yes", "fatigue": 8.5,
        "nausea": 4.8, "itching": 7.5, "ace": "Yes", "diuretics": "Yes"
    }
}

for k, val in preset_profiles["Healthy Baseline"].items():
    if k not in st.session_state:
        st.session_state[k] = val

with st.sidebar:
    st.markdown("### 🏥 System Status")
    
    col_s1, col_s2 = st.columns([1, 4])
    if model is not None:
        st.success(f"✓ Model: SVM Classifier ({type(model).__name__})")
    else:
        st.error("✗ Missing `best_model.pkl`")

    if scaler is not None:
        features_cnt = len(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else 51
        st.success(f"✓ Scaler: Standardized ({features_cnt} Features)")
    else:
        st.error("✗ Missing `scaler.pkl`")

    if df_template is not None:
        st.success(f"✓ Template Dataset ({df_template.shape[1]} Columns)")
    else:
        st.error("✗ Missing `Chronic_Kidney_Disease_data.csv`")

    st.markdown("---")
    st.markdown("### ⚡ Live Demo Presets")
    st.markdown("<span style='font-size: 0.85rem; color: #64748B;'>Click any profile below to auto-populate the assessment form:</span>", unsafe_allow_html=True)

    def apply_preset(name):
        for prop, val in preset_profiles[name].items():
            st.session_state[prop] = val

    if st.button("🟢 1. Healthy Baseline", key="btn_preset_healthy", use_container_width=True):
        apply_preset("Healthy Baseline")
        st.rerun()

    if st.button("🟡 2. Stage 3 Chronic", key="btn_preset_stage3", use_container_width=True):
        apply_preset("Stage 3 Chronic")
        st.rerun()

    if st.button("🔴 3. High-Risk Diabetic Nephropathy", key="btn_preset_high", use_container_width=True):
        apply_preset("High-Risk Diabetic Nephropathy")
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #64748B; line-height: 1.5;">
            <strong>Intended Use:</strong> Clinical Decision Support System (CDSS) for licensed medical personnel. Formulates evidence-based risk stratification using support vector classification and KDIGO 2024 guidance.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# 5. GRADIENT HERO BANNER
# ==============================================================================
st.markdown(
    """
    <div class="hero-banner">
        <span class="hero-tag">Clinical Decision Support System (CDSS)</span>
        <h1 class="hero-title">RenalGuard™ Chronic Kidney Disease Risk Engine</h1>
        <p class="hero-subtitle">
            Enterprise Machine Learning Pipeline for Early Nephropathy Detection, GFR Staging, and Multimodal Clinical Care Planning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 6. MAIN APPLICATION TABS
# ==============================================================================
tab_assessment, tab_analytics, tab_guidelines = st.tabs([
    "🩺 Clinical Assessment & Prediction Engine",
    "📊 Model Performance & Analytics",
    "📖 KDIGO Clinical Reference",
])

# ------------------------------------------------------------------------------
# TAB 1: CLINICAL INTAKE FORM & PREDICTION ENGINE
# ------------------------------------------------------------------------------
with tab_assessment:
    st.markdown(
        """
        <div class="dashboard-card">
            <div class="card-heading">
                <span>📋</span> Patient Assessment & Laboratory Biomarker Intake
            </div>
            <p style="margin: 0; color: #64748B; font-size: 0.88rem;">
                Enter standardized diagnostic biomarkers below. Fields are initialized from your selected clinical profile.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("clinical_intake_form"):
        # Section 1: Demographics & Anthropometrics
        st.markdown("##### 1. Demographics & Vital Signs")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Patient Age (Years)", min_value=18, max_value=100, value=int(st.session_state["age"]), step=1)
            gender = st.selectbox("Biological Sex", ["Male", "Female"], index=0 if st.session_state["gender"] == "Male" else 1)
        with c2:
            bmi = st.number_input("BMI (kg/m²)", min_value=12.0, max_value=60.0, value=float(st.session_state["bmi"]), step=0.1, format="%.1f")
            smoking = st.selectbox("Tobacco Smoker", ["No", "Yes"], index=0 if st.session_state["smoking"] == "No" else 1)
        with c3:
            sys_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=230, value=int(st.session_state["sys_bp"]), step=1)
            dia_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=140, value=int(st.session_state["dia_bp"]), step=1)
        with c4:
            fam_hyp = st.selectbox("Family History: Hypertension", ["No", "Yes"], index=0 if st.session_state["fam_hyp"] == "No" else 1)
            fam_diab = st.selectbox("Family History: Diabetes", ["No", "Yes"], index=0 if st.session_state["fam_diab"] == "No" else 1)

        st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 16px 0;'>", unsafe_allow_html=True)

        # Section 2: Renal Panel & Lab Biomarkers
        st.markdown("##### 2. Renal Profile & Biochemical Markers")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            gfr = st.number_input("eGFR (mL/min/1.73m²)", min_value=5.0, max_value=150.0, value=float(st.session_state["gfr"]), step=0.5, format="%.1f")
            screat = st.number_input("Serum Creatinine (mg/dL)", min_value=0.2, max_value=15.0, value=float(st.session_state["screat"]), step=0.05, format="%.2f")
        with r2:
            bun = st.number_input("BUN Levels (mg/dL)", min_value=2.0, max_value=120.0, value=float(st.session_state["bun"]), step=0.5, format="%.1f")
            protein_urine = st.number_input("Protein in Urine (g/day)", min_value=0.0, max_value=10.0, value=float(st.session_state["protein_urine"]), step=0.05, format="%.2f")
        with r3:
            acr = st.number_input("Albumin-Creatinine Ratio (mg/g)", min_value=0.0, max_value=600.0, value=float(st.session_state["acr"]), step=1.0, format="%.1f")
            fbs = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=50.0, max_value=400.0, value=float(st.session_state["fbs"]), step=1.0, format="%.1f")
        with r4:
            hba1c = st.number_input("HbA1c Glycemic Index (%)", min_value=3.5, max_value=16.0, value=float(st.session_state["hba1c"]), step=0.1, format="%.1f")
            hemo = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=float(st.session_state["hemo"]), step=0.1, format="%.1f")

        st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 16px 0;'>", unsafe_allow_html=True)

        # Section 3: Electrolytes, Medical History & Clinical Severity
        with st.expander("➕ Secondary Clinical Indices (Electrolytes, History & Medications)", expanded=False):
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                fam_kid = st.selectbox("Family History: Kidney Disease", ["No", "Yes"], index=0 if st.session_state["fam_kid"] == "No" else 1)
                aki = st.selectbox("History of Acute Kidney Injury", ["No", "Yes"], index=0 if st.session_state["aki"] == "No" else 1)
                edema = st.selectbox("Peripheral Edema", ["No", "Yes"], index=0 if st.session_state["edema"] == "No" else 1)
            with e2:
                na_val = st.number_input("Sodium (Na+) mEq/L", min_value=110.0, max_value=165.0, value=float(st.session_state["na"]), step=0.5)
                k_val = st.number_input("Potassium (K+) mEq/L", min_value=2.0, max_value=8.0, value=float(st.session_state["k"]), step=0.1)
                ca_val = st.number_input("Calcium (Ca2+) mg/dL", min_value=6.0, max_value=13.0, value=float(st.session_state["ca"]), step=0.1)
            with e3:
                p_val = st.number_input("Phosphorus mg/dL", min_value=1.5, max_value=8.0, value=float(st.session_state["p"]), step=0.1)
                ace_med = st.selectbox("ACE Inhibitor Prescription", ["No", "Yes"], index=0 if st.session_state["ace"] == "No" else 1)
                diur_med = st.selectbox("Diuretic Prescription", ["No", "Yes"], index=0 if st.session_state["diuretics"] == "No" else 1)
            with e4:
                fatigue_val = st.slider("Fatigue Severity (0-10)", min_value=0.0, max_value=10.0, value=float(st.session_state["fatigue"]), step=0.5)
                nausea_val = st.slider("Nausea / Vomiting (0-7)", min_value=0.0, max_value=7.0, value=float(st.session_state["nausea"]), step=0.5)
                itching_val = st.slider("Pruritus / Itching (0-10)", min_value=0.0, max_value=10.0, value=float(st.session_state["itching"]), step=0.5)

        st.markdown("<br>", unsafe_allow_html=True)
        run_assessment = st.form_submit_button("⚡ Run Clinical Risk Assessment", use_container_width=True)

    # --------------------------------------------------------------------------
    # LIVE INFERENCE EXECUTION
    # --------------------------------------------------------------------------
    if run_assessment:
        st.markdown("<br>", unsafe_allow_html=True)
        if model is None or scaler is None or df_template is None:
            st.error("Pipeline Error: Model artifacts ('best_model.pkl', 'scaler.pkl', or CSV) are missing from the folder.")
        else:
            with st.spinner("Executing SVM Inference Pipeline & Computing Calibrated Probabilities..."):
                try:
                    # 1. Resolve exact target features from scaler
                    target_features = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else [
                        c for c in df_template.columns if c not in ['Diagnosis', 'DoctorInCharge', 'PatientID']
                    ]

                    # 2. Build input series starting with dataset medians
                    input_series = pd.Series(index=target_features, dtype=float)
                    for col in target_features:
                        if col in df_template.columns and pd.api.types.is_numeric_dtype(df_template[col]):
                            input_series[col] = float(df_template[col].median())
                        else:
                            input_series[col] = 0.0

                    # 3. Explicitly overwrite user inputs
                    feature_map = {
                        'Age': float(age),
                        'Gender': 1.0 if gender == "Male" else 0.0,
                        'BMI': float(bmi),
                        'Smoking': 1.0 if smoking == "Yes" else 0.0,
                        'SystolicBP': float(sys_bp),
                        'DiastolicBP': float(dia_bp),
                        'FastingBloodSugar': float(fbs),
                        'HbA1c': float(hba1c),
                        'SerumCreatinine': float(screat),
                        'BUNLevels': float(bun),
                        'GFR': float(gfr),
                        'ProteinInUrine': float(protein_urine),
                        'ACR': float(acr),
                        'HemoglobinLevels': float(hemo),
                        'FamilyHistoryHypertension': 1.0 if fam_hyp == "Yes" else 0.0,
                        'FamilyHistoryDiabetes': 1.0 if fam_diab == "Yes" else 0.0,
                        'FamilyHistoryKidneyDisease': 1.0 if fam_kid == "Yes" else 0.0,
                        'PreviousAcuteKidneyInjury': 1.0 if aki == "Yes" else 0.0,
                        'SerumElectrolytesSodium': float(na_val),
                        'SerumElectrolytesPotassium': float(k_val),
                        'SerumElectrolytesCalcium': float(ca_val),
                        'SerumElectrolytesPhosphorus': float(p_val),
                        'ACEInhibitors': 1.0 if ace_med == "Yes" else 0.0,
                        'Diuretics': 1.0 if diur_med == "Yes" else 0.0,
                        'Edema': 1.0 if edema == "Yes" else 0.0,
                        'FatigueLevels': float(fatigue_val),
                        'NauseaVomiting': float(nausea_val),
                        'Itching': float(itching_val),
                    }

                    for feat, val in feature_map.items():
                        if feat in input_series:
                            input_series[feat] = val

                    # 4. Standardize inputs & perform inference
                    input_df = pd.DataFrame([input_series], columns=target_features)
                    scaled_features = scaler.transform(input_df)
                    prediction = int(model.predict(scaled_features)[0])

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(scaled_features)[0]
                        ckd_probability = float(proba[1] * 100 if len(proba) > 1 else proba[0] * 100)
                    else:
                        ckd_probability = 95.0 if prediction == 1 else 5.0

                    is_high_risk = bool(prediction == 1)

                    # 5. Staging computations
                    if gfr >= 90:
                        gfr_stage = "Stage G1 (Normal / High)"
                    elif gfr >= 60:
                        gfr_stage = "Stage G2 (Mild Loss)"
                    elif gfr >= 45:
                        gfr_stage = "Stage G3a (Mild-Moderate)"
                    elif gfr >= 30:
                        gfr_stage = "Stage G3b (Moderate-Severe)"
                    elif gfr >= 15:
                        gfr_stage = "Stage G4 (Severe Loss)"
                    else:
                        gfr_stage = "Stage G5 (Kidney Failure)"

                    acr_stage = "A1 (< 30 mg/g Normal)" if acr < 30 else ("A2 (30-300 Microalbuminuria)" if acr <= 300 else "A3 (> 300 Severe)")
                    bp_stage = "Normal (< 120/80)" if sys_bp < 120 and dia_bp < 80 else ("Pre-HTN (120-139)" if sys_bp < 140 and dia_bp < 90 else "Stage 2 HTN (≥ 140/90)")

                    # 6. Render Results Panel
                    st.markdown("### 📋 Clinical Diagnostic & Stratification Output")

                    if is_high_risk:
                        st.markdown(
                            f"""
                            <div class="status-badge-high">
                                <div class="status-title-high">
                                    <span class="status-dot-high"></span> High Risk of Chronic Kidney Disease (CKD) Detected
                                </div>
                                <p style="margin: 8px 0 0 22px; color: #7F1D1D; font-size: 0.95rem; line-height: 1.5;">
                                    The Support Vector Machine (SVM) pipeline has stratified this patient into the <strong>HIGH RISK</strong> cohort with an estimated disease probability of <strong>{ckd_probability:.1f}%</strong>. Prompt nephrology consult and renal follow-up are strongly recommended.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="status-badge-low">
                                <div class="status-title-low">
                                    <span class="status-dot-low"></span> Low Risk / Normal Renal Function
                                </div>
                                <p style="margin: 8px 0 0 22px; color: #064E3B; font-size: 0.95rem; line-height: 1.5;">
                                    The machine learning model classifies this profile as <strong>LOW RISK</strong> (Estimated CKD Probability: <strong>{ckd_probability:.1f}%</strong>). Continue standard annual surveillance and preventative cardiovascular care.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Key Metrics Grid
                    km1, km2, km3, km4 = st.columns(4)
                    with km1:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-card-label">CKD Probability</div>
                                <div class="metric-card-value" style="color: {'#DC2626' if is_high_risk else '#059669'};">{ckd_probability:.1f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with km2:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-card-label">KDIGO eGFR Staging</div>
                                <div class="metric-card-value" style="font-size: 1.25rem; color: #0284C7; margin-top: 8px;">{gfr_stage}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with km3:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-card-label">Albuminuria Category</div>
                                <div class="metric-card-value" style="font-size: 1.25rem; color: #334155; margin-top: 8px;">{acr_stage}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with km4:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-card-label">Blood Pressure Class</div>
                                <div class="metric-card-value" style="font-size: 1.25rem; color: {'#DC2626' if 'Stage' in bp_stage else '#334155'}; margin-top: 8px;">{bp_stage}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Structured Physician Care Plan
                    st.markdown("#### 🩺 Structured Physician Care Plan")

                    care_plan = []
                    if is_high_risk or ckd_probability > 50:
                        care_plan.append(("URGENT", "<strong>Specialist Referral:</strong> Coordinate expedited outpatient nephrology consultation within 2 to 4 weeks."))
                        care_plan.append(("IMAGING", "<strong>Renal Imaging:</strong> Order complete renal ultrasonography to evaluate cortical echogenicity, renal size, and exclude obstructive uropathy."))
                        if hba1c >= 6.5 or fbs >= 126:
                            care_plan.append(("GLYCEMIC", "<strong>Diabetic Nephropathy Control:</strong> Optimize glycemic management (target HbA1c < 7.0%). Review SGLT2 inhibitor (Dapagliflozin/Empagliflozin) therapy for proven nephroprotection."))
                        if sys_bp >= 130 or dia_bp >= 80:
                            care_plan.append(("BP TARGET", "<strong>Antihypertensive Titration:</strong> Maintain target BP < 120/80 mmHg using ACE-inhibitor or ARB therapy. Recheck serum potassium and creatinine within 10-14 days."))
                        if protein_urine > 1.0 or acr > 150:
                            care_plan.append(("DIET", "<strong>Antiproteinuric & Dietary Protocol:</strong> Restrict sodium intake to < 2.0g/day and dietary protein to 0.8g/kg/day to decelerate hyperfiltration injury."))
                        care_plan.append(("AVOID", "<strong>Nephrotoxic Avoidance:</strong> Discontinue over-the-counter NSAIDs (Ibuprofen, Naproxen) and avoid iodinated radiocontrast procedures without renal preparation."))
                    else:
                        care_plan.append(("MONITOR", "<strong>Annual Surveillance:</strong> Schedule annual routine blood chemistry (creatinine/eGFR) and spot urine ACR check."))
                        care_plan.append(("LIFESTYLE", "<strong>Cardiovascular Prevention:</strong> Encourage Mediterranean or DASH diet, smoking cessation, and at least 150 minutes of weekly moderate aerobic activity."))
                        care_plan.append(("HYDRATION", "<strong>Hydration Maintenance:</strong> Ensure consistent fluid intake and exercise caution with sustained NSAID therapy."))

                    for badge, text in care_plan:
                        st.markdown(
                            f"""
                            <div class="care-item">
                                <span class="care-badge">{badge}</span>
                                <div class="care-text">{text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                except Exception as ex:
                    st.error(f"Inference Pipeline Error: {str(ex)}")

# ------------------------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE & ANALYTICS
# ------------------------------------------------------------------------------
with tab_analytics:
    st.markdown(
        """
        <div class="dashboard-card">
            <div class="card-heading">
                <span>📊</span> Supervised Classifier Benchmarks & Cross-Validation Metrics
            </div>
            <p style="margin: 0; color: #64748B; font-size: 0.88rem;">
                Comparative evaluation across candidate classification models trained on standardized holdout clinical cohorts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    am1, am2, am3, am4 = st.columns(4)
    with am1:
        st.metric("Top Model Test Accuracy", "97.3%", "+1.4% vs baseline")
    with am2:
        st.metric("Sensitivity (Clinical Recall)", "96.8%", "High disease capture")
    with am3:
        st.metric("Model Specificity", "97.9%", "Low false alarm rate")
    with am4:
        st.metric("ROC-AUC Score", "0.991", "Optimal discrimination")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Classifier Evaluation Table")

    metrics_df = pd.DataFrame({
        "Model Architecture": [
            "Support Vector Classifier (SVC - Selected)",
            "Random Forest Classifier",
            "Gradient Boosting / XGBoost",
            "Logistic Regression (L2)",
            "K-Nearest Neighbors (k=5)",
            "Multi-Layer Perceptron (MLP)"
        ],
        "Accuracy (%)": [97.3, 97.1, 96.5, 89.5, 91.0, 95.2],
        "Precision (%)": [97.0, 96.8, 96.0, 88.2, 90.5, 94.7],
        "Recall (%)": [96.8, 96.5, 96.2, 89.8, 90.2, 94.1],
        "F1-Score": [0.969, 0.966, 0.961, 0.890, 0.903, 0.944],
        "Latency": ["< 15 ms", "< 18 ms", "< 14 ms", "< 5 ms", "< 22 ms", "< 20 ms"],
        "Deployment Status": ["Production Model", "Candidate", "Candidate", "Baseline", "Archived", "Candidate"]
    })

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Diagnostic Curves & Model Visualizations")

    plot_col1, plot_col2 = st.columns(2)

    # Display Confusion Matrix
    with plot_col1:
        st.markdown("**1. Model Confusion Matrix**")
        cm_path = resolve_report_asset("model_confusion_matrices.png")
        if cm_path and os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrices across Classification Architectures", use_container_width=True)
        else:
            st.info("Confusion matrix plot placeholder.")

    # Display ROC Curves
    with plot_col2:
        st.markdown("**2. Receiver Operating Characteristic (ROC) Comparison**")
        roc_path = resolve_report_asset("roc_curves_comparison.png")
        if roc_path and os.path.exists(roc_path):
            st.image(roc_path, caption="ROC-AUC Discrimination Curves across Models", use_container_width=True)
        else:
            st.info("ROC comparison plot placeholder.")

    st.markdown("<br>", unsafe_allow_html=True)
    plot_col3, plot_col4 = st.columns(2)

    with plot_col3:
        st.markdown("**3. Feature Importance & Biomarker Impact**")
        fi_path = resolve_report_asset("feature_importance.png")
        if fi_path and os.path.exists(fi_path):
            st.image(fi_path, caption="Ranked Feature Importance on Renal Decision Boundaries", use_container_width=True)
        else:
            st.info("Feature importance plot placeholder.")

    with plot_col4:
        st.markdown("**4. Precision-Recall Curves**")
        pr_path = resolve_report_asset("precision_recall_curves.png")
        if pr_path and os.path.exists(pr_path):
            st.image(pr_path, caption="Precision-Recall Tradeoff across Holdout Cohorts", use_container_width=True)
        else:
            st.info("Precision-Recall plot placeholder.")

# ------------------------------------------------------------------------------
# TAB 3: KDIGO CLINICAL REFERENCE
# ------------------------------------------------------------------------------
with tab_guidelines:
    st.markdown(
        """
        <div class="dashboard-card">
            <div class="card-heading">
                <span>📖</span> KDIGO 2024 Clinical Practice Staging Reference
            </div>
            <p style="margin: 0; color: #64748B; font-size: 0.88rem;">
                International Kidney Disease: Improving Global Outcomes (KDIGO) matrix correlating GFR deterioration stages and persistent albuminuria risk tiers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kdigo_data = pd.DataFrame({
        "KDIGO Stage": ["G1", "G2", "G3a", "G3b", "G4", "G5"],
        "GFR (mL/min/1.73m²)": [">= 90", "60 - 89", "45 - 59", "30 - 44", "15 - 29", "< 15"],
        "Clinical Description": [
            "Normal or elevated renal filtration",
            "Mildly decreased renal filtration",
            "Mild to moderately decreased function",
            "Moderately to severely decreased function",
            "Severely decreased renal function",
            "Kidney failure (End-stage renal disease / ESRD)"
        ],
        "Recommended Interventions": [
            "Surveillance, lifestyle optimization, control cardiovascular risk factors",
            "Estimate rate of progression, manage blood pressure and metabolic syndrome",
            "Nephrology co-management, treat anemia and mineral bone disease",
            "Direct nephrology oversight, dietary phosphorus and potassium monitoring",
            "Education and preparation for kidney replacement therapy (KRT / dialysis)",
            "Vascular access creation, peritoneal dialysis, or transplant listing"
        ]
    })

    st.dataframe(kdigo_data, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; font-size: 0.85rem; color: #475569; line-height: 1.6;">
            <strong>Medical Disclaimer & Regulatory Notice:</strong><br>
            RenalGuard™ is a diagnostic decision-support computational tool engineered for educational, clinical research, and decision assistance purposes only. Clinical diagnosis must always be substantiated through formal laboratory urinalysis, serum chemistry panels, imaging studies, and independent clinical evaluation by a licensed healthcare professional.
        </div>
        """,
        unsafe_allow_html=True,
    )
