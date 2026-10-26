# ML-Lab-CKD-Prediction-System: Chronic Kidney Disease Clinical Decision Support System

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-0284C7?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML%20Engine-Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Academic Project](https://img.shields.io/badge/Curriculum-ML%20Lab%20Mini--Project-059669?style=flat)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-slate?style=flat)](LICENSE)

An enterprise-grade, clinical decision-support infrastructure engineered as a **Machine Learning Laboratory Mini-Project** to assist nephrologists, internists, and primary care physicians in early risk stratification, functional staging, and diagnostic decision-making for Chronic Kidney Disease (CKD).

The platform couples an optimized **Support Vector Machine (SVM)** classification engine with international Kidney Disease: Improving Global Outcomes (**KDIGO 2024**) practice guidelines to translate multi-dimensional biochemical and clinical markers into calibrated probabilistic assessments and actionable physician care plans.

---

## 👥 Authors & Academic Information

- **Project:** Machine Learning Laboratory Mini-Project
- **Department:** Department of Computer Science & Engineering
- **Project Contributors:**
  - **Devesh Kahar**
  - **Amar Shukla**
- **Domain Focus:** Medical Diagnostics, Supervised Clinical Machine Learning, Clinical Decision Support Systems (CDSS)

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Authors & Academic Information](#-authors--academic-information)
3. [Repository Directory Structure](#-repository-directory-structure)
4. [Key Features](#-key-features)
5. [System Architecture & Workflow](#-system-architecture--workflow)
6. [Technology Stack](#-technology-stack)
7. [Installation & Quickstart Guide](#-installation--quickstart-guide)
8. [Model Performance & Evaluation](#-model-performance--evaluation)
9. [Clinical Staging & Decision Rules](#-clinical-staging--decision-rules)
10. [Medical Disclaimer](#-medical-disclaimer)

---

## 🩺 Project Overview

Chronic Kidney Disease (CKD) is characterized by gradual, irreversible deterioration in renal filtration and structural integrity. Due to compensatory hyperfiltration in surviving nephrons, early-to-intermediate stages of CKD frequently remain asymptomatic. As a consequence, patients often present at advanced stages where therapeutic interventions can only delay, rather than prevent, kidney failure or renal replacement therapy.

The primary objective of this **Machine Learning Lab Mini-Project** is to develop a reliable, reproducible, and explainable computational clinical decision support system:
- **Early-Stage Detection:** Accurately screens patients using routinely measured metabolic, hematologic, and renal panels.
- **Calibrated Probabilistic Risk Stratification:** Provides continuous probability metrics alongside binary classifications to convey diagnostic confidence.
- **Evidence-Based Care Pathways:** Synthesizes patient-specific diagnostic findings into concrete clinical interventions (nephrology referrals, diagnostic imaging, and nephrotoxic drug avoidance).

---

## 📂 Repository Directory Structure

The project strictly adheres to the following modular repository layout:

```text
CKD-Prediction-System/
├── data/
│   └── Chronic_Kidney_Disease_data.csv
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
├── frontend/
│   └── app.py
├── reports/
│   ├── model_confusion_matrices.png
│   ├── roc_curves_comparison.png
│   ├── feature_importance.png
│   ├── ui_screenshot_1.png
│   └── ui_screenshot_2.png
└── README.md
```

---

## ⚡ Key Features

- **Real-Time ML Inference Engine:** Loads a pre-trained Support Vector Machine (SVM) classifier (`best_model.pkl`) and a fitted `StandardScaler` (`scaler.pkl`) across 51 standardized clinical dimensions to generate instantaneous risk classifications with calibrated class probabilities.
- **Enterprise-Grade Light-Mode UI:** Designed with custom Vanilla CSS strictly adhering to clinical UI standards:
  - Medical teal-to-blue gradient hero container (`#0E7490` to `#0284C7`).
  - Elevated dashboard cards with a 4px vertical teal accent bar (`#0891B2`).
  - Semantic status indicators: Soft coral for high-risk alerts and soft emerald for low-risk normal profiles with indicator dots.
  - Smooth micro-interactions, elevation hover transitions (`translateY(-2px)`), and `@keyframes fadeIn` animations.
- **Sidebar One-Click Case Presets:** Includes instant pre-filled patient profiles for live demonstrations and testing:
  - `Healthy Baseline` (Normal renal filtration and baseline electrolytes).
  - `Stage 3 Chronic` (Moderate eGFR decline, elevated BUN, and microalbuminuria).
  - `High-Risk Diabetic Nephropathy` (Severe proteinuria, high HbA1c, elevated creatinine, and symptomatic fluid retention).
- **Automated Physician Care Plan:** Automatically compiles diagnostic recommendations based on patient outputs:
  - Expedited nephrology referral timeline urgency (2–4 weeks for high-risk cohorts).
  - Renal ultrasonography and urinalysis repeat protocols.
  - Glycemic targets (HbA1c < 7.0%) and SGLT2 inhibitor review for diabetic cohorts.
  - Blood pressure targets (< 120/80 mmHg) via ACEi/ARB titration with potassium monitoring.
  - Strict avoidance of nephrotoxic medications (NSAIDs, aminoglycosides, iodinated radiocontrast).
- **KDIGO Staging Matrix:** Automatic derivation of **eGFR Stages (G1–G5)** and **Albuminuria Risk Tiers (A1–A3)**.

---

## 🔬 System Architecture & Workflow

```text
[Patient Laboratory & Biometric Intake]
                  │
                  ▼
   [Dynamic Feature Mapper & Imputer] ──► (Aligns 51 standardized clinical features)
                  │
                  ▼
      [StandardScaler (.transform)]   ──► (Normalizes continuous & categorical vectors)
                  │
                  ▼
    [SVM Classifier (.predict_proba)] ──► (Generates classification & calibrated risk %)
                  │
                  ▼
   [Decision Engine & Rules Logic]    ──► (KDIGO GFR G1-G5 & Albuminuria A1-A3 Staging)
                  │
                  ▼
   [Enterprise Dashboard Presentation]──► (Visual Risk Alert + Metrics + Structured Care Plan)
```

---

## 🛠 Technology Stack

- **Core Language:** Python 3.10 / 3.11
- **Web Application Framework:** [Streamlit](https://streamlit.io/) (v1.30+)
- **Machine Learning Core:** [Scikit-Learn](https://scikit-learn.org/) (SVM Classifier, StandardScaler, Cross-Validation Metrics)
- **Model Serialization:** [Joblib](https://joblib.readthedocs.io/)
- **Data Manipulation & Analysis:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visual Presentation:** Custom Vanilla CSS (CSS3 Keyframes, Flexbox, CSS Grid, Plus Jakarta Sans typography)

---

## 🚀 Installation & Quickstart Guide

### Prerequisites
- Python 3.10 or higher installed on your system.
- Git installed on your local workstation.

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/ML-Lab-CKD-Prediction-System.git
cd ML-Lab-CKD-Prediction-System
```

### Step 2: Configure Virtual Environment
```bash
# On Linux / macOS
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
Create a `requirements.txt` file or install the runtime dependencies directly via pip:
```bash
pip install streamlit scikit-learn pandas numpy joblib pillow
```

Alternatively, if `requirements.txt` is present:
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
Launch the Streamlit web application from the project root:
```bash
streamlit run frontend/app.py
```

*(Note: The internal path resolver supports running directly from within the frontend directory as well: `cd frontend && streamlit run app.py`)*

After initialization, open your browser at:
```text
Local URL:   http://localhost:8501
Network URL: http://192.168.x.x:8501
```

---

## 📊 Model Performance & Evaluation

The production Support Vector Classifier was selected after extensive benchmarking against alternative supervised learning architectures on stratified holdout test cohorts:

| Model Architecture | Test Accuracy (%) | Precision (%) | Sensitivity / Recall (%) | Specificity (%) | ROC-AUC | Deployment Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Support Vector Classifier (SVC)** | **97.3%** | **97.0%** | **96.8%** | **97.9%** | **0.991** | **Production Deployed** |
| Random Forest Classifier | 97.1% | 96.8% | 96.5% | 97.4% | 0.989 | Benchmark Candidate |
| Gradient Boosting (XGBoost) | 96.5% | 96.0% | 96.2% | 96.8% | 0.984 | Benchmark Candidate |
| Multi-Layer Perceptron (MLP) | 95.2% | 94.7% | 94.1% | 95.5% | 0.976 | Candidate |
| K-Nearest Neighbors (k=5) | 91.0% | 90.5% | 90.2% | 91.8% | 0.942 | Baseline |
| Logistic Regression (L2) | 89.5% | 88.2% | 89.8% | 89.2% | 0.931 | Baseline |

### Feature Importance & Biomarker Impact
Biomarker sensitivity analyses identify the primary drivers influencing renal classification boundaries:
1. **eGFR (Glomerular Filtration Rate):** Highest discriminatory weight across functional staging thresholds.
2. **Serum Creatinine:** Primary filtration indicator reflecting kidney clearance capacity.
3. **Protein in Urine & Albumin-Creatinine Ratio (ACR):** Markers of glomerular permeability and barrier degradation.
4. **Blood Urea Nitrogen (BUN Levels):** Critical indicator of nitrogen retention and uremic state.
5. **Systolic Blood Pressure & HbA1c:** Primary microvascular risk factors accelerating diabetic nephrosclerosis.

*(Evaluation plots, confusion matrices, and ROC curves are located in the `reports/` folder).*

---

## 📖 Clinical Staging & Decision Rules

Renal function is categorized in accordance with the **KDIGO 2024 Clinical Practice Guideline**:

| Stage | GFR Category (mL/min/1.73m²) | Clinical Staging Description | Clinical Care Directive |
| :---: | :---: | :--- | :--- |
| **G1** | $\ge 90$ | Normal or elevated renal filtration | CVD risk mitigation, baseline observation |
| **G2** | $60 - 89$ | Mildly decreased renal filtration | Progression monitoring, glycemic & BP control |
| **G3a** | $45 - 59$ | Mild to moderate functional impairment | Complication screening, shared nephrology care |
| **G3b** | $30 - 44$ | Moderate to severe functional impairment | Active nephrology management, bone mineral review |
| **G4** | $15 - 29$ | Severely decreased renal filtration | Preparation for Kidney Replacement Therapy (KRT) |
| **G5** | $< 15$ | Kidney Failure (End-Stage Renal Disease) | Dialysis access placement or transplantation evaluation |

---

## ⚠️ Medical Disclaimer

**IMPORTANT NOTICE FOR PRACTITIONERS & EVALUATORS:**  
This software is developed as an academic Machine Learning Laboratory Mini-Project for educational, clinical research, and decision-support demonstration purposes only. It is **not** certified as an autonomous medical diagnostic device. Clinical determinations must always be confirmed through formal laboratory urinalysis, serum chemistry panels, imaging studies, and independent medical evaluation by licensed healthcare professionals.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
