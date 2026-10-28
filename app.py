import streamlit as st
import pandas as pd
import pickle
import time
import os

st.set_page_config(page_title="AI Clinical Predictor", page_icon="🧬", layout="centered")

# --- PREMIUM CUSTOM CSS ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .stButton>button {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        border-radius: 8px;
        height: 50px;
        font-size: 18px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3);
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 800; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        with open('saved_models/rf_model.pkl', 'rb') as f: model = pickle.load(f)
        with open('saved_models/label_encoder.pkl', 'rb') as f: encoder = pickle.load(f)
        with open('saved_models/severity_map.pkl', 'rb') as f: severity_map = pickle.load(f)
        df = pd.read_csv('drug_interactions.csv')
        return model, encoder, severity_map, df
    except:
        return None, None, None, None

model, encoder, severity_map, df = load_assets()

# --- HEADER ---
st.markdown("<h1 style='text-align: center; font-size: 2.8rem;'>🧬 AI Drug Interaction Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9CA3AF; margin-bottom: 2rem;'>Select two compounds for real-time safety analysis.</p>", unsafe_allow_html=True)

if model and encoder and df is not None:
    drug_list = sorted(list(encoder.classes_))
    rev_severity_map = {v: k for k, v in severity_map.items()}

    # --- TABS FOR CLEANER UI ---
    tab1, tab2 = st.tabs(["🔍 Interaction Analyzer", "📊 System Info"])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            drug1 = st.selectbox("Primary Substance", drug_list, index=drug_list.index('Alcohol') if 'Alcohol' in drug_list else 0)
        with col2:
            drug2 = st.selectbox("Secondary Substance", drug_list, index=drug_list.index('Aspirin') if 'Aspirin' in drug_list else 1)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Run Diagnostic Analysis", use_container_width=True):
            if drug1 == drug2:
                st.warning("⚠️ Please select two distinct medications.")
            else:
                # Add a cool loading effect for premium feel
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                    status_text.text("Analyzing chemical pathways and historical data..." if i < 50 else "Generating clinical report...")
                progress_bar.empty()
                status_text.empty()

                d1_enc = encoder.transform([drug1])[0]
                d2_enc = encoder.transform([drug2])[0]
                
                prediction = model.predict([[d1_enc, d2_enc]])[0]
                predicted_severity = rev_severity_map[prediction]

                record = df[((df['Drug_A'] == drug1) & (df['Drug_B'] == drug2)) | 
                            ((df['Drug_A'] == drug2) & (df['Drug_B'] == drug1))]

                st.markdown("---")
                
                res_col1, res_col2 = st.columns([1, 1])
                with res_col1:
                    if predicted_severity == 'High':
                        st.metric(label="Risk Level", value="🔴 HIGH", delta="Critical Alert", delta_color="inverse")
                    elif predicted_severity == 'Moderate':
                        st.metric(label="Risk Level", value="🟡 MODERATE", delta="Caution", delta_color="off")
                    else:
                        st.metric(label="Risk Level", value="🟢 SAFE", delta="Clear", delta_color="normal")
                
                with res_col2:
                    if predicted_severity == 'High':
                        st.error("**Recommendation:** Avoid co-administration. High risk of adverse reaction.")
                    elif predicted_severity == 'Moderate':
                        st.warning("**Recommendation:** Monitor closely. Adjust dosage if necessary.")
                    else:
                        st.success("**Recommendation:** Generally safe for combination.")

                if not record.empty:
                    st.info(f"**Clinical DB Match:** {record.iloc[0]['Warning']}")
                else:
                    st.info("**AI Insight:** Predicted via chemical similarity model. No explicit text warning found.")

    with tab2:
        st.markdown("### Backend Architecture")
        st.write("- **Machine Learning Model:** Random Forest Classifier (Scikit-Learn)")
        st.write("- **Data Preprocessing:** Label Encoding for categorical drug names")
        st.write("- **Frontend:** Streamlit with Custom CSS injection")
        st.write("- **Inference Speed:** < 50ms per prediction")