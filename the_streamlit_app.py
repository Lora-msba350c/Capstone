import streamlit as st
import pandas as pd
import joblib
import numpy as np
from keras.models import load_model
from the_feature_engineering import apply_feature_engineering
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Shipment Anomaly Detector")

# Enhanced styling with MUCH larger title font and left alignment
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-size: 16px;
        color: #0b1f44;
    }
    
    /* Moderately sized title */
.app-title {
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #0b1f44;
    margin-bottom: 20px;
    text-align: left;
    padding-left: 0;
}
    
    h2 { 
        font-size: 28px; 
        font-weight: bold; 
        margin-top: 1.2em; 
        margin-bottom: 0.8em;
    }
    
    h3 { 
        font-size: 22px; 
        font-weight: 600; 
        margin-top: 1em; 
        margin-bottom: 0.8em;
    }

    /* Consistent metric displays */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 2rem;
    }

    .metric-box {
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        background-color: #fff;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        width: 32%;
    }
    
    .metric-box b {
        font-size: 18px;
        display: block;
        margin-bottom: 10px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 600;
    }
    
    .metric-percent {
        font-size: 18px;
        font-weight: 500;
        margin-top: 8px;
    }

    /* Styled information boxes */
    .info-box {
        background-color: #f0f5ff;
        padding: 25px;
        margin-bottom: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 6px solid #3366ff;
    }
    
    .action-box {
        background-color: #fff8f0;
        padding: 25px;
        margin-bottom: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 6px solid #ff9933;
    }
    
    .roadmap-box {
        background-color: #f0fff5;
        padding: 25px;
        margin-bottom: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 6px solid #33cc66;
    }
    
    /* Recommendation styling */
    .recommendation {
        margin-bottom: 15px;
        padding-bottom: 15px;
        border-bottom: 1px solid #eaeaea;
    }
    
    .recommendation:last-child {
        border-bottom: none;
    }
    
    /* Highlight important text */
    .highlight-text {
        font-weight: 600;
        color: #0b1f44;
        background-color: #fffddd;
        padding: 0 3px;
    }
    
    /* Phase styling */
    .phase {
        margin-bottom: 15px;
    }
    
    .phase-title {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 10px;
        color: #1a5336;
    }
    
    /* Horizontal model metrics */
    .model-metrics {
        display: flex;
        gap: 20px;
        margin-bottom: 20px;
    }
    
    .model-metric {
        background: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        flex: 1;
    }
    
    .model-metric-title {
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 10px;
    }
    
    .model-metric-value {
        font-size: 22px;
        font-weight: 600;
    }
    
    .model-metric-percent {
        font-size: 17px;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# LARGE LEFT-ALIGNED TITLE
st.markdown("<div class='app-title'>🛳️ Shipment Declaration Anomaly Detection</div>", unsafe_allow_html=True)

# Load models - REMOVED LOF MODEL
model_iso = joblib.load("isolation_forest_model.joblib")
scaler_iso = joblib.load("scaler_iso.joblib")
model_ae = load_model("autoencoder_model.keras")
scaler_ae = joblib.load("scaler_ae.joblib")

features_iso = [
    'WEIGHT_PER_TEU', 'TEU_DEVIATION', 'COMMODITY_WEIGHT_RATIO',
    'RARE_ROUTE', 'UNUSUAL_CONTAINER_COMMODITY', 'PARTNER_TEU_DEVIATION',
    'ROUTE_IS_NEW_FOR_PARTNER', 'PARTNER_ACTIVITY_SPAN',
    'HS_CODE_LENGTH', 'SEASONAL_PEAK_FLAG']

features_ae = features_iso

def double_mad_outliers(x, threshold=3.5):
    x = x.dropna()
    median = np.median(x)
    left = x[x <= median]
    right = x[x >= median]
    mad_left = np.median(np.abs(left - median)) or 1e-6
    mad_right = np.median(np.abs(right - median)) or 1e-6

    def is_outlier(value):
        if value <= median:
            return np.abs(value - median) / mad_left > threshold
        else:
            return np.abs(value - median) / mad_right > threshold

    return x.apply(is_outlier)

def pct_color(pct):
    return "green" if pct == 0 else "red"

uploaded_file = st.file_uploader("Upload shipment CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_clean = apply_feature_engineering(df.copy())

    missing = [f for f in features_iso if f not in df_clean.columns]
    if missing:
        st.error(f"Missing required features: {missing}")
        st.stop()

    X_iso = scaler_iso.transform(df_clean[features_iso].values)
    X_ae = scaler_ae.transform(df_clean[features_ae].values)

    df_clean['IF_Score'] = model_iso.decision_function(X_iso)
    df_clean['IF_Anomaly'] = model_iso.predict(X_iso) == -1

    recon = model_ae.predict(X_ae)
    ae_score = np.mean(np.square(X_ae - recon), axis=1)
    ae_thresh = np.percentile(ae_score, 95)
    df_clean['AE_Score'] = ae_score
    df_clean['AE_Anomaly'] = ae_score > ae_thresh

    df_clean['MAD_OUTLIER_WPT'] = double_mad_outliers(df_clean['WEIGHT_PER_TEU'])
    df_clean['MAD_OUTLIER_CWR'] = double_mad_outliers(df_clean['COMMODITY_WEIGHT_RATIO'], threshold=6)
    df_clean['MAD_Anomaly'] = df_clean[['MAD_OUTLIER_WPT', 'MAD_OUTLIER_CWR']].any(axis=1)

    # Updated to use only 3 models (removing LOF)
    df_clean['Flagged_by_All'] = df_clean[['IF_Anomaly', 'AE_Anomaly', 'MAD_Anomaly']].all(axis=1)
    df_clean['Anomaly_Model_Count'] = df_clean[['IF_Anomaly', 'AE_Anomaly', 'MAD_Anomaly']].sum(axis=1)
    df_clean['Flagged_by_Any'] = df_clean[['IF_Anomaly', 'AE_Anomaly', 'MAD_Anomaly']].any(axis=1)

    total_rows = len(df_clean)
    # Changed threshold from 3 to 2 for "strong outliers" since we now have only 3 models
    strong_outliers = df_clean[df_clean['Anomaly_Model_Count'] >= 2]

    top_routes = df_clean[df_clean['Flagged_by_Any']].groupby('POL/POD').size().sort_values(ascending=False).head(2)
    top_commodity = df_clean[df_clean['Flagged_by_Any']].groupby('COMMODITY_CODE').size().sort_values(ascending=False).idxmax()
    top_partner = df_clean[df_clean['Flagged_by_Any']].groupby('PARTNER_CODE').size().sort_values(ascending=False).idxmax()

    # Removed LOF tab
    tabs = st.tabs([
        "📊 Overview", "🌲 Isolation Forest", "🧠 Autoencoder",
        "📏 Double MAD", "✅ Solution Validation", "🔎 Findings & Recommendations"
    ])

    with tabs[0]:
        all_models_count = df_clean['Flagged_by_All'].sum()
        any_model_count = df_clean['Flagged_by_Any'].sum()
        # Changed from "three_plus_models" to "two_plus_models" since we now have only 3 models
        two_plus_models = len(strong_outliers)

        st.markdown("<h2>Model Overlap Summary</h2>", unsafe_allow_html=True)
        
        # Horizontal metrics display - Updated for 3 models
        st.markdown('''
            <div class="metric-container">
                <div class="metric-box">
                    <b>Flagged by All Models</b>
                    <div class="metric-value">{0} of {1}</div>
                    <div class="metric-percent" style="color:{2};">↑ {3:.2%}</div>
                </div>
                <div class="metric-box">
                    <b>Flagged by Any Model</b>
                    <div class="metric-value">{4} of {1}</div>
                    <div class="metric-percent" style="color:{5};">↑ {6:.2%}</div>
                </div>
                <div class="metric-box">
                    <b>Flagged by ≥2 Models</b>
                    <div class="metric-value">{7} of {1}</div>
                    <div class="metric-percent" style="color:{8};">↑ {9:.2%}</div>
                </div>
            </div>
        '''.format(
            all_models_count, total_rows, 
            pct_color(all_models_count/total_rows), all_models_count/total_rows,
            any_model_count, pct_color(any_model_count/total_rows), any_model_count/total_rows,
            two_plus_models, pct_color(two_plus_models/total_rows), two_plus_models/total_rows
        ), unsafe_allow_html=True)

        st.markdown("<h3>High Confidence Anomalies (≥2 Models)</h3>", unsafe_allow_html=True)
        st.dataframe(strong_outliers[['POL/POD','PARTNER_CODE','COMMODITY_CODE','Anomaly_Model_Count']].sort_values(by='Anomaly_Model_Count', ascending=False).head(30))

    with tabs[1]:
        iso_count = int(df_clean['IF_Anomaly'].sum())
        st.markdown("<h2>🌲 Isolation Forest</h2>", unsafe_allow_html=True)
        
        # Horizontal metrics
        st.markdown(f'''
            <div class="model-metrics">
                <div class="model-metric">
                    <div class="model-metric-title">Total Anomalies</div>
                    <div class="model-metric-value">{iso_count} of {total_rows}</div>
                    <div class="model-metric-percent" style="color:{pct_color(iso_count/total_rows)};">
                        {(iso_count/total_rows):.2%}
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
            
        st.dataframe(df_clean[df_clean['IF_Anomaly']][['POL/POD','PARTNER_CODE','COMMODITY_CODE','IF_Score']])

    with tabs[2]:
        ae_count = int(df_clean['AE_Anomaly'].sum())
        st.markdown("<h2>🧠 Autoencoder</h2>", unsafe_allow_html=True)
        
        # Horizontal metrics
        st.markdown(f'''
            <div class="model-metrics">
                <div class="model-metric">
                    <div class="model-metric-title">Total Anomalies</div>
                    <div class="model-metric-value">{ae_count} of {total_rows}</div>
                    <div class="model-metric-percent" style="color:{pct_color(ae_count/total_rows)};">
                        {(ae_count/total_rows):.2%}
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
            
        st.dataframe(df_clean[df_clean['AE_Anomaly']][['POL/POD','PARTNER_CODE','COMMODITY_CODE','AE_Score']])

    with tabs[3]:
        mad_wpt = int(df_clean['MAD_OUTLIER_WPT'].sum())
        mad_cwr = int(df_clean['MAD_OUTLIER_CWR'].sum())
        mad_any = int(df_clean['MAD_Anomaly'].sum())
        st.markdown("<h2>📏 Double MAD</h2>", unsafe_allow_html=True)
        
        # Horizontal metrics
        st.markdown(f'''
            <div class="model-metrics">
                <div class="model-metric">
                    <div class="model-metric-title">WPT Outliers</div>
                    <div class="model-metric-value">{mad_wpt}</div>
                    <div class="model-metric-percent" style="color:{pct_color(mad_wpt/total_rows)};">
                        {(mad_wpt/total_rows):.2%}
                    </div>
                </div>
                <div class="model-metric">
                    <div class="model-metric-title">CWR Outliers</div>
                    <div class="model-metric-value">{mad_cwr}</div>
                    <div class="model-metric-percent" style="color:{pct_color(mad_cwr/total_rows)};">
                        {(mad_cwr/total_rows):.2%}
                    </div>
                </div>
                <div class="model-metric">
                    <div class="model-metric-title">Any MAD Outlier</div>
                    <div class="model-metric-value">{mad_any}</div>
                    <div class="model-metric-percent" style="color:{pct_color(mad_any/total_rows)};">
                        {(mad_any/total_rows):.2%}
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
            
        st.dataframe(df_clean[df_clean['MAD_Anomaly']][['POL/POD','PARTNER_CODE','COMMODITY_CODE','WEIGHT_PER_TEU','COMMODITY_WEIGHT_RATIO']])

    with tabs[4]:
        st.markdown("<h2>✅ Human Feedback</h2>", unsafe_allow_html=True)
        st.markdown("Help us validate flagged anomalies:")
        sample = strong_outliers.head(10)
        for idx, row in sample.iterrows():
            st.radio(f"Row {idx} - Anomaly?", ["Uncertain", "Yes", "No"], key=f"fb_{idx}")

    with tabs[5]:
        st.markdown("### ✅ Results Summary")
    
        # Creating a clean metrics display
        st.markdown(f"""
        * **{all_models_count} shipments** flagged by all 3 models - high-risk 
        * **{two_plus_models} shipments** flagged by 2+ models - requires review
        * **{any_model_count} shipments** flagged by at least 1 model
        """)
        
        # Highlight key patterns
        st.markdown("### 🔍 Key Patterns")
        
        st.markdown(f"""
        * **Most anomalous routes**: `{'`, `'.join(top_routes.index.tolist())}`
        * **Most flagged partner**: `{top_partner}`
        * **Most flagged commodity**: `{top_commodity}`
        * **Anomaly rate**: {(any_model_count/total_rows):.1%} of all shipments
        """)
        
        # Recommended Actions section
        st.markdown("### 🔧 Recommended Actions")
        
        st.markdown("""
        1. **Start pilot audit program** on top 2 flagged routes
        2. **Update risk profile** for the most flagged partner
        3. **Implement pre-clearance requirements** for top anomalous commodity
        """)
        
        # Add a toggle for advanced metrics
        with st.expander("Advanced Metrics"):
            # Create a simple visualization of model overlap
            fig, ax = plt.subplots(figsize=(8, 5))
            model_counts = {
                "IF Only": int((df_clean['IF_Anomaly'] & ~df_clean['AE_Anomaly'] & ~df_clean['MAD_Anomaly']).sum()),
                "AE Only": int((~df_clean['IF_Anomaly'] & df_clean['AE_Anomaly'] & ~df_clean['MAD_Anomaly']).sum()),
                "MAD Only": int((~df_clean['IF_Anomaly'] & ~df_clean['AE_Anomaly'] & df_clean['MAD_Anomaly']).sum()),
                "IF+AE": int((df_clean['IF_Anomaly'] & df_clean['AE_Anomaly'] & ~df_clean['MAD_Anomaly']).sum()),
                "IF+MAD": int((df_clean['IF_Anomaly'] & ~df_clean['AE_Anomaly'] & df_clean['MAD_Anomaly']).sum()),
                "AE+MAD": int((~df_clean['IF_Anomaly'] & df_clean['AE_Anomaly'] & df_clean['MAD_Anomaly']).sum()),
                "All 3": int(df_clean['Flagged_by_All'].sum())
            }
            
            # Sort by count descending
            model_counts = {k: v for k, v in sorted(model_counts.items(), key=lambda item: item[1], reverse=True)}
            
            # Create bar chart
            sns.barplot(x=list(model_counts.keys()), y=list(model_counts.values()), palette="viridis", ax=ax)
            ax.set_ylabel('Count')
            ax.set_title('Model Overlap Distribution')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add a table of top anomalous items
            st.markdown("#### Top Anomalous Combinations")
            
            # Get top combinations of route+commodity with anomalies
            top_combos = df_clean[df_clean['Flagged_by_Any']].groupby(['POL/POD', 'COMMODITY_CODE']).size().reset_index()
            top_combos.columns = ['Route', 'Commodity', 'Count']
            top_combos = top_combos.sort_values('Count', ascending=False).head(5)
            
            st.table(top_combos)
