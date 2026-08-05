import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ---------------------------------------------------------
# Page Configuration & Modern Design System
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Crop Production Forecasting Platform (v3.0 Ultra)",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Ultra-Modern Glassmorphism CSS with Animated Gradients
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0f172a 0%, #064e3b 40%, #022c22 90%);
        color: #f8fafc;
    }

    /* Hero Banner Styling */
    .hero-container {
        position: relative;
        border-radius: 24px;
        overflow: hidden;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
    }
    
    .hero-overlay {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(180deg, rgba(2, 44, 34, 0.3) 0%, rgba(15, 23, 42, 0.85) 100%);
        padding: 40px;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(16, 185, 129, 0.4);
        box-shadow: 0 20px 45px rgba(16, 185, 129, 0.2);
    }

    /* Metric Cards */
    .metric-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 78, 59, 0.3) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-4px) scale(1.02);
        border-color: #10b981;
        box-shadow: 0 12px 25px rgba(16, 185, 129, 0.3);
    }
    
    .metric-val {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 6px 0;
    }
    
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    /* Glowing Badges */
    .glow-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.5);
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
        margin-bottom: 16px;
    }

    /* Custom Form Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #047857 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        box-shadow: 0 10px 20px rgba(16, 185, 129, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 30px rgba(16, 185, 129, 0.5) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper Functions & Resource Loaders
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = r"C:\Users\DELL\Downloads"

@st.cache_resource
def load_models_and_encoders():
    paths = {
        'model': [os.path.join(BASE_DIR, "crop_production_model.pkl"), os.path.join(DOWNLOADS_DIR, "crop_production_model.pkl")],
        'crop_enc': [os.path.join(BASE_DIR, "crop_encoder.pkl"), os.path.join(DOWNLOADS_DIR, "crop_encoder.pkl")],
        'season_enc': [os.path.join(BASE_DIR, "season_encoder.pkl"), os.path.join(DOWNLOADS_DIR, "season_encoder.pkl")],
        'state_enc': [os.path.join(BASE_DIR, "state_encoder.pkl"), os.path.join(DOWNLOADS_DIR, "state_encoder.pkl")],
        'meta': [os.path.join(BASE_DIR, "feature_meta.pkl"), os.path.join(DOWNLOADS_DIR, "feature_meta.pkl")]
    }
    loaded = {}
    for key, p_list in paths.items():
        obj = None
        for p in p_list:
            if os.path.exists(p):
                obj = joblib.load(p)
                break
        loaded[key] = obj
    return loaded['model'], loaded['crop_enc'], loaded['season_enc'], loaded['state_enc'], loaded['meta']

@st.cache_data
def load_dataset():
    possible_paths = [
        os.path.join(BASE_DIR, "crop_yield_processed.csv"),
        os.path.join(DOWNLOADS_DIR, "crop_yield_processed.csv")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['Crop_Clean'] = df['Crop'].str.strip()
            df['Season_Clean'] = df['Season'].str.strip()
            df['State_Clean'] = df['State'].str.strip()
            return df
    return None

def get_image_path(filename):
    for p in [os.path.join(BASE_DIR, filename), os.path.join(DOWNLOADS_DIR, filename)]:
        if os.path.exists(p):
            return p
    return None

# Load Resources
model, crop_enc, season_enc, state_enc, feature_meta = load_models_and_encoders()
df = load_dataset()

hero_img_path = get_image_path("hero_banner.jpg")
analytics_img_path = get_image_path("farm_analytics.jpg")

# Encoder Mappings
crop_clean_map = {cls.strip(): cls for cls in crop_enc.classes_}
season_clean_map = {cls.strip(): cls for cls in season_enc.classes_}
state_clean_map = {cls.strip(): cls for cls in state_enc.classes_}

crops_list = sorted(list(crop_clean_map.keys()))
seasons_list = sorted(list(season_clean_map.keys()))
states_list = sorted(list(state_clean_map.keys()))


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=75)
st.sidebar.title("🌾 AI Crop Intelligence")
st.sidebar.markdown("<span style='color:#94a3b8; font-size:0.85rem;'>Smart Agricultural Yield Forecasting v3.0</span>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📌 Navigation Menu",
    [
        "🏠 Home & Overview",
        "🔮 Crop Yield Predictor",
        "📊 Analytics & Data Insights",
        "⚖️ Crop & State Comparison",
        "⚙️ Model Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("✨ **Super-Enhanced Model**: 500 ExtraTrees Estimators ($R^2 = 99.85\%$) with 14 engineered features.")


# ---------------------------------------------------------
# PAGE 1: Home & Overview (Ultra Visual Redesign)
# ---------------------------------------------------------
if page == "🏠 Home & Overview":
    
    # Hero Graphic Banner
    if hero_img_path:
        st.image(hero_img_path, use_container_width=True)
    
    st.markdown('<div class="glow-badge">🌱 AI-Powered Precision Agriculture Platform</div>', unsafe_allow_html=True)
    st.title("🌾 Smart Crop Production & Yield Forecasting System")
    st.markdown("""
    Transforming agricultural decision-making with **state-of-the-art Ensemble Machine Learning**. 
    Our system empowers farmers, agricultural economists, and government planners to predict harvest production, optimize fertilizer allocation, and mitigate climate risks.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Graphic Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-lbl">Dataset Volume</div>
            <div class="metric-val">19,689</div>
            <div style="color:#94a3b8; font-size:0.8rem;">Historical records across India</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-lbl">Crops Supported</div>
            <div class="metric-val">55</div>
            <div style="color:#94a3b8; font-size:0.8rem;">Cereals, Pulses, Commercial</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-lbl">States Covered</div>
            <div class="metric-val">30</div>
            <div style="color:#94a3b8; font-size:0.8rem;">Indian Agricultural Regions</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-lbl">Model Accuracy</div>
            <div class="metric-val">99.85%</div>
            <div style="color:#94a3b8; font-size:0.8rem;">R² Score Variance Explained</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Rich Visual Features Section
    st.markdown("### 🚀 Core Platform Modules")
    
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("""
        <div class="glass-card" style="border-left: 5px solid #10b981;">
            <h3 style="color:#34d399;">🤖 AI Yield Predictor Engine</h3>
            <p style="color:#cbd5e1;">Input farm area, rainfall, fertilizer, and pesticide metrics to generate high-precision production forecasts in Metric Tonnes and Yield per Hectare.</p>
            <ul style="color:#94a3b8; font-size:0.9rem;">
                <li>⚡ 1-Click presets (Rice, Wheat, Sugarcane, Coconut)</li>
                <li>📊 Benchmark yield status ratings</li>
                <li>📥 Exportable CSV forecast reports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with f2:
        st.markdown("""
        <div class="glass-card" style="border-left: 5px solid #3b82f6;">
            <h3 style="color:#60a5fa;">📊 Exploratory Data Intelligence</h3>
            <p style="color:#cbd5e1;">Explore 23 years of national crop production dynamics with interactive Plotly visualizations, correlation matrices, and comparison tools.</p>
            <ul style="color:#94a3b8; font-size:0.9rem;">
                <li>🗺️ Top producing states & crop leaderboard</li>
                <li>🌤️ Seasonal trend breakdown</li>
                <li>⚖️ Side-by-side state & crop comparison matrix</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # National Production Leaderboard Chart
    if df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🏆 Top 10 Crop Producing States in India")
        state_prod = df.groupby('State_Clean')['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head(10)
        
        fig = px.bar(
            state_prod,
            x='Production',
            y='State_Clean',
            orientation='h',
            title="Total Historical Crop Production by State (Metric Tonnes)",
            labels={'Production': 'Total Production (Tonnes)', 'State_Clean': 'State'},
            color='Production',
            color_continuous_scale='Greens'
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans, sans-serif"),
            yaxis=dict(autorange="reversed"),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# PAGE 2: Crop Yield Predictor
# ---------------------------------------------------------
elif page == "🔮 Crop Yield Predictor":
    st.markdown('<div class="glow-badge">🔮 AI Forecast Engine</div>', unsafe_allow_html=True)
    st.title("🔮 Predict Crop Production & Yield Efficiency")
    st.write("Specify your agricultural parameters below to generate instant yield forecasts.")

    # Presets
    st.markdown("##### ⚡ Quick Scenario Presets")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    
    preset_data = {}
    with p_col1:
        if st.button("🌾 Punjab Rice Farm"):
            preset_data = {"crop": "Rice", "state": "Punjab", "season": "Kharif", "area": 25.0, "rainfall": 1100.0, "fert": 4200.0, "pest": 80.0}
    with p_col2:
        if st.button("🌾 UP Wheat Field"):
            preset_data = {"crop": "Wheat", "state": "Uttar Pradesh", "season": "Rabi", "area": 40.0, "rainfall": 950.0, "fert": 6500.0, "pest": 120.0}
    with p_col3:
        if st.button("🎋 MH Sugarcane Estate"):
            preset_data = {"crop": "Sugarcane", "state": "Maharashtra", "season": "Whole Year", "area": 100.0, "rainfall": 1400.0, "fert": 25000.0, "pest": 450.0}
    with p_col4:
        if st.button("🌴 Kerala Coconut"):
            preset_data = {"crop": "Coconut", "state": "Kerala", "season": "Whole Year", "area": 15.0, "rainfall": 2800.0, "fert": 1800.0, "pest": 30.0}

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("### 📋 Input Farm Parameters")
        
        c1, c2 = st.columns(2)
        
        with c1:
            crop_idx = crops_list.index(preset_data.get("crop", "Rice")) if preset_data.get("crop") in crops_list else 0
            selected_crop_clean = st.selectbox("🌱 Select Crop", crops_list, index=crop_idx)
            
            state_idx = states_list.index(preset_data.get("state", "Punjab")) if preset_data.get("state") in states_list else 0
            selected_state_clean = st.selectbox("📍 Select State", states_list, index=state_idx)
            
            season_idx = seasons_list.index(preset_data.get("season", "Kharif")) if preset_data.get("season") in seasons_list else 0
            selected_season_clean = st.selectbox("🌤️ Select Season", seasons_list, index=season_idx)
            
            crop_year = st.slider("📅 Crop Year", 1997, 2030, 2026)

        with c2:
            area = st.number_input(
                "📐 Cultivated Area (Hectares)",
                min_value=0.1,
                max_value=100000.0,
                value=float(preset_data.get("area", 50.0)),
                step=1.0
            )
            
            rainfall = st.number_input(
                "🌧️ Annual Rainfall (mm)",
                min_value=100.0,
                max_value=7000.0,
                value=float(preset_data.get("rainfall", 1200.0)),
                step=10.0
            )
            
            def_fert = float(preset_data.get("fert", area * 170.0))
            def_pest = float(preset_data.get("pest", area * 3.0))

            fertilizer = st.number_input(
                "🧪 Fertilizer Usage (kg)",
                min_value=0.0,
                max_value=5000000.0,
                value=def_fert,
                step=10.0
            )
            
            pesticide = st.number_input(
                "🪲 Pesticide Usage (kg)",
                min_value=0.0,
                max_value=500000.0,
                value=def_pest,
                step=1.0
            )

        submit_btn = st.form_submit_button("🚀 Execute AI Prediction", use_container_width=True)

    if submit_btn:
        crop_orig = crop_clean_map[selected_crop_clean]
        season_orig = season_clean_map[selected_season_clean]
        state_orig = state_clean_map[selected_state_clean]

        try:
            crop_val = crop_enc.transform([crop_orig])[0]
            season_val = season_enc.transform([season_orig])[0]
            state_val = state_enc.transform([state_orig])[0]

            if feature_meta and 'features' in feature_meta:
                fert_per_area = fertilizer / (area + 1e-5)
                pest_per_area = pesticide / (area + 1e-5)
                rainfall_per_area = rainfall / (area + 1e-5)
                
                nat_avg = feature_meta.get('national_avg_yield', 10.0)
                crop_avg = feature_meta.get('crop_yield_map', {}).get(crop_orig, nat_avg)
                state_avg = feature_meta.get('state_yield_map', {}).get(state_orig, nat_avg)
                expected_prod = crop_avg * area
                
                input_df = pd.DataFrame([[
                    crop_val, crop_year, season_val, state_val, area, rainfall, fertilizer, pesticide,
                    fert_per_area, pest_per_area, rainfall_per_area, crop_avg, state_avg, expected_prod
                ]], columns=feature_meta['features'])
            else:
                input_df = pd.DataFrame([[
                    crop_val, crop_year, season_val, state_val, area, rainfall, fertilizer, pesticide
                ]], columns=['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide'])

            prediction = model.predict(input_df)[0]
            predicted_production = max(0.0, float(prediction))
            yield_per_ha = predicted_production / area if area > 0 else 0.0

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🎯 Forecast Results")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.markdown(f"""
                <div class="glass-card" style="border-left: 5px solid #10b981;">
                    <div style="color:#94a3b8; font-size:0.85rem; font-weight:600;">PREDICTED PRODUCTION</div>
                    <div style="font-size:2.4rem; font-weight:800; color:#10b981;">{predicted_production:,.2f}</div>
                    <div style="color:#cbd5e1; font-weight:600;">Metric Tonnes</div>
                </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                st.markdown(f"""
                <div class="glass-card" style="border-left: 5px solid #3b82f6;">
                    <div style="color:#94a3b8; font-size:0.85rem; font-weight:600;">ESTIMATED YIELD EFFICIENCY</div>
                    <div style="font-size:2.4rem; font-weight:800; color:#60a5fa;">{yield_per_ha:,.2f}</div>
                    <div style="color:#cbd5e1; font-weight:600;">Tonnes per Hectare</div>
                </div>
                """, unsafe_allow_html=True)

            with res_col3:
                if yield_per_ha > 8.0:
                    status_color = "#10b981"
                    status_text = "🌟 High Yield Potential"
                    desc = "Optimal parameters for maximum output."
                elif yield_per_ha > 3.0:
                    status_color = "#f59e0b"
                    status_text = "⚡ Moderate Yield"
                    desc = "Standard output yield expected."
                else:
                    status_color = "#ef4444"
                    status_text = "⚠️ Low Yield Output"
                    desc = "Adjust fertilizer or irrigation inputs."

                st.markdown(f"""
                <div class="glass-card" style="border-left: 5px solid {status_color};">
                    <div style="color:#94a3b8; font-size:0.85rem; font-weight:600;">YIELD BENCHMARK RATING</div>
                    <div style="font-size:1.4rem; font-weight:800; color:{status_color}; margin-top:4px;">{status_text}</div>
                    <div style="color:#94a3b8; font-size:0.8rem; margin-top:4px;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

            # Recommendations Card
            st.markdown("""
            <div class="glass-card">
                <h4>💡 Agronomic Insights & Action Plan</h4>
            """, unsafe_allow_html=True)
            
            fert_per_ha = fertilizer / area
            pest_per_ha = pesticide / area
            
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.write(f"• **Fertilizer Density**: `{fert_per_ha:.1f} kg/Ha`")
                st.write(f"• **Pesticide Density**: `{pest_per_ha:.2f} kg/Ha`")
                st.write(f"• **Rainfall Index**: `{rainfall:.0f} mm/year`")
            with r_col2:
                if fert_per_ha < 100:
                    st.info("💡 Fertilizer density is below recommended averages. Balanced N-P-K addition advised.")
                elif fert_per_ha > 350:
                    st.warning("⚠️ High fertilizer usage detected. Monitor soil salinity.")
                else:
                    st.success("✅ Balanced nutrient application rate detected.")
            st.markdown("</div>", unsafe_allow_html=True)

            # Export Report
            report_df = pd.DataFrame([{
                "Crop": selected_crop_clean,
                "State": selected_state_clean,
                "Season": selected_season_clean,
                "Year": crop_year,
                "Area_Ha": area,
                "Rainfall_mm": rainfall,
                "Fertilizer_kg": fertilizer,
                "Pesticide_kg": pesticide,
                "Predicted_Production_Tonnes": predicted_production,
                "Yield_Tonnes_per_Ha": yield_per_ha
            }])
            csv_data = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Prediction Summary (CSV)",
                data=csv_data,
                file_name=f"crop_prediction_{selected_crop_clean}_{selected_state_clean}_{crop_year}.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error executing prediction model: {str(e)}")


# ---------------------------------------------------------
# PAGE 3: Analytics & Data Insights
# ---------------------------------------------------------
elif page == "📊 Analytics & Data Insights":
    
    if analytics_img_path:
        st.image(analytics_img_path, use_container_width=True)
        
    st.markdown('<div class="glow-badge">📊 Data Analytics</div>', unsafe_allow_html=True)
    st.title("📊 Exploratory Data Analysis & Visualizations")
    
    if df is not None:
        tab1, tab2, tab3, tab4 = st.tabs([
            "🗺️ State & Crop Distribution",
            "🌤️ Seasonal Trends",
            "🧪 Inputs vs Production",
            "📋 Raw Data Explorer"
        ])
        
        with tab1:
            st.markdown("### Top Producing Crops & States")
            c1, c2 = st.columns(2)
            
            with c1:
                top_crops = df.groupby('Crop_Clean')['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head(10)
                fig1 = px.pie(
                    top_crops,
                    names='Crop_Clean',
                    values='Production',
                    title="Top 10 Crops by Historical Production Share",
                    hole=0.45,
                    color_discrete_sequence=px.colors.sequential.Greens_r
                )
                fig1.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig1, use_container_width=True)
                
            with c2:
                top_states = df.groupby('State_Clean')['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head(10)
                fig2 = px.bar(
                    top_states,
                    x='State_Clean',
                    y='Production',
                    title="Top 10 States by Total Production (Tonnes)",
                    color='Production',
                    color_continuous_scale='Greens'
                )
                fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-45)
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.markdown("### Seasonal Crop Dynamics")
            season_summary = df.groupby('Season_Clean')['Production'].agg(['sum', 'count']).reset_index()
            season_summary.columns = ['Season', 'Total Production', 'Record Count']
            
            fig3 = px.bar(
                season_summary,
                x='Season',
                y='Total Production',
                color='Total Production',
                title="Total Production by Season across India",
                color_continuous_scale='Viridis'
            )
            fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)
            
            # Yearly Trend
            yearly = df.groupby('Crop_Year')['Production'].sum().reset_index()
            fig_year = px.line(
                yearly,
                x='Crop_Year',
                y='Production',
                title="Annual Crop Production Trend (1997 - 2020)",
                markers=True,
                line_shape="spline"
            )
            fig_year.update_traces(line_color="#10b981", line_width=3)
            fig_year.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_year, use_container_width=True)

        with tab3:
            st.markdown("### Correlation of Inputs to Production")
            numeric_cols = ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Production']
            corr = df[numeric_cols].corr()
            
            fig4 = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale='RdBu_r',
                title="Correlation Matrix of Key Agricultural Parameters"
            )
            fig4.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)

        with tab4:
            st.markdown("### Filterable Dataset Explorer")
            filter_state = st.multiselect("Filter by State", states_list, default=[])
            filter_crop = st.multiselect("Filter by Crop", crops_list, default=[])
            
            df_filtered = df.copy()
            if filter_state:
                df_filtered = df_filtered[df_filtered['State_Clean'].isin(filter_state)]
            if filter_crop:
                df_filtered = df_filtered[df_filtered['Crop_Clean'].isin(filter_crop)]
                
            st.write(f"Showing **{len(df_filtered):,}** records:")
            st.dataframe(
                df_filtered[['Crop_Clean', 'State_Clean', 'Season_Clean', 'Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Production']].head(1000),
                use_container_width=True
            )


# ---------------------------------------------------------
# PAGE 4: Crop & State Comparison
# ---------------------------------------------------------
elif page == "⚖️ Crop & State Comparison":
    st.markdown('<div class="glow-badge">⚖️ Comparative Matrix</div>', unsafe_allow_html=True)
    st.title("⚖️ Side-by-Side Crop & State Comparison")
    
    if df is not None:
        comp_type = st.radio("Select Comparison Domain", ["🌾 Compare Two Crops", "🗺️ Compare Two States"], horizontal=True)
        
        if comp_type == "🌾 Compare Two Crops":
            c1, c2 = st.columns(2)
            with c1:
                crop1 = st.selectbox("Select Crop 1", crops_list, index=crops_list.index("Rice") if "Rice" in crops_list else 0)
            with c2:
                crop2 = st.selectbox("Select Crop 2", crops_list, index=crops_list.index("Wheat") if "Wheat" in crops_list else 1)
                
            d1 = df[df['Crop_Clean'] == crop1]
            d2 = df[df['Crop_Clean'] == crop2]
            
            m1_col, m2_col = st.columns(2)
            with m1_col:
                st.markdown(f"""
                <div class="glass-card" style="border-top: 5px solid #10b981;">
                    <h3>{crop1}</h3>
                    <p>• Total Production: <b>{d1['Production'].sum():,.0f} Tonnes</b></p>
                    <p>• Avg Cultivated Area: <b>{d1['Area'].mean():,.1f} Ha</b></p>
                    <p>• Avg Annual Rainfall: <b>{d1['Annual_Rainfall'].mean():,.1f} mm</b></p>
                    <p>• Avg Fertilizer Usage: <b>{d1['Fertilizer'].mean():,.1f} kg</b></p>
                </div>
                """, unsafe_allow_html=True)
            with m2_col:
                st.markdown(f"""
                <div class="glass-card" style="border-top: 5px solid #3b82f6;">
                    <h3>{crop2}</h3>
                    <p>• Total Production: <b>{d2['Production'].sum():,.0f} Tonnes</b></p>
                    <p>• Avg Cultivated Area: <b>{d2['Area'].mean():,.1f} Ha</b></p>
                    <p>• Avg Annual Rainfall: <b>{d2['Annual_Rainfall'].mean():,.1f} mm</b></p>
                    <p>• Avg Fertilizer Usage: <b>{d2['Fertilizer'].mean():,.1f} kg</b></p>
                </div>
                """, unsafe_allow_html=True)
                
        else:
            c1, c2 = st.columns(2)
            with c1:
                state1 = st.selectbox("Select State 1", states_list, index=states_list.index("Punjab") if "Punjab" in states_list else 0)
            with c2:
                state2 = st.selectbox("Select State 2", states_list, index=states_list.index("Karnataka") if "Karnataka" in states_list else 1)
                
            d1 = df[df['State_Clean'] == state1]
            d2 = df[df['State_Clean'] == state2]
            
            m1_col, m2_col = st.columns(2)
            with m1_col:
                st.markdown(f"""
                <div class="glass-card" style="border-top: 5px solid #10b981;">
                    <h3>{state1}</h3>
                    <p>• Total Production: <b>{d1['Production'].sum():,.0f} Tonnes</b></p>
                    <p>• Top Crop: <b>{d1.groupby('Crop_Clean')['Production'].sum().idxmax() if not d1.empty else 'N/A'}</b></p>
                    <p>• Records Count: <b>{len(d1):,}</b></p>
                    <p>• Avg Rainfall: <b>{d1['Annual_Rainfall'].mean():,.1f} mm</b></p>
                </div>
                """, unsafe_allow_html=True)
            with m2_col:
                st.markdown(f"""
                <div class="glass-card" style="border-top: 5px solid #3b82f6;">
                    <h3>{state2}</h3>
                    <p>• Total Production: <b>{d2['Production'].sum():,.0f} Tonnes</b></p>
                    <p>• Top Crop: <b>{d2.groupby('Crop_Clean')['Production'].sum().idxmax() if not d2.empty else 'N/A'}</b></p>
                    <p>• Records Count: <b>{len(d2):,}</b></p>
                    <p>• Avg Rainfall: <b>{d2['Annual_Rainfall'].mean():,.1f} mm</b></p>
                </div>
                """, unsafe_allow_html=True)


# ---------------------------------------------------------
# PAGE 5: Model Performance
# ---------------------------------------------------------
elif page == "⚙️ Model Performance":
    st.markdown('<div class="glow-badge">⚙️ Technical Architecture</div>', unsafe_allow_html=True)
    st.title("⚙️ Machine Learning Model Technical Overview")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-lbl">R² Accuracy Score</div>
            <div class="metric-val">99.85%</div>
            <div style="color:#94a3b8; font-size:0.8rem;">0.998501 Variance Explained</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-lbl">Tree Estimators</div>
            <div class="metric-val">500</div>
            <div style="color:#94a3b8; font-size:0.8rem;">Randomized Decision Trees</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-lbl">Mean Absolute Error</div>
            <div class="metric-val">520,381</div>
            <div style="color:#94a3b8; font-size:0.8rem;">Reduced by 35% vs Baseline</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if model is not None and hasattr(model, 'feature_importances_'):
        st.markdown("### 🔑 Feature Importance Breakdown (14 Features)")
        feature_names = feature_meta['features'] if (feature_meta and 'features' in feature_meta) else [f"Feature {i+1}" for i in range(len(model.feature_importances_))]
        importances = model.feature_importances_
        
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)
        
        fig_fi = px.bar(
            fi_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance Weights in Enhanced Model",
            color='Importance',
            color_continuous_scale='Greens'
        )
        fig_fi.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=500)
        st.plotly_chart(fig_fi, use_container_width=True)


# Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "🌾 AI Crop Production Prediction System v3.0 | Powered by Streamlit & Scikit-Learn"
    "</div>",
    unsafe_allow_html=True
)
