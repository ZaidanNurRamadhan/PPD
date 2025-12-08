import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import sys
import os
import json
from PIL import Image

# --- PATH SETUP FOR MONOLITH MODE ---
# Add parent directory to path so we can import 'appheart' and 'ml'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

# --- DIRECT IMPORTS (No API) ---
from appheart.database import SessionLocal, engine
from appheart import crud, models, schemas
from ml.cardio_model import CardioRiskModel

# Initialize DB
models.Base.metadata.create_all(bind=engine)

# --- SEED ADMIN USER (For Fresh DB) ---
def seed_admin():
    db = SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.email == "admin@admin.com").first()
        if not user:
            # Create default admin
            admin_user = models.User(
                email="admin@admin.com",
                name="Administrator",
                password_hash="hashed_secret", # In real app use hash, here simplified as we check hardcoded pass in Login View
                role="ADMIN"
            )
            db.add(admin_user)
            db.commit()
            print("Admin user seeded.")
    except Exception as e:
        print(f"Seeding ignored: {e}")
    finally:
        db.close()

seed_admin()

# --- DB HELPERS ---
def get_db():
    db = SessionLocal()
    try:
        return db
    except:
        db.close()
        raise

try:
    icon_path = os.path.join(parent_dir, "assets", "heart.png")
    icon = Image.open(icon_path)
except:
    icon = "🫀"

st.set_page_config(
    page_title="SIAGA Jantung Pro",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Clean Aesthetic */
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; color: #1e293b; }
    .stApp { 
        background-color: #f8fafc; 
    }
    
    /* Main Layout Padding Adjustment */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important; 
    }
    
    /* Premium Sidebar */
    section[data-testid="stSidebar"] { 
        background-color: #ffffff; 
        border-right: 1px solid #e2e8f0;
        box-shadow: 4px 0 24px rgba(0,0,0,0.02);
    }
    
    /* Glassy/Card Metrics */
    .metric-card {
        background: white;
        border-radius: 16px; 
        padding: 24px;
        border: 1px solid #e2e8f0; 
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        transition: all 0.2s;
        position: relative; 
        overflow: hidden;
    }
    .metric-card:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); 
        border-color: #3b82f6;
    }
    .metric-card::before { 
        content: ""; position: absolute; top: 0; left: 0; width: 6px; height: 100%; 
        background: #3b82f6; 
    }
    .metric-value { font-size: 36px; font-weight: 800; color: #0f172a; margin: 4px 0; letter-spacing: -0.02em; }
    .metric-label { font-size: 14px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    
    /* Modern Tabs */
    .stTabs {
        margin-top: 2rem; /* Add spacing from top */
    }
    .stTabs [data-baseweb="tab-list"] { 
        gap: 8px; border-bottom: none; margin-bottom: 24px;
    }
    .stTabs [data-baseweb="tab"] { 
        height: 40px; border-radius: 8px; color: #64748b; font-weight: 600; border: none; padding: 0 24px;
        background-color: transparent;
    }
    .stTabs [aria-selected="true"] { 
        color: #2563eb; background-color: #eff6ff;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    
    /* Buttons */
    .stButton > button { 
        border-radius: 8px; font-weight: 600; border: none !important;
        transition: all 0.2s; padding: 0.5rem 1rem;
    }
    /* Submit Button Specifics */
    div[data-testid="stFormSubmitButton"] > button {
        background: #2563eb;
        color: white;
        width: auto; /* Revert full width */
        min-width: 150px;
        padding: 0.6rem 2rem; /* Better internal padding */
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #1d4ed8;
        transform: translateY(-1px);
        box-shadow: 0 8px 12px -3px rgba(37, 99, 235, 0.3);
    }

    /* Custom Alert Boxes */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "selected_patient" not in st.session_state: st.session_state.selected_patient = None
if "user_id" not in st.session_state: st.session_state.user_id = 1
if "logged_in" not in st.session_state: st.session_state.logged_in = False

# --- LOGIC ACTIONS ---

# Visualization Helpers (Restored)
def create_gauge_chart(value, title):
    color = "green"
    if value >= 30: color = "orange"
    if value >= 60: color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 18, 'color': '#2c3e50'}},
        number = {'suffix': "%", 'font': {'color': color}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#333"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#eee",
            'steps': [
                {'range': [0, 30], 'color': "#e8f5e9"},
                {'range': [30, 60], 'color': "#fff3e0"},
                {'range': [60, 100], 'color': "#ffebee"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value}}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def create_radar_chart(input_data):
    # Normalize values to 0-1 scale for visualization where 1 is "Bad/High Risk"
    categories = ['BMI', 'Tekanan Darah (MAP)', 'Kolesterol', 'Glukosa', 'Gaya Hidup']
    
    # Logic normalization (Simplified)
    norm_bmi = min(max((input_data['bmi'] - 18.5) / (35 - 18.5), 0), 1)
    norm_map = min(max((input_data['map'] - 70) / (130 - 70), 0), 1)
    norm_chol = (input_data['cholesterol'] - 1) / 2
    norm_gluc = (input_data['gluc'] - 1) / 2
    
    # Lifestyle score: Smoke(1) + Alco(1) + Inactive(1) -> Max 3
    lifestyle_score = input_data['smoke'] + input_data['alco'] + (1 - input_data['active'])
    norm_lifestyle = min(lifestyle_score / 3, 1)
    
    values = [norm_bmi, norm_map, norm_chol, norm_gluc, norm_lifestyle]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Profil Pasien',
        line_color='#3498db',
        fillcolor='rgba(52, 152, 219, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False
            )),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        title=dict(text="Peta Faktor Risiko", x=0.5, y=0.95, font=dict(size=14))
    )
    return fig

def perform_analysis(p, age_years, bmi, map_val, chol_map, gluc_map, chol, gluc, smoke, alco, active):
    try:
        # Load Model
        model = CardioRiskModel()
        
        # Prepare Data
        input_data = {
            "age_years": age_years,
            "gender": 1 if p['gender'] == 'F' else 2,
            "bmi": bmi,
            "map": map_val,
            "cholesterol": chol_map[chol],
            "gluc": gluc_map[gluc],
            "smoke": int(smoke),
            "alco": int(alco),
            "active": int(active)
        }
        
        # Predict
        proba = model.predict_proba(input_data)
        label = model.predict_label(input_data, threshold=0.5)
        shap_dict = model.get_shap_values(input_data)
        shap_json = json.dumps(shap_dict)
        
        # Categories & Recommendations
        risk_percent = proba * 100
        if risk_percent < 30: risk_cat = "Rendah"
        elif risk_percent < 60: risk_cat = "Sedang"
        else: risk_cat = "Tinggi"
        
        recs = []
        if risk_cat == "Tinggi":
            recs.append("⚠️ **PROTOKOL RISIKO TINGGI**: Rujuk segera ke Spesialis Jantung.")
            recs.append("Lakukan EKG 12-lead dan Panel Lipid Lengkap.")
        elif risk_cat == "Sedang":
            recs.append("⚠️ **PROTOKOL RISIKO SEDANG**: Jadwalkan kontrol ulang dalam 3 bulan.")
            recs.append("Evaluasi gaya hidup ketat.")
        else:
            recs.append("✅ **PROTOKOL RISIKO RENDAH**: Edukasi gaya hidup sehat.")
            recs.append("Kontrol rutin tahunan.")
            
        if smoke: recs.append("🚭 **STOP MEROKOK**: Wajib program berhenti merokok.")
        if bmi >= 30: recs.append("⚖️ **BERAT BADAN**: Rujuk Ahli Gizi (Target turun 5-10%).")
        if map_val > 105: recs.append("🩺 **HIPERTENSI**: Monitoring tekanan darah harian.")
        if chol_map[chol] >= 3: recs.append("🍔 **KOLESTEROL**: Diet rendah lemak jenuh.")
        if gluc_map[gluc] >= 3: recs.append("🍬 **DIABETES**: Cek HbA1c.")
        
        recommendations_str = "\n".join(recs)
        
        # Save to DB
        db = SessionLocal()
        try:
            checkup_data = schemas.CheckupCreate(
                checked_by_user_id=st.session_state.user_id,
                **input_data
            )
            # Remove extra keys for schema if needed, but schema creation ignores extras usually or validates.
            # We constructed input_data to match model features, which might differ slightly from schema.
            # Actually schema checkupcreate needs all fields. ensuring input_data has them.
            # input_data keys match schema keys for clinical data.
            
            db_checkup = crud.create_checkup(
                db=db,
                checkup=checkup_data,
                patient_id=p['id'],
                probability=proba,
                risk_label=label,
                risk_category=risk_cat,
                model_version="monolith_v1",
                recommendations=recommendations_str,
                shap_values=shap_json
            )
            
            # Return dict format for frontend to render
            return {
                "probability": proba,
                "risk_category": risk_cat,
                "recommendations": recommendations_str,
                "shap_values": shap_json
            }
        finally:
            db.close()
            
    except Exception as e:
        st.error(f"Error during analysis: {e}")
        return None

# --- LOGIN SCREEN ---
def login_page():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        try:
            st.image(icon, width=100)
        except:
            st.title("🫀")
            
        st.title("Admin Login")
        st.caption("Masuk untuk mengakses Dashboard SIAGA Jantung (Monolith Mode)")
        
        with st.form("login_form"):
            username = st.text_input("Email", placeholder="admin@admin.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submit = st.form_submit_button("Masuk", type="primary", use_container_width=True)
            
            if submit:
                if username.lower().strip() == "admin@admin.com" and password == "ADMIN123":
                    st.session_state.logged_in = True
                    st.toast("Login Berhasil!", icon="🔓")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Email atau Password salah!")

if not st.session_state.logged_in:
    login_page()
    st.stop()

# ==========================================
# MAIN APP
# ==========================================
from streamlit_option_menu import option_menu

with st.sidebar:
    try:
        st.image(icon, width=70)
    except:
        st.header("🫀")
    st.markdown("### SIAGA Jantung")
    st.caption("v2.2 (Cloud Ready)")
    st.divider()
    
    menu = option_menu(
        menu_title=None,
        options=["Dashboard", "Pasien", "Laporan", "Bantuan"],
        icons=["speedometer2", "people-fill", "file-earmark-bar-graph", "info-circle"],
        default_index=0,
    )
    
    if menu == "Pasien":
        st.markdown("#### Cari Pasien")
        search = st.text_input("Nama / MRN", placeholder="Ketik nama...", label_visibility="collapsed")
        
        db = SessionLocal()
        if search:
            with st.spinner("Mencari..."):
                patients = crud.get_patients(db, name=search)
            if patients:
                for p_obj in patients:
                    # Convert to dict for easier session handling
                    p = p_obj.__dict__
                    if st.button(f"{p['full_name']}", key=p['id'], use_container_width=True):
                        st.session_state.selected_patient = p
                        st.rerun()
            else:
                st.warning("Tidak ditemukan.")
        db.close()
        
        st.markdown("---")
        with st.expander("Daftar Pasien Baru"):
            with st.form("add_patient"):
                new_name = st.text_input("Nama Lengkap")
                new_dob = st.date_input("Tanggal Lahir", min_value=datetime(1900,1,1))
                new_gender = st.selectbox("Jenis Kelamin", ["M", "F"])
                new_mrn = st.text_input("No. Rekam Medis (Opsional)")
                
                if st.form_submit_button("Simpan Data", type="primary"):
                    db = SessionLocal()
                    try:
                        p_create = schemas.PatientCreate(
                            full_name=new_name,
                            date_of_birth=str(new_dob),
                            gender=new_gender,
                            medical_record_number=new_mrn if new_mrn else None
                        )
                        new_p = crud.create_patient(db, p_create)
                        st.session_state.selected_patient = new_p.__dict__
                        st.toast("Pasien berhasil didaftarkan!", icon="✅")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal: {e}")
                    finally:
                        db.close()

    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
    if st.button("Logout", type="secondary"):
        st.session_state.logged_in = False
        st.rerun()

# --- CONTENT ---
if menu == "Dashboard":
    st.title("Dashboard Klinik")
    
    db = SessionLocal()
    stats = crud.get_checkup_stats(db)
    db.close()
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Pasien Terdaftar", stats['total_patients'])
    with c2: st.metric("Total Pemeriksaan", stats['total_checkups'])
    risk = (stats['averages']['risk'] or 0) * 100
    with c3: st.metric("Rata-rata Risiko", f"{risk:.1f}%")
    with c4: st.metric("Kasus Hipertensi", stats['risk_factors']['Hipertensi'])
    
    st.divider()
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribusi Risiko")
        rd = stats['risk_distribution']
        if rd:
            fig = px.pie(values=list(rd.values()), names=list(rd.keys()), hole=0.5,
                         color_discrete_sequence=['#2ecc71', '#f1c40f', '#e74c3c'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Belum ada data.")
            
    with col_b:
        st.subheader("Faktor Risiko")
        rf = stats['risk_factors']
        df_rf = pd.DataFrame(list(rf.items()), columns=["Faktor", "Jumlah"]).sort_values("Jumlah")
        fig = px.bar(df_rf, x="Jumlah", y="Faktor", orientation='h')
        st.plotly_chart(fig, use_container_width=True)

elif menu == "Laporan":
    st.title("Laporan Data")
    
    db = SessionLocal()
    try:
        all_checkups = crud.get_all_checkups(db)
        
        # Convert SQLAlchemy objects to Dict/DataFrame via Pydantic schema or direct dict
        data_list = []
        for c in all_checkups:
            # Simple dict conversion
            d = {k: v for k, v in c.__dict__.items() if not k.startswith('_')}
            
            # Add Patient Info (Lazy load from relationship - Safe here because session is open)
            if c.patient:
                d['Nama Pasien'] = c.patient.full_name
                d['No RM'] = c.patient.medical_record_number
            else:
                d['Nama Pasien'] = "Unknown"
                d['No RM'] = "-"
                
            data_list.append(d)
    finally:
        db.close()
    
    df = pd.DataFrame(data_list)
    
    # Reorder columns
    if not df.empty and 'Nama Pasien' in df.columns:
        cols = ['Nama Pasien', 'No RM'] + [col for col in df.columns if col not in ['Nama Pasien', 'No RM']]
        df = df[cols]
    
    if not df.empty:
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        date_str = datetime.now().strftime("%Y%m%d")
        st.download_button("📥 Unduh Laporan Lengkap (.csv)", csv, f"Laporan_Klinik_{date_str}.csv", "text/csv", type="primary")
    else:
        st.info("Belum ada data.")

elif menu == "Pasien":
    if not st.session_state.selected_patient:
        st.info("Pilih pasien di sidebar.")
    else:
        p = st.session_state.selected_patient
        
        # Header
        st.markdown(f"""
        <div style="padding: 20px; background: white; border-radius: 10px; border-left: 5px solid #3b82f6;">
            <h2>{p['full_name']}</h2>
            <p>MRN: {p.get('medical_record_number', '-') or 'N/A'} | {p['gender']} | {p['date_of_birth']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Dashboard", "Pemeriksaan Baru", "Riwayat"])
        
        # Get History
        db = SessionLocal()
        hist = crud.get_checkups_by_patient(db, p['id'])
        db.close()
        
        df_hist = pd.DataFrame([h.__dict__ for h in hist])
        if not df_hist.empty:
            if 'created_at' in df_hist.columns:
                df_hist['created_at'] = pd.to_datetime(df_hist['created_at'])
            last = df_hist.iloc[0] # Ordered desc in crud
        else:
            last = None
            
        with tab1:
            if last is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric("Risiko Terakhir", f"{last['probability']:.1%}", last['risk_category'])
                c2.metric("BMI", f"{last['bmi']:.1f}")
                c3.metric("MAP (Tekanan Darah)", f"{last['map']:.0f}")
                
                if 'created_at' in df_hist.columns:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("##### <i class='fa-solid fa-chart-line'></i> Tren Risiko", unsafe_allow_html=True)
                        fig_risk = px.area(df_hist, x='created_at', y='probability', markers=True)
                        fig_risk.update_traces(line_color='#ef4444', fillcolor='rgba(239, 68, 68, 0.1)')
                        fig_risk.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), xaxis_title=None, yaxis_title=None)
                        st.plotly_chart(fig_risk, use_container_width=True)
                    
                    with c2:
                        st.markdown("##### <i class='fa-solid fa-heart-pulse'></i> Tren Tekanan Darah", unsafe_allow_html=True)
                        fig_map = px.line(df_hist, x='created_at', y='map', markers=True)
                        fig_map.update_traces(line_color='#3b82f6', line_width=3)
                        fig_map.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), xaxis_title=None, yaxis_title=None)
                        st.plotly_chart(fig_map, use_container_width=True)
                
                st.info(f"Rekomendasi:\n{last['recommendations']}")
            else:
                st.info("Belum ada data.")
                
        with tab2:
            with st.form("checkup"):
                dob = datetime.strptime(p['date_of_birth'], "%Y-%m-%d")
                age = (datetime.now() - dob).days // 365
                st.caption(f"Usia: {age} tahun")
                
                c1, c2 = st.columns(2)
                with c1:
                    h = st.number_input("Tinggi (cm)", 100, 250, 170)
                    w = st.number_input("Berat (kg)", 30.0, 200.0, 70.0)
                    chol = st.selectbox("Kolesterol", ["Normal", "Sedang", "Tinggi"])
                with c2:
                    sys_bp = st.number_input("Sistolik", 90, 250, 120)
                    dia_bp = st.number_input("Diastolik", 50, 150, 80)
                    gluc = st.selectbox("Glukosa", ["Normal", "Sedang", "Tinggi"])
                
                smoke = st.checkbox("Merokok")
                alco = st.checkbox("Alkohol")
                active = st.checkbox("Aktif Fisik")
                
                submitted = st.form_submit_button("Analisis", type="primary")
            
            if submitted:
                if age < 5:
                    st.error("Pasien terlalu muda.")
                else:
                    with st.spinner("Menganalisis..."):
                        bmi = w / ((h/100)**2)
                        map_val = (2*dia_bp + sys_bp)/3
                        chol_map = {"Normal": 1, "Sedang": 2, "Tinggi": 3}
                        gluc_map = {"Normal": 1, "Sedang": 2, "Tinggi": 3}
                        
                        res = perform_analysis(p, age, bmi, map_val, chol_map, gluc_map, chol, gluc, smoke, alco, active)
                        
                        if res:
                            st.toast("Analisis Selesai!", icon="✅")
                            st.markdown("### 📊 Hasil Analisis")
                            
                            # Row 1: Main Visuals (Gauge & Radar)
                            r1, r2 = st.columns(2)
                            with r1:
                                # 1. GAUGE CHART (Risk Probability)
                                prob_val = res['probability'] * 100
                                st.plotly_chart(create_gauge_chart(prob_val, "Probabilitas Risiko"), use_container_width=True)
                                st.caption(f"Status: **{res['risk_category']}**")
                                
                            with r2:
                                # 2. RADAR CHART (Risk Profile)
                                # Reconstruct input data for visualization
                                radar_input = {
                                    "bmi": bmi,
                                    "map": map_val,
                                    "cholesterol": chol_map[chol], # 1, 2, 3
                                    "gluc": gluc_map[gluc],       # 1, 2, 3
                                    "smoke": int(smoke),
                                    "alco": int(alco),
                                    "active": int(active)
                                }
                                st.plotly_chart(create_radar_chart(radar_input), use_container_width=True)

                            # Row 2: Recommendations & Explanation
                            st.markdown("#### Rekomendasi Medis")
                            st.info(res['recommendations'])
                            
                            # Row 3: Detail SHAP (Expandable)
                            if res.get('shap_values'):
                                with st.expander("Lihat Detail Faktor Penentu (AI Explanation)"):
                                    try:
                                        shap_data = json.loads(res['shap_values'])
                                        if shap_data:
                                            # Sort by absolute impact
                                            sorted_shap = sorted(shap_data.items(), key=lambda x: abs(x[1]), reverse=True)
                                            # Convert to DF for chart
                                            shap_df = pd.DataFrame(sorted_shap, columns=['Faktor', 'Impact'])
                                            
                                            fig_shap = px.bar(shap_df, x='Impact', y='Faktor', orientation='h',
                                                            title="Kontribusi Faktor Risiko (SHAP)",
                                                            color='Impact', color_continuous_scale=['#2ecc71', '#e74c3c'])
                                            fig_shap.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0), showlegend=False)
                                            st.plotly_chart(fig_shap, use_container_width=True)
                                    except:
                                        pass
                                
                            st.button("🔄 Reset / Analisis Ulang", on_click=lambda: st.rerun())

        with tab3:
            if not df_hist.empty:
                # Custom Filename Download
                safe_name = p['full_name'].replace(" ", "_").upper()
                safe_mrn = str(p.get('medical_record_number', 'NA')).replace(" ", "")
                date_str = datetime.now().strftime("%d%m%Y")
                fname = f"HEARTREPORT_{safe_name}_{safe_mrn}_{date_str}_.csv"
                
                csv = df_hist.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Unduh Riwayat Pasien (.csv)", csv, fname, "text/csv", type="primary")
                
                # Clean dataframe for display
                display_cols = ['created_at', 'risk_category', 'probability', 'bmi', 'map', 'recommendations']
                # Filter cols that actually exist
                valid_cols = [c for c in display_cols if c in df_hist.columns]
                st.dataframe(df_hist[valid_cols])
            else:
                st.info("Kosong.")

elif menu == "Bantuan":
    st.markdown("### 📖 Panduan Penggunaan Sistem")
    
    with st.expander("1. Manajemen Pasien", expanded=True):
        st.markdown("""
        - **Pencarian**: Gunakan kolom pencarian di Sidebar untuk mencari pasien berdasarkan Nama atau Nomor Rekam Medis (MRN).
        - **Pendaftaran**: Jika pasien belum ada, buka menu "Daftar Pasien Baru" di Sidebar, isi formulir, dan klik Simpan.
        - **Status**: Pasien yang dipilih akan muncul di atas (Header Biru) dengan detail lengkap.
        """)
        
    with st.expander("2. Melakukan Pemeriksaan (Analisis AI)", expanded=True):
        st.markdown("""
        1. Pilih menu **Pasien** -> Tab **Pemeriksaan Baru**.
        2. Masukkan data klinis pasien (Tinggi, Berat, Tensi, Hasil Lab).
        3. Klik tombol **Analisis**. Sistem akan memproses data menggunakan model *Machine Learning*.
        4. **Interpretasi Hasil**:
           - **Gauge Chart** (Kiri): Menunjukkan *probability* atau kemungkinan risiko penyakit jantung (0-100%).
           - **Radar Chart** (Kanan): Menunjukkan peta profil pasien. Area yang melebar ke luar menunjukkan faktor risiko dominan (misal: Gaya Hidup buruk atau Tensi tinggi).
           - **Rekomendasi**: Ikuti saran medis yang muncul secara otomatis.
        """)
        
    with st.expander("3. Laporan & Ekspor Data"):
        st.markdown("""
        - **Download CSV**: Anda dapat mengunduh seluruh data pemeriksaan di menu **Laporan**. File CSV kini menyertakan Nama Pasien dan No RM untuk memudahkan administrasi.
        - **Riwayat Pasien**: Untuk mengunduh riwayat **satu pasien saja**, buka menu Pasien -> Tab Riwayat -> Unduh Riwayat.
        """)
        
    st.info("💡 Sistem ini menggunakan model XGBoost yang dilatih pada dataset kardiovaskular standar. Gunakan hasil sebagai pendukung keputusan klinis, bukan pengganti diagnosis dokter.")
