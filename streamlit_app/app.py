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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #1e293b; }
    .stApp { background-color: #f1f5f9; }
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    
    .metric-card {
        background-color: white; border-radius: 12px; padding: 24px;
        border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        transition: all 0.2s; position: relative; overflow: hidden;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }
    .metric-card::before { content: ""; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: #3b82f6; }
    .metric-value { font-size: 32px; font-weight: 700; color: #0f172a; margin: 8px 0; }
    .metric-label { font-size: 13px; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .metric-sub { font-size: 12px; color: #94a3b8; display: flex; align-items: center; gap: 4px; }
    
    /* Tabs & Buttons */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { height: 50px; border-radius: 0; color: #64748b; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #3b82f6; border-bottom: 2px solid #3b82f6; }
    .stButton > button { border-radius: 8px; font-weight: 600; border: none; box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05); }
    
    .login-container { display: flex; justify-content: center; align-items: center; height: 100vh; }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "selected_patient" not in st.session_state: st.session_state.selected_patient = None
if "user_id" not in st.session_state: st.session_state.user_id = 1
if "logged_in" not in st.session_state: st.session_state.logged_in = False

# --- LOGIC ACTIONS ---
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
    all_checkups = crud.get_all_checkups(db)
    db.close()
    
    # Convert SQLAlchemy objects to Dict/DataFrame via Pydantic schema or direct dict
    data_list = []
    for c in all_checkups:
        # Simple dict conversion
        d = {k: v for k, v in c.__dict__.items() if not k.startswith('_')}
        data_list.append(d)
        
    df = pd.DataFrame(data_list)
    
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
                    fig = px.line(df_hist, x='created_at', y='probability', title="Tren Risiko")
                    st.plotly_chart(fig, use_container_width=True)
                
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
                            st.markdown("### 📊 Hasil Analisis Saat Ini")
                            
                            # Visual Results
                            r1, r2 = st.columns(2)
                            with r1:
                                # Gauge-like display using metric and progress
                                risk_color = "green" if res['risk_category'] == "Rendah" else "orange" if res['risk_category'] == "Sedang" else "red"
                                st.metric("Probabilitas Risiko", f"{res['probability']:.1%}", res['risk_category'])
                                st.progress(res['probability'])
                                st.caption(f"Kategori: **{res['risk_category']}**")
                                
                            with r2:
                                # Recommendations
                                st.markdown("#### Rekomendasi Medis")
                                st.info(res['recommendations'])
                            
                            # SHAP if available
                            if res.get('shap_values'):
                                try:
                                    shap_data = json.loads(res['shap_values'])
                                    if shap_data:
                                        st.markdown("#### Faktor Kontribusi Utama")
                                        # Sort by absolute impact
                                        sorted_shap = sorted(shap_data.items(), key=lambda x: abs(x[1]), reverse=True)
                                        # Take top 3
                                        for k, v in sorted_shap:
                                            kn = k.replace("age_years", "Usia").replace("bmi", "BMI").replace("map", "Tekanan Darah")
                                            impact = "Meningkatkan Risiko ⬆️" if v > 0 else "Menurunkan Risiko ⬇️"
                                            color = "red" if v > 0 else "green"
                                            st.markdown(f"- **{kn}**: :{color}[{impact}]")
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
