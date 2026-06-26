import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import io

# Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi Listrik", layout="wide")

st.title("⚡ Analisis & Klasifikasi Penggunaan Listrik")
st.write("Aplikasi ini dapat memprediksi kategori penggunaan listrik melalui input manual atau upload file Excel.")

# ==========================================
# 1. SIMULASI TRAINING MODEL (Di balik layar)
# ==========================================
# Agar aplikasi bisa memprediksi input manual, model harus dilatih terlebih dahulu.
# Dalam praktiknya, kamu bisa me-load model yang sudah di-save (misal menggunakan joblib/pickle).
# Di sini, kita buat dataset dummy kecil untuk melatih model Logistic Regression secara instan.
@st.cache_resource # Cache agar model tidak di-train ulang setiap kali ada interaksi
def train_model():
    # Membuat data dummy sederhana berdasarkan logika usage_score
    np.random.seed(42)
    df_dummy = pd.DataFrame({
        'household_size': np.random.randint(1, 6, 100),
        'occupancy_count': np.random.randint(1, 6, 100),
        'power_watts': np.random.uniform(500, 5000, 100),
        'duration_minutes': np.random.randint(10, 120, 100)
    })
    df_dummy['usage_score'] = df_dummy['power_watts'] * df_dummy['duration_minutes'] * df_dummy['occupancy_count']
    
    q1 = df_dummy['usage_score'].quantile(0.33)
    q2 = df_dummy['usage_score'].quantile(0.66)
    
    def get_kategori(score):
        if score <= q1: return "Rendah"
        elif score <= q2: return "Normal"
        else: return "Boros"
        
    df_dummy['kategori'] = df_dummy['usage_score'].apply(get_kategori)
    
    X = df_dummy[['household_size', 'occupancy_count', 'power_watts', 'duration_minutes']]
    y = df_dummy['kategori']
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, q1, q2

# Panggil fungsi training
model, q1, q2 = train_model()


# ==========================================
# 2. MEMBUAT 2 MODE DENGAN TABS
# ==========================================
tab1, tab2 = st.tabs(["📝 Mode Input Manual", "📂 Mode Upload Excel"])

# ------------------------------------------
# TAB 1: INPUT MANUAL
# ------------------------------------------
with tab1:
    st.subheader("Prediksi Penggunaan Listrik (Satu Data)")
    
    # Menggunakan form agar halaman tidak ter-refresh sebelum tombol Submit ditekan
    with st.form("form_manual"):
        col1, col2 = st.columns(2)
        
        with col1:
            household_size = st.number_input("Household Size (Jumlah Anggota Keluarga)", min_value=1, value=4)
            occupancy_count = st.number_input("Occupancy Count (Jumlah Orang di Rumah)", min_value=1, value=4)
        
        with col2:
            power_watts = st.number_input("Power Watts (Daya Listrik dalam Watt)", min_value=0.0, value=1500.0)
            duration_minutes = st.number_input("Duration Minutes (Durasi Pemakaian - Menit)", min_value=1, value=60)
            
        submitted = st.form_submit_button("Prediksi Sekarang")
        
    if submitted:
        # Menghitung usage score
        usage_score = power_watts * duration_minutes * occupancy_count
        
        # Format input untuk model
        input_data = pd.DataFrame({
            'household_size': [household_size],
            'occupancy_count': [occupancy_count],
            'power_watts': [power_watts],
            'duration_minutes': [duration_minutes]
        })
        
        # Prediksi menggunakan model
        prediksi = model.predict(input_data)[0]
        
        # Menampilkan Hasil
        st.write("---")
        st.subheader("Hasil Prediksi")
        st.write(f"**Usage Score:** {usage_score:,.2f}")
        
        if prediksi == "Boros":
            st.error(f"Kategori Penggunaan: **{prediksi}** 🔥")
        elif prediksi == "Normal":
            st.warning(f"Kategori Penggunaan: **{prediksi}** ⚖️")
        else:
            st.success(f"Kategori Penggunaan: **{prediksi}** ❄️")

# ------------------------------------------
# TAB 2: UPLOAD EXCEL (Data Banyak / Bulk)
# ------------------------------------------
with tab2:
    st.subheader("Prediksi Sekaligus (Banyak Data)")
    uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.success(f"File berhasil diunggah! Jumlah Data: {len(df)} baris.")
        
        # Cek apakah kolom yang dibutuhkan ada
        kolom_wajib = ['household_size', 'occupancy_count', 'power_watts', 'duration_minutes']
        if all(kolom in df.columns for kolom in kolom_wajib):
            
            # Hitung Usage Score
            df['usage_score'] = df['power_watts'] * df['duration_minutes'] * df['occupancy_count']
            
            # Prediksi
            X_baru = df[kolom_wajib]
            df['Prediksi_Kategori'] = model.predict(X_baru)
            
            st.write("**Hasil Prediksi Data Excel:**")
            st.dataframe(df)
            
            # Tombol Download Hasil
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Hasil Prediksi')
            
            st.download_button(
                label="📥 Download Hasil Prediksi (Excel)",
                data=output.getvalue(),
                file_name="hasil_prediksi_listrik.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error(f"File Excel harus memiliki kolom: {', '.join(kolom_wajib)}")
