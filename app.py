import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import io

# Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi Listrik", layout="wide")

st.title("⚡ Analisis & Klasifikasi Penggunaan Listrik")
st.write("Aplikasi ini menggunakan Machine Learning yang dilatih langsung dengan data asli Anda untuk akurasi prediksi yang lebih baik.")

# ==========================================
# PANTAUAN KOLOM (Sesuaikan dengan file Excel Anda)
# ==========================================
# CATATAN: Sesuaikan nama kolom di bawah ini agar sama persis dengan yang ada di file Excel Anda.
KOLOM_WAJIB = ['household_size', 'occupancy_count', 'power_watts', 'duration_minutes']

# ==========================================
# FUNGSI UNTUK MELATIH MODEL DENGAN DATA
# ==========================================
def train_model_on_data(df_input):
    df = df_input.copy()
    
    # Hitung usage_score berdasarkan rumus data asli
    if 'usage_score' not in df.columns:
        df['usage_score'] = df['power_watts'] * df['duration_minutes'] * df['occupancy_count']
        
    # Menghitung batas kuartil langsung dari distribusi DATA ASLI
    q1 = df['usage_score'].quantile(0.33)
    q2 = df['usage_score'].quantile(0.66)
    
    def get_kategori(score):
        if score <= q1: return "Rendah"
        elif score <= q2: return "Normal"
        else: return "Boros"
        
    df['kategori'] = df['usage_score'].apply(get_kategori)
    
    X = df[KOLOM_WAJIB]
    y = df['kategori']
    
    # Latih model Logistic Regression dengan data ini
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X, y)
    
    return model, q1, q2

# ==========================================
# INISIALISASI MODEL DI SESSION STATE
# ==========================================
# Membuat model acak awal agar aplikasi tidak error saat pertama kali dibuka (sebelum upload file)
if 'model' not in st.session_state:
    np.random.seed(42)
    df_dummy_awal = pd.DataFrame({
        'household_size': np.random.randint(1, 6, 200),
        'occupancy_count': np.random.randint(1, 6, 200),
        'power_watts': np.random.uniform(100, 3000, 200),
        'duration_minutes': np.random.randint(10, 180, 200)
    })
    model_awal, q1_awal, q2_awal = train_model_on_data(df_dummy_awal)
    st.session_state['model'] = model_awal
    st.session_state['is_using_real_data'] = False

# ==========================================
# MEMBUAT 2 MODE DENGAN TABS
# ==========================================
tab1, tab2 = st.tabs(["📝 Mode Input Manual", "📂 Mode Upload Excel"])

# ------------------------------------------
# TAB 1: INPUT MANUAL
# ------------------------------------------
with tab1:
    st.subheader("Prediksi Penggunaan Listrik (Satu Data)")
    
    if st.session_state['is_using_real_data']:
        st.success("🟢 Mode ini sekarang menggunakan model pintar yang telah dilatih dengan **Data Excel Asli** Anda.")
    else:
        st.info("ℹ️ Mode ini saat ini menggunakan model default. Silakan unggah file Excel di Tab 2 untuk menyesuaikan model dengan data asli Anda.")
    
    with st.form("form_manual"):
        col1, col2 = st.columns(2)
        
        with col1:
            household_size = st.number_input("Household Size (Jumlah Anggota Keluarga)", min_value=1, value=4)
            occupancy_count = st.number_input("Occupancy Count (Jumlah Orang di Rumah)", min_value=1, value=4)
        
        with col2:
            power_watts = st.number_input("Power Watts (Daya Listrik dalam Watt)", min_value=0.0, value=150.0)
            duration_minutes = st.number_input("Duration Minutes (Durasi Pemakaian - Menit)", min_value=1, value=60)
            
        submitted = st.form_submit_button("Prediksi Sekarang")
        
    if submitted:
        usage_score = power_watts * duration_minutes * occupancy_count
        
        input_data = pd.DataFrame({
            'household_size': [household_size],
            'occupancy_count': [occupancy_count],
            'power_watts': [power_watts],
            'duration_minutes': [duration_minutes]
        })
        
        # Prediksi menggunakan model terbaru yang ada di session state
        prediksi = st.session_state['model'].predict(input_data)[0]
        
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
# TAB 2: UPLOAD EXCEL (Latih Otomatis & Bulk Predict)
# ------------------------------------------
with tab2:
    st.subheader("Prediksi Sekaligus (Banyak Data)")
    uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        
        # Validasi apakah kolom yang dibutuhkan ada di Excel
        if all(kolom in df.columns for kolom in KOLOM_WAJIB):
            
            # PROSES OPTIMAL: Otomatis melatih model menggunakan data dari Excel yang diunggah
            model_asli, q1_asli, q2_asli = train_model_on_data(df)
            
            # Update ke session state agar Tab 1 (Manual) juga ikut menggunakan model pintar ini
            st.session_state['model'] = model_asli
            st.session_state['is_using_real_data'] = True
            
            st.success(f"🎉 Berhasil! Model Machine Learning telah otomatis dilatih ulang menggunakan {len(df)} baris dari data asli Anda.")
            
            # Hitung ulang score dan lakukan prediksi bulk
            df['usage_score'] = df['power_watts'] * df['duration_minutes'] * df['occupancy_count']
            df['Prediksi_Kategori'] = st.session_state['model'].predict(df[KOLOM_WAJIB])
            
            st.write("**Hasil Prediksi Data Excel (Kategori kini terbagi rata & akurat):**")
            st.dataframe(df)
            
            # Tombol Download Hasil
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Hasil Prediksi')
            
            st.download_button(
                label="📥 Download Hasil Prediksi (Excel)",
                data=output.getvalue(),
                file_name="hasil_prediksi_listrik_asli.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error(f"File Excel harus memiliki kolom wajib: {', '.join(KOLOM_WAJIB)}")
            st.info("💡 **Tips Tambahan:** Berdasarkan tangkapan layar Anda, kolom Anda mungkin bernama berbeda (seperti `room_temp_c` atau `daily_cumulative_kwh`). Jika Anda ingin menggunakan kolom-kolom tersebut sebagai fitur training, silakan ubah daftar isi variabel `KOLOM_WAJIB` di bagian atas kode program ini.")
