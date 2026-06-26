import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import io

# Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi Listrik", layout="wide")

st.title("⚡ Analisis & Klasifikasi Penggunaan Listrik Rumah Tangga")
st.write("Aplikasi ini menganalisis pola penggunaan listrik dan mengklasifikasikannya menggunakan Logistic Regression.")

# 1. Upload File
uploaded_file = st.file_uploader("Upload file Excel (Data Penggunaan Listrik)", type=["xlsx"])

if uploaded_file is not None:
    # Membaca Data
    df = pd.read_excel(uploaded_file)
    st.success(f"File berhasil diunggah! Jumlah Data: {len(df)} baris.")

    # Menghitung Usage Score
    df['usage_score'] = df['power_watts'] * df['duration_minutes'] * df['occupancy_count']

    # Menentukan Kuartil (Q1 & Q2)
    q1 = df['usage_score'].quantile(0.33)
    q2 = df['usage_score'].quantile(0.66)

    # Fungsi Kategori
    def kategori(score):
        if score <= q1:
            return "Rendah"
        elif score <= q2:
            return "Normal"
        else:
            return "Boros"

    df['kategori'] = df['usage_score'].apply(kategori)

    # Membuat Tab untuk Navigasi UI
    tab1, tab2, tab3 = st.tabs(["📊 Eksplorasi Data", "🤖 Model Prediksi", "📈 Visualisasi & Analisis"])

    # ==========================
    # TAB 1: EKSPLORASI DATA
    # ==========================
    with tab1:
        st.subheader("Batas Kategori")
        st.write(f"- **Rendah**: <= {q1:,.2f} (Q1 33%)")
        st.write(f"- **Normal**: > {q1:,.2f} dan <= {q2:,.2f} (Q2 66%)")
        st.write(f"- **Boros**: > {q2:,.2f}")

        st.subheader("Jumlah per Kategori")
        st.dataframe(df['kategori'].value_counts().reset_index())

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("🔥 **Top 5 Paling Boros**")
            st.dataframe(df.sort_values(by='usage_score', ascending=False)[['household_id', 'power_watts', 'usage_score', 'kategori']].head(5))
        
        with col2:
            st.write("❄️ **Top 5 Paling Rendah**")
            st.dataframe(df.sort_values(by='usage_score', ascending=True)[['household_id', 'power_watts', 'usage_score', 'kategori']].head(5))

        with col3:
            st.write("⚖️ **Top 5 Paling Normal**")
            median_score = df['usage_score'].median()
            top_normal = df.assign(selisih=np.abs(df['usage_score'] - median_score)).sort_values(by='selisih').head(5)
            st.dataframe(top_normal[['household_id', 'power_watts', 'usage_score', 'kategori']])

    # ==========================
    # TAB 2: MODEL PREDIKSI
    # ==========================
    with tab2:
        st.subheader("Pelatihan Model Logistic Regression")
        
        # Penentuan Variabel
        X = df[['household_size', 'occupancy_count', 'power_watts', 'duration_minutes']]
        y = df['kategori']

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        
        st.write(f"**Data Training:** {len(X_train)} | **Data Testing:** {len(X_test)}")

        # Training Model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        prediksi = model.predict(X_test)

        # Evaluasi
        akurasi = accuracy_score(y_test, prediksi)
        st.metric(label="Akurasi Model", value=f"{akurasi*100:.2f}%")

        col_mat, col_rep = st.columns(2)
        with col_mat:
            st.text("Confusion Matrix:")
            st.code(confusion_matrix(y_test, prediksi))
        
        with col_rep:
            st.text("Classification Report:")
            st.code(classification_report(y_test, prediksi))

        # Tombol Download Excel
        st.subheader("Download Hasil")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Klasifikasi')
        
        st.download_button(
            label="📥 Download Hasil Klasifikasi (Excel)",
            data=output.getvalue(),
            file_name="hasil_klasifikasi_listrik.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # ==========================
    # TAB 3: VISUALISASI & ANALISIS
    # ==========================
    with tab3:
        st.subheader("Interpretasi Usage Score")
        st.info("`usage_score = power_watts × duration_minutes × occupancy_count`")

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.write("**Perbandingan Kategori**")
            summary = df['kategori'].value_counts().reindex(['Rendah', 'Normal', 'Boros'])
            fig1, ax1 = plt.subplots()
            bars = ax1.bar(summary.index, summary.values, color=['green', 'orange', 'red'])
            ax1.set_ylabel('Jumlah Data')
            for bar in bars:
                tinggi = bar.get_height()
                ax1.text(bar.get_x() + 0.25, tinggi + 50, int(tinggi))
            st.pyplot(fig1)

        with col_chart2:
            st.write("**Pengaruh Variabel Terhadap Usage Score**")
            corr = df[['power_watts', 'duration_minutes', 'occupancy_count', 'usage_score']].corr()
            pengaruh = corr['usage_score'].drop('usage_score').sort_values(ascending=False)
            
            fig2, ax2 = plt.subplots()
            ax2.bar(pengaruh.index, pengaruh.values, color='purple')
            ax2.set_ylabel('Korelasi')
            st.pyplot(fig2)

        st.write("**Distribusi Usage Score**")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.hist(df['usage_score'], bins=30, color='skyblue', edgecolor='black')
        ax3.axvline(q1, linestyle='--', color='green', label=f'Q1 (Rendah batas)')
        ax3.axvline(q2, linestyle='--', color='red', label=f'Q2 (Normal batas)')
        ax3.legend()
        st.pyplot(fig3)

        terbesar = pengaruh.idxmax()
        st.success(f"**Kesimpulan:** Variabel yang paling berkorelasi positif dan berpengaruh terhadap `usage_score` adalah **{terbesar}**.")
