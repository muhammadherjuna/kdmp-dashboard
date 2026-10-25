import os
import re
import joblib
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# Konfigurasi Halaman Web Akademik (Bebas Elemen Dekoratif / Emoji)
st.set_page_config(
    page_title="Sistem Pendukung Keputusan Analisis Sentimen KDMP",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

REPO_ID = "herjunamhd/kdmp-sentiment-models"
ID_TO_LABEL = {0: "Negatif", 1: "Netral", 2: "Positif"}

# Fungsi Pra-pemrosesan Ringan untuk Inferensi Langsung
def bersihkan_teks_input(teks):
    teks = str(teks).lower()
    teks = re.sub(r'http\S+|www\.\S+', ' ', teks)
    teks = re.sub(r'@\w+|#\w+', ' ', teks)
    teks = re.sub(r'[^a-z0-9\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

# Fungsi Pemuatan Tabel & Gambar Evaluasi dengan Cache
@st.cache_data(show_spinner=False)
def muat_artefak_evaluasi():
    bench_path = hf_hub_download(repo_id=REPO_ID, filename="benchmark_metrics_v2.csv")
    sub_path = hf_hub_download(repo_id=REPO_ID, filename="subgroup_error_analysis_v2.csv")
    cm_path = hf_hub_download(repo_id=REPO_ID, filename="confusion_matrices_comparison_v2.png")
    yt_test_path = hf_hub_download(repo_id=REPO_ID, filename="test_predictions_youtube_v2.csv")
    tt_test_path = hf_hub_download(repo_id=REPO_ID, filename="test_predictions_tiktok_v2.csv")

    df_bench = pd.read_csv(bench_path)
    df_sub = pd.read_csv(sub_path)
    img_cm = Image.open(cm_path)
    df_test_yt = pd.read_csv(yt_test_path)
    df_test_tt = pd.read_csv(tt_test_path)

    return df_bench, df_sub, img_cm, df_test_yt, df_test_tt

# Fungsi Pemuatan Model Machine Learning & Deep Learning dengan Cache
@st.cache_resource(show_spinner=False)
def muat_model_prediksi(platform):
    kode = "youtube" if platform == "YouTube" else "tiktok"

    # 1. Muat Linear SVM & TF-IDF Vectorizer
    svm_path = hf_hub_download(repo_id=REPO_ID, filename=f"svm_model_{kode}_v2.pkl")
    vec_path = hf_hub_download(repo_id=REPO_ID, filename=f"tfidf_vectorizer_{kode}_v2.pkl")
    svm_model = joblib.load(svm_path)
    tfidf_vec = joblib.load(vec_path)

    # 2. Muat IndoBERT & Tokenizer dari Subfolder Repositori
    subfolder_bert = f"indobert_{kode}_v2"
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, subfolder=subfolder_bert)
    bert_model = AutoModelForSequenceClassification.from_pretrained(REPO_ID, subfolder=subfolder_bert)
    bert_model.eval()

    return svm_model, tfidf_vec, tokenizer, bert_model

# Header Utama Aplikasi
st.title("Dashboard Analisis Opini Publik & Purwarupa Sistem Pendukung Keputusan")
st.markdown("**Evaluasi Kebijakan Koperasi Desa Merah Putih (KDMP) Lintas-Platform (YouTube vs. TikTok)**")
st.markdown("---")

# Navigasi Sidebar
st.sidebar.subheader("Panel Navigasi Sistem")
menu = st.sidebar.radio(
    "Pilih Modul Analisis:",
    (
        "1. Ringkasan Eksekutif",
        "2. Eksplorasi Topik Utama",
        "3. Komparasi Kinerja Model",
        "4. Uji Prediksi Langsung"
    )
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Informasi Sistem:**\n"
    "- Korpus Setara: 20.701 / Platform\n"
    "- Data Uji Terkunci: 4.141 / Platform\n"
    "- Arsitektur Uji: Linear SVM vs. IndoBERT"
)

# Pemuatan data evaluasi dasar
try:
    df_bench, df_sub, img_cm, df_test_yt, df_test_tt = muat_artefak_evaluasi()
except Exception as e:
    st.error(f"Gagal memuat artefak evaluasi dari repositori cloud: {e}")
    st.stop()

# ------------------------------------------------------------------------------
# MODUL 1: RINGKASAN EKSEKUTIF
# ------------------------------------------------------------------------------
if menu == "1. Ringkasan Eksekutif":
    st.header("1. Ringkasan Eksekutif Distribusi Sentimen Lintas-Platform")
    st.markdown(
        "Modul ini menyajikan ringkasan kuantitatif distribusi polaritas opini masyarakat "
        "(*Negatif, Netral, Positif*) terhadap kebijakan Koperasi Desa Merah Putih (KDMP) "
        "pada korpus YouTube dan TikTok yang telah disetarakan."
    )

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total Korpus Gabungan", "41.402 Komentar")
    col_m2.metric("Sampel Setara YouTube", "20.701 Komentar")
    col_m3.metric("Sampel Setara TikTok", "20.701 Komentar")

    st.markdown("---")

    data_distribusi = pd.DataFrame([
        {"Platform": "YouTube", "Sentimen": "Negatif", "Jumlah": 10656, "Persentase (%)": 51.48},
        {"Platform": "YouTube", "Sentimen": "Netral", "Jumlah": 2245, "Persentase (%)": 10.84},
        {"Platform": "YouTube", "Sentimen": "Positif", "Jumlah": 7800, "Persentase (%)": 37.68},
        {"Platform": "TikTok", "Sentimen": "Negatif", "Jumlah": 8912, "Persentase (%)": 43.05},
        {"Platform": "TikTok", "Sentimen": "Netral", "Jumlah": 3151, "Persentase (%)": 15.22},
        {"Platform": "TikTok", "Sentimen": "Positif", "Jumlah": 8638, "Persentase (%)": 41.73}
    ])

    col_kiri, col_kanan = st.columns([1.2, 1])
    with col_kiri:
        st.subheader("Grafik Distribusi Sentimen per Platform")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        sns.barplot(
            data=data_distribusi,
            x="Platform",
            y="Jumlah",
            hue="Sentimen",
            palette={"Negatif": "#c0392b", "Netral": "#7f8c8d", "Positif": "#27ae60"},
            ax=ax
        )
        ax.set_title("Perbandingan Jumlah Opini Publik KDMP (YouTube vs. TikTok)", fontweight="bold")
        ax.set_xlabel("Platform Media Sosial")
        ax.set_ylabel("Jumlah Observasi Komentar")
        st.pyplot(fig)

    with col_kanan:
        st.subheader("Tabel Proporsi Korpus Setara")
        st.dataframe(data_distribusi, use_container_width=True, hide_index=True)
        st.markdown(
            "**Temuan Utama:**\n"
            "1. **YouTube** didominasi oleh diskursus kritis/skeptis (*Negatif: 51,48%*), "
            "yang umumnya menyoroti aspek pengawasan dana dan tata kelola koperasi.\n"
            "2. **TikTok** memiliki distribusi yang lebih berimbang antara sentimen "
            "*Negatif (43,05%)* dan *Positif (41,73%)*, dengan proporsi kelas *Netral (15,22%)* yang lebih tinggi."
        )

    st.markdown("---")
    st.subheader("Inspeksi Sampel Data Uji Terstandarisasi")
    col_f1, col_f2 = st.columns(2)
    pilih_plat_inspeksi = col_f1.selectbox("Pilih Platform Data Uji:", ("YouTube", "TikTok"))
    pilih_sent_inspeksi = col_f2.selectbox("Filter Label Sentimen:", ("Semua", "Negatif", "Netral", "Positif"))

    df_inspeksi = df_test_yt.copy() if pilih_plat_inspeksi == "YouTube" else df_test_tt.copy()
    if pilih_sent_inspeksi != "Semua":
        df_inspeksi = df_inspeksi[df_inspeksi["sentimen"] == pilih_sent_inspeksi]

    df_tampil = df_inspeksi[["text_bert", "tipe_komentar", "sentimen"]].head(15).rename(
        columns={"text_bert": "Teks Komentar (Normalisasi)", "tipe_komentar": "Tipe Komentar", "sentimen": "Label Aktual"}
    )
    st.dataframe(df_tampil, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# MODUL 2: EKSPLORASI TOPIK UTAMA
# ------------------------------------------------------------------------------
elif menu == "2. Eksplorasi Topik Utama":
    st.header("2. Eksplorasi Topik Utama & Pemetaan Kata Kunci")
    st.markdown(
        "Modul ini mengekstraksi frekuensi kemunculan istilah (kata kunci) paling dominan "
        "pada masing-masing platform dan kelas sentimen untuk memetakan fokus perhatian publik."
    )

    col_t1, col_t2 = st.columns(2)
    plat_topik = col_t1.selectbox("Pilih Platform Analisis Topik:", ("YouTube", "TikTok"))
    sent_topik = col_t2.selectbox("Pilih Kategori Sentimen:", ("Negatif", "Netral", "Positif"))

    df_sumber_topik = df_test_yt if plat_topik == "YouTube" else df_test_tt
    df_sub_topik = df_sumber_topik[df_sumber_topik["sentimen"] == sent_topik]

    # Ekstraksi kata dari kolom text_svm (telah bersih dari stopword umum)
    semua_kata = " ".join(df_sub_topik["text_svm"].dropna().astype(str)).split()
    kata_terfilter = [k for k in semua_kata if len(k) > 2]
    hitung_kata = Counter(kata_terfilter).most_common(15)
    df_top_kata = pd.DataFrame(hitung_kata, columns=["Kata Kunci", "Frekuensi Kemunculan"])

    col_grafik, col_tabel = st.columns([1.3, 1])
    with col_grafik:
        st.subheader(f"15 Kata Kunci Dominan — {plat_topik} ({sent_topik})")
        fig_top, ax_top = plt.subplots(figsize=(8, 5))
        warna_bar = "#c0392b" if sent_topik == "Negatif" else ("#7f8c8d" if sent_topik == "Netral" else "#27ae60")
        sns.barplot(
            data=df_top_kata,
            x="Frekuensi Kemunculan",
            y="Kata Kunci",
            color=warna_bar,
            ax=ax_top
        )
        ax_top.set_title(f"Distribusi Frekuensi Leksikal ({plat_topik} - {sent_topik})", fontweight="bold")
        ax_top.set_xlabel("Frekuensi pada Korpus Uji")
        ax_top.set_ylabel("Kata Kunci")
        st.pyplot(fig_top)

    with col_tabel:
        st.subheader("Tabel Frekuensi Kata Kunci")
        st.dataframe(df_top_kata, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader(f"Contoh Komentar Aktual Kategori {sent_topik} ({plat_topik})")
    sampel_komentar = df_sub_topik[["text_bert", "tipe_komentar"]].head(10).rename(
        columns={"text_bert": "Teks Komentar", "tipe_komentar": "Struktur Diskusi"}
    )
    st.dataframe(sampel_komentar, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# MODUL 3: KOMPARASI KINERJA MODEL
# ------------------------------------------------------------------------------
elif menu == "3. Komparasi Kinerja Model":
    st.header("3. Transparansi Komparasi Kinerja Model (Model Benchmark)")
    st.markdown(
        "Modul ini menampilkan hasil pengujian komparatif secara *apple-to-apple* antara "
        "model berbasis frekuensi leksikal (**Linear SVM TF-IDF**) dan model kontekstual "
        "(*Deep Learning* **IndoBERT**) pada 20% data uji bersama (`4.141` observasi per platform)."
    )

    st.subheader("A. Tabel Perbandingan Metrik Evaluasi Utama")
    st.dataframe(df_bench, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("B. Evaluasi Ketahanan Sub-Kelompok (Lexical Brevity & Nested Replies)")
    st.markdown(
        "Pengujian spesifik pada komentar pendek (`< 8 kata`) dan komentar balasan berjenjang "
        "(`tipe_komentar == 'balasan'`) untuk membuktikan keunggulan mekanisme *self-attention* IndoBERT."
    )
    st.dataframe(df_sub, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("C. Perbandingan Matriks Konfusi (Confusion Matrix)")
    st.image(img_cm, caption="Perbandingan Matriks Konfusi Linear SVM vs. IndoBERT pada Data Uji YouTube dan TikTok", use_container_width=True)

# ------------------------------------------------------------------------------
# MODUL 4: UJI PREDIKSI LANGSUNG (LIVE INFERENCE)
# ------------------------------------------------------------------------------
elif menu == "4. Uji Prediksi Langsung":
    st.header("4. Modul Simulasi Uji Prediksi Komentar Baru (Live Inference)")
    st.markdown(
        "Modul ini memungkinkan pengujian kalimat opini secara *real-time* untuk membandingkan "
        "keputusan klasifikasi antara **IndoBERT** dan **Linear SVM** secara langsung."
    )

    col_p1, col_p2 = st.columns(2)
    selected_platform = col_p1.selectbox("Pilih Model Platform Konteks:", ("YouTube", "TikTok"))
    contoh_pilihan = col_p2.selectbox(
        "Pilih Contoh Kalimat Uji Cepat (Opsional):",
        (
            "Input Manual Bebas",
            "Contoh Positif: Koperasi desa merah putih sangat membantu permodalan petani kecil di daerah.",
            "Contoh Negatif: Jangan sampai program koperasi desa ini cuma jadi ladang korupsi oknum aparat desa.",
            "Contoh Netral: Bagaimana mekanisme pendaftaran dan syarat menjadi anggota koperasi desa di wilayah Jawa Tengah?"
        )
    )

    teks_awal = "Koperasi desa merah putih sangat membantu permodalan petani kecil di daerah."
    if contoh_pilihan.startswith("Contoh Positif:"):
        teks_awal = contoh_pilihan.replace("Contoh Positif: ", "")
    elif contoh_pilihan.startswith("Contoh Negatif:"):
        teks_awal = contoh_pilihan.replace("Contoh Negatif: ", "")
    elif contoh_pilihan.startswith("Contoh Netral:"):
        teks_awal = contoh_pilihan.replace("Contoh Netral: ", "")

    user_input = st.text_area("Masukkan Teks Komentar Opini Publik:", value=teks_awal, height=110)

    if st.button("Jalankan Inferensi Model"):
        if not user_input.strip():
            st.warning("Peringatan: Kolom masukan teks tidak boleh kosong.")
        else:
            with st.spinner(f"Memuat model {selected_platform} dan menghitung inferensi..."):
                svm_model, tfidf_vec, tokenizer, bert_model = muat_model_prediksi(selected_platform)
                teks_bersih = bersihkan_teks_input(user_input)

                # 1. Inferensi IndoBERT
                inputs = tokenizer(
                    teks_bersih,
                    truncation=True,
                    max_length=128,
                    padding="max_length",
                    return_tensors="pt"
                )
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().numpy()

                pred_id_bert = int(np.argmax(probs))
                label_bert = ID_TO_LABEL[pred_id_bert]
                conf_bert = float(probs[pred_id_bert]) * 100

                # 2. Inferensi Linear SVM
                vec_input = tfidf_vec.transform([teks_bersih])
                pred_id_svm = int(svm_model.predict(vec_input)[0])
                label_svm = ID_TO_LABEL[pred_id_svm]
                decision_scores = svm_model.decision_function(vec_input).squeeze()

            st.markdown("---")
            st.subheader(f"Hasil Inferensi Komparatif (Konteks Platform: {selected_platform})")
            st.markdown(f"**Teks Ternormalisasi:** `{teks_bersih}`")

            col_res1, col_res2 = st.columns(2)

            with col_res1:
                st.markdown("#### Model Kontekstual: IndoBERT")
                st.metric("Keputusan Sentimen (IndoBERT)", label_bert, f"Confidence: {conf_bert:.2f}%")
                df_prob_bert = pd.DataFrame({
                    "Kelas Sentimen": ["Negatif", "Netral", "Positif"],
                    "Probabilitas (%)": np.round(probs * 100, 2)
                })
                st.dataframe(df_prob_bert, use_container_width=True, hide_index=True)

            with col_res2:
                st.markdown("#### Model Leksikal: Linear SVM (TF-IDF)")
                st.metric("Keputusan Sentimen (Linear SVM)", label_svm)
                df_margin_svm = pd.DataFrame({
                    "Kelas Sentimen": ["Negatif", "Netral", "Positif"],
                    "Skor Margin Hyperplane": np.round(decision_scores, 4)
                })
                st.dataframe(df_margin_svm, use_container_width=True, hide_index=True)
