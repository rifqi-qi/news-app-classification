import streamlit as st
import joblib
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
from nltk.corpus import stopwords

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    # 1. Case Folding: Mengubah semua huruf menjadi huruf kecil
    text = text.lower()
    
    # 2. Menghilangkan angka dan karakter khusus
    text = re.sub(r'\d+', '', text)  # Menghapus angka
    text = re.sub(r'[^\w\s]', '', text)  # Menghapus tanda baca dan karakter khusus
    
    # 3. Tokenisasi: Memecah teks menjadi kata-kata menggunakan regex
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)

    # 4. Menghapus stopwords: kata-kata umum yang tidak membawa banyak informasi
    stop_words = set(stopwords.words('indonesian'))
    words = [word for word in words if word not in stop_words]
    
    # Menggabungkan kata-kata kembali menjadi satu kalimat
    processed_text = ' '.join(words)
    
    return processed_text

# Memuat model dan TF-IDF vectorizer yang telah dilatih
model = joblib.load('logistic_regression_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Judul aplikasi
st.title("Aplikasi Klasifikasi Berita")

# Input teks dari pengguna
st.subheader("Masukkan Berita yang Akan Diklasifikasikan")
user_input = st.text_area("Masukkan teks berita di bawah ini:")

# Tombol untuk memproses input
if st.button("Klasifikasikan"):
    if user_input:
        # Preprocessing input
        preprocessed_text = preprocess_text(user_input)

        # Transformasi teks menggunakan TF-IDF
        text_tfidf = tfidf.transform([preprocessed_text])

        # Melakukan prediksi
        prediction = model.predict(text_tfidf)
        predicted_category = "Kesehatan" if prediction[0] == "Kesehatan" else "Olahraga"

        # Menampilkan hasil prediksi
        st.write(f"Hasil Klasifikasi: **{predicted_category}**")
    else:
        st.write("Silakan masukkan teks berita terlebih dahulu.")
