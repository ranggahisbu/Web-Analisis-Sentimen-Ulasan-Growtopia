import streamlit as st
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# === Load model pipeline yang sudah dilatih (gabungan TF-IDF + Random Forest) ===
model = joblib.load('model_random_forest.pkl')

# === Fungsi Preprocessing Sederhana ===
def preprocess(text):
    # Lowercase
    text = text.lower()
    # Hapus karakter selain huruf dan spasi
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenisasi + stopword removal + stemming
    stemmer = StemmerFactory().create_stemmer()
    stopwords = StopWordRemoverFactory().get_stop_words()
    tokens = text.split()
    filtered = [stemmer.stem(w) for w in tokens if w not in stopwords]
    return ' '.join(filtered)

# === Streamlit UI ===
st.image("growtopia_logo.png", width=150)
st.title("Klasifikasi Sentimen Komentar Growtopia")
st.markdown("Masukkan komentar pengguna untuk melihat sentimennya (positif, atau negatif).")

input_text = st.text_area("Komentar Pengguna:")

if st.button("Prediksi Sentimen"):
    if not input_text.strip():
        st.warning("Komentar tidak boleh kosong.")
    else:
        cleaned = preprocess(input_text)
        prediction = model.predict([cleaned])[0]
        st.success(f"Prediksi Sentimen: **{prediction.capitalize()}**")


        label_map = {
            2: "Negatif",
            3: "Netral",
            5: "Positif"
        }
