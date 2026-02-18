import streamlit as st
import pickle
import re
import nltk
import os

# Download only if not already downloaded
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(os.path.join(nltk_data_path, "corpora/stopwords")):
    nltk.download('stopwords', quiet=True)
if not os.path.exists(os.path.join(nltk_data_path, "corpora/wordnet")):
    nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="🐦",
    layout="centered"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }

/* Background */
.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(29, 155, 240, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(120, 40, 200, 0.08) 0%, transparent 50%);
}

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 720px; }

/* Title */
.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #1d9bf0, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: -1px;
}

.sub-title {
    text-align: center;
    color: #6b7280;
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2.5rem;
    letter-spacing: 0.5px;
}

/* Card */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.5rem;
}

/* Text area */
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1.5px solid rgba(29, 155, 240, 0.3) !important;
    border-radius: 14px !important;
    color: #e5e7eb !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 1rem !important;
    transition: border-color 0.3s ease !important;
}
.stTextArea textarea:focus {
    border-color: #1d9bf0 !important;
    box-shadow: 0 0 0 3px rgba(29, 155, 240, 0.15) !important;
}
.stTextArea textarea::placeholder { color: #4b5563 !important; }
.stTextArea label { color: #9ca3af !important; font-size: 0.85rem !important; letter-spacing: 0.5px !important; }

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #1d9bf0, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(29, 155, 240, 0.35) !important;
}

/* Result boxes */
.result-positive {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05));
    border: 1.5px solid rgba(16, 185, 129, 0.4);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-top: 1.5rem;
    animation: fadeIn 0.5s ease;
}
.result-negative {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05));
    border: 1.5px solid rgba(239, 68, 68, 0.4);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-top: 1.5rem;
    animation: fadeIn 0.5s ease;
}
.result-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
.result-label-pos {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #10b981;
}
.result-label-neg {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #ef4444;
}
.result-sub { color: #6b7280; font-size: 0.9rem; margin-top: 0.3rem; }

/* Stats row */
.stats-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
}
.stat-box {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
}
.stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #1d9bf0;
}
.stat-label { color: #6b7280; font-size: 0.75rem; margin-top: 0.2rem; }

/* Cleaned tweet box */
.cleaned-box {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-top: 1rem;
    color: #4b5563;
    font-size: 0.82rem;
    font-style: italic;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = pickle.load(open('sentiment_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vectorizer

model, vectorizer = load_model()

@st.cache_resource
def load_nlp():
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return lemmatizer, stop_words

lemmatizer, stop_words = load_nlp()

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)


# ─── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🐦 Tweet Sentiment</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by Machine Learning · NLP · Twitter Data</div>', unsafe_allow_html=True)

# Stats
st.markdown("""
<div class="stats-row">
    <div class="stat-box">
        <div class="stat-number">78%</div>
        <div class="stat-label">Model Accuracy</div>
    </div>
    <div class="stat-box">
        <div class="stat-number">1.6M</div>
        <div class="stat-label">Tweets Trained</div>
    </div>
    <div class="stat-box">
        <div class="stat-number">2</div>
        <div class="stat-label">Sentiment Classes</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Input Card
st.markdown('<div class="card">', unsafe_allow_html=True)
tweet = st.text_area(
    "ENTER YOUR TWEET",
    placeholder="e.g. I love this product, it's amazing! 🚀",
    height=130
)
analyze = st.button("⚡ Analyze Sentiment")
st.markdown('</div>', unsafe_allow_html=True)

# Result
if analyze:
    if tweet.strip() == "":
        st.warning("⚠️ Please enter a tweet first!")
    else:
        cleaned = clean_tweet(tweet)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.markdown("""
            <div class="result-positive">
                <div class="result-emoji">😊</div>
                <div class="result-label-pos">Positive Tweet!</div>
                <div class="result-sub">This tweet expresses a positive sentiment</div>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown("""
            <div class="result-negative">
                <div class="result-emoji">😞</div>
                <div class="result-label-neg">Negative Tweet!</div>
                <div class="result-sub">This tweet expresses a negative sentiment</div>
            </div>
            """, unsafe_allow_html=True)

        # Show cleaned tweet
        st.markdown(f'<div class="cleaned-box">🔍 Processed text: "{cleaned}"</div>', unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div style="text-align:center; color:#374151; font-size:0.8rem;">Built with ❤️ using Streamlit · Scikit-learn · NLTK</div>', unsafe_allow_html=True)
