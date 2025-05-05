import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load custom CSS for styling 
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('./Supervised_Learning/Real_project/Spam_detection/xgboost_Spam_email_model.pkl')
    return model

@st.cache_data
def load_vectorizer():
    df = pd.read_csv("./Supervised_Learning/Real_project/Spam_detection/emails.csv")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text'])
    return vectorizer

# Set up UI
st.set_page_config(page_title="📧 Spam Email Detector", page_icon="📧", layout="centered")

st.title("📬 Spam Email Detector 🔍")
st.markdown("#### Detect whether an email is *Spam* or *Ham* using a trained **XGBoost** model.")

# Sidebar
st.sidebar.header("📝 About This App")
st.sidebar.info(
    "This app uses an **XGBoost Classifier** to detect spam emails.\n\n"
    "Built using:\n"
    "- Streamlit\n"
    "- Scikit-learn\n"
    "- XGBoost\n"
    "- CountVectorizer"
)

# Load model and vectorizer
model = load_model()
vectorizer = load_vectorizer()

# Input box
email_input = st.text_area("✉️ Enter the email text below:", height=200, placeholder="Type or paste your email content here...")

if st.button("🔍 Predict"):
    if email_input.strip() == "":
        st.warning("⚠️ Please enter some email text to analyze.")
    else:
        # Transform input
        email_vec = vectorizer.transform([email_input])
        prediction = model.predict(email_vec)[0]
        proba = model.predict_proba(email_vec)[0]

        # Decode prediction
        label = "Spam" if prediction == 1 else "Ham"
        spam_confidence = proba[1] * 100
        ham_confidence = proba[0] * 100

        # Display result
        st.markdown("### 🧾 Prediction Result:")
        if label == "Spam":
            st.error(f"🚨 The email is classified as: **{label}**")
            st.progress(int(spam_confidence))
            st.write(f"**Spam Confidence:** {spam_confidence:.2f}%")
        else:
            st.success(f"✅ The email is classified as: **{label}**")
            st.progress(int(ham_confidence))
            st.write(f"**Ham Confidence:** {ham_confidence:.2f}%")
else:
    st.info("📝 Type or paste an email above and click **Predict** to see results.")

# Footer
st.markdown("---")
st.markdown("Built by Deep Knowledge")