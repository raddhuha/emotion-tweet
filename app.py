import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import joblib
import re
from transformers import BertTokenizer, BertModel
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
import nltk
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Analisis Emosi Tweet Berbahasa Indonesia",
    page_icon="😡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Download NLTK data (hanya sekali)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        with st.spinner("Downloading NLTK data..."):
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
        st.success("✅ NLTK data downloaded!")

# Load Word2Vec
@st.cache_resource
def load_word2vec(path):
    word2vec = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype='float32')
            word2vec[word] = vector
    return word2vec

# Model IndoBERT Class
class IndoBertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(IndoBertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("indobenchmark/indobert-base-p1")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.pooler_output
        return self.classifier(self.dropout(pooled))

# Load semua model dan komponen
@st.cache_resource
def load_models():
    with st.spinner("Loading models..."):
        # Load preprocessing components
        with open('saved_models/preprocessing_components.pkl', 'rb') as f:
            preprocessing = pickle.load(f)
        
        # Load label encoder
        le = joblib.load('saved_models/label_encoder.pkl')
        
        # Load Word2Vec model
        w2v_model = joblib.load('saved_models/word2vec_logistic_model.pkl')
        
        # Load Word2Vec embeddings
        word2vec = load_word2vec("Word2Vec_400dim.txt")
        
        # Load IndoBERT components
        tokenizer = BertTokenizer.from_pretrained('saved_models/tokenizer')
        
        # Load IndoBERT model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load('saved_models/indobert_emotion_model.pth', map_location=device, weights_only=False)
        
        indobert_model = IndoBertClassifier(num_labels=checkpoint['num_classes'])
        indobert_model.load_state_dict(checkpoint['model_state_dict'])
        indobert_model.to(device)
        indobert_model.eval()
        
        models = {
            'preprocessing': preprocessing,
            'label_encoder': le,
            'w2v_model': w2v_model,
            'word2vec': word2vec,
            'tokenizer': tokenizer,
            'indobert_model': indobert_model,
            'device': device,
            'max_len': checkpoint['max_len']
        }
        
        st.success("✅ Models loaded successfully!")
        return models

# Fungsi preprocessing
def preprocess_text(text, components):
    kamus_dict = components['preprocessing']['kamus_dict']
    lemmatizer = components['preprocessing']['lemmatizer']
    
    # Bersihkan teks
    def bersihkan_teks(teks):
        teks = re.sub(r'\S*\.\S*\.\S*', '', teks)
        teks = re.sub(r'\.', ' ', teks)
        teks = re.sub(r'\s+', ' ', teks)
        teks = teks.strip().lower()
        teks = re.sub(r"http\S+|www\S+", "", teks)
        teks = re.sub(r"@\w+|#\w+", "", teks)
        teks = re.sub(r"[^a-z\s]", "", teks)
        return teks
    
    # Substitusi kamus
    def substitusi_kamus(teks):
        tokens = teks.split()
        tokens = [kamus_dict.get(token, token) for token in tokens]
        return ' '.join(tokens)
    
    # Lemmatization
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize_text(text):
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
        return ' '.join(lemmatized)
    
    # Proses step by step
    original = text
    cleaned = bersihkan_teks(text)
    substituted = substitusi_kamus(cleaned)
    lemmatized = lemmatize_text(substituted)
    
    return {
        'original': original,
        'cleaned': cleaned,  
        'substituted': substituted,
        'final': lemmatized
    }

# Prediksi dengan Word2Vec
def predict_w2v(text, models):
    def get_avg_embedding(text, word2vec, dim=400):
        words = text.split()
        vectors = [word2vec[w] for w in words if w in word2vec]
        if not vectors:
            return np.zeros(dim)
        return np.mean(vectors, axis=0)
    
    embedding = get_avg_embedding(text, models['word2vec'])
    embedding = embedding.reshape(1, -1)
    
    proba = models['w2v_model'].predict_proba(embedding)[0]
    pred_class = models['w2v_model'].predict(embedding)[0]
    
    return pred_class, proba

# Prediksi dengan IndoBERT
def predict_indobert(text, models):
    tokenizer = models['tokenizer']
    model = models['indobert_model']
    device = models['device']
    max_len = models['max_len']
    
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        proba = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class = torch.argmax(outputs, dim=1).cpu().numpy()[0]
    
    return pred_class, proba

# Buat plot probabilitas dengan Plotly
def create_probability_plot(probabilities, class_names, title):
    fig = go.Figure(data=[
        go.Bar(
            y=class_names,
            x=probabilities,
            orientation='h',
            marker=dict(
                color='rgba(58, 71, 80, 0.6)',
                line=dict(color='rgba(58, 71, 80, 1.0)', width=1)
            ),
            text=[f"{prob:.3f}" for prob in probabilities],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Probability",
        yaxis_title="Emotion",
        height=400,
        template="plotly_white"
    )
    
    return fig

# Main App
def main():
    # Download NLTK data
    download_nltk_data()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Analisis Emosi Tweet Berbahasa Indonesia</h1>
        <p>Aplikasi ini menganalisis emosi dari teks bahasa Indonesia menggunakan dua model:</p>
        <p><b>IndoBERT</b>: Model transformer yang di-fine-tune khusus</p>
        <p><b>Word2Vec + Logistic Regression</b>: Model tradisional dengan embedding</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    try:
        models = load_models()
        le = models['label_encoder']
        class_names = list(le.classes_)
        
        # Sidebar info
        with st.sidebar:
            st.header("📊 Model Info")
            st.write(f"**Device:** {models['device']}")
            st.write(f"**Available Emotions:** {', '.join(class_names)}")
            
            st.header("🎯 Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **IndoBERT**
                - Train: 99.9%
                - Val: 71.0%
                - Test: 70.5%
                """)
            with col2:
                st.markdown("""
                **Word2Vec**
                - Train: 73.6%
                - Val: 61.2%
                - Test: 58.7%
                """)
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Analisis Teks", "🔧 Detail Preprocessing", "📁 Analisis Batch", "ℹ️ Info Model"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💬 Input")
                text_input = st.text_area(
                    "Masukkan Teks",
                    placeholder="Contoh: Hari ini aku merasa sangat bahagia sekali!",
                    height=175
                )
                
                model_choice = st.radio(
                    "👊🏻 Pilih Model",
                    ["IndoBERT", "Word2Vec"],
                    index=0
                )
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    analyze_btn = st.button("🔍 Analisis Emosi", type="primary", use_container_width=True)
                with col_btn2:
                    clear_btn = st.button("🗑️ Clear", use_container_width=True)
                
                if clear_btn:
                    st.rerun()
                
                # Contoh teks
                st.subheader("💡 Contoh Teks")
                examples = [
                    "Aku sangat senang hari ini!",
                    "Sedih banget rasanya ditinggal teman",
                    "Marah sekali sama kelakuan dia",
                    "Takut banget ngeliat film horror",
                    "Biasa aja sih hari ini"
                ]
                
                selected_example = st.selectbox("Pilih contoh:", [""] + examples)
                if selected_example:
                    text_input = selected_example
                    st.experimental_rerun()
            
            with col2:
                st.subheader("📊 Hasil Analisis")
                
                if analyze_btn and text_input.strip():
                    try:
                        with st.spinner("Menganalisis emosi..."):
                            # Preprocessing
                            processed = preprocess_text(text_input, models)
                            
                            # Prediksi
                            if model_choice == "IndoBERT":
                                pred_class, probabilities = predict_indobert(processed['final'], models)
                                model_name = "IndoBERT"
                            else:
                                pred_class, probabilities = predict_w2v(processed['final'], models)
                                model_name = "Word2Vec + Logistic Regression"
                            
                            predicted_emotion = le.inverse_transform([pred_class])[0]
                            confidence = probabilities[pred_class]
                            
                            # Display results
                            st.success("✅ Analisis berhasil!")
                            
                            # Metrics
                            col_m1, col_m2 = st.columns(2)
                            with col_m1:
                                st.metric("Emosi Terdeteksi", predicted_emotion.title())
                            with col_m2:
                                st.metric("Confidence Score", f"{confidence:.4f} ({confidence*100:.2f}%)")
                            
                            # Plot
                            fig = create_probability_plot(probabilities, class_names, 
                                                        f'Distribusi Probabilitas Emosi - {model_name}')
                            # Ubah warna bar menjadi biru
                            fig.update_traces(marker_color='rgba(54, 162, 235, 0.7)', 
                                             marker_line_color='rgba(54, 162, 235, 1.0)')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Store results in session state for preprocessing tab
                            st.session_state['processed'] = processed
                            st.session_state['model_name'] = model_name
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                
                elif analyze_btn and not text_input.strip():
                    st.warning("⚠️ Masukkan teks untuk dianalisis")
        
        with tab2:
            st.subheader("🔧 Detail Preprocessing")
            
            if 'processed' in st.session_state:
                processed = st.session_state['processed']
                
                st.write("**Tahapan Preprocessing:**")
                
                steps = [
                    ("Original", processed['original']),
                    ("Cleaned", processed['cleaned']),
                    ("Substituted", processed['substituted']),
                    ("Final", processed['final'])
                ]
                
                for step, text in steps:
                    with st.expander(f"**{step}**"):
                        st.write(text)
            else:
                st.info("💡 Lakukan analisis teks terlebih dahulu untuk melihat detail preprocessing")
        
        with tab3:
            st.subheader("📁 Analisis Batch")
            st.write("Upload file CSV dengan kolom 'text' untuk analisis batch")
            
            uploaded_file = st.file_uploader("📂 Upload File CSV", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("Preview data:")
                    st.dataframe(df.head())
                    
                    if 'text' not in df.columns:
                        st.error("❌ File CSV harus memiliki kolom 'text'")
                    else:
                        if st.button("🔍 Analisis Batch", type="primary"):
                            with st.spinner("Menganalisis batch..."):
                                results = []
                                progress_bar = st.progress(0)
                                
                                for idx, text in enumerate(df['text']):
                                    try:
                                        processed = preprocess_text(str(text), models)
                                        pred_class, probabilities = predict_indobert(processed['final'], models)
                                        predicted_emotion = le.inverse_transform([pred_class])[0]
                                        confidence = probabilities[pred_class]
                                        
                                        results.append({
                                            'No': idx + 1,
                                            'Original Text': text,
                                            'Predicted Emotion': predicted_emotion,
                                            'Confidence': f"{confidence:.4f}"
                                        })
                                    except:
                                        results.append({
                                            'No': idx + 1,
                                            'Original Text': text,
                                            'Predicted Emotion': 'Error',
                                            'Confidence': '0.0000'
                                        })
                                    
                                    progress_bar.progress((idx + 1) / len(df))
                                
                                results_df = pd.DataFrame(results)
                                
                                st.success(f"✅ Berhasil menganalisis {len(results)} teks")
                                st.dataframe(results_df)
                                
                                # Download results
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Hasil",
                                    data=csv,
                                    file_name="emotion_analysis_results.csv",
                                    mime="text/csv"
                                )
                                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with tab4:
            st.subheader("📈 Informasi Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### IndoBERT Model")
                st.write("- **Akurasi Training**: ~99.9%")
                st.write("- **Akurasi Validation**: ~71.0%")
                st.write("- **Akurasi Test**: ~70.5%")
                st.write("")
                st.write("")

                st.write("### Label Emosi")
                st.write("Dataset mengandung emosi:")
                for emotion in class_names:
                    st.write(f"- **{emotion}**")
            
            with col2:
                st.write("### Word2Vec + Logistic Regression")
                st.write("- **Akurasi Training**: ~73.6%")
                st.write("- **Akurasi Validation**: ~61.2%")
                st.write("- **Akurasi Test**: ~58.7%")
                st.write("")
                st.write("")

                st.write("### 🔄 Preprocessing Steps")
                st.write("1. Pembersihan teks (URL, mention, hashtag)")
                st.write("2. Substitusi singkatan menggunakan kamus")
                st.write("3. Lemmatization")
                st.write("4. Tokenization")
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.error("⚠️ Please check if all model files exist in 'saved_models/' directory")

if __name__ == "__main__":
    main()
