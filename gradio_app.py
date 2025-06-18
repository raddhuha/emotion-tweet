import gradio as gr
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import joblib
import re
from transformers import BertTokenizer, BertModel
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import nltk

# Download NLTK data (hanya sekali)
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        print("✅ NLTK data downloaded!")

# Download NLTK data at startup
download_nltk_data()

# Load Word2Vec
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

# Global variable untuk cache models
LOADED_MODELS = None

# Load semua model dan komponen
def load_models():
    global LOADED_MODELS
    
    if LOADED_MODELS is not None:
        return LOADED_MODELS
    
    print("Loading models...")
    
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
    
    LOADED_MODELS = {
        'preprocessing': preprocessing,
        'label_encoder': le,
        'w2v_model': w2v_model,
        'word2vec': word2vec,
        'tokenizer': tokenizer,
        'indobert_model': indobert_model,
        'device': device,
        'max_len': checkpoint['max_len']
    }
    
    print("✅ Models loaded successfully!")
    return LOADED_MODELS

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

# Buat plot probabilitas
def create_probability_plot(probabilities, class_names, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(class_names, probabilities, color='skyblue', alpha=0.8)
    ax.set_xlabel('Probability')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    
    # Tambahkan nilai di ujung bar
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

# Fungsi utama prediksi
def analyze_emotion(text, model_choice):
    if not text.strip():
        return "⚠️ Masukkan teks untuk dianalisis", "", "", None, None
    
    try:
        models = load_models()
        
        # Preprocessing
        processed = preprocess_text(text, models)
        preprocessing_info = f"""
**Tahapan Preprocessing:**
- **Original:** {processed['original']}
- **Cleaned:** {processed['cleaned']}
- **Substituted:** {processed['substituted']}
- **Final:** {processed['final']}
        """
        
        le = models['label_encoder']
        class_names = list(le.classes_)
        
        if model_choice == "IndoBERT":
            pred_class, probabilities = predict_indobert(processed['final'], models)
            model_name = "IndoBERT"
        else:
            pred_class, probabilities = predict_w2v(processed['final'], models)
            model_name = "Word2Vec + Logistic Regression"
        
        predicted_emotion = le.inverse_transform([pred_class])[0]
        confidence = probabilities[pred_class]
        
        # Hasil prediksi
        result = f"""
## 🎯 Hasil Analisis Emosi ({model_name})

**Emosi Terdeteksi:** {predicted_emotion.title()}  
**Confidence Score:** {confidence:.4f} ({confidence*100:.2f}%)
        """
        
        # Buat plot
        prob_plot = create_probability_plot(probabilities, class_names, 
                                          f'Distribusi Probabilitas Emosi - {model_name}')
        
        return result, preprocessing_info, "✅ Analisis berhasil!", prob_plot, prob_plot
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", f"Error: {str(e)}", None, None

# Analisis batch dari file
def analyze_batch_file(file):
    if file is None:
        return "⚠️ Upload file CSV terlebih dahulu", None
    
    try:
        df = pd.read_csv(file.name)
        
        if 'text' not in df.columns:
            return "❌ File CSV harus memiliki kolom 'text'", None
        
        models = load_models()
        le = models['label_encoder']
        
        results = []
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
        
        results_df = pd.DataFrame(results)
        return f"✅ Berhasil menganalisis {len(results)} teks", results_df
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None

# Interface Gradio
def create_interface():
     # CSS custom untuk styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    """
    with gr.Blocks(css=css, title="🎭 Analisis Emosi Tweet Indonesia", theme=gr.themes.Soft()) as demo:
        
        gr.HTML("""
        <div class ="main-header">
            <h1>🎭 Analisis Emosi Tweet Indonesia </h1>

            <p>Aplikasi ini menganalisis emosi dari teks bahasa Indonesia menggunakan dua model:
            <p><b>IndoBERT</b>: Model transformer yang di-fine-tune khusus</p>
            <p><b>Word2Vec + Logistic Regression</b>: Model tradisional dengan embedding</p>
        </div>
        """)
        
        with gr.Tab("📝 Analisis Teks"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="💬 Masukkan Teks",
                        placeholder="Contoh: Hari ini aku merasa sangat bahagia sekali!",
                        lines=3
                    )
                    
                    model_choice = gr.Radio(
                        choices=["IndoBERT", "Word2Vec"],
                        value="IndoBERT",
                        label="🤖 Pilih Model"
                    )
                    
                    analyze_btn = gr.Button("🔍 Analisis Emosi", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
                    # Contoh teks
                    gr.Examples(
                        examples=[
                            ["Aku sangat senang hari ini!", "IndoBERT"],
                            ["Sedih banget rasanya ditinggal teman", "IndoBERT"],
                            ["Marah sekali sama kelakuan dia", "Word2Vec"],
                            ["Takut banget ngeliat film horror", "Word2Vec"],
                            ["Biasa aja sih hari ini", "IndoBERT"]
                        ],
                        inputs=[text_input, model_choice]
                    )
                
                with gr.Column(scale=2):
                    result_output = gr.Markdown(label="📊 Hasil Analisis")
                    status_output = gr.Textbox(label="Status", visible=False)
                    prob_plot = gr.Plot(label="📈 Grafik Probabilitas")
        
        with gr.Tab("🔧 Detail Preprocessing"):
            preprocessing_output = gr.Markdown(label="Tahapan Preprocessing")
        
        with gr.Tab("📁 Analisis Batch"):
            gr.Markdown("Upload file CSV dengan kolom 'text' untuk analisis batch")
            
            with gr.Row():
                file_input = gr.File(
                    label="📂 Upload File CSV",
                    file_types=[".csv"]
                )
                batch_btn = gr.Button("🔍 Analisis Batch", variant="primary")
            
            batch_status = gr.Textbox(label="Status")
            batch_results = gr.Dataframe(
                label="📊 Hasil Analisis Batch",    
                headers=["No", "Original Text", "Predicted Emotion", "Confidence"]
            )
        
        with gr.Tab("ℹ️ Info Model"):
            gr.Markdown("""
            ## 📈 Performa Model
            
            ### IndoBERT Model
            - **Akurasi Training**: ~99.9%
            - **Akurasi Validation**: ~71.0%
            - **Akurasi Test**: ~70.5%
            
            ### Word2Vec + Logistic Regression
            - **Akurasi Training**: ~73.6%
            - **Akurasi Validation**: ~61.2%
            - **Akurasi Test**: ~58.7%
            
            ## 🎯 Label Emosi
            Dataset mengandung emosi: **anger, fear, happy, love, sadness**
            
            ## 🔄 Preprocessing Steps
            1. Pembersihan teks (URL, mention, hashtag)
            2. Substitusi singkatan menggunakan kamus
            3. Lemmatization
            4. Tokenization
            """)
        
        # Event handlers
        analyze_btn.click(
            analyze_emotion,
            inputs=[text_input, model_choice],
            outputs=[result_output, preprocessing_output, status_output, prob_plot, prob_plot]
        )
        
        clear_btn.click(
            lambda: ("", "", "", None),
            outputs=[text_input, result_output, preprocessing_output, prob_plot]
        )
        
        batch_btn.click(
            analyze_batch_file,
            inputs=[file_input],
            outputs=[batch_status, batch_results]
        )
    
    return demo

# Jalankan aplikasi
if __name__ == "__main__":
    # Load models saat startup
    print("🔄 Initializing application...")
    try:
        models = load_models()
        print(f"✅ Models loaded successfully!")
        print(f"📊 Available emotions: {list(models['label_encoder'].classes_)}")
        print(f"🔧 Device: {models['device']}")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("⚠️  Please check if all model files exist in 'saved_models/' directory")
        exit(1)
    
    # Buat dan jalankan interface
    print("🚀 Starting Gradio interface...")
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",  # Localhost only untuk development
        server_port=7860,
        share=False,  # Set True jika ingin link publik
        debug=False,  # Set True untuk debug mode
        inbrowser=True  # Otomatis buka browser
    )