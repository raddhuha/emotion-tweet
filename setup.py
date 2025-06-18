# setup.py - Script untuk setup environment dan download dependencies

import os
import subprocess
import sys
import nltk

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("✅ NLTK data downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False
    return True

def check_model_files():
    """Check if all required model files exist"""
    print("🔍 Checking model files...")
    required_files = [
        'saved_models/indobert_emotion_model.pth',
        'saved_models/word2vec_logistic_model.pkl',
        'saved_models/label_encoder.pkl',
        'saved_models/preprocessing_components.pkl',
        'saved_models/tokenizer/',
        'Word2Vec_400dim.txt',
        'kamus_singkatan.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n📝 Please ensure you have:")
        print("   1. Trained and saved your models using the export code")
        print("   2. Downloaded Word2Vec_400dim.txt file")
        print("   3. Created kamus_singkatan.csv file")
        return False
    else:
        print("✅ All required files found!")
        return True

def main():
    """Main setup function"""
    print("🚀 Setting up Gradio Sentiment Analysis App...")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        return
    
    # Download NLTK data
    if not download_nltk_data():
        return
    
    # Check model files
    if not check_model_files():
        return
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("🏃 You can now run: python gradio_app.py")
    print("🌐 Access the app at: http://localhost:7860")

if __name__ == "__main__":
    main()