import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"[OK] Python version: {sys.version}")
    return True

def install_dependencies():
    print("\n[SETUP] Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[OK] Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install dependencies: {e}")
        return False

def setup_environment():
    print("\n[SETUP] Setting up environment...")
    
    env_file = Path(".env")
    env_example = Path("env_example.txt")
    
    if not env_file.exists() and env_example.exists():
        with open(env_example, 'r') as f:
            content = f.read()
        with open(env_file, 'w') as f:
            f.write(content)
        print("[OK] Created .env file from template")
        print("[WARNING] Please edit .env file and add your OpenAI API key")
    elif env_file.exists():
        print("[OK] .env file already exists")
    else:
        print("[ERROR] No environment template found")

def check_ollama():
    print("\n[SETUP] Checking Ollama...")
    
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK] Ollama is installed")
            
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if "llama2" in result.stdout:
                print("[OK] Llama2 model is available")
                return True
            else:
                print("[WARNING] Llama2 model not found")
                print("Run: ollama pull llama2")
                return False
        else:
            print("[ERROR] Ollama not found")
            print("Please install Ollama from: https://ollama.com/")
            return False
    except FileNotFoundError:
        print("[ERROR] Ollama not found")
        print("Please install Ollama from: https://ollama.com/")
        return False

def check_data_files():
    print("\n[SETUP] Checking data files...")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("[ERROR] Data directory not found")
        return False
    
    required_files = ["nsw_crime_data.json", "rag_documents.json"]
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            print(f"[OK] {file} found")
        else:
            print(f"[ERROR] {file} not found")
            return False
    
    return True

def main(): 
    print("[SETUP] NSW Crime Data RAG System Setup")
    print("=" * 50)
    
    if not check_python_version():
        return False
    
    if not install_dependencies():
        return False
    
    setup_environment()
    
    ollama_ok = check_ollama()
    
    data_ok = check_data_files()
    
    print("\n" + "=" * 50)
    print("[SUMMARY] Setup Summary:")
    print(f"[OK] Python dependencies: Installed")
    print(f"[OK] Environment file: Created")
    print(f"{'[OK]' if ollama_ok else '[ERROR]'} Ollama: {'Ready' if ollama_ok else 'Needs setup'}")
    print(f"{'[OK]' if data_ok else '[ERROR]'} Data files: {'Ready' if data_ok else 'Missing'}")
    
    if ollama_ok and data_ok:
        print("\n[SUCCESS] Setup complete! You can now run:")
        print("   streamlit run app.py")
    else:
        print("\n[WARNING] Setup incomplete. Please address the issues above.")
        if not ollama_ok:
            print("   1. Install Ollama from https://ollama.com/")
            print("   2. Run: ollama pull llama2")
            print("   3. Run: ollama serve")
        if not data_ok:
            print("   4. Ensure data files are present in data/ directory")
    
    return ollama_ok and data_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)