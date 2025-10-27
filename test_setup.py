import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

def test_imports():
    print("[TEST] Testing imports...")
    
    required_modules = [
        'streamlit',
        'langchain',
        'langchain_openai', 
        'langchain_community',
        'openai',
        'chromadb',
        'pandas',
        'requests',
        'tiktoken'
    ]
    
    failed_imports = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  [OK] {module}")
        except ImportError as e:
            print(f"  [ERROR] {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n[ERROR] Failed to import: {failed_imports}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("[OK] All imports successful")
    return True

def test_environment():
    print("\n[TEST] Testing environment...")
    
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    if not openai_key or openai_key == "your_openai_api_key_here":
        print("[ERROR] OpenAI API key not configured")
        print("Please set OPENAI_API_KEY in your .env file")
        return False
    else:
        print("[OK] OpenAI API key configured")
    
    print(f"[OK] Ollama URL: {ollama_url}")
    return True

def test_data_files():
    print("\n[TEST] Testing data files...")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("[ERROR] Data directory not found")
        return False
    
    crime_file = data_dir / "nsw_crime_data.json"
    if not crime_file.exists():
        print("[ERROR] nsw_crime_data.json not found")
        return False
    
    try:
        with open(crime_file, 'r') as f:
            crime_data = json.load(f)
        
        if 'crime_records' not in crime_data:
            print("[ERROR] Invalid crime data format")
            return False
        
        record_count = len(crime_data['crime_records'])
        print(f"[OK] Crime data: {record_count} records")
        
    except json.JSONDecodeError:
        print("[ERROR] Invalid JSON in crime data file")
        return False
    
    rag_file = data_dir / "rag_documents.json"
    if not rag_file.exists():
        print("[ERROR] rag_documents.json not found")
        return False
    
    try:
        with open(rag_file, 'r') as f:
            rag_docs = json.load(f)
        
        if not isinstance(rag_docs, list):
            print("[ERROR] Invalid RAG documents format")
            return False
        
        doc_count = len(rag_docs)
        print(f"[OK] RAG documents: {doc_count} documents")
        
    except json.JSONDecodeError:
        print("[ERROR] Invalid JSON in RAG documents file")
        return False
    
    return True

def test_ollama_connection():
    print("\n[TEST] Testing Ollama connection...")
    
    try:
        import requests
        
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            llama2_available = any('llama2' in model.get('name', '') for model in models)
            
            if llama2_available:
                print("[OK] Ollama connection successful")
                print("[OK] Llama2 model available")
                return True
            else:
                print("[OK] Ollama connection successful")
                print("[ERROR] Llama2 model not found")
                print("Run: ollama pull llama2")
                return False
        else:
            print(f"[ERROR] Ollama connection failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to Ollama")
        print("Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"[ERROR] Ollama test failed: {e}")
        return False

def test_rag_pipeline():
    print("\n[TEST] Testing RAG pipeline...")
    
    try:
        from rag_pipeline import NSWCrimeRAGPipeline
        
        pipeline = NSWCrimeRAGPipeline()
        print("[OK] RAG pipeline initialized")
        return True
        
    except Exception as e:
        print(f"[ERROR] RAG pipeline test failed: {e}")
        return False

def main(): 
    print("[TEST] NSW Crime Data RAG System - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Environment", test_environment),
        ("Data Files", test_data_files),
        ("Ollama Connection", test_ollama_connection),
        ("RAG Pipeline", test_rag_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("[SUMMARY] Test Results:")
    
    passed = 0
    for test_name, result in results:
        status = "[OK] PASS" if result else "[ERROR] FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n[SUCCESS] All tests passed! Your RAG system is ready to use.")
        print("Run: streamlit run app.py")
    else:
        print("\n[WARNING] Some tests failed. Please address the issues above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
