# Installation Guide

## Quick Start

### 1. Prerequisites
- Python 3.8 or higher
- OpenAI API key (get from https://platform.openai.com/)
- Ollama installed (get from https://ollama.com/)

### 2. Clone/Download Project
```bash
git clone <repository-url>
cd RAG-3
```

### 3. Automated Setup
```bash
python setup.py
```

### 4. Manual Setup (if automated fails)

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Configure Environment
```bash
# Copy environment template
cp env_example.txt .env

# Edit .env file and add your OpenAI API key
# OPENAI_API_KEY=your_actual_api_key_here
# OLLAMA_BASE_URL=http://localhost:11434
```

#### Install Ollama Model
```bash
# Install Ollama from https://ollama.com/
# Then pull the Llama2 model
ollama pull llama2

# Start Ollama service
ollama serve
```

### 5. Test Setup
```bash
python test_setup.py
```

### 6. Run Application
```bash
streamlit run app.py
```

## Troubleshooting

### Common Issues

**"OpenAI API key not found"**
- Make sure you've created a `.env` file
- Add your OpenAI API key to the `.env` file
- Ensure the key is valid and has credits

**"Ollama connection failed"**
- Install Ollama from https://ollama.com/
- Run `ollama serve` to start the service
- Run `ollama pull llama2` to download the model

**"Data files not found"**
- Ensure the `data/` directory exists
- Check that `nsw_crime_data.json` and `rag_documents.json` are present

**"Module not found"**
- Run `pip install -r requirements.txt`
- Make sure you're using Python 3.8+

### Getting Help

1. Run the test script: `python test_setup.py`
2. Check the error messages carefully
3. Verify all prerequisites are installed
4. Ensure API keys and services are configured correctly

