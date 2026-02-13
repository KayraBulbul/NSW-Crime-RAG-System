# NSW Crime Data RAG System

A comprehensive **Retrieval Augmented Generation (RAG)** application for analyzing NSW crime statistics using AI-powered natural language queries.

## Project Overview

This RAG system provides intelligent, conversational access to NSW Bureau of Crime Statistics and Research data, enabling users to ask complex questions about crime trends, regional comparisons, and statistical patterns.

## Architecture

### Technology Stack

- **LangChain**: RAG pipeline orchestration and document management
- **OpenAI**: Text embeddings (text-embedding-3-small) for semantic search
- **Ollama**: Local generative model (Llama2) for cost-effective inference
- **ChromaDB**: Vector database for efficient semantic search
- **Streamlit**: Interactive web interface
- **Python 3.13**: Modern Python features and performance

### System Components

1. **Data Layer**: Preprocessed NSW crime statistics with metadata
2. **Embedding Layer**: OpenAI embeddings for semantic understanding
3. **Vector Store**: ChromaDB for retrieval-optimized storage
4. **Generation Layer**: Ollama + Llama2 for local LLM inference
5. **Interface Layer**: Streamlit web application

## Features

- **Semantic Search**: Find relevant information based on meaning, not just keywords
- **Source Attribution**: Every answer includes source documents for transparency
- **Interactive Interface**: User-friendly web interface with filtering and visualization
- **Local Processing**: Uses Ollama for privacy and cost-effectiveness
- **Real-time Insights**: Instant answers to complex questions about crime data
- **Evaluation Framework**: Built-in metrics for system performance assessment

## Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/))
- Ollama installed ([Install here](https://ollama.com/))

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd RAG-3
   ```

2. **Automated Setup**

   ```bash
   python setup.py
   ```

3. **Manual Setup** (if automated fails)

   ```bash
   # Install dependencies
   pip install -r requirements.txt

   # Configure environment
   cp env_example.txt .env
   # Edit .env and add your OpenAI API key

   # Install Ollama model
   ollama pull llama2
   ollama serve
   ```

4. **Test Setup**
   ```bash
   python test_setup.py
   ```

### Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Data Sources

- **NSW Bureau of Crime Statistics and Research**: Official government crime data
- Regions covered: Greater Sydney, NSW Regional, New South Wales
- Crime types: Assault, Robbery, Break and enter, Motor vehicle theft, Drug offenses, and more
- Time period: 2015-2024 with trend analysis

## Usage Examples

### Sample Queries

- "What are the crime statistics for Greater Sydney?"
- "Compare crime rates between Greater Sydney and NSW Regional areas"
- "Which region has the highest assault rates?"
- "What are the most common types of crimes in NSW?"
- "How has domestic violence related assault changed over time?"
- "What crime trends have occurred from 2015 to 2024?"

### Key Features in Action

1. **Query Interface**: Ask natural language questions about crime data
2. **Data Explorer**: Browse raw data with filters and visualizations
3. **Source Documents**: View the exact sources used for each answer

## Evaluation Metrics

The system includes a comprehensive evaluation framework measuring:

- **Effectiveness**: Answer relevance to questions
- **Faithfulness**: Grounding in source documents
- **Source Attribution**: Accuracy of document references

Run the evaluation:

```bash
python evaluation_framework.py
```

## Development

### Key Features Implemented

- Document chunking and preprocessing
- Vector store creation and persistence
- Retrieval augmented generation pipeline
- Interactive query interface
- Source attribution
- Performance evaluation framework

## Business Value

This RAG solution demonstrates practical applications for:

- **Policy Makers**: Quick insights into crime trends and patterns
- **Researchers**: Efficient data exploration and analysis
- **Public**: Accessible crime statistics with natural language queries
- **Law Enforcement**: Rapid information retrieval for operational planning

## Troubleshooting

### Common Issues

**"OpenAI API key not found"**

- Create a `.env` file
- Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`

**"Ollama connection failed"**

- Install Ollama from https://ollama.com/
- Run `ollama pull llama2`
- Start service: `ollama serve`

**"Data files not found"**

- Ensure `data/` directory exists
- Check that `nsw_crime_data.json` and `rag_documents.json` are present

**"Module not found"**

- Run `pip install -r requirements.txt`
- Ensure Python 3.8+ is installed

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and share this project with proper attribution.

## Author

Created as a portfolio project showcasing expertise in RAG systems, AI/ML integration, and data analysis.

## Technologies Demonstrated

- **Retrieval Augmented Generation (RAG)** architecture
- **Large Language Model** integration (Llama2)
- **Vector embeddings** and semantic search
- **Natural Language Processing** for data analysis
- **Streamlit** web development
- **API integration** (OpenAI, Ollama)
- **Database management** (ChromaDB)
- **Data preprocessing** and pipeline development

---

_Built with modern AI/ML technologies to demonstrate real-world RAG application development_
