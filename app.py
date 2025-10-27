import streamlit as st
import os
import json
from dotenv import load_dotenv
from rag_pipeline import NSWCrimeRAGPipeline

load_dotenv()

st.set_page_config(
    page_title="NSW Crime Data RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        color: #333333;
    }
    .answer-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .answer-text {
        color: #2c3e50;
        font-size: 1.1rem;
        line-height: 1.6;
        margin: 0;
    }
    .answer-header {
        color: #1f77b4;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
    }
    .source-doc {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .source-doc h4 {
        color: #495057;
        margin-bottom: 0.5rem;
    }
    .source-doc p {
        color: #6c757d;
        line-height: 1.5;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .query-input {
        background-color: #ffffff;
        border: 2px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 0.75rem;
        font-size: 1.1rem;
    }
    .query-input:focus {
        border-color: #1f77b4;
        box-shadow: 0 0 0 0.2rem rgba(31, 119, 180, 0.25);
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_pipeline():
    try:
        pipeline = NSWCrimeRAGPipeline()
        documents_file = "data/rag_documents.json"
        
        if not os.path.exists("chroma_db"):
            with st.spinner("Initializing RAG pipeline and creating vector store..."):
                pipeline.initialize_pipeline(documents_file, force_recreate=True)
        else:
            with st.spinner("Loading existing RAG pipeline..."):
                pipeline.initialize_pipeline(documents_file, force_recreate=False)
        
        return pipeline
    except Exception as e:
        st.error(f"Error initializing pipeline: {str(e)}")
        return None

def display_metrics():
    try:
        with open("data/rag_documents.json", 'r') as f:
            documents = json.load(f)
        
        total_documents = len(documents) - 1
        unique_lgas = set()
        crime_types = set()
        
        for doc in documents:
            if doc.get("metadata", {}).get("lga"):
                unique_lgas.add(doc["metadata"]["lga"])
                crime_types.add(doc["metadata"]["crime_type"])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Crime Records", total_documents)
        
        with col2:
            st.metric("Local Government Areas", len(unique_lgas))
        
        with col3:
            st.metric("Crime Types", len(crime_types))
        
        with col4:
            st.metric("Data Source", "NSW Bureau of Crime Statistics")
    
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")

def main():
    
    st.markdown('<h1 class="main-header">NSW Crime Data RAG System</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("About This System")
        st.markdown("""
        This RAG (Retrieval Augmented Generation) system provides intelligent insights into NSW crime statistics.
        
        **Technology Stack:**
        - **LangChain**: RAG pipeline orchestration
        - **OpenAI**: Text embeddings (text-embedding-3-small)
        - **Ollama**: Local generative model (Llama2)
        - **ChromaDB**: Vector storage
        - **Streamlit**: Web interface
        
        **Data Source:**
        NSW Bureau of Crime Statistics and Research
        """)
        
        st.header("Configuration")
        
        openai_key = os.getenv("OPENAI_API_KEY")
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        if openai_key:
            st.success("OpenAI API Key configured")
        else:
            st.error("OpenAI API Key not found")
            st.info("Please set OPENAI_API_KEY in your .env file")
        
        st.info(f"Ollama URL: {ollama_url}")
        
        st.header("Quick Stats")
        display_metrics()
    
    tab1, tab2, tab3 = st.tabs(["Query Interface", "Data Explorer", "About"])
    
    with tab1:
        st.header("Ask Questions About NSW Crime Data")
        
        pipeline = initialize_pipeline()
        
        if pipeline is None:
            st.error("Failed to initialize RAG pipeline. Please check your configuration.")
            return
        
        st.markdown("### Enter your question:")
        
        example_queries = [
            "What are the crime statistics for Greater Sydney?",
            "Compare crime rates between Greater Sydney and NSW Regional areas",
            "Which region has the highest assault rates?",
            "What are the most common types of crimes in NSW?",
            "How has domestic violence related assault changed over time?",
            "What crime trends have occurred from 2015 to 2024?",
            "How did crime rates change during COVID-19 (2020-2021)?",
            "What are the murder rates across NSW regions?",
            "How prevalent is motor vehicle theft in Greater Sydney?",
            "Which areas have the highest drug-related crime rates?",
            "Show me all drug-related offenses in NSW",
            "What crimes have the highest rates per 100,000 population?"
        ]
        
        selected_example = st.selectbox("Or choose an example query:", [""] + example_queries)
        
        if selected_example:
            user_question = selected_example
        else:
            user_question = st.text_input("Your question:", placeholder="e.g., What are the crime statistics for Greater Sydney?")
        
        if st.button("Ask Question", type="primary"):
            if user_question.strip():
                with st.spinner("Processing your question..."):
                    try:
                        result = pipeline.query(user_question)
                        
                        st.markdown("### Answer:")
                        st.markdown(f"""
                        <div class='answer-container'>
                            <div class='answer-header'>Analysis Results</div>
                            <div class='answer-text'>{result['answer']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("### Source Documents:")
                        for i, source_doc in enumerate(result['source_documents']):
                            with st.expander(f"Source {i+1}: {source_doc['metadata'].get('lga', 'Unknown')} - {source_doc['metadata'].get('crime_type', 'Unknown')}"):
                                st.markdown(f"""
                                <div class='source-doc'>
                                    <h4>Document Content</h4>
                                    <p>{source_doc['content']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown("**Metadata:**")
                                metadata_html = "<div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;'>"
                                for key, value in source_doc['metadata'].items():
                                    metadata_html += f"<p><strong>{key}:</strong> {value}</p>"
                                metadata_html += "</div>"
                                st.markdown(metadata_html, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        st.info("Please ensure Ollama is running and the Llama2 model is installed.")
            else:
                st.warning("Please enter a question.")
    
    with tab2:
        st.header("Data Explorer")
        
        try:
            with open("data/nsw_crime_data.json", 'r') as f:
                data = json.load(f)
            
            st.markdown("### Raw Crime Data")
            
            import pandas as pd
            
            records = data['crime_records']
            df = pd.DataFrame(records)
            
            st.markdown("### Data Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Crime Types:**")
                crime_type_counts = df['crime_type'].value_counts()
                st.bar_chart(crime_type_counts)
            
            with col2:
                st.markdown("**LGA Distribution:**")
                lga_counts = df['lga'].value_counts()
                st.bar_chart(lga_counts)
            
            st.markdown("### Interactive Filters")
            
            selected_lga = st.selectbox("Filter by LGA:", ["All"] + list(df['lga'].unique()))
            selected_crime_type = st.selectbox("Filter by Crime Type:", ["All"] + list(df['crime_type'].unique()))
            selected_trend = st.selectbox("Filter by Trend:", ["All"] + list(df['trend'].unique()))
            
            filtered_df = df.copy()
            
            if selected_lga != "All":
                filtered_df = filtered_df[filtered_df['lga'] == selected_lga]
            
            if selected_crime_type != "All":
                filtered_df = filtered_df[filtered_df['crime_type'] == selected_crime_type]
            
            if selected_trend != "All":
                filtered_df = filtered_df[filtered_df['trend'] == selected_trend]
            
            st.markdown(f"### Filtered Results ({len(filtered_df)} records)")
            st.dataframe(filtered_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    with tab3:
        st.header("About This RAG Solution")
        
        st.markdown("""
        ### Project Objective
        
        This project demonstrates a **Retrieval Augmented Generation (RAG)** solution for analyzing NSW crime data, 
        showcasing how AI can add value to domain-specific data analysis.
        
        ### Architecture
        
        **1. Data Layer:**
        - NSW crime statistics from official government sources
        - Document preparation and chunking for optimal retrieval
        
        **2. Embedding Layer:**
        - OpenAI text-embedding-3-small for semantic understanding
        - ChromaDB for efficient vector storage and retrieval
        
        **3. Generation Layer:**
        - Ollama with Llama2 model for local, cost-effective generation
        - LangChain for pipeline orchestration
        
        **4. Interface Layer:**
        - Streamlit web application for interactive querying
        
        ### Key Features
        
        - **Semantic Search**: Find relevant information based on meaning, not just keywords
        - **Source Attribution**: Every answer includes source documents for transparency
        - **Interactive Interface**: User-friendly web interface for querying
        - **Local Processing**: Uses Ollama for privacy and cost-effectiveness
        - **Real-time Insights**: Instant answers to complex questions about crime data
        
        ### Evaluation Framework
        
        The system includes metrics for measuring:
        - **Effectiveness**: Percentage of successfully answered questions
        - **Faithfulness**: Accuracy of information relative to source data
        - **Source Attribution**: Correctness of document references
        
        ### Business Value
        
        This RAG solution demonstrates value for:
        - **Policy Makers**: Quick insights into crime trends and patterns
        - **Researchers**: Efficient data exploration and analysis
        - **Public**: Accessible crime statistics with natural language queries
        - **Law Enforcement**: Rapid information retrieval for operational planning
        
        ### Technical Implementation
        
        Built using modern AI/ML stack:
        - Python 3.13
        - LangChain 0.3.27
        - OpenAI Embeddings API
        - Ollama with Llama2
        - ChromaDB for vector storage
        - Streamlit for web interface
        
        ### Future Enhancements
        
        - Integration with real-time crime data feeds
        - Advanced visualization capabilities
        - Multi-modal analysis (text + geographic data)
        - Automated report generation
        - API endpoints for integration with other systems
        """)

if __name__ == "__main__":
    main()
