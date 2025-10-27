import os
import json
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class NSWCrimeRAGPipeline:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.llm = Ollama(
            model="llama2",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        
        self.vectorstore = None
        self.qa_chain = None
        
    def load_documents(self, documents_file: str) -> List[Document]:
        with open(documents_file, 'r') as f:
            data = json.load(f)
        
        documents = []
        for doc_data in data:
            doc = Document(
                page_content=doc_data["content"],
                metadata=doc_data["metadata"]
            )
            documents.append(doc)
        
        return documents
    
    def create_vector_store(self, documents: List[Document], persist_directory: str = "chroma_db"):
        print("Creating vector store...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        splits = text_splitter.split_documents(documents)
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        self.vectorstore.persist()
        print(f"Vector store created and persisted to {persist_directory}")
        
        return self.vectorstore
    
    def load_vector_store(self, persist_directory: str = "chroma_db"):
        print("Loading existing vector store...")
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        return self.vectorstore
    
    def setup_qa_chain(self):
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        print("Setting up QA chain...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            return_source_documents=True
        )
        
        return self.qa_chain
    
    def query(self, question: str) -> Dict:
        if not self.qa_chain:
            raise ValueError("QA chain not initialized")
        
        print(f"Processing query: {question}")
        result = self.qa_chain({"query": question})
        
        return {
            "question": question,
            "answer": result["result"],
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        }
    
    def initialize_pipeline(self, documents_file: str, force_recreate: bool = False):
        persist_directory = "chroma_db"
        
        if force_recreate or not os.path.exists(persist_directory):
            documents = self.load_documents(documents_file)
            self.create_vector_store(documents, persist_directory)
        else:
            self.load_vector_store(persist_directory)
        self.setup_qa_chain()
        
        print("RAG pipeline initialized successfully!")
        return self

def main():
    pipeline = NSWCrimeRAGPipeline()
    documents_file = "data/rag_documents.json"
    pipeline.initialize_pipeline(documents_file, force_recreate=True)
    test_queries = [
        "What are the crime statistics for Sydney?",
        "Which areas have increasing crime rates?",
        "What types of crimes are most common in NSW?",
        "Compare crime rates between different LGAs"
    ]
    
    print("\n" + "="*50)
    print("TESTING RAG PIPELINE")
    print("="*50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        try:
            result = pipeline.query(query)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {len(result['source_documents'])} documents")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
