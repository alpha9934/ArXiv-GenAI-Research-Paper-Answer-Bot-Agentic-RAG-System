"""Vector store module for document embedding and retrieval"""

import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class VectorStore:
    """Manages vector store operations"""
    
    def __init__(self, persist_dir: str = "faiss_index"):
        """Initialize vector store with OpenAI embeddings"""
        self.embedding = OpenAIEmbeddings()
        self.vectorstore = None
        self.retriever = None
        self.persist_dir = persist_dir # Directory to save/load the database
    
    def create_vectorstore(self, documents: List[Document]):
        """
        Create vector store from documents
        """
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever()
        
    def save_local(self):
        """Saves the FAISS index to disk so we don't have to re-embed every time"""
        if self.vectorstore is not None:
            self.vectorstore.save_local(self.persist_dir)
            
    def load_local(self):
        """Loads a previously saved FAISS index from disk"""
        if os.path.exists(self.persist_dir):
            self.vectorstore = FAISS.load_local(
                self.persist_dir, 
                self.embedding, 
                allow_dangerous_deserialization=True # Required for FAISS loading in Langchain
            )
            self.retriever = self.vectorstore.as_retriever()
            return True
        return False

    def get_retriever(self):
        """
        Get the retriever instance
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Load or create it first.")
        return self.retriever
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Load or create it first.")
        return self.retriever.invoke(query)