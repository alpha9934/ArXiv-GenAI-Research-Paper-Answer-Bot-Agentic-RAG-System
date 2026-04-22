import streamlit as st
from src.graph_builder.graph_builder import GraphBuilder
from src.config.config import Config 
from src.vectorstore.vectorstore import VectorStore
from src.document_ingestion.document_processor import DocumentProcessor

# 1. Initialize Page
st.set_page_config(page_title="Agentic RAG System", page_icon="✨")
st.title(" ✨ ArXiv-GenAI Research Paper Answer Bot | Agentic RAG System")

# 2. Load the Graph Builder
@st.cache_resource
def get_graph_system():
    # Initialize LLM and VectorStore
    llm = Config.get_llm() 
    vs = VectorStore() 
    
    # Auto-ingest if database doesn't exist yet
    if not vs.load_local():
        with st.spinner("First run detected: Ingesting documents to build database. This may take a minute..."):
            processor = DocumentProcessor()
            # Assuming your PDFs/text files are in the "data" directory
            raw_docs = processor.load_documents(["data"]) 
            docs = processor.split_documents(raw_docs)
            
            # Inject chunk_ids for the UI References
            for i, chunk in enumerate(docs):
                chunk.metadata["chunk_id"] = f"chunk_{i}"
                
            vs.create_vectorstore(docs)
            vs.save_local()
            st.success("Database built successfully!")
    
    # Get retriever and configure it to fetch top 4 chunks
    retriever = vs.get_retriever()
    retriever.search_kwargs = {"k": 4} 
    
    # Initialize and compile the LangGraph workflow
    builder = GraphBuilder(retriever=retriever, llm=llm)
    builder.build() 
    return builder

# Safely initialize the application
try:
    graph_system = get_graph_system()
except Exception as e:
    st.error(f"Error building the application: {str(e)}")
    st.stop()

# 3. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display Chat History on App Rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if they were saved in the assistant's message history
        if "sources" in message and message["sources"]:
            with st.expander("View Source Chunks"):
                for doc in message["sources"]:
                    chunk_id = doc.metadata.get("chunk_id", "Unknown ID")
                    source = doc.metadata.get("source", "Unknown Source")
                    st.caption(f"**Chunk ID: {chunk_id}** | Source: {source}")
                    st.write(f"_{doc.page_content[:250]}..._") 
                    st.divider()

# 5. Handle User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    
    # Add user message to chat UI & state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 6. Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating answer..."):
            
            # Run the LangGraph workflow using your custom run() method
            result = graph_system.run(prompt)
            
            # Extract keys exactly as defined in rag_state.py
            final_answer = result.get("answer", "Sorry, no answer was generated.")
            source_documents = result.get("retrieved_docs", [])
            
            # Display the final answer
            st.markdown(final_answer)
            
            # Display Expandable Sources Section immediately for the new response
            if source_documents:
                with st.expander("View Source Chunks"):
                    for doc in source_documents:
                        chunk_id = doc.metadata.get("chunk_id", "Unknown ID")
                        source = doc.metadata.get("source", "Unknown Source")
                        st.caption(f"**Chunk ID: {chunk_id}** | Source: {source}")
                        st.write(f"_{doc.page_content[:250]}..._")
                        st.divider()

    # 7. Save Assistant Message & Sources to State
    st.session_state.messages.append({
        "role": "assistant", 
        "content": final_answer,
        "sources": source_documents 
    })