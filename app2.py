import streamlit as st
import os
from typing import List, Dict
from neo4j import GraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import networkx as nx
from pyvis.network import Network
import tempfile
import plotly.graph_objects as go
# from streamlit_plotly import plotly_chart
from streamlit import plotly_chart
import base64
import re
from dotenv import load_dotenv
import pandas as pd
import io

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(layout="wide", page_title="Knowledge Graph RAG System")

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to right, #1a1a1a, #2d2d2d);
        color: #ffffff;
    }
    /* Sidebar styling - fixing transparency */
    section[data-testid="stSidebar"] {
        background-color: #1e1e1e;
        border-right: 1px solid #333;
    }
    section[data-testid="stSidebar"] > div {
        background-color: #1e1e1e;
        padding: 2rem 1rem;
    }
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        margin: 0.5rem 0;
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    /* Sidebar headers */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #4CAF50;
        font-weight: 600;
        margin-top: 1rem;
    }
    /* Sidebar text */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    /* Divider in sidebar */
    section[data-testid="stSidebar"] hr {
        margin: 2rem 0;
        border-color: #333;
    }
    /* Main content buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 10px 24px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .graph-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    /* Additional sidebar elements */
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div[role="button"] {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    /* Ensure all text in sidebar is visible */
    section[data-testid="stSidebar"] * {
        color: #ffffff;
    }
    /* Reset and override all button styles in sidebar */
    section[data-testid="stSidebar"] .stButton button {
        background-color: #4CAF50 !important;
        border: none !important;
        color: white !important;
        width: 100% !important;
        margin: 0.5rem 0 !important;
        padding: 0.5rem 1rem !important;
        border-radius: 10px !important;
        font-weight: normal !important;
        opacity: 1 !important;
    }

    /* Specific styling for Delete Database button */
    section[data-testid="stSidebar"] .stButton button[data-testid="baseButton-secondary"] {
        background-color: #dc3545 !important;
        color: white !important;
        font-weight: bold !important;
        opacity: 1 !important;
    }

    /* Hover state for Delete Database button */
    section[data-testid="stSidebar"] .stButton button[data-testid="baseButton-secondary"]:hover {
        background-color: #c82333 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
    }

    /* Ensure no transparency in button elements */
    section[data-testid="stSidebar"] .stButton button * {
        opacity: 1 !important;
        color: white !important;
    }

    /* Additional style to force solid background */
    section[data-testid="stSidebar"] .stButton {
        opacity: 1 !important;
        background: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY in the .env file")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-exp-1206",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
    st.error("Please set your NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD in the .env file")
    st.stop()

class KnowledgeGraphRAG:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
    
    def delete_database(self):
        """Delete all nodes and relationships in the database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            try:
                session.run("CALL db.index.vector.drop('document_vectors')")
            except Exception as e:
                st.warning("Vector index might not exist or was already deleted.")
            
    def create_vector_store(self, documents: List):
        """Create vector store in Neo4j"""
        vector_store = Neo4jVector.from_documents(
            documents,
            self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="document_vectors",
            node_label="Document",
            embedding_node_property="embedding",
            text_node_property="text"
        )
        return vector_store

    def _parse_relationships(self, llm_response: str) -> List[Dict]:
        """Parse LLM relationship extraction response"""
        relationships = []
        # Regular expression pattern to match the relationship format
        pattern = r'\(([^)]+)\)-\[([^\]]+)\]->\(([^)]+)\)'
        
        # Split the response into lines and process each line
        for line in llm_response.split('\n'):
            line = line.strip()
            matches = re.findall(pattern, line)
            
            for match in matches:
                if len(match) == 3:  # Ensure we have all three components
                    entity1, relationship, entity2 = match
                    # Clean up the extracted texts
                    entity1 = entity1.strip()
                    relationship = relationship.strip()
                    entity2 = entity2.strip()
                    
                    if entity1 and relationship and entity2:  # Ensure none are empty
                        relationships.append({
                            'entity1': entity1,
                            'relationship': relationship,
                            'entity2': entity2
                        })
        
        return relationships

    def create_knowledge_graph(self, documents: List):
        """Extract entities and relationships to create knowledge graph"""
        with self.driver.session() as session:
            for doc in documents:
                prompt = f"""
                Extract key entities and their relationships from this text. 
                Format each relationship exactly as: (entity1)-[relationship]->(entity2)
                Return one relationship per line.
                Only include clear, explicit relationships from the text.
                Text: {doc.page_content}
                """
                response = llm.predict(prompt)
                
                relationships = self._parse_relationships(response)
                for rel in relationships:
                    if all(rel.values()):  # Check that no values are empty
                        session.run("""
                        MERGE (e1:Entity {name: $entity1})
                        MERGE (e2:Entity {name: $entity2})
                        MERGE (e1)-[:RELATES {type: $relationship}]->(e2)
                        """, rel)

    def create_3d_graph(self):
        """Generate 3D graph visualization using Plotly"""
        G = nx.DiGraph()
        
        with self.driver.session() as session:
            result = session.run("""
            MATCH (e1:Entity)-[r:RELATES]->(e2:Entity)
            RETURN e1.name as source, r.type as relationship, e2.name as target
            """)
            records = list(result)
            
        if not records:  # If no relationships exist
            return None
            
        for record in records:
            G.add_edge(record["source"], record["target"])
            
        pos = nx.spring_layout(G, dim=3)
        
        edge_x = []
        edge_y = []
        edge_z = []
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        edges_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_z = []
        for node in G.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

        nodes_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            hovertext=list(G.nodes()),
            hoverinfo='text',
            marker=dict(
                size=8,
                color='#00ff00',
                line_width=2))

        fig = go.Figure(data=[edges_trace, nodes_trace])
        fig.update_layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig

    def query(self, question: str) -> Dict:
        """Query both vector store and knowledge graph"""
        vector_store = Neo4jVector(
            self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="document_vectors"
        )
        retriever = vector_store.as_retriever()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        rag_answer = qa_chain.run(question)
        
        kg_data = []
        with self.driver.session() as session:
            result = session.run("""
            MATCH path = (e1:Entity)-[r:RELATES]->(e2:Entity)
            WHERE e1.name CONTAINS $question OR e2.name CONTAINS $question
            RETURN e1.name as source, r.type as relationship, e2.name as target
            LIMIT 10
            """, question=question)
            kg_data = [dict(record) for record in result]
        
        return {
            "rag_answer": rag_answer,
            "knowledge_graph": kg_data
        }

    def process_csv(self, csv_file):
        """Process CSV file and store data in Neo4j"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Clean column names: replace spaces and special characters with underscores
            df.columns = [col.strip().replace(' ', '_').replace('-', '_').lower() for col in df.columns]
            
            # Create nodes and relationships in Neo4j
            with self.driver.session() as session:
                # Create constraints if they don't exist
                session.run("CREATE CONSTRAINT csv_data_id IF NOT EXISTS FOR (n:CSVData) REQUIRE n.id IS UNIQUE")
                
                # Process each row in the CSV
                for index, row in df.iterrows():
                    # Convert row to dictionary and clean up values
                    row_dict = row.to_dict()
                    properties = {k: str(v) if pd.notna(v) else '' for k, v in row_dict.items()}
                    
                    # Add unique identifier
                    properties['id'] = f"row_{index}"
                    
                    # Create Cypher query with proper property names
                    props_list = [f"{k}: ${k}" for k in properties.keys()]
                    props_string = ", ".join(props_list)
                    
                    query = f"""
                    CREATE (n:CSVData {{{props_string}}})
                    """
                    
                    session.run(query, properties)
                
            return True, "CSV data successfully loaded into Neo4j"
        except Exception as e:
            return False, f"Error processing CSV: {str(e)}"

    def query_csv_data(self, question: str) -> Dict:
        """Query CSV data from Neo4j using natural language"""
        # Generate Cypher query based on the question
        prompt = f"""
        Convert this question into a Cypher query for Neo4j.
        The data is stored in nodes labeled 'CSVData'.
        Column names are lowercase with spaces replaced by underscores.
        Common columns: id, customer_id, first_name, last_name, company, city, country, phone1, phone2, email, subscription_date, website
        
        Question: {question}
        Return only the Cypher query without any explanation, markdown formatting, or code blocks.
        """
        
        # Get the query from LLM
        cypher_query = llm.predict(prompt).strip()
        
        # Clean the query by removing any markdown code block syntax or extra whitespace
        cypher_query = cypher_query.replace('```cypher', '').replace('```', '').strip()
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                records = [record.data() for record in result]
                
                # Format the response
                if records:
                    # Clean up the records for better display
                    cleaned_records = []
                    for record in records:
                        # Remove None values and empty strings
                        cleaned_record = {k: v for k, v in record.items() if v is not None and v != ''}
                        if cleaned_record:
                            cleaned_records.append(cleaned_record)
                    
                    response = {
                        'answer': cleaned_records,
                        'query': cypher_query
                    }
                else:
                    response = {
                        'answer': "No data found matching your query.",
                        'query': cypher_query
                    }
                return response
        except Exception as e:
            return {
                'answer': f"Error executing query: {str(e)}",
                'query': cypher_query
            }

def main():
    st.title("🌟 Advanced RAG System with 3D Knowledge Graph")
    
    # Initialize system
    rag_system = KnowledgeGraphRAG()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        if st.button(
            "🗑️ Delete Database",
            help="Clear all data and start fresh",
            key="delete_db",
            type="secondary",
            use_container_width=True
        ):
            with st.spinner("Deleting database..."):
                rag_system.delete_database()
            st.success("Database cleared successfully!")
        
        st.markdown("---")
        st.markdown("### 📊 Graph Settings")
        st.markdown("Customize your graph visualization here")
    
    # Main content area
    st.markdown("### 📄 Document Upload")
    
    # Create two columns for file uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Upload PDF")
        pdf_file = st.file_uploader(
            "Drop your PDF document here",
            type="pdf",
            help="Upload a PDF document to analyze",
            key="pdf_uploader"
        )
    
    with col2:
        st.markdown("#### Upload CSV")
        csv_file = st.file_uploader(
            "Drop your CSV file here",
            type="csv",
            help="Upload a CSV file to analyze",
            key="csv_uploader"
        )
    
    # Process button for both file types
    if st.button("Process Files"):
        if pdf_file or csv_file:
            # Process PDF file
            if pdf_file:
                with st.spinner("🔄 Processing PDF document..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        tmp.write(pdf_file.getvalue())
                        loader = PyPDFLoader(tmp.name)
                        documents = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    splits = text_splitter.split_documents(documents)
                    
                    rag_system.create_vector_store(splits)
                    rag_system.create_knowledge_graph(splits)
                    os.unlink(tmp.name)  # Clean up temp file
                st.success("✅ PDF document processed successfully!")
            
            # Process CSV file
            if csv_file:
                with st.spinner("🔄 Processing CSV file..."):
                    success, message = rag_system.process_csv(csv_file)
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
        else:
            st.warning("Please upload at least one file (PDF or CSV) to process.")
    
    # Query Interface
    st.markdown("### 🔍 Query Interface")
    question = st.text_input("Ask a question:", placeholder="Type your question here...")
    
    if question:
        with st.spinner("🤔 Analyzing..."):
            # Determine if the question is about CSV data
            csv_related_keywords = ['customer', 'csv', 'data', 'list', 'show', 'get', 'first', 'last', 'count', 'how many']
            is_csv_query = any(keyword in question.lower() for keyword in csv_related_keywords)
            
            if is_csv_query:
                # Query CSV data
                results = rag_system.query_csv_data(question)
                st.markdown("### 📝 Answer from CSV Data")
                if isinstance(results['answer'], list):
                    # Display as a table if it's a list of records
                    df = pd.DataFrame(results['answer'])
                    st.dataframe(df)
                else:
                    st.write(results['answer'])
                
                # Show the Cypher query used
                with st.expander("Show Cypher Query"):
                    st.code(results['query'], language='cypher')
            else:
                # Query PDF data
                results = rag_system.query(question)
                st.markdown("### 📝 Answer from PDF")
                st.markdown(f"```\n{results['rag_answer']}\n```")
                
                st.markdown("### 🔗 Knowledge Graph Connections")
                for rel in results["knowledge_graph"]:
                    st.markdown(f"🔹 {rel['source']} → *{rel['relationship']}* → {rel['target']}")
    
    # 3D Graph Visualization
    st.markdown("### 🌐 Knowledge Graph Visualization")
    with st.container():
        fig = rag_system.create_3d_graph()
        if fig is not None:
            plotly_chart(fig, use_container_width=True, height=600)
        else:
            st.info("No relationships to visualize yet. Upload a document to create the knowledge graph.")

if __name__ == "__main__":
    main()
