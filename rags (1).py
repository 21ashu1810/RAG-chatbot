import sqlite3
import fitz  
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)


logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)


def authenticate_huggingface(token):
    print("Authenticating Hugging Face...")
    try:
        login(token=token)
        print("Hugging Face authentication successful")
    except Exception as e:
        print(f"Error during Hugging Face authentication: {e}")
        # Optionally log this error
        app.logger.error(f"Authentication error: {e}")

#  Extract text from PDF
def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from PDF: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        if not text.strip():
            print(f"Error: No text found in PDF {pdf_path}")
        print(f"Extracted PDF Text: {text[:200]}...")  
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def initialize_models():
    print("Initializing models...")
    try:
        model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Models initialized successfully")
        return tokenizer, model, embedding_model
    except Exception as e:
        print(f"Error initializing models: {e}")
        return None, None, None

# . Initialize SQLite and FAISS
def init_faiss_sqlite(db_path='knowledge_base.db'):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS documents 
                          (id INTEGER PRIMARY KEY, doc_text TEXT, embedding BLOB)''')
        conn.commit()
        print("SQLite initialized successfully")
        return conn, cursor
    except Exception as e:
        print(f"Error initializing SQLite: {e}")
        return None, None

# Initialize FAISS
def init_faiss_index(embedding_dim):
    try:
        index = faiss.IndexFlatIP(embedding_dim)  
        print("FAISS index initialized successfully")
        return index
    except Exception as e:
        print(f"Error initializing FAISS: {e}")
        return None

# Load embeddings into FAISS from SQLite
def load_embeddings_into_faiss(cursor, faiss_index):
    try:
        cursor.execute("SELECT embedding FROM documents")
        docs = cursor.fetchall()

        if docs:
            embeddings = [np.frombuffer(doc[0], dtype=np.float32) for doc in docs]
            faiss_index.add(np.array(embeddings))
            print(f"Loaded {len(embeddings)} embeddings into FAISS index.")
        else:
            print("No embeddings found in the database.")
    except Exception as e:
        print(f"Error loading embeddings into FAISS: {e}")

# Get embedding from text
def get_embedding(text, embedding_model):
    try:
        return embedding_model.encode(text, normalize_embeddings=True).astype(np.float32)
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None

# Insert text and embedding into SQLite
def insert_document_into_sqlite(text, embedding_model, cursor, conn):
    try:
        embedding = get_embedding(text, embedding_model)
        if embedding is not None:
            cursor.execute("INSERT INTO documents (doc_text, embedding) VALUES (?, ?)", 
                           (text, embedding.tobytes()))
            conn.commit()
            print(f"Inserted document into SQLite: {text[:100]}...")
    except Exception as e:
        print(f"Error inserting document into SQLite: {e}")

# Search documents using FAISS
def search_documents(query, embedding_model, cursor, faiss_index):
    try:
        query_embedding = get_embedding(query, embedding_model)
        cursor.execute("SELECT doc_text, embedding FROM documents")
        docs = cursor.fetchall()

        if not docs:
            print("No documents found in the database.")
            return []

        embeddings = [np.frombuffer(doc[1], dtype=np.float32) for doc in docs]
        texts = [doc[0] for doc in docs]

        if faiss_index.ntotal == 0:
            faiss_index.add(np.array(embeddings))
            print("Added embeddings to FAISS index for search.")

        query_embedding /= np.linalg.norm(query_embedding)

        D, I = faiss_index.search(np.array([query_embedding]), k=5)
        results = [texts[i] for i in I[0] if i < len(texts)]
        print(f"Search results: {results}")
        return results
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []

#  Generate response using retrieved documents
def generate_response(query, retrieved_docs, tokenizer, model):
    try:
        context = "\n".join(retrieved_docs)
        input_text = f"Question: {query}\nContext: {context}\nAnswer:"
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated Response: {response}")
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error generating response"

# 11. Full chatbot pipeline
def chatbot_pipeline(pdf_path, query, tokenizer, model, embedding_model, cursor, conn, faiss_index):
    try:
        print(f"Starting pipeline with PDF: {pdf_path}, Query: {query}")
        pdf_text = extract_text_from_pdf(pdf_path)
        if pdf_text:
            insert_document_into_sqlite(pdf_text, embedding_model, cursor, conn)
        
        retrieved_docs = search_documents(query, embedding_model, cursor, faiss_index)
        if not retrieved_docs:
            return "No relevant documents found."

        response = generate_response(query, retrieved_docs, tokenizer, model)
        return response
    except Exception as e:
        print(f"Error in chatbot_pipeline: {e}")
        return "Error processing the request."

# Initialize models, FAISS, and database connections
tokenizer, model, embedding_model = initialize_models()
if tokenizer and model and embedding_model:
    conn, cursor = init_faiss_sqlite()
    if conn and cursor:
        faiss_index = init_faiss_index(embedding_model.get_sentence_embedding_dimension())
        if faiss_index:
            load_embeddings_into_faiss(cursor, faiss_index)

# Flask route for the chatbot API
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get('query', '')
        pdf_path = data.get('pdf_path', '')

        if not query or not pdf_path:
            app.logger.error("Missing query or PDF path")
            return jsonify({'error': 'Please provide both query and PDF path'}), 400

        app.logger.info(f"Received chat request. Query: {query}, PDF Path: {pdf_path}")
        response = chatbot_pipeline(pdf_path, query, tokenizer, model, embedding_model, cursor, conn, faiss_index)
        app.logger.info(f"Chatbot response: {response}")
        return jsonify({'response': response})
    except Exception as e:
        app.logger.error(f"Error in /chat route: {e}")
        return jsonify({'error': 'An error occurred'}), 500


if __name__ == "__main__":
    hf_token = 'Hugging face api key'  
    authenticate_huggingface(hf_token)
    
    
    print("Starting Flask app...")
    
    
    if tokenizer and model and embedding_model and conn and cursor and faiss_index:
        print("Initialization complete. Starting Flask server...")
        app.run(debug=True)
    else:
        print("Initialization failed. Check earlier errors.")
    
    app.run(debug=True)
