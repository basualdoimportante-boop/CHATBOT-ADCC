import os
import pymupdf  # Reemplaza al antiguo 'fitz'
import chromadb
import certifi
from google import genai

# Forzamos a Python a usar los certificados de seguridad correctos
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# =====================================================================
# PEGA TU CLAVE AQUÍ ABAJO ENTRE LAS COMILLAS
# =====================================================================
API_KEY = "PON_TU_CLAVE_AQUI_PARA_PROBAR_LOCAL"

client = genai.Client(api_key=API_KEY)

PDF_PATH = "documento.pdf"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "adcc_rules"

def extraer_y_dividir_texto(pdf_path, chunk_size=800, overlap=100):
    """Extrae texto del PDF y lo divide en fragmentos con solapamiento."""
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        print(f"Error al abrir el PDF: {e}")
        return []
        
    chunks = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        
        # Dividir el texto de la página en chunks
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Guardar el chunk con su número de página
            chunks.append({
                "text": chunk.strip(),
                "page": page_num + 1
            })
            start += chunk_size - overlap
            
    return chunks

def main():
    print("1. Extrayendo texto del PDF...")
    chunks = extraer_y_dividir_texto(PDF_PATH)
    
    if not chunks:
        print("No se pudo extraer texto. Verifica que el archivo 'documento.pdf' esté en la carpeta.")
        return
        
    print(f"   Se generaron {len(chunks)} fragmentos.")

    print("2. Generando embeddings con Gemini...")
    texts = [c["text"] for c in chunks]
    
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts
        )
        embeddings = [e.values for e in response.embeddings]
        print(f"   Se generaron {len(embeddings)} vectores.")
    except Exception as e:
        print(f"Error al generar embeddings: {e}")
        return

    print("3. Guardando en ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    
    if COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"page": c["page"]} for c in chunks]
    
    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"   ¡Éxito! {len(chunks)} fragmentos guardados en la base de datos vectorial.")

if __name__ == "__main__":
    main()
