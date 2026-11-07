import os
import chromadb
import certifi
from google import genai
from google.genai import types

# Configuración de seguridad para Windows
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# =====================================================================
# PEGA TU CLAVE AQUÍ ABAJO (Reemplaza el texto de ejemplo)
# =====================================================================
API_KEY = "PON_TU_CLAVE_AQUI_PARA_PROBAR_LOCAL"

client = genai.Client(api_key=API_KEY)

DB_PATH = "./chroma_db"
COLLECTION_NAME = "adcc_rules"

# 1. Conectar a la base de datos vectorial
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

def preguntar_al_chatbot(pregunta, top_k=4):
    print(f"\nBuscando información para: '{pregunta}'...")
    
    # 2. Convertir la pregunta del usuario en un vector
    response_embed = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[pregunta]
    )
    vector_pregunta = response_embed.embeddings[0].values

    # 3. Buscar en ChromaDB los 'top_k' fragmentos más parecidos
    results = collection.query(
        query_embeddings=[vector_pregunta],
        n_results=top_k
    )
    
    # Extraer los textos y las páginas
    chunks_recuperados = results['documents'][0]
    paginas = [meta['page'] for meta in results['metadatas'][0]]
    
    # Unir los fragmentos en un solo bloque de contexto
    contexto = "\n\n---\n\n".join(chunks_recuperados)
    
    # (Opcional) Imprimir el contexto para auditar qué leyó la IA
    print("\n--- CONTEXTO RECUPERADO DE CHROMADB ---")
    print(contexto)
    
    # 4. Construir el Prompt (Instrucción)
    prompt = f"""Eres un asistente experto. Responde la pregunta del usuario usando EXCLUSIVAMENTE la información del siguiente contexto.
Si la respuesta no está en el contexto, di: "No tengo información sobre eso en el documento".
No inventes datos bajo ninguna circunstancia.

CONTEXTO:
{contexto}

PREGUNTA: {pregunta}
"""
    
    print("\nGenerando respuesta con Gemini...")
    # 5. Llamar al modelo de lenguaje para que redacte
    response_llm = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
        )
    )
    
    print("\n--- RESPUESTA ---")
    print(response_llm.text)
    print("\n--- FUENTES (Páginas del PDF) ---")
    paginas_unicas = sorted(list(set(paginas)))
    print(f"Páginas consultadas: {paginas_unicas}")

if __name__ == "__main__":
    # =====================================================================
    # HAZ UNA PREGUNTA AQUÍ
    # =====================================================================
    pregunta_test = "¿Está permitido usar algo resbaladizo en el cuerpo o la ropa?" 
    preguntar_al_chatbot(pregunta_test, top_k=4)