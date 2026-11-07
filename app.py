import os
import streamlit as st
import chromadb
import certifi
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Configuración de seguridad para Windows
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Cargar variables de entorno locales (.env)
load_dotenv()

# Lógica para obtener la API Key (Local o en la Nube)
try:
    API_KEY = st.secrets["GEMINI_KEY"]
except Exception:
    API_KEY = os.getenv("GEMINI_KEY", "AQ.Ab8RN6LMd1CP_ZmaMB569pMezDvRozNs6Tb9r6qikLWKS4L32g")

# Inicializar cliente forzando v1beta (necesario para claves AQ.)
client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(api_version='v1beta')
)

# Configuración de la página web
st.set_page_config(page_title="Chatbot Reglamento ADCC", page_icon="🥋")
st.title("🥋 Asistente del Reglamento ADCC")
st.markdown("Haz preguntas sobre las reglas del torneo. Las respuestas están basadas **exclusivamente** en el documento oficial.")

# Conectar a la base de datos
@st.cache_resource
def cargar_db():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return chroma_client.get_collection(name="adcc_rules")

collection = cargar_db()

# Input del usuario
pregunta = st.text_input("Escribe tu pregunta aquí:")

if st.button("Buscar respuesta") and pregunta:
    with st.spinner("Buscando en el reglamento..."):
        try:
            # 1. Vectorizar pregunta
            response_embed = client.models.embed_content(
                model="gemini-embedding-001",
                contents=[pregunta]
            )
            vector_pregunta = response_embed.embeddings[0].values

            # 2. Buscar en ChromaDB (n_results=6 para traer más contexto)
            results = collection.query(query_embeddings=[vector_pregunta], n_results=6)
            chunks = results['documents'][0]
            paginas = [meta['page'] for meta in results['metadatas'][0]]
            
            contexto = "\n\n---\n\n".join(chunks)
            
            # 3. Generar respuesta (usando el modelo que sabemos que funciona)
            prompt = f"""Eres un asistente experto. Responde la pregunta usando EXCLUSIVAMENTE el contexto.
Si no está en el contexto, di: "No tengo información sobre eso en el documento".
No inventes datos.

CONTEXTO:
{contexto}

PREGUNTA: {pregunta}
"""
            response_llm = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
            
            # 4. Mostrar resultados en la web
            st.success("Respuesta generada:")
            st.write(response_llm.text)
            
            # Mostrar fuentes
            paginas_unicas = sorted(list(set(paginas)))
            st.info(f"📄 **Fuentes consultadas:** Páginas {paginas_unicas} del PDF oficial.")
            
            # Un "expander" para ver el contexto crudo
            with st.expander("Ver fragmentos exactos extraídos del PDF"):
                for i, (chunk, pag) in enumerate(zip(chunks, paginas)):
                    st.markdown(f"**Fragmento {i+1} (Página {pag}):**")
                    st.text(chunk)
                    st.divider()

        except Exception as e:
            st.error(f"Ocurrió un error al procesar tu pregunta: {e}")