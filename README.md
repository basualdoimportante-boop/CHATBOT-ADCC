# 🥋 Chatbot RAG - Reglamento ADCC 2026

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-adcc-bf3qydrrzkjvootbnxfkn6.streamlit.app/)

Sistema de Generación Aumentada por Recuperación (RAG) diseñado para responder preguntas sobre el reglamento oficial de Jiu-Jitsu ADCC, garantizando **cero alucinaciones** y citando las fuentes exactas (páginas) del documento original.

## 🔗 Demo en vivo
Puedes probar la aplicación desplegada en la nube aquí: **[Chatbot ADCC en Streamlit Cloud](https://chatbot-adcc-bf3qydrrzkjvootbnxfkn6.streamlit.app/)**

## 🏗️ Arquitectura
- **LLM de Generación:** `gemini-3.5-flash-lite` (optimizado para bajo costo/latencia).
- **Modelo de Embeddings:** `gemini-embedding-001`.
- **Base de Datos Vectorial:** `ChromaDB` (persistencia local con distancia coseno).
- **Interfaz de Usuario:** `Streamlit`.
- **Framework de Evaluación:** `RAGAS` (métricas de calidad y alucinación).

## 📊 Estrategia y Resultados de Evaluación (Harness)
Para validar la fiabilidad del sistema, diseñé un *benchmark* estratificado de 10 preguntas divididas en 3 categorías. El objetivo era aislar los fallos de recuperación (Retrieval) de los fallos de generación (Generation).

| Categoría | Preguntas | Faithfulness (Cero Alucinaciones) | Context Recall (No perder info) | Context Precision (No traer ruido) |
| :--- | :--- | :--- | :--- | :--- |
| **Factual** | 6 | `1.00` | `0.67` | `0.75` |
| **Multi-hop** | 2 | `1.00` | `0.75` | `0.25` |
| **Negativas** (Abstención) | 2 | `1.00` | `1.00` | `0.00*` |

*\*Nota técnica: La precisión en preguntas negativas es 0.00 porque, al no existir la respuesta en el documento, matemáticamente ningún fragmento recuperado es "relevante". Sin embargo, el sistema respondió correctamente "No tengo información" (Faithfulness 1.0).*

### Conclusiones del Benchmark:
1. **Cero Alucinaciones:** El sistema es 100% confiable en cuanto a no inventar datos.
2. **Área de mejora (Precisión):** El *Context Precision* en preguntas Multi-hop (0.25) indica que el sistema trae fragmentos de más. La solución propuesta a futuro es implementar un **Reranker** para filtrar el ruido antes de enviarlo al LLM.

## 🚀 Cómo ejecutar el proyecto localmente
1. Clonar el repositorio.
2. Crear un entorno virtual e instalar dependencias: `pip install -r requirements.txt`
3. Configurar la API Key en un archivo `.env` (`GEMINI_KEY=tu_clave`)
4. Ingestar el PDF: `python ingest.py`
5. Ejecutar la interfaz: `streamlit run app.py`