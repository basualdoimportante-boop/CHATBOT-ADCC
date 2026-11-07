# 🥋 Chatbot RAG - Reglamento ADCC 2026

Sistema de Generación Aumentada por Recuperación (RAG) diseñado para responder preguntas sobre el reglamento oficial de Jiu-Jitsu ADCC, garantizando cero alucinaciones y citando las fuentes exactas.

## 🏗️ Arquitectura
- **LLM:** `gemini-3.5-flash-lite` (vía API de Google).
- **Embeddings:** `gemini-embedding-001`.
- **Base de Datos Vectorial:** `ChromaDB` (almacenamiento local con distancia coseno).
- **Interfaz:** `Streamlit`.
- **Framework de Evaluación:** `RAGAS` (Faithfulness, Context Recall, Context Precision).

## 📊 Estrategia y Resultados de Evaluación
Para validar la fiabilidad del sistema, diseñé un *benchmark* estratificado de 10 preguntas divididas en 3 categorías para aislar fallos de generación vs. fallos de recuperación:

| Categoría | Preguntas | Faithfulness (Cero Alucinaciones) | Context Recall (No perder info) | Context Precision (No traer ruido) |
| :--- | :--- | :--- | :--- | :--- |
| **Factual** | 6 | `1.00` | `0.67` | `0.75` |
| **Multi-hop** | 2 | `1.00` | `0.75` | `0.25` |
| **Negativas** (Abstención) | 2 | `1.00` | `1.00` | `0.00`* |

*\*Nota: La precisión en preguntas negativas es 0.00 porque, al no existir la respuesta en el documento, matemáticamente ningún fragmento recuperado es "relevante". El sistema respondió correctamente "No tengo información" (Faithfulness 1.0).*

### Conclusiones del Benchmark:
1. **Cero Alucinaciones:** El sistema es 100% fiable en cuanto a no inventar datos (Faithfulness perfecto en todas las categorías).
2. **Área de mejora (Precisión):** El *Context Precision* en preguntas Multi-hop (0.25) indica que el sistema trae fragmentos de más. La solución propuesta a futuro es implementar un **Reranker** para filtrar el ruido antes de enviarlo al LLM.

## 🚀 Cómo ejecutar el proyecto localmente

1. Clonar el repositorio.
2. Crear un entorno virtual e instalar dependencias:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .\.venv\Scripts\activate
   pip install -r requirements.txt