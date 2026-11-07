import os
import json
import asyncio
import certifi
import chromadb
import argparse
import httpx
from openai import AsyncOpenAI
from google import genai
from google.genai import types
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, ContextRecall, ContextPrecision

# Configuración de seguridad y telemetría
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["RAGAS_DO_NOT_TRACK"] = "True"

# =====================================================================
# PEGA TU CLAVE AQUÍ (o asegúrate de que esté en tu .env)
# =====================================================================
GEMINI_KEY = os.getenv("GEMINI_KEY", "AQ.Ab8RN6LMd1CP_ZmaMB569pMezDvRozNs6Tb9r6qikLWKS4L32g")

# Cliente generador (Google GenAI) con v1beta
client = genai.Client(
    api_key=GEMINI_KEY,
    http_options=types.HttpOptions(api_version='v1beta')
)

# Cliente para RAGAS (Juez) - Usa httpx para evitar problemas de SSL en Windows
ragas_client = AsyncOpenAI(
    api_key=GEMINI_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    http_client=httpx.AsyncClient(verify=False)
)
judge = llm_factory("gemini-3.5-flash-lite", provider="openai", client=ragas_client)

# Conectar a ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="adcc_rules")

async def evaluar_pregunta(item):
    pregunta = item["question"]
    ground_truth = item["ground_truth"]
    tipo = item.get("type", "desconocido")
    
    # 1. Búsqueda (Retrieval)
    embed_res = client.models.embed_content(model="gemini-embedding-001", contents=[pregunta])
    vector = embed_res.embeddings[0].values
    results = collection.query(query_embeddings=[vector], n_results=6)
    contextos = results['documents'][0]
    contexto_unido = "\n\n".join(contextos)

    # 2. Generación (Generation)
    prompt = f"Responde EXCLUSIVAMENTE con el contexto:\n{contexto_unido}\nPregunta: {pregunta}"
    llm_res = client.models.generate_content(
        model="gemini-3.5-flash-lite", contents=prompt, 
        config=types.GenerateContentConfig(temperature=0.0)
    )
    respuesta_generada = llm_res.text

    # 3. Evaluación (RAGAS)
    print(f"  -> Evaluando [{tipo}]: '{pregunta[:40]}...'")
    
    metric_faith = Faithfulness(llm=judge)
    metric_recall = ContextRecall(llm=judge)
    metric_precision = ContextPrecision(llm=judge)

    score_faith = await metric_faith.ascore(user_input=pregunta, response=respuesta_generada, retrieved_contexts=contextos)
    score_recall = await metric_recall.ascore(user_input=pregunta, retrieved_contexts=contextos, reference=ground_truth)
    score_precision = await metric_precision.ascore(user_input=pregunta, retrieved_contexts=contextos, reference=ground_truth)

    return {
        "type": tipo,
        "pregunta": pregunta,
        "faithfulness": score_faith.value,
        "context_recall": score_recall.value,
        "context_precision": score_precision.value
    }

async def evaluar_con_reintentos(item, max_intentos=3):
    for intento in range(max_intentos):
        try:
            return await evaluar_pregunta(item)
        except Exception as e:
            if "429" in str(e) or "503" in str(e):
                print(f"     [!] Límite de cuota o saturación (503/429). Esperando 30 segundos...")
                await asyncio.sleep(30)
            else:
                print(f"     [!] Error: {e}")
                return None
    return None

async def main():
    parser = argparse.ArgumentParser(description="Evaluador RAGAS segmentado")
    parser.add_argument("--type", type=str, help="Filtrar por tipo: factual, multi_hop, negative")
    args = parser.parse_args()

    with open("benchmark.json", "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    
    if args.type:
        print(f"🔍 Filtrando solo preguntas de tipo: '{args.type}'")
        benchmark = [item for item in benchmark if item.get("type") == args.type]
        if not benchmark:
            print(f"No se encontraron preguntas de tipo '{args.type}'.")
            return

    resultados = []
    for item in benchmark:
        res = await evaluar_con_reintentos(item)
        if res:
            resultados.append(res)
        await asyncio.sleep(5) # Pausa entre preguntas

    print("\n" + "="*50)
    print(f"       RESULTADOS: {args.type.upper() if args.type else 'TODAS'}")
    print("="*50)

    if resultados:
        avg_faith = sum(r["faithfulness"] for r in resultados) / len(resultados)
        avg_recall = sum(r["context_recall"] for r in resultados) / len(resultados)
        avg_precision = sum(r["context_precision"] for r in resultados) / len(resultados)
        
        print(f"   - Faithfulness: {avg_faith:.2f}")
        print(f"   - Context Recall: {avg_recall:.2f}")
        print(f"   - Context Precision: {avg_precision:.2f}")

if __name__ == "__main__":
    asyncio.run(main())