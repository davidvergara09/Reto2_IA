"""Almacén vectorial compartido con soporte ChromaDB y aceleración CUDA."""
import os
from pathlib import Path
import torch
import chromadb
from chromadb.utils import embedding_functions

# Evita consultas innecesarias a Hugging Face si el modelo ya está en caché local
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"


def crear_cliente(ruta: str = "datos/chroma") -> chromadb.ClientAPI:
    """Crea (o abre) una base vectorial persistente en disco."""
    Path(ruta).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=ruta)


def funcion_embeddings(device: str | None = None):
    """Modelo multilingüe ejecutado sobre GPU NVIDIA (CUDA) si está disponible."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        device=device,
    )


def obtener_coleccion(cliente, nombre: str = "manual_macbook", funcion_emb=None):
    """Devuelve una colección que mide la similitud con la distancia coseno."""
    return cliente.get_or_create_collection(
        name=nombre,
        embedding_function=funcion_emb or funcion_embeddings(),
        metadata={"hnsw:space": "cosine"},
    )
