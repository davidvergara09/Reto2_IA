#!/usr/bin/env python3
"""Script directo para iniciar el Chatbot RAG con Ollama y CUDA."""
import sys
from pathlib import Path

# Agregar src al path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from memoria_rag.chat import iniciar_chat
from memoria_rag.llm import MODELO_DEFAULT

if __name__ == "__main__":
    modelo = sys.argv[1] if len(sys.argv) > 1 else MODELO_DEFAULT
    iniciar_chat(modelo=modelo)
