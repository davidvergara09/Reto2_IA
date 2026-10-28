"""Punto de entrada principal para el chatbot RAG."""
import sys
from memoria_rag.chat import iniciar_chat
from memoria_rag.llm import MODELO_DEFAULT


def main():
    modelo = sys.argv[1] if len(sys.argv) > 1 else MODELO_DEFAULT
    iniciar_chat(modelo=modelo)


if __name__ == "__main__":
    main()
