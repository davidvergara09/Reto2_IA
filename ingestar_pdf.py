#!/usr/bin/env python3
"""Script directo para convertir el PDF e indexarlo en ChromaDB con CUDA."""
import sys
from pathlib import Path

# Agregar src al path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from memoria_rag.ingesta import indexar_pdf

if __name__ == "__main__":
    pdf_archivo = sys.argv[1] if len(sys.argv) > 1 else "ma144_macbook_pro_users_guide.pdf"
    print(f"Iniciando proceso de conversión e indexación para: {pdf_archivo}")
    indexar_pdf(ruta_pdf=pdf_archivo, reiniciar_coleccion=True)
