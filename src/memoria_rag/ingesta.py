"""Módulo de conversión e indexación de PDFs hacia ChromaDB con aceleración CUDA."""
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from memoria_rag.almacen import crear_cliente, funcion_embeddings, obtener_coleccion


def extraer_texto_pdf(ruta_pdf: str | Path) -> list[dict[str, Any]]:
    """Extrae el contenido textual de cada página de un archivo PDF usando pdftotext."""
    ruta = Path(ruta_pdf).resolve()
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo PDF en la ruta: {ruta}")

    if not shutil.which("pdftotext"):
        raise RuntimeError(
            "La utilidad 'pdftotext' no está instalada en el sistema. "
            "Instálela mediante: sudo dnf install poppler-utils (o apt install poppler-utils)."
        )

    proceso = subprocess.run(
        ["pdftotext", str(ruta), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )

    paginas_raw = proceso.stdout.split("\x0c")
    paginas: list[dict[str, Any]] = []

    for indice, texto_raw in enumerate(paginas_raw):
        num_pagina = indice + 1
        # Limpiar encabezados repetitivos de plantilla del libro (ej. E3614.book Page ...)
        lineas = [
            linea.strip()
            for linea in texto_raw.splitlines()
            if linea.strip() and not re.match(r"^E3614\.book Page \d+", linea.strip())
        ]
        texto_limpio = "\n".join(lineas).strip()
        if texto_limpio:
            paginas.append({"pagina": num_pagina, "texto": texto_limpio})

    return paginas


def dividir_en_fragmentos(
    paginas: list[dict[str, Any]],
    nombre_archivo: str = "manual_macbook",
    palabras_por_chunk: int = 150,
    solapamiento: int = 30,
) -> list[dict[str, Any]]:
    """Divide las páginas en fragmentos (chunks) con solapamiento conservando metadatos."""
    chunks: list[dict[str, Any]] = []
    chunk_global_id = 0

    for pag in paginas:
        pagina_num = pag["pagina"]
        texto = pag["texto"]
        palabras = texto.split()

        if len(palabras) <= palabras_por_chunk:
            chunks.append(
                {
                    "id": f"p{pagina_num}_c{chunk_global_id}",
                    "document": texto,
                    "metadata": {
                        "pagina": pagina_num,
                        "fuente": nombre_archivo,
                        "chunk": chunk_global_id,
                    },
                }
            )
            chunk_global_id += 1
            continue

        paso = palabras_por_chunk - solapamiento
        for i in range(0, len(palabras), paso):
            palabras_chunk = palabras[i : i + palabras_por_chunk]
            if len(palabras_chunk) < 15 and chunks:
                continue

            chunk_texto = " ".join(palabras_chunk)
            chunks.append(
                {
                    "id": f"p{pagina_num}_c{chunk_global_id}",
                    "document": chunk_texto,
                    "metadata": {
                        "pagina": pagina_num,
                        "fuente": nombre_archivo,
                        "chunk": chunk_global_id,
                    },
                }
            )
            chunk_global_id += 1

    return chunks


def indexar_pdf(
    ruta_pdf: str | Path = "ma144_macbook_pro_users_guide.pdf",
    nombre_coleccion: str = "manual_macbook",
    ruta_chroma: str = "datos/chroma",
    reiniciar_coleccion: bool = False,
    lote_tamano: int = 64,
) -> int:
    """Extrae texto, divide en fragmentos y almacena embeddings en ChromaDB con CUDA."""
    ruta = Path(ruta_pdf).resolve()
    print(f"[*] Leyendo y convirtiendo PDF: {ruta.name}...")
    paginas = extraer_texto_pdf(ruta)
    print(f"[✓] Se extrajeron {len(paginas)} páginas con contenido.")

    print("[*] Segmentando texto en fragmentos (chunks)...")
    chunks = dividir_en_fragmentos(paginas, nombre_archivo=ruta.name)
    print(f"[✓] Se generaron {len(chunks)} fragmentos.")

    print(f"[*] Conectando a ChromaDB en '{ruta_chroma}' con aceleración CUDA...")
    cliente = crear_cliente(ruta_chroma)
    emb_fn = funcion_embeddings()

    if reiniciar_coleccion:
        try:
            cliente.delete_collection(nombre_coleccion)
            print(f"[*] Colección '{nombre_coleccion}' reiniciada.")
        except Exception:
            pass

    coleccion = obtener_coleccion(cliente, nombre_coleccion, emb_fn)

    print(f"[*] Indexando {len(chunks)} fragmentos en ChromaDB...")
    for i in range(0, len(chunks), lote_tamano):
        lote = chunks[i : i + lote_tamano]
        coleccion.upsert(
            ids=[item["id"] for item in lote],
            documents=[item["document"] for item in lote],
            metadatas=[item["metadata"] for item in lote],
        )
        print(f"    - Indexados {min(i + lote_tamano, len(chunks))}/{len(chunks)} fragmentos...")

    total = coleccion.count()
    print(f"[✓] Indexación finalizada exitosamente. Total en colección '{nombre_coleccion}': {total} registros.\n")
    return total


if __name__ == "__main__":
    import sys

    ruta = sys.argv[1] if len(sys.argv) > 1 else "ma144_macbook_pro_users_guide.pdf"
    indexar_pdf(ruta_pdf=ruta, reiniciar_coleccion=True)
