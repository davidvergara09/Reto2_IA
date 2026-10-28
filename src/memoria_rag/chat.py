"""Chatbot interactivo con RAG sobre el manual de MacBook Pro usando Ollama y CUDA."""
import os
import sys
import torch
from memoria_rag.almacen import crear_cliente
from memoria_rag.ingesta import indexar_pdf
from memoria_rag.llm import MODELO_DEFAULT, OLLAMA_HOST
from memoria_rag.rag import RAGMacBook


def verificar_estado(ruta_chroma: str = "datos/chroma", nombre_coleccion: str = "manual_macbook"):
    """Verifica si ChromaDB ya cuenta con los fragmentos del PDF indexados."""
    cliente = crear_cliente(ruta_chroma)
    try:
        coleccion = cliente.get_collection(nombre_coleccion)
        conteo = coleccion.count()
        return conteo
    except Exception:
        return 0


def iniciar_chat(modelo: str = MODELO_DEFAULT):
    print("=" * 70)
    print("  🤖 CHATBOT RAG - MANUAL DE USUARIO MACBOOK PRO")
    print("=" * 70)

    # Verificación de CUDA
    cuda_disponible = torch.cuda.is_available()
    dispositivo = torch.cuda.get_device_name(0) if cuda_disponible else "CPU (sin CUDA)"
    print(f" • Aceleración GPU : {'CUDA Activado' if cuda_disponible else 'Desactivado'} ({dispositivo})")
    print(f" • Servidor LLM   : Ollama en {OLLAMA_HOST}")
    print(f" • Modelo inicial : {modelo} (Óptimo para 8GB VRAM)")
    print(f" • Base vectorial : ChromaDB (datos/chroma)")

    total_chunks = verificar_estado()
    if total_chunks == 0:
        print("\n[!] No se encontraron documentos indexados en ChromaDB.")
        respuesta = input("¿Desea procesar e indexar el archivo 'ma144_macbook_pro_users_guide.pdf' ahora? [S/n]: ").strip().lower()
        if respuesta in {"", "s", "si", "y", "yes"}:
            indexar_pdf()
            total_chunks = verificar_estado()
        else:
            print("No se puede iniciar el RAG sin documentos indexados. Saliendo...")
            return

    print(f" • Fragmentos RAG : {total_chunks} fragmentos listos para consulta")
    print("-" * 70)
    print("Comandos disponibles:")
    print("  'salir' / 'exit'      - Terminar el chat")
    print("  'limpiar' / 'clear'   - Reiniciar el historial de la conversación")
    print("  'fuentes'             - Ver el contenido exacto de los fragmentos citados")
    print("  'modelo <nombre>'     - Cambiar el modelo de Ollama (ej: 'modelo qwen2.5:7b')")
    print("=" * 70)
    print()

    rag = RAGMacBook(modelo_llm=modelo)
    historial: list[dict[str, str]] = []
    ultimos_fragmentos: list[dict] = []

    while True:
        try:
            pregunta = input("\n\033[1;36mUsted\033[0m: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nHasta pronto.")
            break

        if not pregunta:
            continue

        comando = pregunta.lower()
        if comando in {"salir", "exit", "quit"}:
            print("Sesión finalizada. ¡Hasta luego!")
            break

        if comando in {"limpiar", "clear"}:
            historial.clear()
            ultimos_fragmentos.clear()
            print("[✓] Historial de conversación reiniciado.")
            continue

        if comando == "fuentes":
            if not ultimos_fragmentos:
                print("[!] Aún no se han consultado fragmentos en esta sesión.")
            else:
                print("\n--- FRAGMENTOS DE LA ÚLTIMA CONSULTA ---")
                for idx, f in enumerate(ultimos_fragmentos, 1):
                    print(f"\n[Fuente {idx} | Página {f['pagina']} | Similitud: {f['similitud']}]")
                    print(f["texto"])
                print("-" * 40)
            continue

        if comando.startswith("modelo "):
            nuevo_modelo = pregunta.split(" ", 1)[1].strip()
            if nuevo_modelo:
                rag.modelo_llm = nuevo_modelo
                print(f"[✓] Modelo LLM cambiado a: {nuevo_modelo}")
            continue

        if comando == "ayuda":
            print("Escriba su pregunta sobre el MacBook Pro o use: 'salir', 'limpiar', 'fuentes', 'modelo <nombre>'")
            continue

        # Proceso de RAG con streaming
        print("\n\033[1;32mAsistente (RAG)\033[0m: ", end="", flush=True)

        try:
            generador, fragmentos = rag.responder_stream(
                pregunta=pregunta,
                historial=historial[-6:],  # ventana de memoria reciente
                k=4,
            )
            ultimos_fragmentos = fragmentos

            partes_respuesta = []
            for chunk in generador:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                partes_respuesta.append(chunk)
            print()

            respuesta_completa = "".join(partes_respuesta)

            # Mostrar resumen de páginas referenciadas
            paginas = sorted(list({str(f["pagina"]) for f in fragmentos if f.get("pagina")}))
            if paginas:
                print(f"\033[90m[Páginas consultadas en el manual: {', '.join(paginas)}]\033[0m")

            # Actualizar historial
            historial.append({"role": "user", "content": pregunta})
            historial.append({"role": "assistant", "content": respuesta_completa})

        except Exception as e:
            print(f"\n[Error durante la generación]: {e}")


if __name__ == "__main__":
    modelo_arg = sys.argv[1] if len(sys.argv) > 1 else MODELO_DEFAULT
    iniciar_chat(modelo=modelo_arg)
