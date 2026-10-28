"""Módulo RAG para consultas sobre el manual de MacBook Pro."""
from typing import Any, Generator
from memoria_rag.almacen import crear_cliente, funcion_embeddings, obtener_coleccion
from memoria_rag.llm import llamar_llm, stream_llm, MODELO_DEFAULT

PROMPT_SISTEMA_RAG = (
    "Usted es un asistente técnico experto en el manual de usuario del MacBook Pro "
    "(ma144_macbook_pro_users_guide.pdf).\n"
    "Instrucciones:\n"
    "1. Responda siempre en español de forma concisa, clara y estructurada.\n"
    "2. Base su respuesta ÚNICAMENTE en los fragmentos de contexto provistos.\n"
    "3. CITE SIEMPRE la página o páginas de donde extrajo los datos clave (ejemplo: [Pág. 92]).\n"
    "4. Si la respuesta no está en el contexto, indique amablemente que dicha información no figura en el manual."
)


class RAGMacBook:
    def __init__(
        self,
        ruta_chroma: str = "datos/chroma",
        nombre_coleccion: str = "manual_macbook",
        modelo_llm: str = MODELO_DEFAULT,
    ):
        self.cliente = crear_cliente(ruta_chroma)
        self.emb_fn = funcion_embeddings()
        self.coleccion = obtener_coleccion(self.cliente, nombre_coleccion, self.emb_fn)
        self.modelo_llm = modelo_llm

    def recuperar(self, pregunta: str, k: int = 4) -> list[dict[str, Any]]:
        """Recupera los k fragmentos más relevantes de ChromaDB para la pregunta."""
        total = self.coleccion.count()
        if total == 0:
            return []

        resultados = self.coleccion.query(
            query_texts=[pregunta],
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )

        fragmentos = []
        documentos = resultados.get("documents", [[]])[0]
        metadatas = resultados.get("metadatas", [[]])[0]
        distancias = resultados.get("distances", [[]])[0]

        for doc, meta, dist in zip(documentos, metadatas, distancias):
            fragmentos.append(
                {
                    "texto": doc,
                    "pagina": meta.get("pagina", "N/A"),
                    "fuente": meta.get("fuente", "manual"),
                    "distancia": dist,
                    "similitud": round(1.0 - dist, 4) if dist is not None else 0.0,
                }
            )

        return fragmentos

    def armar_contexto(self, fragmentos: list[dict[str, Any]]) -> str:
        """Formatea los fragmentos recuperados en un bloque de contexto legible."""
        if not fragmentos:
            return "No se encontraron fragmentos relevantes en el manual."

        lineas = ["### INFORMACIÓN EXTRAÍDA DEL MANUAL DEL MACBOOK PRO:"]
        for i, f in enumerate(fragmentos, 1):
            lineas.append(f"\n--- [Fragmento {i} | Página {f['pagina']}] ---")
            lineas.append(f["texto"])
        return "\n".join(lineas)

    def responder(
        self,
        pregunta: str,
        historial: list[dict[str, str]] | None = None,
        k: int = 4,
    ) -> dict[str, Any]:
        """Responde una consulta recuperando contexto y llamando al LLM local."""
        fragmentos = self.recuperar(pregunta, k=k)
        contexto = self.armar_contexto(fragmentos)

        mensajes: list[dict[str, str]] = []
        if historial:
            mensajes.extend(historial)

        prompt_usuario = (
            f"Contexto del manual:\n{contexto}\n\n"
            f"Pregunta del usuario: {pregunta}\n\n"
            "Respuesta detallada con citas de páginas:"
        )
        mensajes.append({"role": "user", "content": prompt_usuario})

        respuesta = llamar_llm(
            sistema=PROMPT_SISTEMA_RAG,
            mensajes=mensajes,
            modelo=self.modelo_llm,
        )

        paginas_citadas = sorted(list({f["pagina"] for f in fragmentos if f["pagina"] != "N/A"}))

        return {
            "respuesta": respuesta,
            "fragmentos": fragmentos,
            "paginas": paginas_citadas,
        }

    def responder_stream(
        self,
        pregunta: str,
        historial: list[dict[str, str]] | None = None,
        k: int = 4,
    ) -> tuple[Generator[str, None, None], list[dict[str, Any]]]:
        """Versión con streaming para interactuar en consola en tiempo real."""
        fragmentos = self.recuperar(pregunta, k=k)
        contexto = self.armar_contexto(fragmentos)

        mensajes: list[dict[str, str]] = []
        if historial:
            mensajes.extend(historial)

        prompt_usuario = (
            f"Contexto del manual:\n{contexto}\n\n"
            f"Pregunta del usuario: {pregunta}\n\n"
            "Respuesta estructurada y citando las páginas correspondientes:"
        )
        mensajes.append({"role": "user", "content": prompt_usuario})

        generador = stream_llm(
            sistema=PROMPT_SISTEMA_RAG,
            mensajes=mensajes,
            modelo=self.modelo_llm,
        )

        return generador, fragmentos
