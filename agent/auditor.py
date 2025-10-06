# auditor.py
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# ========================
# 1. Cargar entorno y API
# ========================
load_dotenv()
api_key = os.getenv('GROQ_API_KEY2')
hf_api_key = os.getenv('HUGGINGFACE_API_KEY2')

if not api_key:
    raise ValueError('No hay una API key válida de Groq en el .env')

if not hf_api_key:
    raise ValueError('No hay una API key válida de Hugging Face en el .env')

# ========================
# 2. Inicializar LLM principal (Groq)
# ========================
llm = ChatGroq(
    model="Llama-3.1-8B-Instant",
    api_key=api_key,
    temperature=0.7,
    max_tokens=1500
)

# ========================
# 2b. Fallback Hugging Face
# ========================
def llm_huggingface_fallback(prompt_text: str) -> str:
    try:
        from transformers import pipeline
        generator = pipeline(
            "text-generation",
            model="tiiuae/falcon-7b-instruct",
            device=-1,
            use_auth_token=hf_api_key
        )
        output = generator(
            prompt_text,
            max_length=500,
            do_sample=True,
            top_p=0.95
        )
        return output[0]["generated_text"]
    except Exception as e:
        print("❌ Error fallback Hugging Face:", e)
        return "Lo siento, no pude generar la auditoría."

# ========================
# 3. Prompt de auditoría
# ========================
PROMPT_SOLUCIONES_GLYNNE = """
[META]
Fecha del reporte: {fecha}
Tu meta es analizar el negocio del usuario usando la conversación histórica.
Genera un documento profesional, corporativo y estructurado, centrado en **proponer soluciones de software e inteligencia artificial** para optimizar los procesos, eliminar cuellos de botella y mejorar la eficiencia del negocio.

Además, en cada sección donde sea relevante, menciona explícitamente **cómo GLYNNE AI como empresa puede implementar estas soluciones**, adaptarlas a la organización del cliente y garantizar la automatización efectiva de sus procesos.

Sigue estos apartados:

1. Portada - Incluye el nombre de la empresa (si se menciona), el consultor (GLY-AI) y la fecha.
2. Resumen ejecutivo - Breve descripción de los procesos actuales, retos detectados y cómo GLYNNE AI puede ayudar a solucionarlos.
3. Alcance y objetivos - Define qué procesos o áreas se pueden optimizar mediante software e IA según la conversación, y cómo GLYNNE AI adaptaría estas soluciones.
4. Metodología - Explica cómo se analiza la información de la conversación para proponer soluciones concretas y escalables, y cómo GLYNNE AI garantiza su correcta implementación.
5. Procesos y oportunidades de mejora - Para cada proceso mencionado:
    - Describe los cuellos de botella o problemas detectados.
    - Propón soluciones tecnológicas específicas (agentes inteligentes, automatización, flujos de datos, integración de APIs, dashboards, etc.).
    - Explica cómo **GLYNNE AI implementaría y adaptaría estas soluciones** al cliente, asegurando resultados prácticos.
6. Recomendaciones - Estrategias concretas de implementación: tecnologías sugeridas, arquitecturas posibles, flujos automatizables y prioridades, siempre integrando el enfoque de GLYNNE AI como consultor activo.
7. Conclusiones - Beneficios esperados al implementar estas soluciones de software e IA y cómo GLYNNE AI ayuda a escalar y optimizar el negocio del cliente.
8. Anexos - Incluye fragmentos relevantes de la conversación que sirvan como evidencia o contexto de las soluciones propuestas.

Cada apartado debe tener al menos un párrafo completo, técnico y contextual, basado únicamente en lo que el usuario comunicó en el historial. No inventes datos, pero sí extrapola soluciones prácticas y la forma en que GLYNNE AI las aplicaría.

[ENTRADA DEL USUARIO]
Historial de conversación: {historial}

Respuesta:
"""

prompt_template = PromptTemplate(
    input_variables=["historial", "fecha"],
    template=PROMPT_SOLUCIONES_GLYNNE.strip()
)

# ========================
# 4. Función para generar auditoría
# ========================
def generar_auditoria():
    json_path = "conversacion_temp.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No se encontró el archivo: {json_path}")

    # Leer conversación
    with open(json_path, "r", encoding="utf-8") as f:
        conversacion = json.load(f)

    # Formatear conversación
    historial_texto = ""
    for intercambio in conversacion:
        if isinstance(intercambio, dict):
            historial_texto += f"Usuario: {intercambio.get('user', '')}\n"
            historial_texto += f"GLY-AI: {intercambio.get('ai', '')}\n"

    # Obtener fecha actual
    fecha_actual = datetime.now().strftime("%d/%m/%Y")

    # Crear prompt con fecha
    prompt_text = prompt_template.format(historial=historial_texto, fecha=fecha_actual)

    # Llamar LLM principal con fallback
    try:
        respuesta = llm.invoke(prompt_text)
        texto_final = respuesta.content if hasattr(respuesta, "content") else str(respuesta)
    except Exception as e:
        print("❌ Error en Groq LLM:", e)
        texto_final = llm_huggingface_fallback(prompt_text)

    # Limpiar el archivo JSON después de usarlo
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print("✅ Archivo de conversación limpiado después de generar la auditoría.")
    except Exception as e:
        print("❌ Error al limpiar el archivo JSON:", e)

    return texto_final

# ========================
# 5. CLI opcional
# ========================
if __name__ == "__main__":
    print("LLM Auditoría iniciado")
    try:
        resultado = generar_auditoria()
        print("\n===== AUDITORÍA =====\n")
        print(resultado)
        print("\n=====================\n")
    except Exception as e:
        print("❌ Error general:", e)
