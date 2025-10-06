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
# 2b. LLM de fallback Hugging Face vía API
# ========================
def llm_huggingface_fallback(prompt_text: str) -> str:
    """
    Fallback a Hugging Face usando API Key
    """
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
Prompt_estructura = """
[META]
Fecha del reporte: {fecha}
Tu meta es analizar el negocio del usuario usando la conversación histórica.
Genera un documento de auditoría profesional, corporativo y estructurado, centrado en identificar
cómo las soluciones de automatización impulsadas por inteligencia artificial pueden integrarse
en los procesos, problemas o necesidades expresadas por el usuario.

El documento debe tener un enfoque estratégico y técnico, mostrando cómo la IA puede
mejorar eficiencia, escalabilidad, comunicación interna, toma de decisiones y reducción de tareas repetitivas.

Estructura el informe con los siguientes apartados:

1. Portada  
   - Incluye el nombre del usuario o empresa si se menciona, el auditor (GLY-IA), y la fecha.  

2. Resumen ejecutivo  
   - Resume brevemente la situación actual del negocio y las oportunidades de automatización con IA detectadas.  

3. Alcance y objetivos  
   - Define los procesos o áreas que se pueden beneficiar de la automatización según el historial de conversación.  

4. Metodología  
   - Explica cómo se analizaron los datos y cómo se plantea identificar flujos automatizables usando modelos de lenguaje, agentes inteligentes, o integración de sistemas.  

5. Procesos auditados y hallazgos  
   - Describe cada proceso o área mencionada por el usuario.  
   - Detalla los puntos críticos, tareas repetitivas o cuellos de botella y cómo pueden automatizarse mediante IA (por ejemplo, agentes, APIs, flujos conversacionales o sistemas de orquestación).  

6. Recomendaciones  
   - Propón estrategias concretas para aplicar IA en los flujos de trabajo del usuario.  
   - Menciona posibles arquitecturas, integración de agentes, automatización de departamentos o conexión de datos empresariales.  

7. Conclusiones  
   - Sintetiza los beneficios esperados al implementar la automatización basada en IA y cómo esto puede escalar el negocio.  

8. Anexos  
   - Incluye fragmentos relevantes de la conversación que sirvan como evidencia o contexto.  

Cada apartado debe tener al menos un párrafo completo, técnico y contextual.  
No inventes información, usa únicamente lo que el usuario comunicó en el historial, extrapolando cómo se podrían aplicar soluciones inteligentes.

[ENTRADA DEL USUARIO]
Historial de conversación: {historial}

Respuesta:
"""


prompt_template = PromptTemplate(
    input_variables=["historial", "fecha"],
    template=Prompt_estructura.strip()
)

# ========================
# 4. Función para generar auditoría por usuario
# ========================
def generar_auditoria(user_id: str):
    """
    Genera una auditoría basada en el JSON correspondiente a un usuario específico.
    Compatible con la estructura del main.py
    """
    json_path = os.path.join("conversaciones", f"conversacion_{user_id}.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No se encontró la conversación del usuario {user_id}")

    # Leer conversación del usuario
    with open(json_path, "r", encoding="utf-8") as f:
        conversacion = json.load(f)

    if not conversacion:
        raise ValueError(f"La conversación del usuario {user_id} está vacía.")

    # Formatear conversación para el prompt
    historial_texto = ""
    for intercambio in conversacion:
        historial_texto += f"Usuario: {intercambio.get('user', '')}\n"
        historial_texto += f"GLY-AI: {intercambio.get('ai', '')}\n"

    # Obtener fecha actual
    fecha_actual = datetime.now().strftime("%d/%m/%Y")

    # Crear prompt con fecha y conversación
    prompt_text = prompt_template.format(historial=historial_texto, fecha=fecha_actual)

    # ===========================
    # Ejecutar LLM principal con fallback
    # ===========================
    try:
        respuesta = llm.invoke(prompt_text)
        texto_final = respuesta.content if hasattr(respuesta, "content") else str(respuesta)
    except Exception as e:
        print("❌ Error en Groq LLM:", e)
        texto_final = llm_huggingface_fallback(prompt_text)

    # ===========================
    # Guardar auditoría generada en archivo separado
    # ===========================
    auditoria_path = os.path.join("conversaciones", f"auditoria_{user_id}.json")
    try:
        with open(auditoria_path, "w", encoding="utf-8") as f:
            json.dump({
                "user_id": user_id,
                "fecha": fecha_actual,
                "auditoria": texto_final
            }, f, ensure_ascii=False, indent=4)
        print(f"✅ Auditoría guardada en: {auditoria_path}")
    except Exception as e:
        print(f"❌ Error al guardar auditoría de {user_id}:", e)

    return texto_final

# ========================
# 5. CLI opcional para pruebas
# ========================
if __name__ == "__main__":
    print("LLM Auditoría iniciado")
    try:
        test_user = input("🧩 Ingrese user_id para generar la auditoría: ").strip()
        resultado = generar_auditoria(test_user)
        print("\n===== AUDITORÍA =====\n")
        print(resultado)
        print("\n=====================\n")
    except Exception as e:
        print("❌ Error general:", e)
