import os
import random
import json
import time
import threading
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# ========================
# 1. Configuración
# ========================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("en el .env no hay una api valida")

# LLM principal: Groq
llm = ChatGroq(
    model="Llama-3.1-8B-Instant",
    api_key=api_key,
    temperature=0.4,
    max_tokens=110
)

# LLM de respaldo: Hugging Face
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


def llm_huggingface_fallback(prompt_text: str) -> str:
    try:
        from transformers import pipeline
        hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not hf_api_key:
            raise ValueError("No se encontró HUGGINGFACE_API_KEY en el .env")

        generator = pipeline(
            "text-generation",
            model="tiiuae/falcon-7b-instruct",
            device=-1,
            use_auth_token=hf_api_key
        )

        output = generator(
            prompt_text,
            max_length=150,
            do_sample=True,
            top_p=0.95
        )
        return output[0]["generated_text"]
    except Exception as e:
        print("❌ Error fallback Hugging Face:", e)
        return "Lo siento, no pude generar la respuesta."


# ========================
# 2. Prompt optimizado
# ========================
Prompt_estructura = """
[META]
Eres GLY-AI, agente de GLYNNE. Tu misión: conducir una conversación con el usuario para mapear procesos empresariales y detectar oportunidades de automatización con IA. No propongas soluciones todavía. Tu objetivo es recopilar datos claros, precisos y accionables sobre procesos, roles, herramientas y dificultades.

[COMPORTAMIENTO]
1. Reconoce lo que dice el usuario brevemente (1 frase).
2. Haz 1 pregunta concreta y directa sobre procesos, roles, datos involucrados, herramientas o dificultades.
3. Profundiza en cada respuesta con preguntas de seguimiento solo cuando sea necesario.
4. Mantén un tono cercano, humano, empático y profesional.
5. Evita suposiciones, no inventes datos.
6. Responde cualquier pregunta que el usuario haga.
7. Pregunta el nombre de la empresa si aún no se ha dicho (revisa {historial}).

[FORMATO]
- Respuesta máxima: 100 palabras.
- Solo 1 pregunta por turno.
- Evita saludos repetidos.

[MEMORIA]
Últimos 2 mensajes: {historial}

[ENTRADA DEL USUARIO]
{mensaje}

RESPUESTA:
"""

prompt = PromptTemplate(
    input_variables=["rol", "mensaje", "historial"],
    template=Prompt_estructura.strip(),
)

# ========================
# 3. Estado global
# ========================
class State(TypedDict):
    mensaje: str
    rol: str
    historial: str
    respuesta: str
    user_id: str


usuarios = {}
USER_ACTIVITY = {}  # 🔹 Controla última actividad por usuario
USER_TIMEOUT = 600  # 🔹 Tiempo en segundos (10 minutos)

# ========================
# 4. Funciones auxiliares
# ========================
def get_memory(user_id: str):
    if user_id not in usuarios:
        usuarios[user_id] = ConversationBufferMemory(
            memory_key="historial",
            input_key="mensaje",
            output_key="respuesta",
            k=2
        )
    USER_ACTIVITY[user_id] = time.time()
    return usuarios[user_id]


def get_json_path(user_id: str) -> str:
    """Devuelve la ruta del JSON correspondiente a un usuario"""
    return f"conversacion_{user_id}.json"


def guardar_conversacion(user_id: str, user_msg: str, ai_resp: str):
    """Guarda cada mensaje en el JSON correspondiente al usuario"""
    path = get_json_path(user_id)

    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f)

    with open(path, "r+", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                data = []
        except json.JSONDecodeError:
            data = []

        data.append({"user": user_msg, "ai": ai_resp})
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()

    USER_ACTIVITY[user_id] = time.time()


def eliminar_json_inactivo():
    """Hilo que elimina JSON de usuarios inactivos"""
    while True:
        now = time.time()
        inactive_users = []
        for user_id, last_active in list(USER_ACTIVITY.items()):
            if now - last_active > USER_TIMEOUT:
                path = get_json_path(user_id)
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        print(f"🗑️ Eliminado JSON inactivo: {path}")
                    except Exception as e:
                        print(f"⚠️ Error al eliminar {path}: {e}")
                inactive_users.append(user_id)

        # Limpiamos estructuras
        for uid in inactive_users:
            USER_ACTIVITY.pop(uid, None)
            usuarios.pop(uid, None)

        time.sleep(60)  # 🔁 Revisión cada minuto


# 🔹 Lanzamos el hilo de limpieza en segundo plano
cleanup_thread = threading.Thread(target=eliminar_json_inactivo, daemon=True)
cleanup_thread.start()

# ========================
# 5. Nodo principal del agente
# ========================
def agente_node(state: State) -> State:
    user_id = state.get("user_id", "default")
    memory = get_memory(user_id)
    historial = memory.load_memory_variables({}).get("historial", "")

    texto_prompt = prompt.format(
        rol=state["rol"],
        mensaje=state["mensaje"],
        historial=historial
    )

    try:
        respuesta = llm.invoke(texto_prompt).content
    except Exception as e:
        print("❌ Error en Groq LLM:", e)
        respuesta = llm_huggingface_fallback(texto_prompt)

    memory.save_context({"mensaje": state["mensaje"]}, {"respuesta": respuesta})
    guardar_conversacion(user_id, state["mensaje"], respuesta)

    state["respuesta"] = respuesta
    state["historial"] = historial
    return state


# ========================
# 6. Construcción del grafo
# ========================
workflow = StateGraph(State)
workflow.add_node("agente", agente_node)
workflow.set_entry_point("agente")
workflow.add_edge("agente", END)
app = workflow.compile()


# ========================
# 7. CLI interactiva
# ========================
print("LLM iniciado con LangGraph ✅")

user_id = str(random.randint(10000, 90000))
print(f"🧩 Tu user_id es: {user_id}")

rol = "auditor"

