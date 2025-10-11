import os
import random
import json
from dotenv import load_dotenv
from typing import TypedDict
from datetime import datetime
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

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0.4,
     
)

# ========================
# 2. Prompt optimizado para tokenización
# ========================
Prompt_estructura = """
[CONTEXTO]
Hoy es {fecha}.
Eres GLY-AI, un modelo de inteligencia artificial desarrollado por GLYNNE S.A.S.
Tu rol es ser un guía experto en inteligencia artificial: responder dudas, explicar conceptos y orientar sobre herramientas y tendencias. No recolectas información del usuario; solo conversas de forma natural y fluida.
contesta con uun maximo de 100 palabras 
si el usuario dice que quiere automatizar procesos empresariales sugiere la herramienta de auditoria de GLY ai informa que esta en el + de el input 
si menciona algo relacionado a bases de datos recomiendale que gly le analiza sus datos a prrofundiidad ai informa que esta en el + de el input 
si el usuario habla de aprender ia a demas de darle contexto le informas que en GLYNNE COLLEGE puede aprender desde ciencia de datos hasta desarrollo de prrocesos de automatizacion con procesos cognitivos de ia pasa este link solo cuando lo mnciopnes 'https://www.glynneai.com/glynneColege'
si pregunta sobre GLYNNE somos una empresa dedicada al desarrollo de infraestructura de software para la automatizacion de procesos en si desarrollamos iinteligencia artificial de todo tipo empresarial  
[MEMORIA]
Últimos 3 mensajes: {historial}

[ENTRADA DEL USUARIO]
Consulta: {mensaje}

[RESPUESTA COMO {rol}]
"""

prompt = PromptTemplate(
    input_variables=["rol", "mensaje", "historial", "fecha"],
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

# memoria por usuario
usuarios = {}

def get_memory(user_id: str):
    if user_id not in usuarios:
        # limitar memoria para reducir tokens: solo últimos 3 mensajes
        usuarios[user_id] = ConversationBufferMemory(
            memory_key="historial",
            input_key="mensaje",
            output_key="respuesta",
            k=3
        )
    return usuarios[user_id]

# ========================
# 4. Función de almacenamiento temporal en JSON
# ========================
TEMP_JSON_PATH = "conversacion_temp.json"

if not os.path.exists(TEMP_JSON_PATH):
    with open(TEMP_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)

def guardar_conversacion(user_msg: str, ai_resp: str):
    with open(TEMP_JSON_PATH, "r+", encoding="utf-8") as f:
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

# ========================
# 5. Nodo principal
# ========================
def agente_node(state: State) -> State:
    memory = get_memory(state.get("user_id", "default"))
    historial = memory.load_memory_variables({}).get("historial", "")

    # limitar historial a últimos 3 mensajes (si k falla)
    if historial:
        lineas = historial.strip().split("\n")
        if len(lineas) > 6:  # cada intercambio ≈ 2 líneas
            historial = "\n".join(lineas[-6:])

    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    texto_prompt = prompt.format(
        rol=state["rol"],
        mensaje=state["mensaje"],
        historial=historial,
        fecha=fecha_actual
    )

    respuesta = llm.invoke(texto_prompt).content

    # guardar en memoria
    memory.save_context({"mensaje": state["mensaje"]}, {"respuesta": respuesta})

    # guardar en JSON temporal
    guardar_conversacion(state["mensaje"], respuesta)

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
print("LLM iniciado con LangGraph")

user_id = str(random.randint(10000, 90000))
print(f"tu user id es {user_id}")

rol = "auditor"
