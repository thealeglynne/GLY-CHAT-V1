import os, random, requests, datetime
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
serper_api_key = os.getenv("SERPER_API_KEY")

if not api_key:
    raise ValueError("en el .env no hay una api valida de GROQ")

if not serper_api_key:
    raise ValueError("en el .env no hay una api valida de SERPER")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0.4,
)

# ========================
# 2. Prompt
# ========================
Prompt_estructura = """
[CONTEXTO]
Hoy es {fecha}.
Eres GLY-AI, un modelo de inteligencia artificial desarrollado por GLYNNE S.A.S.
Tu rol es ser un guía experto en todo lo relacionado con inteligencia artificial: responder dudas, explicar conceptos, orientar sobre herramientas y tendencias.
No estás recolectando información del usuario; tu objetivo es conversar de forma natural y fluida, generando una interacción amigable y educativa.
Si el usuario menciona procesos de automatización empresarial, sugiere el uso de nuestro módulo de análisis y generación de auditorías.
Si habla sobre bases de datos o manejo de datos, sugiere nuestro analizador de DB.

Responde de forma natural y conversacional:
- Mantén la conversación guiada hacia temas de IA.
- Introduce preguntas tuyas únicamente si ayudan a profundizar o aclarar conceptos.
- Si hay resultados de búsqueda disponibles, intégralos sin sonar forzado.
- Si no hay resultados de búsqueda, responde con tu conocimiento previo, indicando brevemente si puede estar desactualizado.
- Prioriza claridad, utilidad y concisión, evitando advertencias repetitivas innecesarias.

[BUSQUEDA WEB]
{busqueda}

[MEMORIA]
{historial}

[ENTRADA DEL USUARIO]
Consulta: {mensaje}

[RESPUESTA COMO {rol}]
"""


prompt = PromptTemplate(
    input_variables=["rol", "mensaje", "historial", "busqueda", "fecha"],
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
    busqueda: str
    user_id: str

usuarios = {}

def get_memory(user_id: str):
    if user_id not in usuarios:
        usuarios[user_id] = ConversationBufferMemory(
            memory_key="historial", input_key="mensaje"
        )
    return usuarios[user_id]

# ========================
# 4. Nodo de búsqueda Serper
# ========================
def serper_node(state: State) -> State:
    try:
        q = state.get("mensaje", "")
        headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}
        resp = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            json={"q": q},
        )
        data = resp.json()
        if "organic" in data and len(data["organic"]) > 0:
            resumen = [
                f"{item.get('title')} - {item.get('link','')}"
                for item in data["organic"][:3]
            ]
            state["busqueda"] = " | ".join(resumen)
        else:
            state["busqueda"] = "No hubo resultados"
    except Exception as e:
        state["busqueda"] = f"Error en búsqueda: {e}"
    
    print("DEBUG búsqueda:", state["busqueda"])  # 👈 Debug
    return state

# ========================
# 5. Nodo agente (usa búsqueda + historial)
# ========================
def agente_node(state: State) -> State:
    memory = get_memory(state.get("user_id", "default"))
    historial = memory.load_memory_variables({}).get("historial", "")
    fecha = datetime.date.today().strftime("%d/%m/%Y")

    # 🔑 Lógica: decidir si activar búsqueda
    activar_busqueda = any(
        palabra in state["mensaje"].lower()
        for palabra in ["quién", "cuándo", "actual", "último", "presidente", "hoy", "noticia", "última hora"]
    )

    if activar_busqueda:
        state = serper_node(state)  # ejecuta búsqueda solo si hace falta

  

    texto_prompt = prompt.format(
        rol=state["rol"],
        mensaje=state["mensaje"],
        historial=historial,
        busqueda=state.get("busqueda", ""),
        fecha=fecha,
    )
    respuesta = llm.invoke(texto_prompt).content

    # Guardar en memoria
    memory.save_context({"mensaje": state["mensaje"]}, {"respuesta": respuesta})

    state["respuesta"] = respuesta
    state["historial"] = historial
    return state

# ========================
# 6. Grafo
# ========================
workflow = StateGraph(State)
workflow.add_node("serper", serper_node)
workflow.add_node("agente", agente_node)

workflow.set_entry_point("agente")   # 👈 ahora empieza en el agente
workflow.add_edge("agente", END)

app = workflow.compile()

# ========================
# 7. CLI
# ========================
print("LLM iniciado con LangGraph + Serper dinámico")
user_id = str(random.randint(10000, 90000))
rol = "auditor"
print(f"tu user id es {user_id}")

