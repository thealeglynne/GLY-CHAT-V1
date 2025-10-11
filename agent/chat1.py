
import os, random
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

# memoria por usuario
usuarios = {}

def get_memory(user_id: str):
    if user_id not in usuarios:
        usuarios[user_id] = ConversationBufferMemory(
            memory_key="historial", input_key="mensaje"
        )
    return usuarios[user_id]

# ========================
# 4. Nodo principal
# ========================
def agente_node(state: State) -> State:
    memory = get_memory(state.get("user_id", "default"))
    historial = memory.load_memory_variables({}).get("historial", "")

    texto_prompt = prompt.format(
        rol=state["rol"], mensaje=state["mensaje"], historial=historial
    )
    respuesta = llm.invoke(texto_prompt).content

    # guardar en memoria
    memory.save_context({"mensaje": state["mensaje"]}, {"respuesta": respuesta})

    # actualizar estado
    state["respuesta"] = respuesta
    state["historial"] = historial
    return state

# ========================
# 5. Construcción del grafo
# ========================
workflow = StateGraph(State)
workflow.add_node("agente", agente_node)
workflow.set_entry_point("agente")
workflow.add_edge("agente", END)
app = workflow.compile()

# ========================
# 6. CLI interactiva
# ========================
print("LLM iniciado con LangGraph")

roles = {
    "auditor": "actua como un auditor empresarial...",
    "desarrollador": "explica con detalle técnico...",
    "vendedor": "vende software con mala técnica...",
}

user_id = str(random.randint(10000, 90000))
print(f"tu user id es {user_id}")

rol = "auditor"

while True:
    try:
        user_input = input("Tu: ")
        if user_input.lower() == "salir":
            break

        if user_input.startswith("/rol "):
            nuevo_rol = user_input.split("/rol ", 1)[1].lower().strip()
            if nuevo_rol in roles:
                rol = nuevo_rol
                print(f"✅ tu nuevo rol es {nuevo_rol}")
            else:
                print("⚠️ rol no disponible")
            continue

        # ejecutar grafo
        result = app.invoke(
            {"mensaje": user_input, "rol": rol, "historial": "", "user_id": user_id}
        )
        print("LLM:", result["respuesta"])
        print("📝 memoria:", get_memory(user_id).load_memory_variables({}))
    except Exception as e:
        print("❌ Error:", str(e))
        break
