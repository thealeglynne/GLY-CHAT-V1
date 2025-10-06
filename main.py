import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import json
import traceback
import time
import threading

# Importamos el agente desde tu script
from agent.chat import agente_node, get_memory, State

# 🚀 Importar función de auditoría
from agent.auditor import generar_auditoria as auditor_llm

# 🚀 Importar generador de ecosistema
from agent.diagrama import generar_ecosistema

# ========================
# 🔹 Gestión de JSON por usuario
# ========================
USER_ACTIVITY = {}
USER_TIMEOUT = 7200  # 2 horas
JSON_DIR = "conversaciones"

if not os.path.exists(JSON_DIR):
    os.makedirs(JSON_DIR)

def get_json_path(user_id: str) -> str:
    """Devuelve la ruta del JSON correspondiente a un usuario"""
    return os.path.join(JSON_DIR, f"conversacion_{user_id}.json")

def actualizar_actividad(user_id: str):
    USER_ACTIVITY[user_id] = time.time()

def eliminar_json_inactivo():
    """Hilo que elimina los JSON que no se han usado en más de 2 horas"""
    while True:
        ahora = time.time()
        for user_id, last_used in list(USER_ACTIVITY.items()):
            if ahora - last_used > USER_TIMEOUT:
                path = get_json_path(user_id)
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        print(f"🗑️ Eliminado JSON inactivo: {path}")
                    except Exception as e:
                        print(f"⚠️ Error al eliminar {path}: {e}")
                USER_ACTIVITY.pop(user_id, None)
        time.sleep(300)  # cada 5 minutos

cleanup_thread = threading.Thread(target=eliminar_json_inactivo, daemon=True)
cleanup_thread.start()

# ========================
# 1. Inicialización FastAPI
# ========================
app = FastAPI(
    title="GLYNNE LLM API",
    description="API para interactuar con GLY-AI mediante LangGraph",
    version="1.0"
)

# ========================
# 2. Middleware CORS
# ========================
origins = [
    "https://glynne-sst-ai-hsiy.vercel.app",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# 3. Modelo de datos
# ========================
class ChatRequest(BaseModel):
    mensaje: str
    rol: Optional[str] = "auditor"
    user_id: str  # obligatorio desde frontend

class ChatResponse(BaseModel):
    respuesta: str
    historial: dict

# ========================
# 4. Endpoints Chat
# ========================
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.user_id:
        raise HTTPException(status_code=400, detail="user_id es obligatorio")

    state: State = {
        "mensaje": request.mensaje,
        "rol": request.rol,
        "historial": "",
        "respuesta": "",
        "user_id": request.user_id
    }

    try:
        result = agente_node(state)
        memoria = get_memory(request.user_id).load_memory_variables({})
        actualizar_actividad(request.user_id)
        return ChatResponse(respuesta=result.get("respuesta", ""), historial=memoria)

    except Exception as e:
        print("❌ Error en /chat endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.get("/user/{user_id}/memory")
def get_user_memory(user_id: str):
    try:
        memoria = get_memory(user_id).load_memory_variables({})
        actualizar_actividad(user_id)
        return {"user_id": user_id, "historial": memoria}
    except Exception as e:
        print("❌ Error en /user/{user_id}/memory:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.get("/reset/{user_id}")
def reset_conversacion(user_id: str):
    """Elimina el JSON temporal del usuario y reinicia memoria"""
    try:
        path = get_json_path(user_id)
        if os.path.exists(path):
            os.remove(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f)
        actualizar_actividad(user_id)
        return {"status": "ok", "message": f"Conversación reiniciada para {user_id}"}
    except Exception as e:
        print("❌ Error en /reset endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ========================
# 5. Endpoint Auditoría
# ========================
@app.post("/generar_auditoria")
def generar_auditoria(user_id: str):
    """
    Genera auditoría real llamando al LLM con la conversación
    y también genera el ecosistema de nodos.
    """
    try:
        path = get_json_path(user_id)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="No hay conversación para generar auditoría")

        # 1️⃣ Generar auditoría
        resultado_auditoria = auditor_llm(user_id)

        # 2️⃣ Generar ecosistema con base en la auditoría
        resultado_ecosistema = generar_ecosistema(json.dumps(resultado_auditoria, ensure_ascii=False))
        actualizar_actividad(user_id)

        return {
            "mensaje": "✅ Auditoría y Ecosistema generados correctamente",
            "auditoria": resultado_auditoria,
            "ecosistema": resultado_ecosistema
        }

    except Exception as e:
        print("❌ Error en /generar_auditoria endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ========================
# 6. Endpoint Auditoría JSON
# ========================
@app.get("/generar_auditoria/json/{user_id}")
def generar_auditoria_json(user_id: str):
    """
    Devuelve la misma auditoría pero directamente en formato JSON.
    """
    try:
        path = get_json_path(user_id)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="No hay conversación para generar auditoría")

        resultado = auditor_llm(user_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(resultado, f, ensure_ascii=False, indent=4)
        actualizar_actividad(user_id)

        return resultado

    except Exception as e:
        print("❌ Error en /generar_auditoria/json endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ========================
# 7. Endpoint Ecosistema
# ========================
@app.post("/generar_ecosistema/{user_id}")
def generar_ecosistema_endpoint(user_id: str):
    """
    Genera el ecosistema de nodos basado en la conversación actual.
    """
    try:
        path = get_json_path(user_id)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="No hay conversación para generar el ecosistema")

        with open(path, "r", encoding="utf-8") as f:
            conversacion = json.load(f)

        if not conversacion:
            raise HTTPException(status_code=400, detail="La conversación está vacía")

        historial_texto = ""
        for intercambio in conversacion:
            historial_texto += f"Usuario: {intercambio.get('user', '')}\n"
            historial_texto += f"GLY-AI: {intercambio.get('ai', '')}\n"

        resultado_ecosistema = generar_ecosistema(historial_texto)
        actualizar_actividad(user_id)

        return {
            "mensaje": "✅ Ecosistema generado correctamente",
            "ecosistema": resultado_ecosistema
        }

    except Exception as e:
        print("❌ Error en /generar_ecosistema endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ========================
# 8. Entrypoint uvicorn
# ========================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
