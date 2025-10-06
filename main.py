# main.py
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
from datetime import datetime
from dotenv import load_dotenv

# ========================
# 🔹 Configuración y LLM
# ========================
load_dotenv()
from agent.chat import agente_node, get_memory, State
from agent.auditor import generar_auditoria as auditor_llm
from agent.diagrama import generar_ecosistema

USER_ACTIVITY = {}
USER_TIMEOUT = 7200  # 2 horas
JSON_DIR = "conversaciones"
if not os.path.exists(JSON_DIR):
    os.makedirs(JSON_DIR)

def get_json_path(user_id: str) -> str:
    return os.path.join(JSON_DIR, f"conversacion_{user_id}.json")

def actualizar_actividad(user_id: str):
    USER_ACTIVITY[user_id] = time.time()

def eliminar_json_inactivo():
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

# Lanzar hilo de limpieza
cleanup_thread = threading.Thread(target=eliminar_json_inactivo, daemon=True)
cleanup_thread.start()

# ========================
# 🔹 FastAPI
# ========================
app = FastAPI(
    title="GLYNNE LLM API",
    description="API para interactuar con GLY-AI mediante LangGraph",
    version="1.0"
)

origins = [
    "https://glynne-sst-ai-hsiy.vercel.app",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ========================
# 🔹 Modelos de datos
# ========================
class ChatRequest(BaseModel):
    mensaje: str
    rol: Optional[str] = "auditor"
    user_id: str

class ChatResponse(BaseModel):
    respuesta: str
    historial: dict

# ========================
# 🔹 Endpoints
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

@app.post("/generar_auditoria")
def generar_auditoria_endpoint(user_id: str):
    try:
        path = get_json_path(user_id)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="No hay conversación para generar auditoría")

        # Generar auditoría
        resultado_auditoria = auditor_llm(user_id)
        # Generar ecosistema
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
# 🔹 Entrypoint
# ========================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
