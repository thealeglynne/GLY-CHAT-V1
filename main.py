# main.py
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import json
import traceback

# Importamos el agente desde tu script
from agent.chat import agente_node, get_memory, State, TEMP_JSON_PATH

# 🚀 Importar función de auditoría
from agent.auditor import generar_auditoria as auditor_llm

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
    "https://glynne-sst-ai-hsiy.vercel.app",  # tu frontend
    "http://localhost:3000",  # opcional para testing local
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
        return ChatResponse(respuesta=result.get("respuesta", ""), historial=memoria)

    except Exception as e:
        print("❌ Error en /chat endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.get("/user/{user_id}/memory")
def get_user_memory(user_id: str):
    try:
        memoria = get_memory(user_id).load_memory_variables({})
        return {"user_id": user_id, "historial": memoria}
    except Exception as e:
        print("❌ Error en /user/{user_id}/memory:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.get("/reset")
def reset_conversacion():
    """Elimina el JSON temporal y reinicia memoria"""
    try:
        if os.path.exists(TEMP_JSON_PATH):
            os.remove(TEMP_JSON_PATH)
        with open(TEMP_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)
        for user_id in list(get_memory.__defaults__[0].keys()) if get_memory.__defaults__ else []:
            get_memory(user_id).clear()
        return {"status": "ok", "message": "Conversación temporal reiniciada"}
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
    Genera auditoría real llamando al LLM con la conversación.
    """
    try:
        if not os.path.exists(TEMP_JSON_PATH):
            raise HTTPException(status_code=404, detail="No hay conversación para generar auditoría")

        resultado = auditor_llm()

        return {
            "mensaje": "✅ Auditoría generada correctamente",
            "auditoria": resultado
        }

    except Exception as e:
        print("❌ Error en /generar_auditoria endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ========================
# 6. Nuevo endpoint JSON
# ========================
@app.get("/generar_auditoria/json")
def generar_auditoria_json():
    """
    Devuelve la misma auditoría pero directamente en formato JSON.
    """
    try:
        if not os.path.exists(TEMP_JSON_PATH):
            raise HTTPException(status_code=404, detail="No hay conversación para generar auditoría")

        resultado = auditor_llm()

        # Guardar temporalmente en JSON por si se necesita
        with open(TEMP_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(resultado, f, ensure_ascii=False, indent=4)

        return resultado

    except Exception as e:
        print("❌ Error en /generar_auditoria/json endpoint:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ========================
# 7. Entrypoint uvicorn
# ========================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
