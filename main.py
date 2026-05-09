from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import os
import httpx

app = FastAPI(title="Agente Psicólogo IA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """Eres un psicólogo empático, cálido y profesional llamado Dr. Alma. Tu rol es:

1. **Escuchar activamente** con empatía genuina, validando las emociones del usuario.
2. **Hacer preguntas abiertas** que inviten a la reflexión profunda.
3. **Usar técnicas terapéuticas** como TCC (Terapia Cognitivo-Conductual), mindfulness y psicología positiva cuando sea apropiado.
4. **Normalizar las emociones** sin juzgar ni minimizar lo que siente el usuario.
5. **Identificar patrones de pensamiento** negativos y ayudar a reformularlos.
6. **Mantener límites profesionales** y recordar al usuario que no reemplazas la terapia presencial.

Reglas importantes:
- Nunca diagnostiques enfermedades mentales formalmente.
- Si detectas una crisis (ideación suicida, autolesión), proporciona inmediatamente la línea de crisis: En Colombia: 106 (Línea 106 de Salud Mental). En cualquier país: sugiere llamar a servicios de emergencia.
- Responde siempre en español, de forma cálida y accesible.
- Usa el nombre del usuario si lo menciona.
- Respuestas concisas (máximo 3 párrafos) para mantener la conversación fluida.
- Cierra cada respuesta con UNA sola pregunta reflexiva cuando sea apropiado.

Recuerda: tu misión es crear un espacio seguro donde el usuario se sienta escuchado y comprendido."""


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    reply: str
    model: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY no configurada en el servidor.")

    groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in request.messages:
        groq_messages.append({"role": msg.role, "content": msg.content})

    payload = {
        "model": GROQ_MODEL,
        "messages": groq_messages,
        "temperature": 0.75,
        "max_tokens": 600,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(GROQ_API_URL, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Error de Groq API: {e.response.text}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"No se pudo conectar con Groq: {str(e)}")

    data = response.json()
    reply = data["choices"][0]["message"]["content"]
    return ChatResponse(reply=reply, model=GROQ_MODEL)


@app.get("/health")
async def health():
    return {"status": "ok", "model": GROQ_MODEL}


app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.exception_handler(404)
async def not_found(request, exc):
    return FileResponse("static/index.html")
