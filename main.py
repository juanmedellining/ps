import os
import logging
from typing import List

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dr-alma")

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Agente Psicólogo IA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

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


# ── Modelos Pydantic ─────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    reply: str
    model: str


# ── Rutas API (DEFINIR ANTES del mount de StaticFiles) ───────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": GROQ_MODEL,
        "groq_key_configured": bool(GROQ_API_KEY),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY no configurada")
        raise HTTPException(status_code=500, detail="GROQ_API_KEY no configurada en el servidor.")

    if not request.messages:
        raise HTTPException(status_code=400, detail="No se enviaron mensajes.")

    groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in request.messages:
        if msg.role not in ("user", "assistant"):
            continue
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

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(GROQ_API_URL, json=payload, headers=headers)

        if response.status_code != 200:
            logger.error("Groq devolvió %s: %s", response.status_code, response.text)
            raise HTTPException(
                status_code=502,
                detail=f"Groq API error {response.status_code}: {response.text[:300]}",
            )

        data = response.json()
        reply = data["choices"][0]["message"]["content"]
        return ChatResponse(reply=reply, model=GROQ_MODEL)

    except httpx.TimeoutException:
        logger.exception("Timeout llamando a Groq")
        raise HTTPException(status_code=504, detail="Tiempo de espera agotado con Groq.")
    except httpx.RequestError as e:
        logger.exception("Error de red con Groq")
        raise HTTPException(status_code=503, detail=f"No se pudo conectar con Groq: {e}")
    except (KeyError, IndexError) as e:
        logger.exception("Respuesta de Groq con formato inesperado")
        raise HTTPException(status_code=502, detail=f"Respuesta inválida de Groq: {e}")


# ── Static files (DESPUÉS de las rutas API) ──────────────────────────────────
# Verificamos que el directorio exista antes de montar para evitar crash al arrancar
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    logger.info("Static files montados desde: %s", STATIC_DIR)
else:
    logger.warning("Directorio static/ no encontrado en %s — el frontend no se servirá", STATIC_DIR)

    @app.get("/")
    async def root_fallback():
        return {"status": "ok", "warning": "static/ no encontrado", "api": "/api/chat"}
