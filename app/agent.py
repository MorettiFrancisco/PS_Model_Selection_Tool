import os

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.providers.openai import OpenAIProvider

GOOGLE_API_KEY = os.getenv("GCP_API_KEY", "NONE")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")


def create_ollama_model(model_name: str):
    """Crea un modelo de Ollama con el nombre especificado"""
    return OpenAIModel(
        model_name,
        provider=OpenAIProvider(base_url=OLLAMA_BASE_URL, api_key="your-api-key"),
    )


def create_gemini_model():
    """Crea el modelo de Gemini"""
    return GeminiModel(
        "gemini-1.5-flash", provider=GoogleGLAProvider(api_key=GOOGLE_API_KEY)
    )


agent_system_prompt = """
Eres un experto en analizar imagenes.
Tu tarea es extraer y detallar solo el contenido explícito y visible de una imagen.

Guías:
- Concéntrate en la idea principal o tema de la imagen.
- Usa un lenguaje claro y factual basado estrictamente en lo que aparece en la imagen.
- NO hagas suposiciones ni agregues conocimiento externo.
- NO especules ni interpretes significados implícitos.
Devuelve solo un resumen bien estructurado del contenido visible de la imagen, **en español**.
"""


def get_agent(provider: str = "ollama", model_name: str = "qwen2.5vl:7b"):
    """
    Crea un agente basado en el proveedor y modelo especificado.

    Args:
        provider: "ollama" o "gemini"
        model_name: Nombre del modelo (solo usado para Ollama)

    Returns:
        Agent configurado con el modelo especificado
    """
    if provider.lower() == "gemini":
        model = create_gemini_model()
    elif provider.lower() == "ollama":
        model = create_ollama_model(model_name)
    else:
        raise ValueError(f"Proveedor no soportado: {provider}. Use 'ollama' o 'gemini'")

    return Agent(
        model,
        output_type=str,
        system_prompt=agent_system_prompt,
    )


# Función de conveniencia para obtener modelos disponibles
def get_available_models():
    """Retorna una lista de modelos disponibles específicos del proyecto"""
    return {
        "ollama": [
            "qwen2.5vl:7b",  # Modelo multimodal de visión Qwen
            "llava:7b",  # Modelo multimodal de visión Llava
            "gemma3:4b",  # Modelo multimodal Gemma
        ],
        "gemini": ["gemini-1.5-flash"],  # Modelo de visión de Google
    }
