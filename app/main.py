import time
import base64
from datetime import datetime
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from .agent import get_agent, get_available_models

app = FastAPI(
    title="Model Selection Tool",
    description="API para análisis de imágenes con múltiples modelos de IA",
    version="1.0.0",
)

# Configurar templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Página principal con formulario para seleccionar modelo y subir imagen"""
    available_models = get_available_models()
    return templates.TemplateResponse(
        "index_simple.html", {"request": request, "models": available_models}
    )


@app.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    """Página de test para diagnosticar JavaScript"""
    return templates.TemplateResponse("test.html", {"request": request})


@app.get("/api")
async def api_info():
    """Endpoint de información de la API"""
    return {
        "message": "Model Selection Tool API",
        "available_endpoints": {
            "/": "GET - Interfaz web principal",
            "/api/analyze": "POST - Analizar imagen con modelo seleccionado",
            "/api/models": "GET - Obtener modelos disponibles",
            "/docs": "GET - Documentación interactiva",
        },
    }


@app.get("/api/models")
async def get_models():
    """Obtiene la lista de modelos disponibles por proveedor"""
    return get_available_models()


@app.post("/api/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    provider: str = Form(..., description="Proveedor del modelo: 'ollama' o 'gemini'"),
    model_name: str = Form(..., description="Nombre del modelo"),
):
    """
    Endpoint para análisis de imágenes con métricas de rendimiento.

    Parámetros:
    - image: Archivo de imagen a analizar
    - provider: "ollama" o "gemini"
    - model_name: Nombre del modelo

    Retorna:
    - Análisis de la imagen
    - Métricas de rendimiento (tiempo, tokens, etc.)
    - Información del modelo usado
    """

    # Métricas de inicio
    start_time = time.time()
    analysis_timestamp = datetime.now().isoformat()

    try:
        # Validar el tipo de archivo
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="El archivo debe ser una imagen"
            )

        # Leer el contenido de la imagen
        image_content = await image.read()
        image_size_mb = len(image_content) / (1024 * 1024)

        # Validar tamaño de imagen (máximo 10MB)
        if image_size_mb > 10:
            raise HTTPException(
                status_code=400, detail="La imagen es demasiado grande (máximo 10MB)"
            )

        # Crear el agente con el modelo especificado
        try:
            agent = get_agent(provider=provider, model_name=model_name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Convertir imagen a base64 para el análisis
        image_b64 = base64.b64encode(image_content).decode("utf-8")

        # Tiempo de preparación
        prep_time = time.time() - start_time

        # Procesar la imagen con el agente
        inference_start = time.time()
        try:
            # Para modelos de visión, necesitamos pasar la imagen correctamente
            if provider.lower() == "ollama" and "llava" in model_name.lower():
                # Para modelos LLaVA, incluir la imagen en el prompt
                prompt = "Analiza esta imagen y describe detalladamente lo que ves."
                result = await agent.run(
                    prompt,
                    message_history=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{image.content_type};base64,{image_b64}"
                                    },
                                },
                            ],
                        }
                    ],
                )
            else:
                # Para otros modelos, usar solo texto
                prompt = f"Analiza esta imagen (tamaño: {image_size_mb:.2f}MB, tipo: {image.content_type}) y describe lo que puedes inferir de ella."
                result = await agent.run(prompt)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error procesando imagen con {provider}/{model_name}: {str(e)}",
            )

        # Calcular métricas finales
        inference_time = time.time() - inference_start
        total_time = time.time() - start_time

        # Métricas del análisis
        analysis_metrics = {
            "total_time_seconds": round(total_time, 3),
            "preparation_time_seconds": round(prep_time, 3),
            "inference_time_seconds": round(inference_time, 3),
            "processing_speed_mb_per_sec": round(image_size_mb / total_time, 3)
            if total_time > 0
            else 0,
            "response_length_chars": len(str(result)),
            "words_per_second": round(len(str(result).split()) / inference_time, 2)
            if inference_time > 0
            else 0,
        }

        # Información del modelo
        model_info = {
            "provider": provider,
            "model_name": model_name,
            "supports_vision": provider.lower() == "ollama"
            and "llava" in model_name.lower()
            or provider.lower() == "gemini",
            "timestamp": analysis_timestamp,
        }

        # Información de la imagen
        image_info = {
            "filename": image.filename,
            "content_type": image.content_type,
            "size_bytes": len(image_content),
            "size_mb": round(image_size_mb, 3),
        }

        return JSONResponse(
            content={
                "status": "success",
                "analysis": str(result),
                "metrics": analysis_metrics,
                "model_info": model_info,
                "image_info": image_info,
                "provider": provider,
                "model": model_name,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        error_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Error interno del servidor: {str(e)}",
                "processing_time": round(error_time, 3),
                "timestamp": datetime.now().isoformat(),
            },
        )


@app.post("/api/compare")
async def compare_models(
    image: UploadFile = File(...),
    providers: str = Form(
        ..., description="Proveedores separados por coma, ej: 'ollama,gemini'"
    ),
    models: str = Form(
        ..., description="Modelos separados por coma, ej: 'llava:13b,gemini-1.5-flash'"
    ),
):
    """
    Endpoint para comparar múltiples modelos en la misma imagen.

    Parámetros:
    - image: Archivo de imagen a analizar
    - providers: Lista de proveedores separados por coma
    - models: Lista de modelos separados por coma (deben corresponder con providers)

    Retorna:
    - Comparación detallada entre modelos
    - Métricas de rendimiento de cada uno
    - Análisis comparativo
    """

    start_time = time.time()

    try:
        # Validar imagen
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="El archivo debe ser una imagen"
            )

        # Parsear listas
        provider_list = [p.strip() for p in providers.split(",")]
        model_list = [m.strip() for m in models.split(",")]

        if len(provider_list) != len(model_list):
            raise HTTPException(
                status_code=400,
                detail="El número de proveedores debe coincidir con el número de modelos",
            )

        # Leer imagen una sola vez
        image_content = await image.read()
        image_size_mb = len(image_content) / (1024 * 1024)

        if image_size_mb > 10:
            raise HTTPException(
                status_code=400, detail="Imagen demasiado grande (máximo 10MB)"
            )

        # Preparar imagen
        image_b64 = base64.b64encode(image_content).decode("utf-8")

        # Resultados por modelo
        results = []

        for provider, model_name in zip(provider_list, model_list):
            model_start = time.time()

            try:
                # Crear agente
                agent = get_agent(provider=provider, model_name=model_name)

                # Procesar imagen
                inference_start = time.time()

                if provider.lower() == "ollama" and "llava" in model_name.lower():
                    prompt = "Analiza esta imagen y describe detalladamente lo que ves."
                    result = await agent.run(
                        prompt,
                        message_history=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{image.content_type};base64,{image_b64}"
                                        },
                                    },
                                ],
                            }
                        ],
                    )
                else:
                    prompt = f"Analiza esta imagen (tamaño: {image_size_mb:.2f}MB) y describe lo que ves."
                    result = await agent.run(prompt)

                inference_time = time.time() - inference_start
                total_model_time = time.time() - model_start

                # Métricas del modelo
                model_metrics = {
                    "provider": provider,
                    "model": model_name,
                    "inference_time": round(inference_time, 3),
                    "total_time": round(total_model_time, 3),
                    "response_length": len(str(result)),
                    "words_count": len(str(result).split()),
                    "words_per_second": round(
                        len(str(result).split()) / inference_time, 2
                    )
                    if inference_time > 0
                    else 0,
                    "success": True,
                    "analysis": str(result),
                }

                results.append(model_metrics)

            except Exception as e:
                # Error en modelo específico
                model_time = time.time() - model_start
                results.append(
                    {
                        "provider": provider,
                        "model": model_name,
                        "inference_time": round(model_time, 3),
                        "total_time": round(model_time, 3),
                        "success": False,
                        "error": str(e),
                        "response_length": 0,
                        "words_count": 0,
                        "words_per_second": 0,
                        "analysis": None,
                    }
                )

        # Análisis comparativo
        successful_results = [r for r in results if r["success"]]

        if not successful_results:
            raise HTTPException(
                status_code=500, detail="Ningún modelo pudo procesar la imagen"
            )

        # Métricas comparativas
        comparison_metrics = {
            "fastest_model": min(successful_results, key=lambda x: x["inference_time"]),
            "slowest_model": max(successful_results, key=lambda x: x["inference_time"]),
            "most_verbose": max(successful_results, key=lambda x: x["words_count"]),
            "most_concise": min(successful_results, key=lambda x: x["words_count"]),
            "average_inference_time": round(
                sum(r["inference_time"] for r in successful_results)
                / len(successful_results),
                3,
            ),
            "average_response_length": round(
                sum(r["words_count"] for r in successful_results)
                / len(successful_results),
                1,
            ),
            "total_comparison_time": round(time.time() - start_time, 3),
            "models_tested": len(results),
            "successful_models": len(successful_results),
            "failed_models": len(results) - len(successful_results),
        }

        return JSONResponse(
            content={
                "status": "success",
                "image_info": {
                    "filename": image.filename,
                    "size_mb": round(image_size_mb, 3),
                    "content_type": image.content_type,
                },
                "individual_results": results,
                "comparison_metrics": comparison_metrics,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Error en comparación: {str(e)}",
                "processing_time": round(total_time, 3),
            },
        )


@app.get("/health")
async def health_check():
    """Endpoint de verificación de salud"""
    return {"status": "healthy", "message": "API funcionando correctamente"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
