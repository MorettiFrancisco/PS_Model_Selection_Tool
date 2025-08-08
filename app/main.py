import time
from datetime import datetime
from enum import Enum
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic_ai import BinaryContent
from .agent import get_agent, get_available_models
from .model_evaluator import ModelEvaluator

app = FastAPI(
    title="Model Selection Tool",
    description="API para análisis de imágenes con múltiples modelos de IA",
    version="1.0.0",
)

# Configurar templates
templates = Jinja2Templates(directory="templates")


class AIProvider(str, Enum):
    OLLAMA = "ollama"
    GEMINI = "gemini"


async def analyze_with_unified_approach(
    provider: AIProvider,
    model_name: str,
    image_content: bytes,
    prompt: str = "Analiza detalladamente esta imagen y describe todo lo que ves.",
) -> str:
    """
    Función unificada para analizar imágenes con cualquier modelo.
    Usa el patrón correcto de pydantic-ai con BinaryContent para todos los proveedores.
    """
    try:
        match provider:
            case AIProvider.OLLAMA | AIProvider.GEMINI:
                # Ambos proveedores usan pydantic-ai con BinaryContent
                agent = get_agent(provider.value, model_name)
                result = await agent.run(
                    [
                        prompt,
                        BinaryContent(data=image_content, media_type="image/png"),
                    ]
                )
                return result.output.strip()

            case _:
                raise ValueError(f"Proveedor no soportado: {provider}")

    except Exception as e:
        raise Exception(f"Error con {provider.value}/{model_name}: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Página principal con funcionalidades completas de análisis y comparación"""
    available_models = get_available_models()
    return templates.TemplateResponse(
        "index.html", {"request": request, "models": available_models}
    )


@app.get("/advanced", response_class=HTMLResponse)
async def advanced_comparison(request: Request):
    """Redirección a la página principal que ahora incluye todas las funcionalidades"""
    available_models = get_available_models()
    return templates.TemplateResponse(
        "index.html", {"request": request, "models": available_models}
    )


@app.get("/api")
async def api_info():
    """Información general de la API"""
    return {
        "name": "Model Selection Tool API",
        "version": "1.0.0",
        "description": "API para análisis de imágenes con múltiples modelos de IA",
        "endpoints": {
            "/": "GET - Página principal",
            "/api": "GET - Información de la API",
            "/api/models": "GET - Lista de modelos disponibles",
            "/api/analyze": "POST - Analizar imagen con modelo seleccionado",
            "/api/compare": "POST - Comparar múltiples modelos en la misma imagen",
        },
    }


@app.get("/api/models")
async def get_models():
    """Obtiene la lista de modelos disponibles por proveedor"""
    return get_available_models()


@app.get("/health")
async def health_check():
    """Endpoint de health check para monitoreo del contenedor"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    provider: AIProvider = Form(
        ..., description="Proveedor del modelo: 'ollama' o 'gemini'"
    ),
    model_name: str = Form(..., description="Nombre del modelo"),
):
    """
    Endpoint unificado para análisis de imágenes con métricas de rendimiento.
    Usa pattern matching moderno para manejar diferentes proveedores.
    """
    start_time = time.time()

    try:
        # Validar imagen
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="El archivo debe ser una imagen"
            )

        # Leer contenido de imagen
        prep_start = time.time()
        image_content = await image.read()
        image_size_mb = len(image_content) / (1024 * 1024)
        prep_time = time.time() - prep_start

        print(f"📸 Procesando imagen: {image.filename} ({image_size_mb:.2f}MB)")
        print(f"🤖 Modelo: {provider.value}/{model_name}")

        # Análisis con función unificada
        inference_start = time.time()
        analysis_timestamp = datetime.now().isoformat()

        result = await analyze_with_unified_approach(
            provider=provider,
            model_name=model_name,
            image_content=image_content,
            prompt="Analiza detalladamente esta imagen y describe todo lo que ves.",
        )

        inference_time = time.time() - inference_start
        total_time = time.time() - start_time

        print(f"✅ Análisis completado en {total_time:.2f}s")

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

        # Información del modelo (todos los modelos soportan visión)
        model_info = {
            "provider": provider.value,
            "model_name": model_name,
            "supports_vision": True,  # Todos nuestros modelos soportan visión
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
                "provider": provider.value,
                "model": model_name,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Error en análisis: {str(e)}")
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
    Endpoint unificado para comparar múltiples modelos en la misma imagen.
    Usa la función unificada con pattern matching para consistencia entre modelos.
    """
    start_time = time.time()

    try:
        # Validar imagen
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="El archivo debe ser una imagen"
            )

        # Leer contenido de imagen
        image_content = await image.read()
        image_size_mb = len(image_content) / (1024 * 1024)

        # Procesar listas de proveedores y modelos con validación de enum
        provider_strings = [p.strip() for p in providers.split(",") if p.strip()]
        model_list = [m.strip() for m in models.split(",") if m.strip()]

        # Convertir strings a enum AIProvider
        try:
            provider_list = [AIProvider(p.lower()) for p in provider_strings]
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Proveedor inválido. Use 'ollama' o 'gemini': {e}",
            )

        if len(provider_list) != len(model_list):
            raise HTTPException(
                status_code=400,
                detail="El número de proveedores debe coincidir con el número de modelos",
            )

        print(f"📸 Comparando imagen: {image.filename} ({image_size_mb:.2f}MB)")
        print(f"🤖 Modelos a comparar: {len(model_list)}")

        # Análisis con cada modelo usando función unificada
        results = []
        for provider_str, model_name in zip(provider_list, model_list):
            model_start = time.time()

            try:
                # Convertir string a enum
                provider = AIProvider(provider_str.lower())

                print(f"🔄 Analizando con {provider.value}/{model_name}...")

                result = await analyze_with_unified_approach(
                    provider=provider,
                    model_name=model_name,
                    image_content=image_content,
                    prompt="Analiza detalladamente esta imagen y describe todo lo que ves.",
                )

                model_time = time.time() - model_start

                results.append(
                    {
                        "provider": provider.value,
                        "model": model_name,
                        "analysis": str(result),
                        "processing_time_seconds": round(model_time, 3),
                        "response_length_chars": len(str(result)),
                        "words_per_second": round(
                            len(str(result).split()) / model_time, 2
                        )
                        if model_time > 0
                        else 0,
                        "status": "success",
                    }
                )

                print(
                    f"✅ {provider.value}/{model_name} completado en {model_time:.2f}s"
                )

            except ValueError:
                # Error de enum (proveedor no válido)
                model_time = time.time() - model_start
                print(f"❌ Proveedor no válido: {provider_str}")

                results.append(
                    {
                        "provider": provider_str,
                        "model": model_name,
                        "analysis": f"Error: Proveedor no válido '{provider_str}'. Use: {', '.join([p.value for p in AIProvider])}",
                        "processing_time_seconds": round(model_time, 3),
                        "response_length_chars": 0,
                        "words_per_second": 0,
                        "status": "error",
                    }
                )

            except Exception as e:
                model_time = time.time() - model_start
                print(f"❌ Error con {provider_str}/{model_name}: {str(e)}")

                results.append(
                    {
                        "provider": provider_str,
                        "model": model_name,
                        "analysis": f"Error: {str(e)}",
                        "processing_time_seconds": round(model_time, 3),
                        "response_length_chars": 0,
                        "words_per_second": 0,
                        "status": "error",
                    }
                )

        total_time = time.time() - start_time

        # Métricas comparativas
        successful_results = [r for r in results if r["status"] == "success"]

        comparison_metrics = {
            "total_comparison_time_seconds": round(total_time, 3),
            "models_compared": len(results),
            "successful_analyses": len(successful_results),
            "failed_analyses": len(results) - len(successful_results),
            "average_response_time": round(
                sum(r["processing_time_seconds"] for r in successful_results)
                / len(successful_results),
                3,
            )
            if successful_results
            else 0,
            "fastest_model": min(
                successful_results, key=lambda x: x["processing_time_seconds"]
            )["model"]
            if successful_results
            else None,
            "slowest_model": max(
                successful_results, key=lambda x: x["processing_time_seconds"]
            )["model"]
            if successful_results
            else None,
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
                "comparison_results": results,
                "metrics": comparison_metrics,
                "image_info": image_info,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Error en comparación: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Error interno del servidor: {str(e)}",
                "processing_time": round(error_time, 3),
                "timestamp": datetime.now().isoformat(),
            },
        )


@app.post("/api/compare-sequential")
async def compare_models_sequential(
    image: UploadFile = File(...),
    providers: str = Form(..., description="Proveedores separados por coma"),
    models: str = Form(..., description="Modelos separados por coma"),
):
    """
    Endpoint para comparación secuencial de modelos con métricas avanzadas.
    Ejecuta un modelo a la vez para evitar sobrecarga del sistema.
    """
    import asyncio

    start_time = time.time()
    evaluator = ModelEvaluator()

    try:
        # Validar imagen
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="El archivo debe ser una imagen"
            )

        # Leer contenido de imagen
        image_content = await image.read()
        image_size_mb = len(image_content) / (1024 * 1024)

        # Procesar listas de proveedores y modelos con validación de enum
        provider_strings = [p.strip() for p in providers.split(",") if p.strip()]
        model_list = [m.strip() for m in models.split(",") if m.strip()]

        # Convertir strings a enum AIProvider
        try:
            provider_list = [AIProvider(p.lower()) for p in provider_strings]
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Proveedor inválido. Use 'ollama' o 'gemini': {e}",
            )

        if len(provider_list) != len(model_list):
            raise HTTPException(
                status_code=400,
                detail="El número de proveedores debe coincidir con el número de modelos",
            )

        print(f"🔄 Comparación secuencial - {len(model_list)} modelos")
        print(f"📸 Imagen: {image.filename} ({image_size_mb:.2f}MB)")

        # Análisis secuencial con métricas avanzadas
        results = []
        model_metrics = []

        for i, (provider, model_name) in enumerate(zip(provider_list, model_list), 1):
            model_start = time.time()

            try:
                print(
                    f"🤖 [{i}/{len(model_list)}] Analizando con {provider.value}/{model_name}..."
                )

                result = await analyze_with_unified_approach(
                    provider=provider,
                    model_name=model_name,
                    image_content=image_content,
                    prompt="Analiza detalladamente esta imagen y describe todo lo que ves.",
                )

                model_time = time.time() - model_start

                # Evaluación avanzada con métricas
                metrics = evaluator.evaluate_response(
                    response_text=result,
                    response_time=model_time,
                    provider=provider.value,
                    model=model_name,
                    image_size_bytes=len(image_content),
                )

                model_metrics.append(metrics)

                results.append(
                    {
                        "provider": provider.value,
                        "model": model_name,
                        "analysis": str(result),
                        "basic_metrics": {
                            "processing_time_seconds": round(model_time, 3),
                            "response_length_chars": len(str(result)),
                            "words_per_second": round(
                                len(str(result).split()) / model_time, 2
                            )
                            if model_time > 0
                            else 0,
                        },
                        "advanced_metrics": {
                            "overall_score": metrics.overall_score,
                            "detail_score": metrics.detail_score,
                            "coherence_score": metrics.coherence_score,
                            "completeness_score": metrics.completeness_score,
                            "efficiency_score": metrics.efficiency_score,
                            "readability_score": metrics.readability_score,
                            "words_count": metrics.words_count,
                            "sentences_count": metrics.sentences_count,
                        },
                        "status": "success",
                        "position": i,
                    }
                )

                print(
                    f"✅ [{i}/{len(model_list)}] {provider.value}/{model_name} - Score: {metrics.overall_score:.2f} ({model_time:.2f}s)"
                )

                # Pequeña pausa entre modelos para evitar sobrecarga
                if i < len(model_list):  # No pausar después del último
                    await asyncio.sleep(0.5)

            except Exception as e:
                model_time = time.time() - model_start
                print(
                    f"❌ [{i}/{len(model_list)}] Error con {provider.value}/{model_name}: {str(e)}"
                )

                results.append(
                    {
                        "provider": provider.value,
                        "model": model_name,
                        "analysis": f"Error: {str(e)}",
                        "basic_metrics": {
                            "processing_time_seconds": round(model_time, 3),
                            "response_length_chars": 0,
                            "words_per_second": 0,
                        },
                        "advanced_metrics": {
                            "overall_score": 0,
                            "detail_score": 0,
                            "coherence_score": 0,
                            "completeness_score": 0,
                            "efficiency_score": 0,
                            "readability_score": 0,
                            "words_count": 0,
                            "sentences_count": 0,
                        },
                        "status": "error",
                        "position": i,
                    }
                )

        total_time = time.time() - start_time

        print(f"🔍 Depuración - model_metrics count: {len(model_metrics)}")
        for i, metric in enumerate(model_metrics):
            print(
                f"   [{i}] {metric.provider}/{metric.model} - Score: {metric.overall_score}"
            )

        # Generar comparación avanzada
        try:
            comparison_analysis = evaluator.compare_models(model_metrics)
            print(f"🔍 comparison_analysis type: {type(comparison_analysis)}")
            if isinstance(comparison_analysis, dict):
                print(
                    f"🔍 comparison_analysis keys: {list(comparison_analysis.keys())}"
                )
        except Exception as e:
            print(f"❌ Error en compare_models: {e}")
            comparison_analysis = {"error": str(e), "stats": {}}

        # Métricas del conjunto completo
        successful_results = [r for r in results if r["status"] == "success"]

        # Manejo seguro de comparison_analysis
        stats = (
            comparison_analysis.get("stats", {})
            if isinstance(comparison_analysis, dict)
            else {}
        )

        summary_metrics = {
            "total_time_seconds": round(total_time, 3),
            "models_evaluated": len(results),
            "successful_evaluations": len(successful_results),
            "failed_evaluations": len(results) - len(successful_results),
            "average_score": stats.get("avg_score", 0),
            "best_model": stats.get("best_overall", {}).get("model", "N/A")
            if stats.get("best_overall")
            else "N/A",
            "fastest_model": stats.get("fastest", {}).get("model", "N/A")
            if stats.get("fastest")
            else "N/A",
            "most_detailed_model": stats.get("most_detailed", {}).get("model", "N/A")
            if stats.get("most_detailed")
            else "N/A",
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
                "sequential_results": results,
                "summary_metrics": summary_metrics,
                "detailed_comparison": comparison_analysis,
                "image_info": image_info,
                "timestamp": datetime.now().isoformat(),
                "execution_mode": "sequential",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Error en comparación secuencial: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Error interno del servidor: {str(e)}",
                "processing_time": round(error_time, 3),
                "timestamp": datetime.now().isoformat(),
            },
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
