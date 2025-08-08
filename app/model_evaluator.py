import re
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

try:
    import textstat
    import nltk
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Descargar recursos de NLTK si no están disponibles
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

except ImportError as e:
    print(f"⚠️ Algunas librerías de métricas no están disponibles: {e}")


@dataclass
class ModelMetrics:
    """Métricas completas de evaluación de un modelo"""

    # Métricas básicas
    response_time: float
    response_length: int
    words_count: int
    sentences_count: int

    # Métricas de calidad del texto
    readability_score: float
    complexity_score: float
    sentiment_score: float

    # Métricas de contenido
    detail_score: float  # Nivel de detalle en la descripción
    coherence_score: float  # Coherencia del texto
    completeness_score: float  # Completitud de la respuesta

    # Métricas de rendimiento
    words_per_second: float
    efficiency_score: float  # Relación calidad/tiempo

    # Puntaje general
    overall_score: float

    # Metadatos
    timestamp: str
    provider: str
    model: str


class ModelEvaluator:
    """Evaluador de métricas para modelos de IA multimodales"""

    def __init__(self):
        self.evaluation_criteria = {
            "min_words": 20,  # Mínimo de palabras esperadas
            "max_response_time": 30.0,  # Tiempo máximo aceptable
            "quality_keywords": [
                "color",
                "forma",
                "tamaño",
                "textura",
                "objeto",
                "persona",
                "fondo",
                "iluminación",
                "composición",
                "detalle",
                "material",
                "ubicación",
                "posición",
                "expresión",
                "movimiento",
            ],
        }

    def evaluate_response(
        self,
        response_text: str,
        response_time: float,
        provider: str,
        model: str,
        image_size_bytes: int = 0,
    ) -> ModelMetrics:
        """
        Evalúa integralmente la respuesta de un modelo
        """
        try:
            # Métricas básicas
            words = self._count_words(response_text)
            sentences = self._count_sentences(response_text)

            # Métricas de calidad
            readability = self._calculate_readability(response_text)
            complexity = self._calculate_complexity(response_text)
            sentiment = self._analyze_sentiment(response_text)

            # Métricas de contenido
            detail_score = self._evaluate_detail_level(response_text)
            coherence = self._evaluate_coherence(response_text)
            completeness = self._evaluate_completeness(response_text)

            # Métricas de rendimiento
            wps = words / response_time if response_time > 0 else 0
            efficiency = self._calculate_efficiency(detail_score, response_time)

            # Puntaje general
            overall = self._calculate_overall_score(
                detail_score, coherence, completeness, efficiency, readability
            )

            return ModelMetrics(
                response_time=round(response_time, 3),
                response_length=len(response_text),
                words_count=words,
                sentences_count=sentences,
                readability_score=round(readability, 2),
                complexity_score=round(complexity, 2),
                sentiment_score=round(sentiment, 2),
                detail_score=round(detail_score, 2),
                coherence_score=round(coherence, 2),
                completeness_score=round(completeness, 2),
                words_per_second=round(wps, 2),
                efficiency_score=round(efficiency, 2),
                overall_score=round(overall, 2),
                timestamp=datetime.now().isoformat(),
                provider=provider,
                model=model,
            )

        except Exception as e:
            print(f"⚠️ Error en evaluación: {e}")
            # Retornar métricas básicas en caso de error
            return ModelMetrics(
                response_time=response_time,
                response_length=len(response_text),
                words_count=len(response_text.split()),
                sentences_count=response_text.count(".")
                + response_text.count("!")
                + response_text.count("?"),
                readability_score=0.5,
                complexity_score=0.5,
                sentiment_score=0.5,
                detail_score=0.5,
                coherence_score=0.5,
                completeness_score=0.5,
                words_per_second=len(response_text.split()) / response_time
                if response_time > 0
                else 0,
                efficiency_score=0.5,
                overall_score=0.5,
                timestamp=datetime.now().isoformat(),
                provider=provider,
                model=model,
            )

    def _count_words(self, text: str) -> int:
        """Cuenta palabras en el texto"""
        return len(re.findall(r"\b\w+\b", text))

    def _count_sentences(self, text: str) -> int:
        """Cuenta oraciones en el texto"""
        return max(1, len(re.findall(r"[.!?]+", text)))

    def _calculate_readability(self, text: str) -> float:
        """Calcula puntuación de legibilidad"""
        try:
            # Usar textstat si está disponible
            if hasattr(textstat, "flesch_reading_ease"):
                score = textstat.flesch_reading_ease(text)
            else:
                # Fallback simple
                sentences = len(re.split(r"[.!?]+", text))
                words = len(re.findall(r"\b\w+\b", text))
                score = (
                    206.835
                    - (1.015 * words / sentences)
                    - (84.6 * sum(len(w) for w in re.findall(r"\b\w+\b", text)) / words)
                    if sentences > 0 and words > 0
                    else 50
                )
            return min(1.0, max(0.0, score / 100.0))
        except Exception:
            # Fallback: longitud promedio de palabras
            words = re.findall(r"\b\w+\b", text)
            if not words:
                return 0.5
            avg_word_length = sum(len(word) for word in words) / len(words)
            return min(1.0, max(0.0, 1.0 - (avg_word_length - 4) / 10))

    def _calculate_complexity(self, text: str) -> float:
        """Calcula complejidad del texto"""
        try:
            # Usar textstat si está disponible
            if hasattr(textstat, "flesch_kincaid_grade"):
                score = textstat.flesch_kincaid_grade(text)
            else:
                # Fallback simple
                sentences = len(re.split(r"[.!?]+", text))
                words = len(re.findall(r"\b\w+\b", text))
                syllables = sum(
                    len(re.findall(r"[aeiouAEIOU]", w))
                    for w in re.findall(r"\b\w+\b", text)
                )
                score = (
                    0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
                    if sentences > 0 and words > 0
                    else 10
                )
            return min(1.0, max(0.0, score / 20.0))
        except Exception:
            # Fallback: ratio de palabras largas
            words = re.findall(r"\b\w+\b", text)
            if not words:
                return 0.5
            long_words = sum(1 for word in words if len(word) > 6)
            return long_words / len(words)

    def _analyze_sentiment(self, text: str) -> float:
        """Analiza sentimiento (0-1, donde 0.5 es neutro)"""
        # Simple análisis de sentimiento basado en palabras
        positive_words = [
            "bueno",
            "excelente",
            "claro",
            "detallado",
            "preciso",
            "completo",
        ]
        negative_words = ["malo", "confuso", "incorrecto", "incompleto", "vago"]

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count + neg_count == 0:
            return 0.5  # Neutro

        return (pos_count + 1) / (pos_count + neg_count + 2)

    def _evaluate_detail_level(self, text: str) -> float:
        """Evalúa el nivel de detalle en la descripción"""
        text_lower = text.lower()

        # Contar menciones de características visuales importantes
        detail_keywords = self.evaluation_criteria["quality_keywords"]
        detail_mentions = sum(1 for keyword in detail_keywords if keyword in text_lower)

        # Normalizar score (máximo esperado: 10 keywords)
        detail_score = min(1.0, detail_mentions / 10.0)

        # Bonus por descripciones largas y detalladas
        words_count = len(re.findall(r"\b\w+\b", text))
        length_bonus = min(0.3, words_count / 200.0)  # Hasta 30% extra por longitud

        return min(1.0, detail_score + length_bonus)

    def _evaluate_coherence(self, text: str) -> float:
        """Evalúa la coherencia del texto"""
        sentences = re.split(r"[.!?]+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.8  # Textos muy cortos son coherentes por defecto

        try:
            # Usar TF-IDF para medir similitud entre oraciones
            vectorizer = TfidfVectorizer(stop_words="english")
            vectors = vectorizer.fit_transform(sentences)

            # Calcular similitud promedio entre oraciones consecutivas
            similarities = []
            for i in range(len(sentences) - 1):
                sim = cosine_similarity(vectors[i : i + 1], vectors[i + 1 : i + 2])[0][
                    0
                ]
                similarities.append(sim)

            return float(np.mean(similarities)) if similarities else 0.5

        except Exception:
            # Fallback: evaluar repetición de palabras clave
            words = re.findall(r"\b\w+\b", text.lower())
            unique_words = set(words)
            repetition_ratio = len(words) / len(unique_words) if unique_words else 1

            # Score más alto para repetición moderada (coherencia temática)
            return min(1.0, max(0.3, 2.0 - repetition_ratio / 3.0))

    def _evaluate_completeness(self, text: str) -> float:
        """Evalúa completitud de la respuesta"""
        words_count = len(re.findall(r"\b\w+\b", text))
        min_words = self.evaluation_criteria["min_words"]

        # Score básico por longitud
        length_score = min(1.0, words_count / (min_words * 2))

        # Evaluar cobertura de aspectos importantes
        coverage_aspects = [
            r"\b(color|colores)\b",
            r"\b(forma|formas|figura)\b",
            r"\b(tamaño|grande|pequeño|mediano)\b",
            r"\b(objeto|objetos|cosa|cosas)\b",
            r"\b(fondo|background|parte posterior)\b",
            r"\b(persona|personas|gente|hombre|mujer)\b",
        ]

        covered_aspects = 0
        text_lower = text.lower()
        for aspect_pattern in coverage_aspects:
            if re.search(aspect_pattern, text_lower):
                covered_aspects += 1

        coverage_score = covered_aspects / len(coverage_aspects)

        # Combinar ambos scores
        return (length_score * 0.6) + (coverage_score * 0.4)

    def _calculate_efficiency(
        self, quality_score: float, response_time: float
    ) -> float:
        """Calcula eficiencia (calidad vs tiempo)"""
        max_time = self.evaluation_criteria["max_response_time"]

        # Penalizar tiempos excesivos
        time_penalty = max(0.1, 1.0 - (response_time / max_time))

        return quality_score * time_penalty

    def _calculate_overall_score(
        self,
        detail: float,
        coherence: float,
        completeness: float,
        efficiency: float,
        readability: float,
    ) -> float:
        """Calcula puntuación general ponderada"""
        weights = {
            "detail": 0.25,
            "coherence": 0.20,
            "completeness": 0.25,
            "efficiency": 0.20,
            "readability": 0.10,
        }

        overall = (
            detail * weights["detail"]
            + coherence * weights["coherence"]
            + completeness * weights["completeness"]
            + efficiency * weights["efficiency"]
            + readability * weights["readability"]
        )

        return overall

    def compare_models(self, metrics_list: List[ModelMetrics]) -> Dict[str, Any]:
        """
        Compara múltiples modelos y genera reporte de comparación
        """
        if not metrics_list:
            return {"error": "No hay métricas para comparar"}

        # Rankings por categoría (convertir a diccionarios)
        rankings = {
            "overall": [self._metrics_to_dict(m) for m in sorted(
                metrics_list, key=lambda x: x.overall_score, reverse=True
            )],
            "speed": [self._metrics_to_dict(m) for m in sorted(
                metrics_list, key=lambda x: x.words_per_second, reverse=True
            )],
            "detail": [self._metrics_to_dict(m) for m in sorted(
                metrics_list, key=lambda x: x.detail_score, reverse=True
            )],
            "efficiency": [self._metrics_to_dict(m) for m in sorted(
                metrics_list, key=lambda x: x.efficiency_score, reverse=True
            )],
            "coherence": [self._metrics_to_dict(m) for m in sorted(
                metrics_list, key=lambda x: x.coherence_score, reverse=True
            )],
        }

        # Estadísticas comparativas
        scores = [m.overall_score for m in metrics_list]
        times = [m.response_time for m in metrics_list]

        comparison_stats = {
            "best_overall": rankings["overall"][0],
            "fastest": rankings["speed"][0],
            "most_detailed": rankings["detail"][0],
            "most_efficient": rankings["efficiency"][0],
            "most_coherent": rankings["coherence"][0],
            "avg_score": round(np.mean(scores), 3),
            "score_std": round(np.std(scores), 3),
            "avg_time": round(np.mean(times), 3),
            "time_std": round(np.std(times), 3),
            "total_models": len(metrics_list),
        }

        return {
            "rankings": rankings,
            "stats": comparison_stats,
            "detailed_metrics": [self._metrics_to_dict(m) for m in metrics_list],
        }

    def _metrics_to_dict(self, metrics: ModelMetrics) -> Dict[str, Any]:
        """Convierte métricas a diccionario"""
        return {
            "provider": metrics.provider,
            "model": metrics.model,
            "overall_score": metrics.overall_score,
            "response_time": metrics.response_time,
            "words_count": metrics.words_count,
            "detail_score": metrics.detail_score,
            "coherence_score": metrics.coherence_score,
            "completeness_score": metrics.completeness_score,
            "efficiency_score": metrics.efficiency_score,
            "readability_score": metrics.readability_score,
            "words_per_second": metrics.words_per_second,
            "timestamp": metrics.timestamp,
        }
