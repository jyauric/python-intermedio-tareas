# Tarea 1: Clasificador de titulares peruanos

**Estudiante:** Josselin Yauri ([@jyauric](https://github.com/jyauric)) · Opción A: La República

## Pregunta del proyecto
¿Pueden los titulares peruanos convertirse en un clasificador de noticias? Se buscó clasificar automáticamente titulares de La República en 4 categorías (`politica`, `economia`, `deportes`, `ciencia_tecnologia`) con un Transformer ligero entrenado a partir de etiquetas generadas por un LLM.

## Medio y fechas
- **Medio:** La República (`larepublica.pe`), descarga del 18/10/2026.
- **Noticias:** 400 (100 por categoría), publicadas entre el 22/01/2024 y el 18/10/2026; 348 son de 2026 y 231 de octubre de 2026.
- **Método:** RSS y sitemaps oficiales como punto de partida, luego visita de noticias individuales con `requests` + `BeautifulSoup`, respetando `robots.txt` (pausa de 1.5 s, 1 reintento).

## Resultados (test reservado, 60 noticias)
| Modelo | F1 macro | Costo | Latencia |
|---|---|---|---|
| Zero-shot (`Recognai/bert-base-spanish-wwm-cased-xnli`)* | 0.571 | Gratis, local | Media |
| LLM: Gemini (`gemini-3.8-flash`) | 0.950 | Capa gratuita con límites | Alta (API) |
| **Transformer fine-tuned (`distilbert-base-multilingual-cased`)** | **0.865** | Gratis, local | Baja |

\*Medido sobre 40 de las 60 noticias del test. Acuerdo LLM vs. auditoría humana: **20/20**.

## Hallazgos analíticos
1. **Sí se puede construir un clasificador útil a partir de etiquetas de un LLM.** DistilBERT, entrenado solo con etiquetas de Gemini, alcanza F1 macro 0.865 en test, 0.294 más que el zero-shot (0.571). Queda por debajo de Gemini (0.950), pero corre local, gratis y en milisegundos.
2. **Economía es la categoría más difícil (F1 0.80) y deportes la más fácil (F1 0.94).** Economía se confunde con política cuando un actor político anuncia una medida económica (por ejemplo, "Pensión 65 se duplicará… anunció Keiko Fujimori") y con ciencia y tecnología cuando el titular trae cifras o lenguaje comercial. Deportes acierta las 15 noticias del test gracias a su vocabulario propio (clubes, jugadores, "fecha, hora y canal").

## Limitaciones técnicas
1. **Etiquetas débiles y test pequeño.** El modelo aprende de etiquetas de Gemini, no humanas: 3 de sus 8 errores en test los heredó del LLM. Además, el test tiene solo 60 noticias (15 por categoría), así que cada error mueve la accuracy en unos 1.7 puntos.
2. **Cobertura temporal desigual.** Todas las noticias de política son de 2026, pero en ciencia y tecnología 30 de 100 son de 2024–2025, porque el scraper siguió enlaces a noticias antiguas. También quedaron 2 titulares repetidos con distinta URL.

## Estructura
| Archivo | Contenido |
|---|---|
| `01_recoleccion.ipynb` | Scraping, limpieza, mapeo y EDA (Parte 1) |
| `02_clasificacion.ipynb` | Zero-shot, etiquetado con LLM, fine-tuning y evaluación (Parte 2) |
| `app.py` | Dashboard en Streamlit (Parte 3) |
| `datos/noticias.csv` | 400 noticias con 7 columnas |
| `datos/etiquetas_llm.csv` | Etiquetas de Gemini |
| `datos/metricas.json` | Métricas que muestra el dashboard |
| `modelo/` | Configuración y tokenizer (los pesos no se suben: pesan más de 100 MB) |

## Cómo ejecutarlo
Requisito: Python 3.12.

```bash
git clone https://github.com/jyauric/python-intermedio-tareas.git
cd python-intermedio-tareas/tarea_1

# Opción A, con Anaconda
conda create -n tarea1 python=3.12 -y
conda activate tarea1
# Opción B, sin Anaconda (Mac/Linux)
python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

**Generar el modelo** (los pesos no están en GitHub): ejecuta en `02_clasificacion.ipynb` las secciones 1 (split), 3.2 y 4 (fine-tuning). No hace falta volver a llamar a Gemini: en vez de la celda de la sección 3, usa `etq = pd.read_csv("datos/etiquetas_llm.csv")`. Esto crea `modelo/model.safetensors`.

**Abrir el dashboard:**
```bash
streamlit run app.py
```