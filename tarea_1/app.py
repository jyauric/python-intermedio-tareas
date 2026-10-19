import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from transformers import pipeline

AQUI = Path(__file__).parent
NOMBRES = {"politica": "Política", "economia": "Economía",
           "deportes": "Deportes", "ciencia_tecnologia": "Ciencia y tecnología"}

st.set_page_config(page_title="Clasificador de titulares peruanos", page_icon="🗞️", layout="wide")


@st.cache_data
def cargar_datos():
    return pd.read_csv(AQUI / "datos/noticias.csv")


@st.cache_resource
def cargar_modelo():
    return pipeline("text-classification", model=str(AQUI / "modelo"), top_k=None)


df = cargar_datos()
with open(AQUI / "datos/metricas.json") as f:
    met = json.load(f)

# 1) Título y metodología
st.title("¿Pueden los titulares peruanos convertirse en un clasificador de noticias?")
st.markdown(
    "Se recolectaron **400 titulares de La República** (100 por categoría) desde sus RSS, sitemaps y "
    "noticias individuales. **Gemini** los etiquetó de forma automática (supervisión débil) y con esas "
    "etiquetas se entrenó **DistilBERT**, un Transformer ligero. El modelo se evaluó sobre un conjunto "
    "de test reservado, comparando sus predicciones con la sección original de cada noticia."
)

# 2) Filtros en la barra lateral
st.sidebar.header("Filtros")
medios = st.sidebar.multiselect("Medio", sorted(df["medio"].unique()), default=sorted(df["medio"].unique()))
cats = st.sidebar.multiselect("Categoría", list(NOMBRES), default=list(NOMBRES), format_func=NOMBRES.get)
f = df[df["medio"].isin(medios) & df["etiqueta"].isin(cats)]

# 3) Indicadores clave
c1, c2, c3 = st.columns(3)
c1.metric("Noticias recolectadas", len(df), f"{len(f)} con los filtros", delta_color="off")
c2.metric("F1 macro del modelo (test)", f'{met["f1_test"]:.3f}',
          f'{met["f1_test"] - met["f1_zero_shot"]:+.3f} vs. zero-shot')
c3.metric("Acuerdo LLM vs. auditoría humana", f'{met["acuerdo_llm_humano"] / 20:.0%}',
          f'{met["acuerdo_llm_humano"]}/20 casos', delta_color="off")

# 4) Gráfico y tabla
conteo = f["etiqueta"].map(NOMBRES).value_counts().rename_axis("Categoría").reset_index(name="Noticias")
st.plotly_chart(px.bar(conteo, x="Categoría", y="Noticias", color="Categoría",
                       title="Noticias por categoría (según los filtros)"),
                width="stretch")

st.subheader("Noticias")
st.dataframe(f, width="stretch", hide_index=True,
             column_config={"url": st.column_config.LinkColumn("url")})
st.download_button("Descargar CSV", f.to_csv(index=False).encode("utf-8"),
                   "noticias_filtradas.csv", "text/csv")

# 5) Probador en vivo
st.subheader("Prueba el clasificador")
texto = st.text_input("Escribe o pega un titular",
                      placeholder="Ej.: BCR mantiene su tasa de referencia en 4,25% por tercer mes consecutivo")
if st.button("Clasificar", type="primary"):
    if not texto.strip():
        st.warning("Escribe un titular primero.")
    else:
        salida = cargar_modelo()(texto)
        probs = salida[0] if isinstance(salida[0], list) else salida
        mejor = max(probs, key=lambda p: p["score"])
        st.success(f"Categoría: **{NOMBRES[mejor['label']]}** (probabilidad {mejor['score']:.1%})")
        tabla = pd.DataFrame({"Categoría": [NOMBRES[p["label"]] for p in probs],
                              "Probabilidad": [p["score"] for p in probs]}).sort_values("Probabilidad", ascending=False)
        st.dataframe(tabla, hide_index=True, width="content",
                     column_config={"Probabilidad": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})