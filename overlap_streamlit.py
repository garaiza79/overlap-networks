"""
Overlap Analysis - Streamlit App (Memory Optimized)
=====================================================
Análisis de overlap entre dos redes de fibra óptica.
Optimizado para Streamlit Cloud (~1GB RAM).

Requisitos:
  pip install streamlit geopandas shapely pyogrio simplekml pandas

Ejecutar:
  streamlit run overlap_streamlit.py
"""

import streamlit as st
import geopandas as gpd
from shapely.strtree import STRtree
from shapely.ops import unary_union
from shapely.geometry import MultiLineString, LineString
from shapely import make_valid
import simplekml
import pandas as pd
import tempfile
import os
import time
import zipfile
import io
import gc

# =============================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================
st.set_page_config(
    page_title="Overlap Analysis - Redes de Fibra Óptica",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================
# ESTILOS CSS
# =============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
    .stApp { font-family: 'DM Sans', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        border-radius: 12px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 { color: #f0f9ff; font-weight: 700; font-size: 1.8rem; margin: 0 0 0.3rem 0; }
    .main-header p { color: #94a3b8; font-size: 0.95rem; margin: 0; }
    .metric-row { display: flex; gap: 1rem; margin: 1rem 0; }
    .metric-card {
        background: #0f172a; border: 1px solid #1e3a5f; border-radius: 10px;
        padding: 1.2rem 1.5rem; flex: 1; text-align: center;
    }
    .metric-card .label { color: #64748b; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { color: #f0f9ff; font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; margin: 0.3rem 0 0.1rem; }
    .metric-card .sub { color: #38bdf8; font-size: 0.85rem; font-weight: 500; }
    .metric-card.overlap { border-color: #ef4444; }
    .metric-card.overlap .sub { color: #f87171; }
    .metric-card.unique { border-color: #22c55e; }
    .metric-card.unique .sub { color: #4ade80; }
    .status-badge { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
    .status-ok { background: rgba(34, 197, 94, 0.15); color: #4ade80; }
    .status-warn { background: rgba(234, 179, 8, 0.15); color: #facc15; }
    section[data-testid="stSidebar"] { background: #0f172a; }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stFileUploader label,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown label { color: #e2e8f0 !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================
# FUNCIONES DE ANÁLISIS (optimizadas para bajo consumo de RAM)
# =============================================================

def cargar_archivo(uploaded_file, nombre_alias):
    """Carga archivo geoespacial, filtra líneas, reproyecta a UTM. Descarta atributos para ahorrar RAM."""
    ext = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.TemporaryDirectory() as tmpdir:
        ruta_tmp = os.path.join(tmpdir, uploaded_file.name)
        with open(ruta_tmp, "wb") as f:
            f.write(uploaded_file.getbuffer())

        read_kwargs = {"engine": "pyogrio", "on_invalid": "ignore"}

        if ext == ".kmz":
            with zipfile.ZipFile(ruta_tmp, "r") as z:
                z.extractall(tmpdir)
            kml_files = [f for f in os.listdir(tmpdir) if f.lower().endswith(".kml")]
            if not kml_files:
                st.error(f"No se encontró KML dentro del KMZ '{uploaded_file.name}'")
                return None
            gdf = gpd.read_file(os.path.join(tmpdir, kml_files[0]), **read_kwargs)
        elif ext == ".kml":
            gdf = gpd.read_file(ruta_tmp, **read_kwargs)
        elif ext == ".zip":
            gdf = gpd.read_file(f"zip://{ruta_tmp}", **read_kwargs)
        else:
            gdf = gpd.read_file(ruta_tmp, **read_kwargs)

    # Limpiar geometrías
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.apply(lambda g: make_valid(g) if g is not None and not g.is_valid else g)

    # Filtrar solo líneas y descartar atributos (ahorro de RAM)
    lineas = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])][["geometry"]].copy()
    del gdf
    gc.collect()

    if len(lineas) == 0:
        st.error(f"❌ No se encontraron líneas en '{nombre_alias}'.")
        return None

    # Reproyectar a UTM
    if lineas.crs is None or lineas.crs.is_geographic:
        centroide = lineas.geometry.iloc[0].centroid
        zona_utm = int((centroide.x + 180) / 6) + 1
        hemisferio = 326 if centroide.y >= 0 else 327
        epsg = hemisferio * 100 + zona_utm
        lineas = lineas.to_crs(epsg=epsg)

    return lineas


def calcular_overlap_solo_km(red_origen, red_consulta, buffer_m, progress_bar=None, progress_text=None, label=""):
    """
    Calcula overlap usando STRtree. Solo retorna kilómetros (no guarda geometrías).
    Esto ahorra MUCHA memoria comparado con guardar todas las geometrías.
    """
    buffers = red_origen.geometry.buffer(buffer_m)
    tree = STRtree(buffers.values)

    km_overlap = 0.0
    km_unico = 0.0
    total = len(red_consulta)

    for i, linea in enumerate(red_consulta.geometry):
        if progress_bar and total > 0:
            progress_bar.progress((i + 1) / total)
        if progress_text and ((i + 1) % 300 == 0 or (i + 1) == total):
            progress_text.text(f"{label} — {i+1:,}/{total:,} ({(i+1)/total*100:.0f}%)")

        candidatos_idx = tree.query(linea)

        if len(candidatos_idx) == 0:
            km_unico += linea.length
            continue

        zona_local = unary_union([buffers.iloc[j] for j in candidatos_idx])
        parte_overlap = linea.intersection(zona_local)
        parte_unica = linea.difference(zona_local)

        if not parte_overlap.is_empty:
            km_overlap += parte_overlap.length
        if not parte_unica.is_empty:
            km_unico += parte_unica.length

    return km_overlap / 1000, km_unico / 1000


def generar_kmz_streaming(red_origen, red_consulta, buffer_m, color, nombre_capa, es_overlap=True):
    """
    Genera KMZ directamente sin guardar geometrías en RAM.
    Reproyecta en lotes para eficiencia.
    """
    buffers = red_origen.geometry.buffer(buffer_m)
    tree = STRtree(buffers.values)
    kml = simplekml.Kml(name=nombre_capa)
    crs_utm = red_consulta.crs
    count = 0
    batch_geoms = []
    BATCH = 2000

    def escribir_lote(geoms_utm):
        nonlocal count
        if not geoms_utm:
            return
        gdf_tmp = gpd.GeoDataFrame(geometry=geoms_utm, crs=crs_utm).to_crs(epsg=4326)
        for geom_wgs in gdf_tmp.geometry:
            if geom_wgs.is_empty:
                continue
            lines = geom_wgs.geoms if geom_wgs.geom_type == "MultiLineString" else [geom_wgs]
            for line in lines:
                coords = [(c[0], c[1]) for c in line.coords]
                ls = kml.newlinestring(coords=coords)
                ls.style.linestyle.color = color
                ls.style.linestyle.width = 3
                count += 1

    for linea_utm in red_consulta.geometry:
        candidatos_idx = tree.query(linea_utm)

        if es_overlap:
            if len(candidatos_idx) == 0:
                continue
            zona_local = unary_union([buffers.iloc[j] for j in candidatos_idx])
            resultado = linea_utm.intersection(zona_local)
        else:
            if len(candidatos_idx) == 0:
                resultado = linea_utm
            else:
                zona_local = unary_union([buffers.iloc[j] for j in candidatos_idx])
                resultado = linea_utm.difference(zona_local)

        if not resultado.is_empty:
            batch_geoms.append(resultado)

        if len(batch_geoms) >= BATCH:
            escribir_lote(batch_geoms)
            batch_geoms = []
            gc.collect()

    escribir_lote(batch_geoms)
    del batch_geoms
    gc.collect()

    with tempfile.TemporaryDirectory() as tmpdir:
        ruta_kmz = os.path.join(tmpdir, f"{nombre_capa}.kmz")
        kml.savekmz(ruta_kmz)
        with open(ruta_kmz, "rb") as f:
            return f.read(), count


def crear_zip_descarga(archivos_dict):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for nombre, contenido in archivos_dict.items():
            zf.writestr(nombre, contenido)
    buffer.seek(0)
    return buffer


# =============================================================
# INTERFAZ PRINCIPAL
# =============================================================

st.markdown("""
<div class="main-header">
    <h1>🔗 Overlap Analysis</h1>
    <p>Análisis de traslape entre redes de fibra óptica · Buffer + Intersección Espacial</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")
    st.markdown("**📂 Red 1**")
    archivo_1 = st.file_uploader("Subir archivo Red 1", type=["kmz", "kml", "gpkg", "shp", "zip"], key="file1")
    alias_1 = st.text_input("Alias Red 1", value="Red A", key="alias1")
    st.markdown("---")
    st.markdown("**📂 Red 2**")
    archivo_2 = st.file_uploader("Subir archivo Red 2", type=["kmz", "kml", "gpkg", "shp", "zip"], key="file2")
    alias_2 = st.text_input("Alias Red 2", value="Red B", key="alias2")
    st.markdown("---")
    st.markdown("**📐 Buffer de tolerancia**")
    buffer_metros = st.slider("Metros de buffer", min_value=5, max_value=100, value=30, step=5)
    st.caption(f"Corredor total: **{buffer_metros * 2}m** de ancho ({buffer_metros}m por lado)")
    st.markdown("---")
    st.markdown("**💾 Archivos de salida**")
    nombre_salida = st.text_input("Prefijo para archivos", value="overlap_resultado")
    generar_kmz = st.checkbox("Generar KMZ (requiere más tiempo y memoria)", value=False)
    st.markdown("---")
    ejecutar = st.button("🚀 Ejecutar Análisis", use_container_width=True, type="primary")

col_s1, col_s2 = st.columns(2)
with col_s1:
    if archivo_1:
        st.markdown(f'<span class="status-badge status-ok">✓ {alias_1}</span> <small>{archivo_1.name}</small>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-badge status-warn">⏳ {alias_1}</span> <small>Esperando archivo...</small>', unsafe_allow_html=True)
with col_s2:
    if archivo_2:
        st.markdown(f'<span class="status-badge status-ok">✓ {alias_2}</span> <small>{archivo_2.name}</small>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-badge status-warn">⏳ {alias_2}</span> <small>Esperando archivo...</small>', unsafe_allow_html=True)

if ejecutar:
    if not archivo_1 or not archivo_2:
        st.error("❌ Sube ambos archivos antes de ejecutar.")
        st.stop()

    inicio = time.time()

    with st.status("📂 Cargando archivos...", expanded=True) as status:
        st.write(f"Cargando **{alias_1}**...")
        red_a = cargar_archivo(archivo_1, alias_1)
        if red_a is None:
            st.stop()
        km_a = red_a.geometry.length.sum() / 1000
        st.write(f"✅ **{alias_1}**: {len(red_a):,} líneas — **{km_a:,.2f} km**")

        st.write(f"Cargando **{alias_2}**...")
        red_b = cargar_archivo(archivo_2, alias_2)
        if red_b is None:
            st.stop()
        km_b = red_b.geometry.length.sum() / 1000
        st.write(f"✅ **{alias_2}**: {len(red_b):,} líneas — **{km_b:,.2f} km**")

        if red_a.crs != red_b.crs:
            red_b = red_b.to_crs(red_a.crs)
        status.update(label="📂 Archivos cargados", state="complete")

    # Paso 1
    st.markdown(f"#### Paso 1: ¿Cuánto de **{alias_2}** cae dentro del buffer de **{alias_1}**?")
    p1_bar = st.progress(0)
    p1_text = st.empty()
    km_overlap_b, km_unico_b = calcular_overlap_solo_km(red_a, red_b, buffer_metros, p1_bar, p1_text, f"{alias_2} vs {alias_1}")
    p1_bar.progress(1.0)
    p1_text.text(f"✅ {alias_2}: {km_overlap_b:,.2f} km overlap — {km_unico_b:,.2f} km único")
    gc.collect()

    # Paso 2
    st.markdown(f"#### Paso 2: ¿Cuánto de **{alias_1}** cae dentro del buffer de **{alias_2}**?")
    p2_bar = st.progress(0)
    p2_text = st.empty()
    km_overlap_a, km_unico_a = calcular_overlap_solo_km(red_b, red_a, buffer_metros, p2_bar, p2_text, f"{alias_1} vs {alias_2}")
    p2_bar.progress(1.0)
    p2_text.text(f"✅ {alias_1}: {km_overlap_a:,.2f} km overlap — {km_unico_a:,.2f} km único")
    gc.collect()

    duracion = time.time() - inicio

    # Resultados
    st.markdown("---")
    st.markdown(f"### 📊 Resultados &nbsp; <small style='color:#64748b;'>({duracion:.1f} seg · buffer {buffer_metros}m)</small>", unsafe_allow_html=True)

    pct_a = km_overlap_a / km_a * 100 if km_a > 0 else 0
    pct_b = km_overlap_b / km_b * 100 if km_b > 0 else 0

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card"><div class="label">{alias_1} — Total</div><div class="value">{km_a:,.2f}</div><div class="sub">km</div></div>
        <div class="metric-card"><div class="label">{alias_2} — Total</div><div class="value">{km_b:,.2f}</div><div class="sub">km</div></div>
    </div>
    <div class="metric-row">
        <div class="metric-card overlap"><div class="label">Overlap {alias_1}</div><div class="value">{km_overlap_a:,.2f}</div><div class="sub">{pct_a:.1f}% de {alias_1}</div></div>
        <div class="metric-card overlap"><div class="label">Overlap {alias_2}</div><div class="value">{km_overlap_b:,.2f}</div><div class="sub">{pct_b:.1f}% de {alias_2}</div></div>
    </div>
    <div class="metric-row">
        <div class="metric-card unique"><div class="label">Único {alias_1}</div><div class="value">{km_unico_a:,.2f}</div><div class="sub">{km_unico_a/km_a*100:.1f}% de {alias_1}</div></div>
        <div class="metric-card unique"><div class="label">Único {alias_2}</div><div class="value">{km_unico_b:,.2f}</div><div class="sub">{km_unico_b/km_b*100:.1f}% de {alias_2}</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📋 Tabla Resumen")
    df_resumen = pd.DataFrame([
        {"Concepto": f"{alias_1} — Total", "Kilómetros": f"{km_a:,.2f}", "Porcentaje": "100%"},
        {"Concepto": f"{alias_2} — Total", "Kilómetros": f"{km_b:,.2f}", "Porcentaje": "100%"},
        {"Concepto": f"Overlap en {alias_1}", "Kilómetros": f"{km_overlap_a:,.2f}", "Porcentaje": f"{pct_a:.1f}%"},
        {"Concepto": f"Overlap en {alias_2}", "Kilómetros": f"{km_overlap_b:,.2f}", "Porcentaje": f"{pct_b:.1f}%"},
        {"Concepto": f"Único {alias_1}", "Kilómetros": f"{km_unico_a:,.2f}", "Porcentaje": f"{km_unico_a/km_a*100:.1f}%"},
        {"Concepto": f"Único {alias_2}", "Kilómetros": f"{km_unico_b:,.2f}", "Porcentaje": f"{km_unico_b/km_b*100:.1f}%"},
        {"Concepto": "Buffer (metros)", "Kilómetros": str(buffer_metros), "Porcentaje": ""},
    ])
    st.dataframe(df_resumen, use_container_width=True, hide_index=True)

    # Exportar
    st.markdown("---")
    st.markdown("### 💾 Descargar Archivos")

    archivos_zip = {}
    prefix = nombre_salida

    # CSV siempre
    csv_buf = io.StringIO()
    csv_buf.write("Concepto,Kilómetros,Porcentaje\n")
    csv_buf.write(f"{alias_1} - Total,{km_a:.2f},100%\n")
    csv_buf.write(f"{alias_2} - Total,{km_b:.2f},100%\n")
    csv_buf.write(f"Overlap en {alias_1},{km_overlap_a:.2f},{pct_a:.1f}%\n")
    csv_buf.write(f"Overlap en {alias_2},{km_overlap_b:.2f},{pct_b:.1f}%\n")
    csv_buf.write(f"Único {alias_1},{km_unico_a:.2f},{km_unico_a/km_a*100:.1f}%\n")
    csv_buf.write(f"Único {alias_2},{km_unico_b:.2f},{km_unico_b/km_b*100:.1f}%\n")
    csv_buf.write(f"Buffer (metros),{buffer_metros},\n")
    csv_bytes = csv_buf.getvalue().encode("utf-8")
    archivos_zip[f"{prefix}_resumen.csv"] = csv_bytes

    st.download_button("📋 Descargar CSV Resumen", data=csv_bytes, file_name=f"{prefix}_resumen.csv", mime="text/csv", use_container_width=True)

    # KMZ opcional
    if generar_kmz:
        with st.status("Generando archivos KMZ (esto toma unos minutos)...", expanded=True) as kmz_status:
            colores = {"overlap": "ff0000ff", "unico": "ff00ff00"}

            for label_tipo, es_ov in [("overlap", True), ("unico", False)]:
                for alias, r_origen, r_consulta in [(alias_1, red_b, red_a), (alias_2, red_a, red_b)]:
                    nombre_f = f"{prefix}_{label_tipo}_{alias.replace(' ', '_')}"
                    st.write(f"Generando: {label_tipo.capitalize()} {alias}...")
                    data, n = generar_kmz_streaming(r_origen, r_consulta, buffer_metros, colores[label_tipo], nombre_f, es_overlap=es_ov)
                    archivos_zip[f"{nombre_f}.kmz"] = data
                    st.write(f"✅ {nombre_f}.kmz ({n:,} elementos)")
                    del data
                    gc.collect()

            kmz_status.update(label="✅ KMZ generados", state="complete")

    if len(archivos_zip) > 1:
        zip_buffer = crear_zip_descarga(archivos_zip)
        st.download_button(f"📦 Descargar TODO ({len(archivos_zip)} archivos)", data=zip_buffer, file_name=f"{prefix}_completo.zip", mime="application/zip", use_container_width=True, type="primary")

    with st.expander("📂 Descargas individuales"):
        for nombre_archivo, contenido in archivos_zip.items():
            mime = "text/csv" if nombre_archivo.endswith(".csv") else "application/vnd.google-earth.kmz" if nombre_archivo.endswith(".kmz") else "application/octet-stream"
            st.download_button(f"⬇️ {nombre_archivo}", data=contenido, file_name=nombre_archivo, mime=mime, key=f"dl_{nombre_archivo}")

    st.success(f"✅ Análisis completo en {time.time() - inicio:.1f} segundos")

st.markdown("---")
st.markdown("<div style='text-align:center; color:#475569; font-size:0.8rem;'>Overlap Analysis · Buffer + Intersección Espacial (STRtree) · KMZ, KML, GPKG, SHP</div>", unsafe_allow_html=True)
