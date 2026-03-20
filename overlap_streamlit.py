"""
Overlap Analysis - Streamlit App
=================================
Análisis de overlap entre dos redes de fibra óptica.
Soporta KMZ, KML, GPKG y SHP.

Requisitos:
  pip install streamlit geopandas shapely fiona simplekml zipfile36

Ejecutar:
  streamlit run overlap_streamlit.py
"""

import streamlit as st
import geopandas as gpd
from shapely.strtree import STRtree
from shapely.ops import unary_union
from shapely.geometry import MultiLineString, LineString
import simplekml
import pandas as pd
import tempfile
import os
import time
import zipfile
import io
import shutil

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

    /* General */
    .stApp {
        font-family: 'DM Sans', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #f0f9ff;
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        font-size: 1.8rem;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: #0f172a;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        flex: 1;
        text-align: center;
    }
    .metric-card .label {
        color: #64748b;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    .metric-card .value {
        color: #f0f9ff;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 500;
        margin: 0.3rem 0 0.1rem;
    }
    .metric-card .sub {
        color: #38bdf8;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .metric-card.overlap { border-color: #ef4444; }
    .metric-card.overlap .sub { color: #f87171; }
    .metric-card.unique { border-color: #22c55e; }
    .metric-card.unique .sub { color: #4ade80; }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-ok { background: rgba(34, 197, 94, 0.15); color: #4ade80; }
    .status-warn { background: rgba(234, 179, 8, 0.15); color: #facc15; }

    /* Progress area */
    .log-box {
        background: #0f172a;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #94a3b8;
        max-height: 300px;
        overflow-y: auto;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #0f172a;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown label {
        color: #cbd5e1;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================
# FUNCIONES DE ANÁLISIS (del script v2 optimizado)
# =============================================================

def cargar_archivo(uploaded_file, nombre_alias):
    """Carga un archivo geoespacial (KMZ, KML, GPKG, SHP) y retorna GeoDataFrame de líneas."""
    ext = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.TemporaryDirectory() as tmpdir:
        ruta_tmp = os.path.join(tmpdir, uploaded_file.name)
        with open(ruta_tmp, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if ext == ".kmz":
            # Descomprimir KMZ y leer el KML interior
            with zipfile.ZipFile(ruta_tmp, "r") as z:
                z.extractall(tmpdir)
            kml_files = [f for f in os.listdir(tmpdir) if f.lower().endswith(".kml")]
            if not kml_files:
                st.error(f"No se encontró un archivo KML dentro del KMZ '{uploaded_file.name}'")
                return None
            ruta_lectura = os.path.join(tmpdir, kml_files[0])
            # Habilitar driver KML en fiona
            import fiona
            fiona.drv["KML"] = "rw"
            fiona.drv["LIBKML"] = "rw"
            gdf = gpd.read_file(ruta_lectura, driver="KML")
        elif ext == ".kml":
            import fiona
            fiona.drv["KML"] = "rw"
            fiona.drv["LIBKML"] = "rw"
            gdf = gpd.read_file(ruta_tmp, driver="KML")
        elif ext == ".zip":
            # Shapefile como ZIP
            gdf = gpd.read_file(f"zip://{ruta_tmp}")
        elif ext == ".shp":
            gdf = gpd.read_file(ruta_tmp)
        else:
            # GPKG u otro formato soportado por fiona
            gdf = gpd.read_file(ruta_tmp)

    # Filtrar solo líneas
    lineas = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()

    if len(lineas) == 0:
        st.error(f"❌ No se encontraron líneas en '{nombre_alias}'. "
                 f"Tipos de geometría: {gdf.geometry.type.unique().tolist()}")
        return None

    # Reproyectar a UTM
    if lineas.crs is None or lineas.crs.is_geographic:
        centroide = lineas.geometry.iloc[0].centroid
        zona_utm = int((centroide.x + 180) / 6) + 1
        hemisferio = 326 if centroide.y >= 0 else 327
        epsg = hemisferio * 100 + zona_utm
        lineas = lineas.to_crs(epsg=epsg)

    return lineas


def calcular_overlap_rapido(red_origen, red_consulta, buffer_m, progress_bar=None, progress_text=None, label=""):
    """Calcula overlap usando índice espacial."""
    buffers = red_origen.geometry.buffer(buffer_m)
    tree = STRtree(buffers.values)

    overlap_geoms = []
    unico_geoms = []
    total = len(red_consulta)

    for i, linea in enumerate(red_consulta.geometry):
        if progress_bar and total > 0:
            progress_bar.progress((i + 1) / total)
        if progress_text and ((i + 1) % 200 == 0 or (i + 1) == total):
            progress_text.text(f"{label} — Procesando {i+1:,}/{total:,} ({(i+1)/total*100:.0f}%)")

        candidatos_idx = tree.query(linea)

        if len(candidatos_idx) == 0:
            unico_geoms.append(linea)
            continue

        zona_local = unary_union([buffers.iloc[j] for j in candidatos_idx])
        parte_overlap = linea.intersection(zona_local)
        parte_unica = linea.difference(zona_local)

        if not parte_overlap.is_empty:
            overlap_geoms.append(parte_overlap)
        if not parte_unica.is_empty:
            unico_geoms.append(parte_unica)

    km_overlap = sum(g.length for g in overlap_geoms) / 1000
    km_unico = sum(g.length for g in unico_geoms) / 1000

    return km_overlap, km_unico, overlap_geoms, unico_geoms


def geoms_a_gdf(geoms, crs):
    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    return gpd.GeoDataFrame(geometry=geoms, crs=crs)


def exportar_kmz_bytes(gdf, color, nombre_capa):
    """Genera un KMZ en memoria y retorna bytes."""
    kml = simplekml.Kml(name=nombre_capa)
    for geom in gdf.geometry:
        if geom.is_empty:
            continue
        lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for line in lines:
            coords = [(c[0], c[1]) for c in line.coords]
            ls = kml.newlinestring(coords=coords)
            ls.style.linestyle.color = color
            ls.style.linestyle.width = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        ruta_kmz = os.path.join(tmpdir, f"{nombre_capa}.kmz")
        kml.savekmz(ruta_kmz)
        with open(ruta_kmz, "rb") as f:
            return f.read()


def exportar_gpkg_bytes(gdf, nombre):
    """Genera un GPKG en memoria y retorna bytes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ruta = os.path.join(tmpdir, f"{nombre}.gpkg")
        gdf.to_file(ruta, driver="GPKG")
        with open(ruta, "rb") as f:
            return f.read()


def crear_zip_descarga(archivos_dict):
    """Crea un ZIP en memoria con múltiples archivos. archivos_dict = {nombre: bytes}"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for nombre, contenido in archivos_dict.items():
            zf.writestr(nombre, contenido)
    buffer.seek(0)
    return buffer


# =============================================================
# INTERFAZ PRINCIPAL
# =============================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🔗 Overlap Analysis</h1>
    <p>Análisis de traslape entre redes de fibra óptica · Buffer + Intersección Espacial</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Configuración
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")

    st.markdown("**📂 Red 1**")
    archivo_1 = st.file_uploader(
        "Subir archivo Red 1",
        type=["kmz", "kml", "gpkg", "shp", "zip"],
        key="file1",
        help="KMZ, KML, GPKG o SHP (ZIP para shapefiles)"
    )
    alias_1 = st.text_input("Alias Red 1", value="Red A", key="alias1")

    st.markdown("---")
    st.markdown("**📂 Red 2**")
    archivo_2 = st.file_uploader(
        "Subir archivo Red 2",
        type=["kmz", "kml", "gpkg", "shp", "zip"],
        key="file2",
        help="KMZ, KML, GPKG o SHP (ZIP para shapefiles)"
    )
    alias_2 = st.text_input("Alias Red 2", value="Red B", key="alias2")

    st.markdown("---")
    st.markdown("**📐 Buffer de tolerancia**")
    buffer_metros = st.slider(
        "Metros de buffer",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        help="Radio del buffer en metros alrededor de cada red. "
             "Un buffer de 30m crea un corredor de 60m de ancho (30m por lado)."
    )
    st.caption(f"Corredor total: **{buffer_metros * 2}m** de ancho ({buffer_metros}m por lado)")

    st.markdown("---")
    st.markdown("**💾 Archivos de salida**")
    nombre_salida = st.text_input(
        "Prefijo para archivos",
        value="overlap_resultado",
        help="Los archivos se nombrarán con este prefijo"
    )

    st.markdown("---")
    ejecutar = st.button("🚀 Ejecutar Análisis", use_container_width=True, type="primary")


# =============================================================
# ÁREA PRINCIPAL
# =============================================================

# Estado de archivos cargados
col_s1, col_s2 = st.columns(2)
with col_s1:
    if archivo_1:
        st.markdown(f'<span class="status-badge status-ok">✓ {alias_1}</span> &nbsp; '
                    f'<small>{archivo_1.name}</small>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-badge status-warn">⏳ {alias_1}</span> &nbsp; '
                    f'<small>Esperando archivo...</small>', unsafe_allow_html=True)
with col_s2:
    if archivo_2:
        st.markdown(f'<span class="status-badge status-ok">✓ {alias_2}</span> &nbsp; '
                    f'<small>{archivo_2.name}</small>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-badge status-warn">⏳ {alias_2}</span> &nbsp; '
                    f'<small>Esperando archivo...</small>', unsafe_allow_html=True)

# Ejecutar análisis
if ejecutar:
    if not archivo_1 or not archivo_2:
        st.error("❌ Debes subir ambos archivos antes de ejecutar el análisis.")
        st.stop()

    inicio = time.time()

    # --- Cargar archivos ---
    with st.status("📂 Cargando archivos...", expanded=True) as status:
        st.write(f"Cargando **{alias_1}** ({archivo_1.name})...")
        red_a = cargar_archivo(archivo_1, alias_1)
        if red_a is None:
            st.stop()
        km_a = red_a.geometry.length.sum() / 1000
        st.write(f"✅ **{alias_1}**: {len(red_a):,} líneas — **{km_a:,.2f} km**")

        st.write(f"Cargando **{alias_2}** ({archivo_2.name})...")
        red_b = cargar_archivo(archivo_2, alias_2)
        if red_b is None:
            st.stop()
        km_b = red_b.geometry.length.sum() / 1000
        st.write(f"✅ **{alias_2}**: {len(red_b):,} líneas — **{km_b:,.2f} km**")

        # Asegurar mismo CRS
        if red_a.crs != red_b.crs:
            st.write("Reproyectando al mismo CRS...")
            red_b = red_b.to_crs(red_a.crs)

        status.update(label="📂 Archivos cargados", state="complete")

    crs_trabajo = red_a.crs

    # --- Paso 1: Overlap de Red B dentro de Red A ---
    st.markdown(f"#### Paso 1: ¿Cuánto de **{alias_2}** cae dentro del buffer de **{alias_1}**?")
    p1_bar = st.progress(0)
    p1_text = st.empty()
    km_overlap_b, km_unico_b, geoms_overlap_b, geoms_unico_b = calcular_overlap_rapido(
        red_a, red_b, buffer_metros, p1_bar, p1_text, f"{alias_2} vs {alias_1}"
    )
    p1_bar.progress(1.0)
    p1_text.text(f"✅ {alias_2}: {km_overlap_b:,.2f} km overlap — {km_unico_b:,.2f} km único")

    # --- Paso 2: Overlap de Red A dentro de Red B ---
    st.markdown(f"#### Paso 2: ¿Cuánto de **{alias_1}** cae dentro del buffer de **{alias_2}**?")
    p2_bar = st.progress(0)
    p2_text = st.empty()
    km_overlap_a, km_unico_a, geoms_overlap_a, geoms_unico_a = calcular_overlap_rapido(
        red_b, red_a, buffer_metros, p2_bar, p2_text, f"{alias_1} vs {alias_2}"
    )
    p2_bar.progress(1.0)
    p2_text.text(f"✅ {alias_1}: {km_overlap_a:,.2f} km overlap — {km_unico_a:,.2f} km único")

    duracion = time.time() - inicio

    # =============================================================
    # RESULTADOS
    # =============================================================
    st.markdown("---")
    st.markdown(f"### 📊 Resultados &nbsp; <small style='color:#64748b;'>({duracion:.1f} seg · buffer {buffer_metros}m)</small>",
                unsafe_allow_html=True)

    # Métricas totales
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">{alias_1} — Total</div>
            <div class="value">{km_a:,.2f}</div>
            <div class="sub">km</div>
        </div>
        <div class="metric-card">
            <div class="label">{alias_2} — Total</div>
            <div class="value">{km_b:,.2f}</div>
            <div class="sub">km</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Overlap
    pct_a = km_overlap_a / km_a * 100 if km_a > 0 else 0
    pct_b = km_overlap_b / km_b * 100 if km_b > 0 else 0

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card overlap">
            <div class="label">Overlap {alias_1}</div>
            <div class="value">{km_overlap_a:,.2f}</div>
            <div class="sub">{pct_a:.1f}% de {alias_1}</div>
        </div>
        <div class="metric-card overlap">
            <div class="label">Overlap {alias_2}</div>
            <div class="value">{km_overlap_b:,.2f}</div>
            <div class="sub">{pct_b:.1f}% de {alias_2}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Únicos
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card unique">
            <div class="label">Único {alias_1}</div>
            <div class="value">{km_unico_a:,.2f}</div>
            <div class="sub">{km_unico_a/km_a*100:.1f}% de {alias_1}</div>
        </div>
        <div class="metric-card unique">
            <div class="label">Único {alias_2}</div>
            <div class="value">{km_unico_b:,.2f}</div>
            <div class="sub">{km_unico_b/km_b*100:.1f}% de {alias_2}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabla resumen
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

    # =============================================================
    # EXPORTAR ARCHIVOS
    # =============================================================
    st.markdown("---")
    st.markdown("### 💾 Descargar Archivos")

    colores_kml = {
        "overlap": "ff0000ff",   # Rojo
        "unico": "ff00ff00",     # Verde
    }

    # Preparar todos los archivos
    with st.status("Generando archivos de salida...", expanded=False) as export_status:
        archivos_zip = {}
        prefix = nombre_salida

        # Overlap Red A
        if geoms_overlap_a:
            gdf_ov_a = geoms_a_gdf(geoms_overlap_a, crs_trabajo).to_crs(epsg=4326)
            nombre_ov_a = f"{prefix}_overlap_{alias_1.replace(' ', '_')}"
            archivos_zip[f"{nombre_ov_a}.gpkg"] = exportar_gpkg_bytes(gdf_ov_a, nombre_ov_a)
            archivos_zip[f"{nombre_ov_a}.kmz"] = exportar_kmz_bytes(gdf_ov_a, colores_kml["overlap"], nombre_ov_a)
            st.write(f"✅ Overlap {alias_1}: GPKG + KMZ")

        # Overlap Red B
        if geoms_overlap_b:
            gdf_ov_b = geoms_a_gdf(geoms_overlap_b, crs_trabajo).to_crs(epsg=4326)
            nombre_ov_b = f"{prefix}_overlap_{alias_2.replace(' ', '_')}"
            archivos_zip[f"{nombre_ov_b}.gpkg"] = exportar_gpkg_bytes(gdf_ov_b, nombre_ov_b)
            archivos_zip[f"{nombre_ov_b}.kmz"] = exportar_kmz_bytes(gdf_ov_b, colores_kml["overlap"], nombre_ov_b)
            st.write(f"✅ Overlap {alias_2}: GPKG + KMZ")

        # Único Red A
        if geoms_unico_a:
            gdf_un_a = geoms_a_gdf(geoms_unico_a, crs_trabajo).to_crs(epsg=4326)
            nombre_un_a = f"{prefix}_unico_{alias_1.replace(' ', '_')}"
            archivos_zip[f"{nombre_un_a}.gpkg"] = exportar_gpkg_bytes(gdf_un_a, nombre_un_a)
            archivos_zip[f"{nombre_un_a}.kmz"] = exportar_kmz_bytes(gdf_un_a, colores_kml["unico"], nombre_un_a)
            st.write(f"✅ Único {alias_1}: GPKG + KMZ")

        # Único Red B
        if geoms_unico_b:
            gdf_un_b = geoms_a_gdf(geoms_unico_b, crs_trabajo).to_crs(epsg=4326)
            nombre_un_b = f"{prefix}_unico_{alias_2.replace(' ', '_')}"
            archivos_zip[f"{nombre_un_b}.gpkg"] = exportar_gpkg_bytes(gdf_un_b, nombre_un_b)
            archivos_zip[f"{nombre_un_b}.kmz"] = exportar_kmz_bytes(gdf_un_b, colores_kml["unico"], nombre_un_b)
            st.write(f"✅ Único {alias_2}: GPKG + KMZ")

        # CSV resumen
        csv_buffer = io.StringIO()
        csv_buffer.write("Concepto,Kilómetros,Porcentaje\n")
        csv_buffer.write(f"{alias_1} - Total,{km_a:.2f},100%\n")
        csv_buffer.write(f"{alias_2} - Total,{km_b:.2f},100%\n")
        csv_buffer.write(f"Overlap en {alias_1},{km_overlap_a:.2f},{pct_a:.1f}%\n")
        csv_buffer.write(f"Overlap en {alias_2},{km_overlap_b:.2f},{pct_b:.1f}%\n")
        csv_buffer.write(f"Único {alias_1},{km_unico_a:.2f},{km_unico_a/km_a*100:.1f}%\n")
        csv_buffer.write(f"Único {alias_2},{km_unico_b:.2f},{km_unico_b/km_b*100:.1f}%\n")
        csv_buffer.write(f"Buffer (metros),{buffer_metros},\n")
        archivos_zip[f"{prefix}_resumen.csv"] = csv_buffer.getvalue().encode("utf-8")
        st.write("✅ Resumen CSV")

        export_status.update(label="💾 Archivos listos", state="complete")

    # Botones de descarga
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        # ZIP con todo
        zip_buffer = crear_zip_descarga(archivos_zip)
        st.download_button(
            label=f"📦 Descargar TODO ({len(archivos_zip)} archivos)",
            data=zip_buffer,
            file_name=f"{prefix}_completo.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
        )

    with col_d2:
        # Solo CSV
        st.download_button(
            label="📋 Descargar solo CSV resumen",
            data=archivos_zip[f"{prefix}_resumen.csv"],
            file_name=f"{prefix}_resumen.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Descargas individuales (expandible)
    with st.expander("📂 Descargas individuales"):
        for nombre_archivo, contenido in archivos_zip.items():
            if nombre_archivo.endswith(".csv"):
                mime = "text/csv"
            elif nombre_archivo.endswith(".kmz"):
                mime = "application/vnd.google-earth.kmz"
            else:
                mime = "application/octet-stream"

            st.download_button(
                label=f"⬇️ {nombre_archivo}",
                data=contenido,
                file_name=nombre_archivo,
                mime=mime,
                key=f"dl_{nombre_archivo}",
            )

    st.success(f"✅ Análisis completo en {duracion:.1f} segundos")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#475569; font-size:0.8rem;'>"
    "Overlap Analysis · Buffer + Intersección Espacial (STRtree) · "
    "Formatos: KMZ, KML, GPKG, SHP"
    "</div>",
    unsafe_allow_html=True,
)
