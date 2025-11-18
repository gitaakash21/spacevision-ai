# app.py
import streamlit as st
import numpy as np
import rasterio
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="SpaceVision AI — Satellite Analyzer")

st.title("SpaceVision AI — Satellite Image Analyzer (demo)")
st.sidebar.header("Input")

tile_url = st.sidebar.text_input("Enter sentinel-cog tile base path:",
                                 value="https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/18/S/UH/2021/4/S2B_18SUH_20210423_0_L2A")
st.sidebar.write("Example band filenames appended: B02.tif, B03.tif, B04.tif, B08.tif, B11.tif")

def readb(url):
    with rasterio.Env():
        with rasterio.open(url) as src:
            return src.read(1).astype('float32')

def safe_idx(a,b):
    denom = (a + b)
    denom[denom==0] = np.nan
    return (a - b) / denom

def plot_to_img(arr, cmap='viridis', vmin=None, vmax=None):
    plt.figure(figsize=(6,4))
    plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, bbox_inches='tight', pad_inches=0.0)
    plt.close()
    buf.seek(0)
    return buf

if st.button("Run analysis"):
    try:
        red = readb(f"{tile_url}/B04.tif")
        green = readb(f"{tile_url}/B03.tif")
        blue = readb(f"{tile_url}/B02.tif")
        nir = readb(f"{tile_url}/B08.tif")
        swir = readb(f"{tile_url}/B11.tif")

        ndvi = safe_idx(nir, red)
        ndwi = safe_idx(green, nir)
        ndbi = safe_idx(swir, nir)

        st.subheader("Indices")
        cols = st.columns(3)
        cols[0].image(plot_to_img(ndvi, cmap='RdYlGn', vmin=-1, vmax=1), use_column_width=True)
        cols[1].image(plot_to_img(ndwi, cmap='Blues', vmin=-1, vmax=1), use_column_width=True)
        cols[2].image(plot_to_img(ndbi, cmap='Oranges', vmin=-1, vmax=1), use_column_width=True)

        # masks
        veg_mask = ndvi > 0.4
        water_mask = ndwi > 0.2
        urban_mask = (ndbi > 0.2) & (ndvi < 0.2)
        red_norm = red / np.nanmax(red)
        nir_norm = nir / np.nanmax(nir)
        hotspot = (red_norm > 0.6) & (nir_norm < 0.25)

        st.subheader("Detections")
        c2 = st.columns(4)
        c2[0].image(plot_to_img(veg_mask.astype('uint8')*255, cmap='gray'), use_column_width=True)
        c2[1].image(plot_to_img(water_mask.astype('uint8')*255, cmap='gray'), use_column_width=True)
        c2[2].image(plot_to_img(urban_mask.astype('uint8')*255, cmap='gray'), use_column_width=True)
        c2[3].image(plot_to_img(hotspot.astype('uint8')*255, cmap='hot'), use_column_width=True)

    except Exception as e:
        st.error(f"Error reading tile/bands: {e}")
