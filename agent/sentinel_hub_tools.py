# geospatial_tools.py

import folium
from rasterio.warp import reproject, Resampling

import rasterio
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from rasterio.mask import mask
import tempfile
import os
import os
import json
import geopandas as gpd
import requests
import osmnx as ox
from dotenv import load_dotenv
from sentinelsat import geojson_to_wkt
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from pydantic import BaseModel
import cdsapi
import xarray as xr
import rasterio
from rasterio.transform import from_origin
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point
from scipy.spatial import cKDTree
from rasterio.features import rasterize
from rasterio.windows import Window
from tqdm import tqdm

# Load environment variables
load_dotenv()
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

##########################
# Authentication Utility #
##########################
def get_access_token():
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)
    token = oauth.fetch_token(
        token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
        client_secret=client_secret,
        include_client_id=True
    )
    return token['access_token']


#####################
# Geo BBox Utilities #
#####################

class ExtractBox(BaseModel):
    location: str
    distance: float = 1000.0  # Default distance in meters
    filepath: str

def extract_bbox(**kwargs):
    input = ExtractBox(**kwargs)
    location = input.location
    distance = input.distance
    filepath = input.filepath

    # Generate GeoDataFrame from location
    gdf = ox.geocode_to_gdf(location)
    gdf.to_file(filepath, driver="GeoJSON")
    print(f"✅ GeoJSON file saved as {filepath}")

    # Compute bounding box
    minx, miny, maxx, maxy = gdf.total_bounds

    # ✅ Return structured output
    return {
        "file_path": filepath,
        "bbox": [minx, miny, maxx, maxy]
    }


class GetBox(BaseModel):
    file_path: str

def get_bbox(**kwargs):
    input = GetBox(**kwargs)
    file_path = input.file_path

    gdf = gpd.read_file(file_path)
    minx, miny, maxx, maxy = gdf.total_bounds

    # ✅ Return both bbox and path
    return {
        "file_path": file_path,
        "bbox": [minx, miny, maxx, maxy]
    }



##########################
# Evalscripts Dictionary #
##########################

evalscripts = {
    "dem": '''
function setup() {
  return { input: ["DEM"], output: { bands: 1 } };
}
function evaluatePixel(sample) {
  return [sample.DEM];
}''',
    "landcover": '''
function setup() {
  return { input: ["Map"], output: { bands: 1 } };
}
function evaluatePixel(sample) {
  return [sample.Map];
}''',
    "ndvi": '''
function setup() {
  return { input: ["B04", "B08"], output: { bands: 1 } };
}
function evaluatePixel(sample) {
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
  return [ndvi];
}''',
    "soil_saturation": '''
function setup() {
  return { input: ["VV", "VH"], output: { bands: 2 } };
}
function evaluatePixel(sample) {
  return [sample.VV, sample.VH];
}''',
    "ndwi": '''
function setup() {
  return { input: ["B03", "B08"], output: { bands: 1 } };
}
function evaluatePixel(sample) {
  let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08);
  return [ndwi];
}''',
    "aod": '''
function setup() {
  return { input: ["AER_AI_340_380"], output: { bands: 1 } };
}
function evaluatePixel(sample) {
  return [sample.AER_AI_340_380];
}'''
}


##########################
# Dataset Type Mapping   #
##########################

SUPPORTED_DATASETS = {
    "dem": "DEM",
    "ndvi": "sentinel-2-l2a",
    "ndwi": "sentinel-2-l2a",
    "landcover": "sentinel-2-l2a",
    "soil_saturation": "sentinel-1-grd",
    "aod": "modis"
}


#####################
# Payload Generator #
#####################

class CreatePayload(BaseModel):
    dataset_type: str
    bbox: list
    time_from: str = "2023-01-01T00:00:00Z"
    time_to: str = "2023-12-31T23:59:59Z"
    evalscript: str = None

def create_payload(**kwargs):
    input = CreatePayload(**kwargs)

    dataset_type = input.dataset_type.lower()
    bbox = input.bbox
    time_from = input.time_from
    time_to = input.time_to
    evalscript = input.evalscript

    if dataset_type not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    if not evalscript:
        if dataset_type not in evalscripts:
            raise ValueError(f"No default evalscript available for dataset_type '{dataset_type}'")
        evalscript = evalscripts[dataset_type]

    print(f"🛠️ Creating payload for dataset: {dataset_type}, time: {time_from} to {time_to}")

    return {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": SUPPORTED_DATASETS[dataset_type],
                "dataFilter": {
                    "timeRange": {
                        "from": time_from,
                        "to": time_to
                    }
                }
            }]
        },
        "output": {
            "width": 512,
            "height": 512,
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/tiff"}
            }]
        },
        "evalscript": evalscript
    }


##################
# Process Handler #
##################

class ProcessRequest(BaseModel):
    payload: dict
    filepath: str

def process_request(**kwargs) -> str:
    input = ProcessRequest(**kwargs)
    payload = input.payload
    filepath = input.filepath

    if not filepath.endswith(".tif"):
        filepath += ".tif"

    access_token = get_access_token()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    print(f"🚀 Sending request to Sentinel Hub...")
    response = requests.post(
        "https://sh.dataspace.copernicus.eu/api/v1/process",
        headers=headers,
        data=json.dumps(payload)
    )

    if response.ok:
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"✅ Processed data saved to {filepath}")
        return filepath
    else:
        print("❌ Request failed:", response.status_code)
        try:
            print("📄 Error details:", response.json())
        except:
            print("📄 Error content:", response.text)
        raise RuntimeError(f"Sentinel Hub request failed with status {response.status_code}")
    
#########################################################


class GetPrecipation(BaseModel):
    bbox: list
    output_path: str

def get_area(bbox):
   
    minx, miny, maxx, maxy = bbox
    area = [maxy, minx, miny, maxx]  # Convert to [N, W, S, E]
    return area
    

def convert_nc_tif(netcdf_path: str, output_tif_path: str):

    # 1. Load the NetCDF file
    ds = xr.open_dataset(netcdf_path)

    # 2. Extract total precipitation (assuming single time step)
    precip = ds['tp'][0]
    lat = precip.latitude.values
    lon = precip.longitude.values
    data = precip.values

    # 3. Flip data if latitude is decreasing
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        data = data[::-1, :]

    # 4. Define geotransform
    res_lon = abs(lon[1] - lon[0])
    res_lat = abs(lat[1] - lat[0])
    transform = from_origin(
        west=lon[0],
        north=lat[-1] + len(lat) * res_lat,
        xsize=res_lon,
        ysize=res_lat
    )

    # 5. Save to GeoTIFF
    with rasterio.open(
        output_tif_path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype='float32',
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(data.astype('float32'), 1)

    print(f"✅ Saved: {output_tif_path}")


def get_precipitation(**kwargs):
    input = GetPrecipation(**kwargs)
    bbox = input.bbox
    output_path = input.output_path
    c = cdsapi.Client()
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    area = get_area(bbox)
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': 'total_precipitation',
            'year': '2023',
            'month': '08',
            'time': '00:00',
            'area': area,  # [North, West, South, East]
            'format': 'netcdf',
        },
        base_name + '.nc'
    )
    convert_nc_tif(base_name + '.nc', base_name + '.tif')
    


def extract_rivers(place_name: str ):
    # --- Fetch OSM features ---
    base_name = os.path.basename(place_name).split("_bbox")[0]
    output_path = base_name + "_rivers.geojson"
    print(f"🔍 Downloading OSM rivers for: {place_name}")
    gdf = ox.features_from_place(base_name, {"waterway": True})

    # --- Filter only rivers and streams ---
    rivers = gdf[gdf['waterway'].isin(['river', 'stream'])]

    # --- Save to GeoJSON ---
    rivers.to_file(output_path, driver="GeoJSON")
    print(f"✅ Saved: {output_path}")
    return output_path

class GenerateDistanceAndDrainageDensity(BaseModel):
    region_path: str
    resolution: int = 100  # in meters
    density_window: int = 1000  # in meters


def generate_distance_and_drainage_density(**kwargs):
    input  = GenerateDistanceAndDrainageDensity(**kwargs)
    region_path = input.region_path
    resolution = input.resolution
    density_window = input.density_window
    # === Load region and rivers ===
    region = gpd.read_file(region_path).to_crs("EPSG:4326")
    base_name = os.path.splitext(os.path.basename(region_path))[0]
    rivers_path = extract_rivers(base_name + " India")
    
    rivers = gpd.read_file(rivers_path).to_crs(region.crs)

    # === Define raster grid ===
    minx, miny, maxx, maxy = region.total_bounds
    width = int((maxx - minx) / (resolution / 111320))  # ~100m in degrees
    height = int((maxy - miny) / (resolution / 111320))
    transform = from_origin(minx, maxy, (maxx - minx) / width, (maxy - miny) / height)

    # === Rasterize river lines ===
    river_raster = rasterize(
        [(geom, 1) for geom in rivers.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype='uint8'
    )

    # === Get pixel centers
    xs = np.linspace(minx + 0.5 * transform.a, maxx - 0.5 * transform.a, width)
    ys = np.linspace(maxy - 0.5 * abs(transform.e), miny + 0.5 * abs(transform.e), height)
    grid_points = np.array([(x, y) for y in ys for x in xs])

    # === Get all river coordinates
    river_coords = []
    for geom in rivers.geometry:
        if geom.geom_type == "LineString":
            river_coords.extend(list(geom.coords))
        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                river_coords.extend(list(line.coords))
    river_coords = np.array(river_coords)

    # === Build KDTree and compute distances
    tree = cKDTree(river_coords)
    dists, _ = tree.query(grid_points, k=1)
    dist_raster = dists.reshape((height, width)).astype('float32') * 111320  # Convert deg to meters

    # === Compute drainage density (using moving window sum of river pixels)
    kernel_size = int(density_window / resolution)
    padded = np.pad(river_raster, kernel_size // 2, mode='constant')
    density_raster = np.zeros_like(river_raster, dtype='float32')

    for i in tqdm(range(height)):
        for j in range(width):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            pixel_area_km2 = (resolution / 1000) ** 2
            river_length_km = window.sum() * (resolution / 1000)
            density_raster[i, j] = river_length_km / (kernel_size**2 * pixel_area_km2)

    # === Save distance raster
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': transform
    }

    with rasterio.open(base_name+"_river_proximity.tif", 'w', **profile) as dst:
        dst.write(dist_raster, 1)

    with rasterio.open(base_name+ "_drainage_density.tif", 'w', **profile) as dst:
        dst.write(density_raster, 1)

    return {
        "distance_raster": base_name + "_river_proximity.tif",
        "density_raster": base_name + "_drainage_density.tif",
    }
    


class SaveMapAsPNG(BaseModel):
    html_path: str
    png_output: str
    delay: int = 3  
    width: int = 1024
    height: int = 768

def save_folium_map_as_png(**kwargs):
    input = SaveMapAsPNG(**kwargs)
    html_path = input.html_path
    png_output = input.png_output
    delay = input.delay
    width = input.width
    height = input.height
    

    options = Options()
    options.add_argument("--headless")
    options.add_argument(f"--window-size={width},{height}")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    # Path setup
    html_file = "file://" + os.path.abspath(html_path)

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(html_file)
        time.sleep(delay)  # wait for tiles to load
        driver.save_screenshot(png_output)
        print(f"✅ Saved PNG: {png_output}")
    finally:
        driver.quit()
        
    return png_output

class OverlayMap(BaseModel):
    raster_path: str
    vector_path: str 
    output_html: str 
    zoom_start: int = 8
    opacity: float = 0.6

def overlay_masked_raster_on_osm(
    **kwargs
):
    
    input = OverlayMap(**kwargs)
    raster_path = input.raster_path
    vector_path = input.vector_path
    output_html = input.output_html
    zoom_start = input.zoom_start
    opacity = input.opacity
    
    name = os.path.splitext(os.path.basename(raster_path))[0]
    # === Load region ===
    region = gpd.read_file(vector_path).to_crs("EPSG:4326")

    # === Load and mask raster ===
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, region.geometry, crop=True)
        data = out_image[0]
        bounds = rasterio.transform.array_bounds(*data.shape, out_transform)
        bounds = [out_transform * (0, 0), out_transform * (data.shape[1], data.shape[0])]
        bounds = [[bounds[0][1], bounds[0][0]], [bounds[1][1], bounds[1][0]]]  # [[south, west], [north, east]]

    # === Normalize for colormap ===
    data = np.where(np.isnan(data), 0, data)
    norm = Normalize(vmin=0, vmax=1)
    img_norm = norm(data)

    # === Save as temporary PNG ===
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        ax.imshow(img_norm, cmap='Reds', interpolation='nearest')
        ax.axis("off")
        fig.savefig(tmp.name, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        img_path = tmp.name

    # === Center map ===
    center_lat = (bounds[0][0] + bounds[1][0]) / 2
    center_lon = (bounds[0][1] + bounds[1][1]) / 2

    # === Create folium map ===
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    folium.raster_layers.ImageOverlay(
        name=name,
        image=img_path,
        bounds=bounds,
        opacity=opacity,
        interactive=True,
        cross_origin=False,
        zindex=1
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_html)
    print(f"✅ Saved: {output_html}")

    # Cleanup temp image
    os.remove(img_path)
    


# --- Normalize raster ---
def normalize(array):
    array = np.where(np.isnan(array), 0, array)
    arr_min = np.min(array)
    arr_max = np.max(array)
    if arr_max == arr_min:
        return np.zeros_like(array)
    return (array - arr_min) / (arr_max - arr_min)

# --- Build reference grid from AOI ---
def create_reference_grid_from_geojson(geojson_path, resolution=0.001):
    gdf = gpd.read_file(geojson_path).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds

    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    transform = from_origin(minx, maxy, resolution, resolution)

    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
    }

    return gdf, meta

# --- Clip and read raster ---
def load_and_clip(path, region):
    with rasterio.open(path) as src:
        out_image, _ = mask(src, region.geometry, crop=True)
        return out_image[0], src.meta.copy()

# --- Resample raster to match reference ---
def resample_to_match(source_array, source_meta, target_meta):
    dst_array = np.zeros((target_meta["height"], target_meta["width"]), dtype=np.float32)
    reproject(
        source=source_array,
        destination=dst_array,
        src_transform=source_meta["transform"],
        src_crs=source_meta["crs"],
        dst_transform=target_meta["transform"],
        dst_crs=target_meta["crs"],
        resampling=Resampling.bilinear
    )
    return dst_array

# --- Flood risk computation ---
class ComputeFloodRisk(BaseModel):
    dem_path: str
    precip_path: str
    dist_path: str
    density_path: str
    region_path: str
    output_path: str
def compute_flood_risk(**kwargs):
    input = ComputeFloodRisk(**kwargs)
    dem_path = input.dem_path
    precip_path = input.precip_path
    dist_path = input.dist_path
    density_path = input.density_path
    region_path = input.region_path
    output_path = input.output_path
    region, ref_meta = create_reference_grid_from_geojson(region_path, resolution=0.001)

    # Load & clip input rasters
    dem_raw, dem_meta = load_and_clip(dem_path, region)
    precip_raw, precip_meta = load_and_clip(precip_path, region)
    dist_raw, dist_meta = load_and_clip(dist_path, region)
    density_raw, density_meta = load_and_clip(density_path, region)

    # Resample to reference grid
    dem = resample_to_match(dem_raw, dem_meta, ref_meta)
    precip = resample_to_match(precip_raw, precip_meta, ref_meta)
    dist = resample_to_match(dist_raw, dist_meta, ref_meta)
    density = resample_to_match(density_raw, density_meta, ref_meta)

    print("DEM shape:", dem.shape)
    print("Precip shape:", precip.shape)
    print("Dist shape:", dist.shape)
    print("Density shape:", density.shape)

    # Normalize
    dem_norm = normalize(dem)
    precip_norm = normalize(precip)
    dist_norm = normalize(dist)
    density_norm = normalize(density)

    # Combine with weights
    flood_risk = (
        0.30 * precip_norm +
        0.25 * (1 - dem_norm) +
        0.25 * (1 - dist_norm) +
        0.20 * density_norm
    )

    # Save output
    with rasterio.open(output_path, 'w', **ref_meta) as dst:
        dst.write(flood_risk.astype('float32'), 1)

    print(f"✅ Saved: {output_path}")
    return output_path

import base64

class GetImageBase64(BaseModel):
    image_path: str
    
def get_image_base64(**kwargs) -> str:
    input = GetImageBase64(**kwargs)
    image_path = input.image_path
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return encoded
