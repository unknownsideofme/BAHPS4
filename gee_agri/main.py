import ee
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle
import os
import io
from googleapiclient.http import MediaIoBaseDownload
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling

state_name = input("Enter state name: ")
# Google Drive Auth Setup
def get_drive_service(creds_path='credentials.json', token_path='token.pickle'):
    creds = None
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, [
                'https://www.googleapis.com/auth/drive'])
            creds = flow.run_local_server(port=0)
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)

# Delete any old .tif with matching name in folder
def delete_drive_file_if_exists(filename, folder_name="GEE_State_Exports"):
    service = get_drive_service()
    folder_results = service.files().list(
        q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
        spaces='drive',
        fields="files(id, name)",
        pageSize=1
    ).execute()
    folders = folder_results.get('files', [])
    if not folders:
        print(f"⚠ Folder '{folder_name}' not found.")
        return
    folder_id = folders[0]['id']
    query = f"name contains '{filename}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    for file in results.get('files', []):
        try:
            service.files().delete(fileId=file['id']).execute()
            print(f"🗑 Deleted old file: {file['name']}")
        except HttpError as e:
            print(f"❌ Error deleting {file['name']}: {e}")

# Automatically delete all old files before starting exports
base_filenames = ["avg_temp", "avg_precip", "avg_et", "lulc"]
for fname in base_filenames:
    delete_drive_file_if_exists(f"{fname}{state_name.replace(' ', '')}")



# ---------------- Step 1: Authenticate & Initialize ----------------
ee.Authenticate()
ee.Initialize() 

# ---------------- Step 2: Load State Boundary ----------------
# GEE GAUL Level 1 = States
 
states = ee.FeatureCollection("FAO/GAUL/2015/level1")
state = states.filter(ee.Filter.eq('ADM1_NAME', state_name))

# ---------------- Step 3: Load Climate & LULC Layers ----------------
# Temperature (1991–2020 average, °C)
temp = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY") \
    .filterDate('1991-01-01', '2020-12-31') \
    .select('temperature_2m') \
    .mean().subtract(273.15)  

# Precipitation (1991–2020 average, mm)
precip = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY") \
    .filterDate('1991-01-01', '2020-12-31') \
    .select('total_precipitation') \
    .mean().multiply(1000)  

# Evapotranspiration (2001–2020 average, mm)
et = ee.ImageCollection("MODIS/006/MOD16A2") \
    .filterDate('2001-01-01', '2020-12-31') \
    .select('ET') \
    .mean().multiply(0.1)  

# Land Use Land Cover (2020)
lulc = ee.ImageCollection("MODIS/006/MCD12Q1") \
    .filterDate('2020-01-01', '2020-12-31') \
    .first().select('LC_Type1')

# ---------------- Step 4: Export Function ----------------
def export_img(image, name, scale):
    task = ee.batch.Export.image.toDrive(
        image=image.clip(state),
        description=f"{name}_export",
        folder='GEE_State_Exports',
        fileNamePrefix=f"{name}{state_name.replace(' ', '')}",
        region=state.geometry(),
        scale=scale,
        fileFormat='GeoTIFF'
    )
    task.start()
    print(f"✅ Export started: {name}")
    return task

# ---------------- Step 5: Start Exports ----------------
tasks = [
    export_img(temp, "avg_temp", 1000),
    export_img(precip, "avg_precip", 1000),
    export_img(et, "avg_et", 500),
    export_img(lulc, "lulc", 500)
]

# ---------------- Step 6: Monitor Progress ----------------
print("⏳ Monitoring export tasks...")
while any(task.status()['state'] in ['READY', 'RUNNING'] for task in tasks):
    for i, task in enumerate(tasks):
        print(f"Task {i+1} - {task.status()['state']}")
    time.sleep(30)

print("✅ All exports complete! Check Google Drive → GEE_State_Exports")

def download_from_drive(filename, folder_name="GEE_State_Exports", download_dir="downloads"):
    service = get_drive_service()
    os.makedirs(download_dir, exist_ok=True)
    query = f"name contains '{filename}' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)', pageSize=5).execute()
    files = results.get('files', [])
    if not files:
        print(f"❌ File '{filename}' not found on Drive")
        return None

    file_id = files[0]['id']
    request = service.files().get_media(fileId=file_id)
    local_path = os.path.join(download_dir, filename + ".tif")
    fh = io.FileIO(local_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            print(f"⬇ Downloading '{filename}': {int(status.progress() * 100)}%")

    print(f"✅ Downloaded: {local_path}")
    return local_path

for fname in base_filenames:
    download_from_drive(f"{fname}{state_name.replace(' ', '')}")

# Automatically download all exported files
for fname in base_filenames:
    delete_drive_file_if_exists(f"{fname}{state_name}")


# ----------- 1. Define Weights ------------
weights = {
    'temp': 0.3,
    'precip': 0.3,
    'et': 0.2,
    'lulc': 0.2
}

# ----------- 2. Normalize ---------------
def normalize(arr):
    arr[arr == 0] = np.nan
    return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))

# ----------- 3. LULC Scoring ------------
lulc_scores = {
    12: 1.0,
    14: 0.8,
    13: 0.2,
    0: 0.3,
    10: 0.6,
    16: 0.2,
    4: 0.7,
    5: 0.5,
}

def score_lulc(arr):
    scored = np.zeros_like(arr, dtype=float)
    for k, v in lulc_scores.items():
        scored[arr == k] = v
    scored[arr == 255] = np.nan
    return scored

# ----------- 4. Reproject to Match Reference --------------
def reproject_to_match(src_path, ref_meta, ref_shape):
    with rasterio.open(src_path) as src:
        data = np.empty(ref_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_meta['transform'],
            dst_crs=ref_meta['crs'],
            resampling=Resampling.bilinear
        )
    return data

# ----------- 5. Load Base Raster (Temperature) ------------
region = state_name
formatted_state = region.replace(" ", "") if " " in region else region
temp_path = f"downloads/avg_temp{formatted_state}.tif"
with rasterio.open(temp_path) as src:
    temp = src.read(1).astype(float)
    ref_meta = src.meta
    ref_shape = temp.shape

# ----------- 6. Load & Reproject Other Rasters -----------
precip = reproject_to_match(f"downloads/avg_precip{formatted_state}.tif", ref_meta, ref_shape)
et = reproject_to_match(f"downloads/avg_et{formatted_state}.tif", ref_meta, ref_shape)
lulc_raw = reproject_to_match(f"downloads/lulc{formatted_state}.tif", ref_meta, ref_shape)

# ----------- 7. Normalize & Score -------------------------
temp_score = normalize(temp)
precip_score = normalize(precip)
et_score = normalize(et)
lulc_score = score_lulc(lulc_raw)

# ----------- 8. Calculate Suitability Score --------------
suitability = (
    weights['temp'] * temp_score +
    weights['precip'] * precip_score +
    weights['et'] * et_score +
    weights['lulc'] * lulc_score
)

# ----------- 9. Visualize -------------------------------
plt.figure(figsize=(10, 6))
plt.imshow(suitability, cmap='YlGn', vmin=0, vmax=1)
plt.colorbar(label='Agricultural Suitability Score')
plt.title(f"Agricultural Suitability - {region.replace('_', ' ')}")
plt.axis('off')
plt.show()

# ----------- 10. Optional Save as TIF -------------------
# from rasterio import Affine
# output_path = f"outputs/agri_suitability_{region}.tif"
# os.makedirs("outputs", exist_ok=True)
# ref_meta.update({"dtype": "float32", "count": 1})
# with rasterio.open(output_path, 'w', **ref_meta) as dst:
#     dst.write(suitability.astype(np.float32), 1)
# print(f"✅ Saved suitability raster to {output_path}")





#Previous Code

# import ee
# import time
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError
# from google.auth.transport.requests import Request
# from google_auth_oauthlib.flow import InstalledAppFlow
# import pickle
# import os
# import io
# from googleapiclient.http import MediaIoBaseDownload
# import numpy as np
# import matplotlib.pyplot as plt
# import rasterio
# from rasterio.warp import reproject, Resampling

# state_name = input("Enter state name: ")
# # Google Drive Auth Setup
# def get_drive_service(creds_path='credentials.json', token_path='token.pickle'):
#     creds = None
#     if os.path.exists(token_path):
#         with open(token_path, 'rb') as token:
#             creds = pickle.load(token)
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(creds_path, [
#                 'https://www.googleapis.com/auth/drive'])
#             creds = flow.run_local_server(port=0)
#         with open(token_path, 'wb') as token:
#             pickle.dump(creds, token)
#     return build('drive', 'v3', credentials=creds)

# # Delete any old .tif with matching name in folder
# def delete_drive_file_if_exists(filename, folder_name="GEE_State_Exports"):
#     service = get_drive_service()
#     folder_results = service.files().list(
#         q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
#         spaces='drive',
#         fields="files(id, name)",
#         pageSize=1
#     ).execute()
#     folders = folder_results.get('files', [])
#     if not folders:
#         print(f"⚠ Folder '{folder_name}' not found.")
#         return
#     folder_id = folders[0]['id']
#     query = f"name contains '{filename}' and '{folder_id}' in parents and trashed=false"
#     results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
#     for file in results.get('files', []):
#         try:
#             service.files().delete(fileId=file['id']).execute()
#             print(f"🗑 Deleted old file: {file['name']}")
#         except HttpError as e:
#             print(f"❌ Error deleting {file['name']}: {e}")

# # Automatically delete all old files before starting exports
# base_filenames = ["avg_temp", "avg_precip", "avg_et", "lulc"]
# for fname in base_filenames:
#     delete_drive_file_if_exists(f"{fname}{state_name.replace(' ', '')}")



# # ---------------- Step 1: Authenticate & Initialize ----------------
# ee.Authenticate()
# ee.Initialize() 

# # ---------------- Step 2: Load State Boundary ----------------
# # GEE GAUL Level 1 = States
 
# states = ee.FeatureCollection("FAO/GAUL/2015/level1")
# state = states.filter(ee.Filter.eq('ADM1_NAME', state_name))

# # ---------------- Step 3: Load Climate & LULC Layers ----------------
# # Temperature (1991–2020 average, °C)
# temp = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY") \
#     .filterDate('1991-01-01', '2020-12-31') \
#     .select('temperature_2m') \
#     .mean().subtract(273.15)  

# # Precipitation (1991–2020 average, mm)
# precip = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY") \
#     .filterDate('1991-01-01', '2020-12-31') \
#     .select('total_precipitation') \
#     .mean().multiply(1000)  

# # Evapotranspiration (2001–2020 average, mm)
# et = ee.ImageCollection("MODIS/006/MOD16A2") \
#     .filterDate('2001-01-01', '2020-12-31') \
#     .select('ET') \
#     .mean().multiply(0.1)  

# # Land Use Land Cover (2020)
# lulc = ee.ImageCollection("MODIS/006/MCD12Q1") \
#     .filterDate('2020-01-01', '2020-12-31') \
#     .first().select('LC_Type1')

# # ---------------- Step 4: Export Function ----------------
# def export_img(image, name, scale):
#     task = ee.batch.Export.image.toDrive(
#         image=image.clip(state),
#         description=f"{name}_export",
#         folder='GEE_State_Exports',
#         fileNamePrefix=f"{name}{state_name.replace(' ', '')}",
#         region=state.geometry(),
#         scale=scale,
#         fileFormat='GeoTIFF'
#     )
#     task.start()
#     print(f"✅ Export started: {name}")
#     return task

# # ---------------- Step 5: Start Exports ----------------
# tasks = [
#     export_img(temp, "avg_temp", 1000),
#     export_img(precip, "avg_precip", 1000),
#     export_img(et, "avg_et", 500),
#     export_img(lulc, "lulc", 500)
# ]

# # ---------------- Step 6: Monitor Progress ----------------
# print("⏳ Monitoring export tasks...")
# while any(task.status()['state'] in ['READY', 'RUNNING'] for task in tasks):
#     for i, task in enumerate(tasks):
#         print(f"Task {i+1} - {task.status()['state']}")
#     time.sleep(30)

# print("✅ All exports complete! Check Google Drive → GEE_State_Exports")

# def download_from_drive(filename, folder_name="GEE_State_Exports", download_dir="downloads"):
#     service = get_drive_service()
#     os.makedirs(download_dir, exist_ok=True)
#     query = f"name contains '{filename}' and trashed=false"
#     results = service.files().list(q=query, spaces='drive', fields='files(id, name)', pageSize=5).execute()
#     files = results.get('files', [])
#     if not files:
#         print(f"❌ File '{filename}' not found on Drive")
#         return None

#     file_id = files[0]['id']
#     request = service.files().get_media(fileId=file_id)
#     local_path = os.path.join(download_dir, filename + ".tif")
#     fh = io.FileIO(local_path, 'wb')
#     downloader = MediaIoBaseDownload(fh, request)

#     done = False
#     while not done:
#         status, done = downloader.next_chunk()
#         if status:
#             print(f"⬇ Downloading '{filename}': {int(status.progress() * 100)}%")

#     print(f"✅ Downloaded: {local_path}")
#     return local_path

# for fname in base_filenames:
#     download_from_drive(f"{fname}{state_name.replace(' ', '')}")

# # Automatically download all exported files
# for fname in base_filenames:
#     delete_drive_file_if_exists(f"{fname}{state_name}")


# # ----------- 1. Define Weights ------------
# weights = {
#     'temp': 0.3,
#     'precip': 0.3,
#     'et': 0.2,
#     'lulc': 0.2
# }

# # ----------- 2. Normalize ---------------
# def normalize(arr):
#     arr[arr == 0] = np.nan
#     return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))

# # ----------- 3. LULC Scoring ------------
# lulc_scores = {
#     12: 1.0,
#     14: 0.8,
#     13: 0.2,
#     0: 0.3,
#     10: 0.6,
#     16: 0.2,
#     4: 0.7,
#     5: 0.5,
# }

# def score_lulc(arr):
#     scored = np.zeros_like(arr, dtype=float)
#     for k, v in lulc_scores.items():
#         scored[arr == k] = v
#     scored[arr == 255] = np.nan
#     return scored

# # ----------- 4. Reproject to Match Reference --------------
# def reproject_to_match(src_path, ref_meta, ref_shape):
#     with rasterio.open(src_path) as src:
#         data = np.empty(ref_shape, dtype=np.float32)
#         reproject(
#             source=rasterio.band(src, 1),
#             destination=data,
#             src_transform=src.transform,
#             src_crs=src.crs,
#             dst_transform=ref_meta['transform'],
#             dst_crs=ref_meta['crs'],
#             resampling=Resampling.bilinear
#         )
#     return data

# # ----------- 5. Load Base Raster (Temperature) ------------
# region = state_name
# formatted_state = region.replace(" ", "_") if " " in region else region
# temp_path = f"downloads/avg_temp_{formatted_state}.tif"
# with rasterio.open(temp_path) as src:
#     temp = src.read(1).astype(float)
#     ref_meta = src.meta
#     ref_shape = temp.shape

# # ----------- 6. Load & Reproject Other Rasters -----------
# precip = reproject_to_match(f"downloads/avg_precip_{formatted_state}.tif", ref_meta, ref_shape)
# et = reproject_to_match(f"downloads/avg_et_{formatted_state}.tif", ref_meta, ref_shape)
# lulc_raw = reproject_to_match(f"downloads/lulc_{formatted_state}.tif", ref_meta, ref_shape)

# # ----------- 7. Normalize & Score -------------------------
# temp_score = normalize(temp)
# precip_score = normalize(precip)
# et_score = normalize(et)
# lulc_score = score_lulc(lulc_raw)

# # ----------- 8. Calculate Suitability Score --------------
# suitability = (
#     weights['temp'] * temp_score +
#     weights['precip'] * precip_score +
#     weights['et'] * et_score +
#     weights['lulc'] * lulc_score
# )

# # ----------- 9. Visualize -------------------------------
# plt.figure(figsize=(10, 6))
# plt.imshow(suitability, cmap='YlGn', vmin=0, vmax=1)
# plt.colorbar(label='Agricultural Suitability Score')
# plt.title(f"Agricultural Suitability - {region.replace('_', ' ')}")
# plt.axis('off')
# plt.show()

# # ----------- 10. Optional Save as TIF -------------------
# # from rasterio import Affine
# # output_path = f"outputs/agri_suitability_{region}.tif"
# # os.makedirs("outputs", exist_ok=True)
# # ref_meta.update({"dtype": "float32", "count": 1})
# # with rasterio.open(output_path, 'w', **ref_meta) as dst:
# #     dst.write(suitability.astype(np.float32), 1)
# # print(f"✅ Saved suitability raster to {output_path}")