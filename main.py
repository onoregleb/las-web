import asyncio
import uuid
import logging
from typing import List, Dict, Any
from pathlib import Path
from urllib.parse import quote
import shutil # Для сохранения UploadFile

from fastapi import FastAPI, UploadFile, Request, HTTPException, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel # Для моделей запросов

import laspy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # Для проверки и сохранения изображений

# --- Конфигурация ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR_LAS = BASE_DIR / "uploads" / "las"
PROCESS_DIR_NPY = BASE_DIR / "processed" / "npy"
PROCESS_DIR_IMG = BASE_DIR / "processed" / "images"
TEMPLATES_DIR = BASE_DIR / "templates"

# Создаем директории, если они не существуют
UPLOAD_DIR_LAS.mkdir(parents=True, exist_ok=True)
PROCESS_DIR_NPY.mkdir(parents=True, exist_ok=True)
PROCESS_DIR_IMG.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="LAS/LAZ Multi-Stage Processor API",
    # Увеличение максимального размера запроса, если нужно обрабатывать очень большие файлы
    max_request_size=512 * 1024 * 1024
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # В продакшене лучше указать конкретный origin фронтенда
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Шаблоны ---
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --- Pydantic Модели для Запросов ---
class LasFileId(BaseModel):
    lasFileId: str
    originalName: str

class LasIdList(BaseModel):
    lasFileIds: List[str]

class NpyIdList(BaseModel):
    npyFileIds: List[str]

class LasFilesInfo(BaseModel):
    lasFilesInfo: List[LasFileId]

# --- Вспомогательные функции ---
def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    """Синхронно сохраняет UploadFile."""
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

def generate_safe_filename(original_name: str, new_extension: str, prefix: str = "") -> str:
    """Генерирует безопасное имя файла с UUID."""
    base_name = Path(original_name).stem # Имя без расширения
    safe_base = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in base_name)
    unique_id = uuid.uuid4()
    return f"{prefix}{safe_base}_{unique_id}{new_extension}"

async def process_las_to_npy_task(las_file_path: Path, npy_file_path: Path):
    """Асинхронная обертка для CPU-bound задачи генерации NPY."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, generate_npy_from_las_sync, las_file_path, npy_file_path)

def generate_npy_from_las_sync(las_file_path: Path, npy_file_path: Path):
    """Синхронная функция генерации NPY из LAS."""
    try:
        logger.info(f"Processing LAS for NPY: {las_file_path}")
        las = laspy.read(las_file_path)
        logger.info(f"LAS file read. Points: {len(las.points)}")

        # Собираем данные: X, Y, Z обязательно. Пытаемся добавить Intensity и RGB, если есть.
        points_data = [las.x, las.y, las.z]
        point_dim_names = ['x', 'y', 'z']

        if 'intensity' in las.point_format.dimension_names:
            points_data.append(las.intensity)
            point_dim_names.append('intensity')
            logger.debug("Intensity data included.")

        if all(c in las.point_format.dimension_names for c in ['red', 'green', 'blue']):
            # Нормализуем RGB к диапазону 0-1 (предполагаем 16 бит)
            max_val = 65535.0
            points_data.extend([las.red / max_val, las.green / max_val, las.blue / max_val])
            point_dim_names.extend(['red_norm', 'green_norm', 'blue_norm'])
            logger.debug("Normalized RGB data included.")

        # Транспонируем для получения массива (N_points, N_dims)
        points_array = np.vstack(points_data).transpose()
        logger.info(f"NPY array shape: {points_array.shape}")

        # Сохраняем в NPY
        np.save(npy_file_path, points_array)
        logger.info(f"Successfully saved NPY file: {npy_file_path}")

    except FileNotFoundError:
        logger.error(f"LAS file not found for NPY generation: {las_file_path}")
        raise # Передаем ошибку выше
    except Exception as e:
        logger.error(f"Error processing LAS to NPY ({las_file_path}): {e}", exc_info=True)
        # Удаляем частично созданный файл, если он есть
        if npy_file_path.exists():
            try:
                npy_file_path.unlink()
            except OSError:
                pass
        raise # Передаем ошибку выше


def generate_image_from_npy_sync(npy_file_path: Path, img_file_path: Path):
    """Синхронная функция генерации изображения из NPY."""
    try:
        logger.info(f"Generating image from NPY: {npy_file_path}")
        data = np.load(npy_file_path)
        logger.info(f"NPY data loaded. Shape: {data.shape}")

        if data.shape[1] < 3:
             raise ValueError("NPY data must have at least X, Y, Z columns.")

        fig, ax = plt.subplots(figsize=(8, 8))

        # Используем X и Y для координат, Z для цвета
        scatter = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap='viridis', s=1, marker='.')
        ax.set_title(f"Visualization from {npy_file_path.name}")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_aspect('equal', adjustable='box') # Равный масштаб осей
        plt.colorbar(scatter, label='Z coordinate')
        plt.tight_layout()
        plt.savefig(img_file_path, dpi=150)
        plt.close(fig) # Закрываем фигуру для освобождения памяти
        logger.info(f"Successfully saved image: {img_file_path}")

    except FileNotFoundError:
        logger.error(f"NPY file not found for image generation: {npy_file_path}")
        raise
    except Exception as e:
        logger.error(f"Error generating image from NPY ({npy_file_path}): {e}", exc_info=True)
        if img_file_path.exists():
            try:
                img_file_path.unlink()
            except OSError:
                pass
        raise


def generate_rgb_image_from_las_sync(las_file_path: Path, img_file_path: Path):
    """Синхронная функция генерации RGB-изображения из LAS."""
    try:
        logger.info(f"Generating RGB image from LAS: {las_file_path}")
        las = laspy.read(las_file_path)
        logger.info(f"LAS file read for RGB image. Points: {len(las.points)}")

        if not all(c in las.point_format.dimension_names for c in ['red', 'green', 'blue']):
            logger.warning(f"LAS file {las_file_path.name} does not contain required RGB dimensions.")
            return False # Сигнализируем, что RGB нет

        # Нормализуем 16-битные цвета к диапазону 0-1
        max_val = 65535.0
        red = las.red / max_val
        green = las.green / max_val
        blue = las.blue / max_val
        colors = np.vstack((red, green, blue)).transpose()
        # Убедимся, что значения в пределах [0, 1]
        colors = np.clip(colors, 0, 1)
        logger.debug(f"RGB colors prepared. Shape: {colors.shape}")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(las.x, las.y, c=colors, s=1, marker='.') # Используем X, Y и массив цветов
        ax.set_title(f"RGB Visualization from {las_file_path.name}")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(img_file_path, dpi=150)
        plt.close(fig)
        logger.info(f"Successfully saved RGB image: {img_file_path}")
        return True # Успешно сгенерировано

    except FileNotFoundError:
        logger.error(f"LAS file not found for RGB image generation: {las_file_path}")
        raise
    except Exception as e:
        logger.error(f"Error generating RGB image from LAS ({las_file_path}): {e}", exc_info=True)
        if img_file_path.exists():
            try:
                img_file_path.unlink()
            except OSError:
                pass
        raise


# --- API Эндпоинты ---

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root(request: Request):
    """Отдает главную HTML страницу (фронтенд)."""
    if not (TEMPLATES_DIR / "index.html").exists():
         logger.error(f"Frontend file not found: {TEMPLATES_DIR / 'index.html'}")
         raise HTTPException(status_code=500, detail="Frontend file (index.html) not found on server.")
    return templates.TemplateResponse("index.html", {"request": request})

# === Этап 1: Загрузка LAS ===
@app.post("/api/v1/las/upload", status_code=201)
async def upload_las_files(files: List[UploadFile] = File(...)):
    """
    Принимает LAS/LAZ файлы, сохраняет их на диск и возвращает их ID и имена.
    """
    uploaded_files_info = []
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    for file in files:
        if not (file.filename.lower().endswith('.las') or file.filename.lower().endswith('.laz')):
            logger.warning(f"Skipping invalid file type: {file.filename}")
            continue

        las_file_id = str(uuid.uuid4())
        # Генерируем имя файла на диске, чтобы избежать коллизий
        # Сохраняем оригинальное расширение
        original_extension = Path(file.filename).suffix
        disk_filename = f"{las_file_id}{original_extension}"
        destination = UPLOAD_DIR_LAS / disk_filename

        logger.info(f"Receiving file: {file.filename} -> Saving as: {disk_filename}")

        try:
            # Используем синхронную функцию сохранения, т.к. работа с UploadFile потоком
            # может быть сложной в чистом async/await без доп. библиотек типа `spooler`
            save_upload_file(file, destination)
            logger.info(f"Successfully saved file: {destination}")
            uploaded_files_info.append({
                "lasFileId": las_file_id, # ID = имя файла без расширения
                "originalName": file.filename
            })
        except Exception as e:
            logger.error(f"Failed to save file {file.filename}: {e}", exc_info=True)
            # Можно добавить логику отката или просто пропустить файл
            # В этом случае не добавляем в uploaded_files_info
            # Можно вернуть ошибку 500, если хотя бы один файл не сохранился
            # raise HTTPException(status_code=500, detail=f"Failed to save file {file.filename}. Error: {e}")

    if not uploaded_files_info:
         raise HTTPException(
            status_code=400,
            detail="No valid LAS/LAZ files were uploaded or saved successfully."
        )

    return JSONResponse(content={
        "success": True,
        "uploaded_files": uploaded_files_info
    })


# === Этап 2: Генерация NPY ===
@app.post("/api/v1/npy/generate")
async def generate_npy_endpoint(payload: LasIdList, background_tasks: BackgroundTasks):
    """
    Принимает список ID LAS файлов, генерирует для них NPY файлы в фоне.
    Сразу возвращает информацию о том, где будут файлы.
    """
    npy_files_results = []
    las_file_ids = payload.lasFileIds

    if not las_file_ids:
         raise HTTPException(status_code=400, detail="No LAS file IDs provided.")

    # Найти соответствующие LAS файлы и запланировать генерацию
    las_files_to_process: List[Dict[str, Any]] = []
    for las_id in las_file_ids:
        # Ищем файл с этим ID и расширением .las или .laz
        found_las = list(UPLOAD_DIR_LAS.glob(f"{las_id}.las")) + list(UPLOAD_DIR_LAS.glob(f"{las_id}.laz"))
        if not found_las:
            logger.warning(f"LAS file with ID {las_id} not found in {UPLOAD_DIR_LAS}. Skipping.")
            continue # Пропускаем этот ID, можно вернуть ошибку позже
        las_file_path = found_las[0]
        # Получаем оригинальное имя из найденного пути (нужно будет передать или хранить маппинг)
        # Для простоты, будем использовать ID в имени npy
        original_name_base = las_file_path.stem # Имя без расширения (это наш ID)

        npy_filename = f"{original_name_base}.npy" # Имя NPY файла совпадает с ID LAS
        npy_file_path = PROCESS_DIR_NPY / npy_filename

        las_files_to_process.append({
             "las_id": las_id,
             "las_path": las_file_path,
             "npy_path": npy_file_path,
             "npy_filename": npy_filename,
             "original_las_name": f"{original_name_base}{las_file_path.suffix}" # Восстанавливаем имя с расширением
        })

    if not las_files_to_process:
        raise HTTPException(status_code=404, detail="None of the provided LAS file IDs were found.")

    # Запускаем генерацию в фоне и формируем ответ
    processing_errors = {}
    for item in las_files_to_process:
        try:
            # Запуск синхронной функции в фоновом потоке
             # !!! Внимание: BackgroundTasks не идеальны для долгих CPU-задач.
             # Лучше использовать Celery или similar для production.
             # background_tasks.add_task(generate_npy_from_las_sync, item["las_path"], item["npy_path"])

             # Альтернатива: Выполняем синхронно здесь (блокирует worker на время выполнения!)
             # Это проще для примера, но хуже для производительности.
             generate_npy_from_las_sync(item["las_path"], item["npy_path"])

             # Формируем результат для успешного файла
             npy_files_results.append({
                 "npyFileId": item["las_id"], # Используем ID LAS файла как ID NPY файла
                 "filename": item["npy_filename"],
                 # URL для скачивания NPY
                 "downloadUrl": f"/download/npy/{item['las_id']}",
                 "sourceLasId": item["las_id"],
                 "originalLasName": item["original_las_name"]
             })
        except Exception as e:
            logger.error(f"Failed to initiate NPY generation for LAS ID {item['las_id']}: {e}", exc_info=True)
            processing_errors[item['las_id']] = str(e)
            # Не добавляем в npy_files_results

    # Если были ошибки при обработке некоторых файлов, сообщаем об этом
    # Фронтенд должен будет обработать частичный успех
    if not npy_files_results and processing_errors:
         raise HTTPException(status_code=500, detail=f"NPY generation failed for all files. Errors: {processing_errors}")

    return JSONResponse(content={
        "success": len(npy_files_results) > 0,
        "npy_files": npy_files_results,
        "errors": processing_errors if processing_errors else None # Сообщаем об ошибках, если были
    })

# === Этап 3: Генерация Изображений из NPY ===
@app.post("/api/v1/images/from-npy")
async def generate_images_from_npy_endpoint(payload: NpyIdList, background_tasks: BackgroundTasks):
    """
    Принимает список ID NPY файлов, генерирует для них изображения.
    """
    image_results = []
    npy_file_ids = payload.npyFileIds
    processing_errors = {}

    if not npy_file_ids:
        raise HTTPException(status_code=400, detail="No NPY file IDs provided.")

    for npy_id in npy_file_ids:
         # NPY файл имеет то же имя (без расширения), что и ID
        npy_filename = f"{npy_id}.npy"
        npy_file_path = PROCESS_DIR_NPY / npy_filename

        if not npy_file_path.exists():
            logger.warning(f"NPY file with ID {npy_id} not found: {npy_file_path}. Skipping.")
            processing_errors[npy_id] = "NPY file not found"
            continue

        img_file_id = str(uuid.uuid4()) # Генерируем новый ID для изображения
        img_filename = f"{npy_id}_npy_{img_file_id}.png" # Имя изображения включает ID NPY
        img_file_path = PROCESS_DIR_IMG / img_filename

        try:
             # Запуск синхронной функции в фоновом потоке (или синхронно)
             # background_tasks.add_task(generate_image_from_npy_sync, npy_file_path, img_file_path)
             generate_image_from_npy_sync(npy_file_path, img_file_path)

             image_results.append({
                "imageId": img_file_id,
                "filename": img_filename,
                # URL для предпросмотра (может быть тем же, что и скачивание)
                "previewUrl": f"/preview/img/{img_file_id}",
                 # URL для скачивания
                "downloadUrl": f"/download/img/{img_file_id}",
                "sourceNpyId": npy_id # ID исходного NPY файла
             })
        except Exception as e:
            logger.error(f"Failed to initiate image generation from NPY ID {npy_id}: {e}", exc_info=True)
            processing_errors[npy_id] = str(e)

    if not image_results and processing_errors:
         raise HTTPException(status_code=500, detail=f"Image generation from NPY failed for all files. Errors: {processing_errors}")

    return JSONResponse(content={
        "success": len(image_results) > 0,
        "images": image_results,
        "errors": processing_errors if processing_errors else None
    })


# === Этап 4: Генерация RGB Изображений из LAS ===
@app.post("/api/v1/images/from-las-rgb")
async def generate_rgb_images_from_las_endpoint(payload: LasFilesInfo, background_tasks: BackgroundTasks):
    """
    Принимает список ID LAS файлов, генерирует для них RGB изображения (если возможно).
    """
    image_results = []
    las_files_info = payload.lasFilesInfo
    processing_errors = {}
    skipped_no_rgb = []

    if not las_files_info:
         raise HTTPException(status_code=400, detail="No LAS files info provided.")

    for las_info in las_files_info:
        las_id = las_info.lasFileId
        # Ищем файл с этим ID и расширением .las или .laz
        found_las = list(UPLOAD_DIR_LAS.glob(f"{las_id}.las")) + list(UPLOAD_DIR_LAS.glob(f"{las_id}.laz"))

        if not found_las:
            logger.warning(f"LAS file with ID {las_id} not found for RGB generation. Skipping.")
            processing_errors[las_id] = "LAS file not found"
            continue

        las_file_path = found_las[0]
        img_file_id = str(uuid.uuid4()) # Новый ID для изображения
        img_filename = f"{las_id}_lasrgb_{img_file_id}.png" # Имя включает ID LAS
        img_file_path = PROCESS_DIR_IMG / img_filename

        try:
             # Запуск синхронной функции в фоновом потоке (или синхронно)
             # rgb_generated = await loop.run_in_executor(None, generate_rgb_image_from_las_sync, las_file_path, img_file_path) # Если хотим асинхронно
             rgb_generated = generate_rgb_image_from_las_sync(las_file_path, img_file_path)

             if rgb_generated: # Функция возвращает True, если RGB было и картинка создана
                 image_results.append({
                    "imageId": img_file_id,
                    "filename": img_filename,
                    "previewUrl": f"/preview/img/{img_file_id}",
                    "downloadUrl": f"/download/img/{img_file_id}",
                    "sourceLasId": las_id # ID исходного LAS файла
                 })
             else:
                 # Файл обработан, но RGB не найдено
                 logger.info(f"Skipped RGB image generation for LAS ID {las_id} (no RGB data).")
                 skipped_no_rgb.append(las_id)

        except Exception as e:
            logger.error(f"Failed to generate RGB image from LAS ID {las_id}: {e}", exc_info=True)
            processing_errors[las_id] = str(e)

    if not image_results and processing_errors and not skipped_no_rgb:
        # Ошибка, если ничего не сгенерировано и не пропущено из-за отсутствия RGB
         raise HTTPException(status_code=500, detail=f"RGB Image generation from LAS failed for all files. Errors: {processing_errors}")

    return JSONResponse(content={
        "success": len(image_results) > 0,
        "images": image_results,
        "skipped_no_rgb": skipped_no_rgb if skipped_no_rgb else None,
        "errors": processing_errors if processing_errors else None
    })


# === Эндпоинты для Скачивания и Предпросмотра ===

@app.get("/download/npy/{file_id}")
async def download_npy_file(file_id: str):
    """Отдает NPY файл для скачивания."""
    # NPY файл имеет имя {file_id}.npy
    file_path = PROCESS_DIR_NPY / f"{file_id}.npy"
    if not file_path.is_file():
        logger.warning(f"Download requested for non-existent NPY: {file_id}")
        raise HTTPException(status_code=404, detail="NPY file not found")

    safe_filename = quote(file_path.name)
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type='application/octet-stream', # Общий тип для бинарных данных
         headers={"Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}"}
    )

@app.get("/download/img/{file_id}")
async def download_image_file(file_id: str):
    """Отдает сгенерированный файл изображения для скачивания."""
    # Изображение может иметь разное имя, ищем по ID в имени
    found_files = list(PROCESS_DIR_IMG.glob(f"*_{file_id}.png")) # Ищем файлы, где ID в конце имени
    if not found_files:
         # Попробуем найти файлы, где ID в начале (для RGB картинок)
         found_files = list(PROCESS_DIR_IMG.glob(f"{file_id}_*.png"))
         if not found_files:
             logger.warning(f"Download requested for non-existent image ID: {file_id}")
             raise HTTPException(status_code=404, detail="Image file not found")

    file_path = found_files[0]
    safe_filename = quote(file_path.name)
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type='image/png',
         headers={"Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}"}
    )

@app.get("/preview/img/{file_id}")
async def preview_image_file(file_id: str):
    """Отдает сгенерированный файл изображения для отображения в <img>."""
     # Ищем файл так же, как при скачивании
    found_files = list(PROCESS_DIR_IMG.glob(f"*_{file_id}.png"))
    if not found_files:
         found_files = list(PROCESS_DIR_IMG.glob(f"{file_id}_*.png"))
         if not found_files:
            logger.warning(f"Preview requested for non-existent image ID: {file_id}")
            # Можно вернуть 404 или placeholder изображение
            raise HTTPException(status_code=404, detail="Image file not found")
            # return FileResponse(path="path/to/placeholder.png", media_type='image/png')

    file_path = found_files[0]
    # Для предпросмотра просто отдаем файл с правильным media_type
    return FileResponse(path=file_path, media_type='image/png')


@app.post("/api/v1/reset")
async def reset_files():
    """
    Удаляет все файлы в директориях uploads и processed.
    """
    try:
        # Удаляем файлы из uploads/las
        for file_path in UPLOAD_DIR_LAS.glob("*"):
            try:
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Ошибка при удалении файла {file_path}: {e}")

        # Удаляем файлы из processed/npy
        for file_path in PROCESS_DIR_NPY.glob("*"):
            try:
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Ошибка при удалении файла {file_path}: {e}")

        # Удаляем файлы из processed/images
        for file_path in PROCESS_DIR_IMG.glob("*"):
            try:
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Ошибка при удалении файла {file_path}: {e}")

        # Если нужно, можно создать директории заново, чтобы не нарушить последующую работу:
        UPLOAD_DIR_LAS.mkdir(parents=True, exist_ok=True)
        PROCESS_DIR_NPY.mkdir(parents=True, exist_ok=True)
        PROCESS_DIR_IMG.mkdir(parents=True, exist_ok=True)

        logger.info("Все файлы успешно удалены.")
        return JSONResponse(content={"success": True, "message": "Файлы успешно удалены."})
    except Exception as e:
        logger.error(f"Ошибка сброса файлов: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Не удалось выполнить сброс файлов.")


# --- Health Check ---
@app.get("/health", include_in_schema=False)
async def health_check():
    return {"status": "healthy", "message": "Server is running!"}
