import os
from fastapi import FastAPI, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import uuid
from typing import List
import matplotlib.pyplot as plt
import laspy
import numpy as np
from io import BytesIO
from fastapi.responses import StreamingResponse
from urllib.parse import quote

app = FastAPI()

# Templates for rendering HTML
templates = Jinja2Templates(directory="templates")

# In-memory storage for files
in_memory_files = {}


def process_las_file(file_content: bytes) -> BytesIO:
    """Process LAS file content in memory and create PNG image"""
    try:
        # Read LAS file from BytesIO
        las_file = laspy.read(BytesIO(file_content))

        # Get available properties
        available_props = set(las_file.point_format.dimension_names)
        depth = las_file.z

        # Create figure
        fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(10, 12))

        # Plot 1: Intensity or similar
        if 'intensity' in available_props:
            axes[0].plot(las_file.intensity, depth)
            axes[0].set_title('Intensity')
        elif 'red' in available_props and 'green' in available_props and 'blue' in available_props:
            rgb_mean = (las_file.red + las_file.green + las_file.blue) / 3
            axes[0].plot(rgb_mean, depth)
            axes[0].set_title('RGB Mean')

        # Plot 2: Classification
        if 'classification' in available_props:
            axes[1].scatter(np.random.rand(len(depth)), depth, c=las_file.classification, s=1)
            axes[1].set_title('Classification')

        # Plot 3: Additional property
        if 'return_number' in available_props:
            axes[2].plot(las_file.return_number, depth)
            axes[2].set_title('Return Number')
        elif 'gps_time' in available_props:
            axes[2].plot(las_file.gps_time, depth)
            axes[2].set_title('GPS Time')

        plt.tight_layout()
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100)
        plt.close()
        img_buffer.seek(0)
        return img_buffer

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing LAS file: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Template error: {str(e)}"}
        )


@app.post("/upload/")
async def upload_files(files: List[UploadFile]):
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    results = []

    for file in files:
        try:
            if not file.filename.lower().endswith('.las') and not file.filename.lower().endswith('.laz'):
                raise HTTPException(status_code=400, detail="Only LAS/LAZ files are allowed")

            file_id = str(uuid.uuid4())
            file_content = await file.read()

            # Process and create visualization
            img_buffer = process_las_file(file_content)

            # Store the processed image in memory
            in_memory_files[file_id] = img_buffer

            original_name = os.path.splitext(file.filename)[0]
            results.append({
                "filename": f"{original_name}_processed.png",
                "download_url": f"/download/?file_id={file_id}"
            })

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({
        "results": results,
        "message": f"Processed {len(results)} file(s)"
    })


@app.get("/download/")
async def download_file(file_id: str):
    if file_id not in in_memory_files:
        raise HTTPException(status_code=404, detail="File not found")

    img_buffer = in_memory_files[file_id]
    del in_memory_files[file_id]

    img_buffer.seek(0)

    # Кодируем имя файла для безопасного использования в URL
    safe_filename = quote(f"processed_{file_id}.png")

    return StreamingResponse(
        img_buffer,
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}"}
    )


@app.get("/health")
async def test():
    return {"message": "Server is running!"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Используйте PORT из Railway или 8000 по умолчанию
    uvicorn.run(app, host="0.0.0.0", port=port)