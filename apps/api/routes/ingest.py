from fastapi import APIRouter, UploadFile, File
from apps.api.schemas import IngestResponse
from apps.services.ingestion import ingestion_service
import shutil
import os

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):

    # 1. save uploaded file to disk
    upload_dir = "./data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = f"{upload_dir}/{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

        # 2. run ingestion pipeline
        result = ingestion_service.ingest(file_path)

        # 3. return stats
        return IngestResponse(**result)
