from fastapi import FastAPI
from apps.api.routes import ingest, query, evaluate
from apps.core.config import settings
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title=settings.app_name, version=settings.app_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# register all routes
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(evaluate.router)


@app.get("/")
async def root():
    return {"status": "running", "app": settings.app_name}
