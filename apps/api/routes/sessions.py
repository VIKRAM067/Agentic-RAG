from fastapi import APIRouter
from apps.services.memory import memory_manager
from pydantic import BaseModel

router = APIRouter()


class RenameRequest(BaseModel):
    name: str


@router.get("/sessions")
async def list_sessions():
    sessions = memory_manager.list_sessions()
    return {"sessions": sessions}


@router.put("/sessions/{session_id}")
async def rename_session(session_id: str, request: RenameRequest):
    memory_manager.rename_session(request.name, session_id)
    return {"status": "success", "session_id": session_id, "name": request.name}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    memory_manager.delete_session(session_id)
    return {"status": "success", "session_id": session_id}


@router.delete("/sessions")
async def delete_all_sessions():
    memory_manager.delete_all_sessions()
    return {"status": "success", "message": "all sessions deleted"}
