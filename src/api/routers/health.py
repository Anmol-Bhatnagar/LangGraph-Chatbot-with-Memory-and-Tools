from fastapi import APIRouter

router = APIRouter(tags=["Health"])

@router.get("/healthz", summary="Liveness probe")
def healthz():
    """Returns 200 OK if application is running."""
    return {"status": "ok"}

@router.get("/ready", summary="Readiness probe")
def ready():
    """Returns 200 OK if service dependencies are initialized."""
    # Placeholder: could check SQLite connection status if needed
    return {"status": "ready"}
