import uvicorn
from src.config.settings import settings

def start():
    """ASGI entrypoint run by script or CLI."""
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=settings.APP_PORT,
        reload=settings.APP_ENV == "development"
    )

if __name__ == "__main__":
    start()
