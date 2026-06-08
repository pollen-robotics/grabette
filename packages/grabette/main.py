import uvicorn

from grabette.app.main import create_app
from grabette.config import settings

app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
