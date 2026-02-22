from fastapi import FastAPI
from backend.routes import upload, analyze, results

app = FastAPI(title="CerebraScan API")

app.include_router(upload.router)
app.include_router(analyze.router)
app.include_router(results.router)