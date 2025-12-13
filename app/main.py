from fastapi import FastAPI
from .core.init_db import init_db

from .fastapi_routers import patients, visits, lab_results, imaging, spirometry, diagnostic, rag

app = FastAPI()

@app.on_event("startup")
def startup():
    init_db()

app.include_router(patients.router)
app.include_router(visits.router)
app.include_router(lab_results.router)
app.include_router(imaging.router)
app.include_router(spirometry.router)
app.include_router(diagnostic.router)
app.include_router(rag.router)
