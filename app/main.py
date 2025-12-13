from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.init_db import init_db

from .fastapi_routers import patients, visits, lab_results, imaging, spirometry, diagnostic, rag, auth

app = FastAPI()

# CORS Configuration
# Allow requests from frontend (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React default
        "http://localhost:3001",  # Alternative React port
        "http://localhost:5173",  # Vite default
        "http://localhost:5174",  # Alternative Vite port
        "http://localhost:8080",  # Vue default
        "http://localhost:4200",  # Angular default
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        # Add production frontend URL here when deployed
        # "https://your-frontend-domain.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.on_event("startup")
def startup():
    init_db()

app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(visits.router)
app.include_router(lab_results.router)
app.include_router(imaging.router)
app.include_router(spirometry.router)
app.include_router(diagnostic.router)
app.include_router(rag.router)
