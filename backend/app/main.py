"""EduPredict AI backend.

Run with:  uvicorn app.main:app --reload

Datasets stay on this server. Nothing is sent to a third-party service: files
are parsed, analysed, trained and scored in this process, and the only things
written to disk are the uploaded file, the fitted model and the prediction
output, all inside the backend directory.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import analysis, dashboard, dataset, explain, intelligence, jobs, model, performance, prediction, students
from app.core.errors import register_exception_handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="EduPredict AI",
    description="Student performance analysis, model training and prediction.",
    version="2.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# The Vite dev server runs on a different port, so the browser needs explicit
# permission to call this API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(dataset.router)
app.include_router(analysis.router)
app.include_router(model.router)
app.include_router(prediction.router)
app.include_router(jobs.router)
app.include_router(dashboard.router)
app.include_router(performance.router)
app.include_router(students.router)
app.include_router(explain.router)
app.include_router(intelligence.router)


@app.get("/", include_in_schema=False)
def root():
    return {
        "success": True,
        "name": "EduPredict AI API",
        "version": app.version,
        "docs": "/api/docs",
    }
