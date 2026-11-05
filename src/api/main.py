from fastapi import FastAPI
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.api.routers import upload_file_router

app = FastAPI(
    title="RAG  System FastAPI",
    description="This is a FastAPI application for the RAG System.",
    version="1.0.0",
)
app.include_router(upload_file_router.router)
@app.get("/")
async def index():
    return {"message": "Welcome to the RAG System FastAPI!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000,reload=True)