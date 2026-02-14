# Imports
from fastapi import FastAPI
from app.routes import router  # import router z routes.py

# Create a FastAPI application
app = FastAPI(title="Iris Zad 8.1")

# Include all endpoints from routes.py
app.include_router(router)
