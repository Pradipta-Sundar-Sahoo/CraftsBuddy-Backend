"""CraftBuddy Backend API"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.settings import config
from src.core.database_service import DatabaseService
from src.apps.routes import auth_router, catalog_router, seller_router

# Create FastAPI app
app = FastAPI(
    title="CraftBuddy API", 
    version="1.0.0",
    description="Backend API for CraftBuddy - Artisan Product Marketplace"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
db_service = DatabaseService()

# Include routers
app.include_router(auth_router, prefix="/api")
app.include_router(catalog_router, prefix="/api")
app.include_router(seller_router, prefix="/api")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "CraftBuddy Backend API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "gcs_enabled": config.USE_GCS,
        "gemini_enabled": config.USE_GEMINI,
        "database_connected": True  # TODO: Add actual DB health check
    }

@app.get("/api/routes")
async def list_routes():
    """List all available API routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": route.name
            })
    return {"routes": routes}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
