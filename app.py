from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import time
from contextlib import asynccontextmanager
from metrics import PerformanceTracker
from optimized_database import OptimizedVectorSearch

# Initialize before app starts
optimizer = OptimizedVectorSearch()
tracker = PerformanceTracker()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models to memory
    print("Loading embeddings model...")
    await optimizer.warmup()
    yield
    # Shutdown
    print("Closing connections...")

app = FastAPI(title="Book Recommender API", lifespan=lifespan)

class BookRequest(BaseModel):
    query: str
    user_id: int

class BookResponse(BaseModel):
    recommendations: list
    latency_ms: float
    accuracy: float

@app.post("/recommend", response_model=BookResponse)
async def recommend_books(req: BookRequest, background_tasks: BackgroundTasks):
    """Endpoint: /recommend"""
    start_time = time.time()
    
    # Get recommendations (cached or fresh)
    books = await optimizer.get_similar_books_cached(req.query, k=5)
    
    # Parallel: Call LLM while returning response
    latency = (time.time() - start_time) * 1000
    
    # Track asynchronously
    background_tasks.add_task(tracker.track_query, req.query, books, None)
    
    return BookResponse(
        recommendations=books,
        latency_ms=latency,
        accuracy=tracker.accuracy()
    )

@app.get("/metrics")
def get_metrics():
    """Endpoint: /metrics (for monitoring)"""
    return {
        "avg_latency_ms": tracker.avg_latency(),
        "accuracy_percent": tracker.accuracy(),
        "total_queries": len(tracker.metrics)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)