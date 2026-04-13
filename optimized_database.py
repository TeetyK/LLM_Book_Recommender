import asyncio
from cachetools import TTLCache
import redis
import json
from langchain_huggingface import HuggingFaceEmbeddings

class OptimizedVectorSearch:
    def __init__(self):
        self.redis_cache = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour cache
    
    async def get_similar_books_cached(self, query_text, k=5):
        """Cache popular queries"""
        cache_key = f"books:{query_text}:{k}"
        
        # Check Redis first (fast)
        cached = self.redis_cache.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Vector search
        results = await self._vector_search(query_text, k)
        
        # Save to cache
        self.redis_cache.setex(cache_key, 3600, json.dumps(results))
        return results
    
    async def _vector_search(self, query_text, k):
        """Parallel embedding + search"""
        embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
        vector = embeddings.embed_query(query_text)
        
        # Async DB query
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, 
            self._db_vector_search, 
            vector, 
            k
        )
        return results