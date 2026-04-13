import time
from datetime import datetime

class PerformanceTracker:
    def __init__(self):
        self.metrics = []
    def track_query(self,query,retrieved_books, user_rating):
        """user rating: 1-5 star"""
        latency = time.time()
        self.metrics.append({
            "timestamp":datetime.now(),
            "query":query,
            "retrieved_count": len(retrieved_books),
            "user_rating":user_rating,
            "latency_ms":latency
        })
    def accuracy(self):
        if not self.metrics:
            return 0
        good_recs = sum( 1 for m in self.metrics if m['user_rating'] >=4 )
        return (good_recs / len(self.metrics)) * 100
    def avg_latency(self):
        if not self.metrics:
            return 0
        return sum(m['latency_ms'] for m in self.metrics) / len(self.metrics)
    