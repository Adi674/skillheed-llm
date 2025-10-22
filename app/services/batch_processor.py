import asyncio
import time
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class BatchItem:
    data: Any
    future: asyncio.Future
    user_id: str
    timestamp: float

class AsyncBatchProcessor:
    """High-performance batch processor"""
    
    def __init__(
        self,
        batch_size: int = 32,
        max_wait_time: float = 0.05,  # 50ms
        max_concurrent_batches: int = 10,
        processor_func: Callable = None
    ):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_concurrent_batches = max_concurrent_batches
        self.processor_func = processor_func
        
        self.queue: List[BatchItem] = []
        self.processing = False
        self.active_batches = 0
        
        # Start background processor
        asyncio.create_task(self._processor_loop())
    
    async def submit(self, data: Any, user_id: str = "unknown") -> Any:
        """Submit item for batch processing"""
        future = asyncio.Future()
        item = BatchItem(
            data=data,
            future=future,
            user_id=user_id,
            timestamp=time.time()
        )
        
        self.queue.append(item)
        return await future
    
    async def _processor_loop(self):
        """Main processing loop"""
        while True:
            try:
                await asyncio.sleep(0.01)
                
                should_process = (
                    len(self.queue) >= self.batch_size or
                    (self.queue and time.time() - self.queue[0].timestamp > self.max_wait_time)
                )
                
                if should_process and self.active_batches < self.max_concurrent_batches:
                    await self._process_batch()
                    
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(self):
        """Process next batch"""
        if not self.queue:
            return
        
        # Extract batch
        batch = self.queue[:self.batch_size]
        self.queue = self.queue[self.batch_size:]
        self.active_batches += 1
        
        # Process in background
        asyncio.create_task(self._execute_batch(batch))
    
    async def _execute_batch(self, batch: List[BatchItem]):
        """Execute batch processing"""
        try:
            start_time = time.time()
            
            # Extract data
            batch_data = [item.data for item in batch]
            
            # Process batch
            if self.processor_func:
                results = await self.processor_func(batch_data)
            else:
                results = batch_data
            
            # Resolve futures
            for item, result in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(result)
            
            logger.info(f"Processed batch of {len(batch)} in {time.time() - start_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)
        finally:
            self.active_batches -= 1