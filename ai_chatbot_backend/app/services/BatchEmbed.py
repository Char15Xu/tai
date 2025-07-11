import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import uuid
from contextlib import asynccontextmanager
from FlagEmbedding import BGEM3FlagModel
import logging

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingRequest:
    """Single embedding request with metadata"""
    request_id: str
    text: str
    timestamp: float
    future: asyncio.Future
    return_dense: bool = True
    return_sparse: bool = True
    return_colbert_vecs: bool = True

@dataclass
class EmbeddingResponse:
    """Single embedding response"""
    request_id: str
    dense_vecs: Optional[Any] = None
    sparse_vecs: Optional[Any] = None
    colbert_vecs: Optional[Any] = None
    error: Optional[str] = None

class BatchEmbed:
    """
    High-throughput embedding service using batching similar to vLLM
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        max_batch_size: int = 32,
        max_wait_time: float = 0.05,  # 50ms max wait
        worker_threads: int = 1,
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.worker_threads = worker_threads
        self.device = device

        # Request queues and processing
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.workers = []
        self.model = None

        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0,
            "throughput_per_second": 0
        }

    async def start(self):
        """Initialize the service and start worker threads"""
        if self.running:
            return

        logger.info(f"Starting BatchEmbed with {self.worker_threads} workers")

        # Initialize model in main thread
        self.model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16, device=self.device)

        self.running = True

        # Start worker threads
        for i in range(self.worker_threads):
            worker = asyncio.create_task(self._worker_loop(worker_id=i))
            self.workers.append(worker)

        logger.info("BatchEmbed started successfully")

    async def stop(self):
        """Stop the service and cleanup"""
        if not self.running:
            return

        logger.info("Stopping BatchEmbed")
        self.running = False

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)

        self.workers.clear()
        logger.info("BatchEmbed stopped")

    async def encode_async(
        self,
        text: str,
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert_vecs: bool = True,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Async encode function that batches requests for higher throughput
        """
        if not self.running:
            raise RuntimeError("Service not started. Call start() first.")

        request_id = str(uuid.uuid4())
        future = asyncio.Future()

        request = EmbeddingRequest(
            request_id=request_id,
            text=text,
            timestamp=time.time(),
            future=future,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert_vecs
        )

        # Add to queue
        await self.request_queue.put(request)
        self.stats["total_requests"] += 1

        # Wait for result with timeout
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            if result.error:
                raise RuntimeError(f"Embedding failed: {result.error}")

            # Format response similar to original BGEM3FlagModel
            response = {}
            if return_dense and result.dense_vecs is not None:
                response['dense_vecs'] = result.dense_vecs
            if return_sparse and result.sparse_vecs is not None:
                response['sparse_vecs'] = result.sparse_vecs
            if return_colbert_vecs and result.colbert_vecs is not None:
                response['colbert_vecs'] = result.colbert_vecs

            return response

        except asyncio.TimeoutError:
            raise RuntimeError(f"Embedding request timed out after {timeout}s")

    async def _worker_loop(self, worker_id: int):
        """Main worker loop that processes batches"""
        logger.info(f"Worker {worker_id} started")

        while self.running:
            try:
                # Collect batch with timeout
                batch = await self._collect_batch()

                if not batch:
                    continue

                # Process batch
                await self._process_batch(batch, worker_id)

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)

        logger.info(f"Worker {worker_id} stopped")

    async def _collect_batch(self) -> List[EmbeddingRequest]:
        """Collect requests into batches with dynamic batching"""
        batch = []
        start_time = time.time()

        # Get first request (blocking)
        try:
            first_request = await asyncio.wait_for(
                self.request_queue.get(),
                timeout=1.0
            )
            batch.append(first_request)
        except asyncio.TimeoutError:
            return batch

        # Collect additional requests up to max_batch_size or max_wait_time
        while len(batch) < self.max_batch_size:
            elapsed = time.time() - start_time
            remaining_time = max(0, self.max_wait_time - elapsed)

            if remaining_time <= 0:
                break

            try:
                request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=remaining_time
                )
                batch.append(request)
            except asyncio.TimeoutError:
                break

        return batch

    async def _process_batch(self, batch: List[EmbeddingRequest], worker_id: int):
        """Process a batch of embedding requests"""
        if not batch:
            return

        batch_size = len(batch)
        logger.debug(f"Worker {worker_id} processing batch of {batch_size} requests")

        try:
            # Extract texts and options
            texts = [req.text for req in batch]

            # Determine what to return (assume all requests want the same for simplicity)
            # In production, you might want to handle mixed requirements
            sample_req = batch[0]
            return_dense = sample_req.return_dense
            return_sparse = sample_req.return_sparse
            return_colbert_vecs = sample_req.return_colbert_vecs

            # Run batch encoding (this is the expensive operation)
            start_time = time.time()

            # Run in thread pool to avoid blocking event loop
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self._encode_batch_sync,
                texts,
                return_dense,
                return_sparse,
                return_colbert_vecs
            )

            encoding_time = time.time() - start_time

            # Distribute results back to individual requests
            for i, request in enumerate(batch):
                try:
                    response = EmbeddingResponse(
                        request_id=request.request_id,
                        dense_vecs=results.get('dense_vecs', [None])[i] if return_dense else None,
                        sparse_vecs=results.get('sparse_vecs', [None])[i] if return_sparse else None,
                        colbert_vecs=results.get('colbert_vecs', [None])[i] if return_colbert_vecs else None
                    )
                    request.future.set_result(response)
                except Exception as e:
                    error_response = EmbeddingResponse(
                        request_id=request.request_id,
                        error=str(e)
                    )
                    request.future.set_result(error_response)

            # Update stats
            self.stats["total_batches"] += 1
            self.stats["avg_batch_size"] = (
                (self.stats["avg_batch_size"] * (self.stats["total_batches"] - 1) + batch_size)
                / self.stats["total_batches"]
            )

            logger.debug(f"Worker {worker_id} completed batch in {encoding_time:.3f}s")

        except Exception as e:
            logger.error(f"Batch processing error: {e}")

            # Set error for all requests in batch
            for request in batch:
                if not request.future.done():
                    error_response = EmbeddingResponse(
                        request_id=request.request_id,
                        error=str(e)
                    )
                    request.future.set_result(error_response)

    def _encode_batch_sync(
        self,
        texts: List[str],
        return_dense: bool,
        return_sparse: bool,
        return_colbert_vecs: bool
    ) -> Dict[str, Any]:
        """Synchronous batch encoding using the actual model"""
        return self.model.encode(
            texts,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert_vecs
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return self.stats.copy()

# Global service instance
_embedding_service: Optional[BatchEmbed] = None

async def get_embedding_service() -> BatchEmbed:
    """Get or create the global embedding service"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = BatchEmbed()
        await _embedding_service.start()
    return _embedding_service

async def encode_with_batching(
    text: str,
    return_dense: bool = True,
    return_sparse: bool = True,
    return_colbert_vecs: bool = True
) -> Dict[str, Any]:
    """
    Drop-in replacement for embedding_model.encode() with batching
    """
    service = await get_embedding_service()
    return await service.encode_async(
        text=text,
        return_dense=return_dense,
        return_sparse=return_sparse,
        return_colbert_vecs=return_colbert_vecs
    )

@asynccontextmanager
async def embedding_service_lifespan():
    """Context manager for service lifecycle"""
    service = await get_embedding_service()
    try:
        yield service
    finally:
        await service.stop()

# Example usage and testing
async def test_batched_service():
    """Test the batched embedding service"""
    service = BatchEmbed(max_batch_size=8, max_wait_time=0.1)
    await service.start()

    try:
        print(f"Testing BatchEmbed with max_batch_size={service.max_batch_size}, max_wait_time={service.max_wait_time}")
        # Test concurrent requests
        tasks = []
        messages = ["hi" for _ in range(100)]

        start_time = time.time()
        for text in messages:
            task = service.encode_async(text)
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        print(f"Processed {len(messages)} requests in {end_time - start_time:.3f}s")
        print(f"Stats: {service.get_stats()}")

        # Verify results
        for i, result in enumerate(results):
            print(f"Result {i}: {list(result.keys())}")

    finally:
        await service.stop()
        print(f"Service Stop")

if __name__ == "__main__":
    asyncio.run(test_batched_service())
