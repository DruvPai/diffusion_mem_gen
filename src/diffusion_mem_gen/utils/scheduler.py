from queue import Queue
from threading import Lock

class GPUJobScheduler:
    """
    Manages job scheduling across multiple GPUs with dynamic load balancing.
    """
    
    def __init__(self, num_gpus: int):
        """
        Initialize the GPU job scheduler.
        
        Args:
            num_gpus: Number of GPUs available
        """
        self.num_gpus = num_gpus
        self.gpu_locks = [Lock() for _ in range(num_gpus)]
        self.available_gpus = Queue()
        for i in range(num_gpus):
            self.available_gpus.put(i)
    
    def acquire_gpu(self) -> int:
        """
        Get an available GPU.
        
        Returns:
            GPU ID
        """
        return self.available_gpus.get()
    
    def release_gpu(self, gpu_id: int) -> None:
        """
        Release a GPU back to the pool.
        
        Args:
            gpu_id: GPU ID to release
        """
        self.available_gpus.put(gpu_id)
    
    def run_job(self, job_func, *args):
        """
        Run a job on an available GPU and release it when done.
        
        Args:
            job_func: Function to run
            *args: Arguments to pass to the function
        """
        gpu_id = self.acquire_gpu()
        try:
            result = job_func(*args, gpu_id=gpu_id)
            return result
        finally:
            self.release_gpu(gpu_id)