"""Batch Loader module."""

import time
import threading
from typing import Iterator, Optional, List, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import random

import torch
from torch.utils.data import DataLoader, Sampler
import numpy as np
from loguru import logger

from .streaming_dataset import StreamingVisionDataset
from ..utils.storage_monitor import StorageMonitor

class IntelligentBatchSampler(Sampler):
    """Intelligent sampler that considers cache locality"""
    
    def __init__(self, dataset: StreamingVisionDataset, batch_size: int, 
                 shuffle: bool = True, locality_factor: float = 0.3):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.locality_factor = locality_factor  # Probability of choosing cached items
        
    def __iter__(self):
        """Generate batches with cache-aware sampling"""
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            # Intelligent shuffling that considers cache locality
            cached_indices = []
            uncached_indices = []
            
            for idx in indices:
                cache_key = self.dataset._get_cache_key(idx)
                if self.dataset.cache.exists(cache_key):
                    cached_indices.append(idx)
                else:
                    uncached_indices.append(idx)
            
            # Create batches with mix of cached and uncached
            batches = []
            while cached_indices or uncached_indices:
                batch = []
                
                # Add cached items with locality factor probability
                cached_in_batch = int(self.batch_size * self.locality_factor)
                while len(batch) < cached_in_batch and cached_indices:
                    batch.append(cached_indices.pop(random.randint(0, len(cached_indices) - 1)))
                
                # Fill remaining with uncached items
                while len(batch) < self.batch_size and uncached_indices:
                    batch.append(uncached_indices.pop(random.randint(0, len(uncached_indices) - 1)))
                
                # Fill any remaining slots with cached items
                while len(batch) < self.batch_size and cached_indices:
                    batch.append(cached_indices.pop(random.randint(0, len(cached_indices) - 1)))
                
                if batch:
                    batches.append(batch)
            
            # Shuffle batch order
            random.shuffle(batches)
            for batch in batches:
                yield batch
        else:
            # Sequential batching
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class StreamingDataLoader:
    """Optimized data loader for streaming datasets"""
    
    def __init__(self, 
                 dataset: StreamingVisionDataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 2,
                 prefetch_factor: int = 2,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 locality_factor: float = 0.3,
                 adaptive_batching: bool = True):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.locality_factor = locality_factor
        self.adaptive_batching = adaptive_batching
        
        self.storage_monitor = StorageMonitor()
        
        # Adaptive batching parameters
        self._current_batch_size = batch_size
        self._load_times = []
        self._performance_window = 10  # batches to consider for performance
        
        # Create sampler
        self.sampler = IntelligentBatchSampler(
            dataset, self._current_batch_size, shuffle, locality_factor
        )
        
        # Threading for batch preparation
        self._batch_queue = Queue(maxsize=prefetch_factor)
        self._batch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="batch_prep")
        self._stop_event = threading.Event()
        
        # Statistics
        self._stats = {
            'batches_loaded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_batch_load_time': 0.0,
            'adaptive_changes': 0
        }
    
    def _prepare_batch(self, indices: List[int]) -> torch.Tensor:
        """Prepare a single batch"""
        batch_data = []
        batch_targets = []
        cache_hits = 0
        
        start_time = time.time()
        
        # Load items in parallel for better performance
        def load_item(idx):
            return self.dataset[idx]
        
        # Use threading for I/O bound operations
        with ThreadPoolExecutor(max_workers=min(4, len(indices))) as executor:
            futures = {executor.submit(load_item, idx): idx for idx in indices}
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    data, target = future.result()
                    batch_data.append(data)
                    batch_targets.append(target)
                    
                    # Check cache hit
                    cache_key = self.dataset._get_cache_key(idx)
                    if self.dataset.cache.exists(cache_key):
                        cache_hits += 1
                        
                except Exception as e:
                    logger.error(f"Failed to load item {idx}: {e}")
                    # Use dummy data to maintain batch consistency
                    dummy_data = torch.zeros_like(batch_data[0] if batch_data else torch.zeros(3, 224, 224))
                    batch_data.append(dummy_data)
                    batch_targets.append(-1)
        
        # Stack tensors
        if batch_data:
            batch_tensor = torch.stack(batch_data)
            target_tensor = torch.tensor(batch_targets)
            
            if self.pin_memory and torch.cuda.is_available():
                batch_tensor = batch_tensor.pin_memory()
                target_tensor = target_tensor.pin_memory()
        else:
            # Empty batch fallback
            batch_tensor = torch.empty(0, 3, 224, 224)
            target_tensor = torch.empty(0, dtype=torch.long)
        
        load_time = time.time() - start_time
        
        # Update statistics
        self._stats['cache_hits'] += cache_hits
        self._stats['cache_misses'] += len(indices) - cache_hits
        self._load_times.append(load_time)
        
        # Keep only recent load times for performance calculation
        if len(self._load_times) > self._performance_window:
            self._load_times.pop(0)
        
        return batch_tensor, target_tensor, load_time
    
    def _batch_preparation_worker(self):
        """Background worker for batch preparation"""
        for batch_indices in self.sampler:
            if self._stop_event.is_set():
                break
            
            try:
                batch_data = self._prepare_batch(batch_indices)
                self._batch_queue.put(batch_data, timeout=30)
            except Exception as e:
                logger.error(f"Batch preparation error: {e}")
    
    def _adapt_batch_size(self):
        """Adapt batch size based on performance and memory usage"""
        if not self.adaptive_batching or len(self._load_times) < 5:
            return
        
        avg_load_time = np.mean(self._load_times[-5:])
        cache_hit_rate = self._stats['cache_hits'] / (self._stats['cache_hits'] + self._stats['cache_misses'])
        
        # Check memory pressure
        available_memory_gb = self.storage_monitor.get_available_memory_gb()
        
        # Adaptive logic
        should_increase = (
            avg_load_time < 0.5 and  # Fast loading
            cache_hit_rate > 0.7 and  # Good cache performance
            available_memory_gb > 4.0 and  # Sufficient memory
            self._current_batch_size < self.batch_size * 2  # Not too large
        )
        
        should_decrease = (
            avg_load_time > 2.0 or  # Slow loading
            cache_hit_rate < 0.3 or  # Poor cache performance
            available_memory_gb < 2.0 or  # Memory pressure
            self._current_batch_size > self.batch_size  # Above original size
        )
        
        old_size = self._current_batch_size
        
        if should_increase:
            self._current_batch_size = min(self._current_batch_size + 4, self.batch_size * 2)
        elif should_decrease:
            self._current_batch_size = max(self._current_batch_size - 4, self.batch_size // 2)
        
        if old_size != self._current_batch_size:
            self._stats['adaptive_changes'] += 1
            logger.info(f"Adapted batch size: {old_size} -> {self._current_batch_size}")
            
            # Update sampler
            self.sampler = IntelligentBatchSampler(
                self.dataset, self._current_batch_size, self.shuffle, self.locality_factor
            )
    
    def __iter__(self):
        """Iterate over batches"""
        # Reset statistics for new epoch
        self._stats['batches_loaded'] = 0
        self._stop_event.clear()
        
        # Start batch preparation worker
        preparation_future = self._batch_executor.submit(self._batch_preparation_worker)
        
        try:
            while True:
                try:
                    # Get prepared batch
                    batch_data, targets, load_time = self._batch_queue.get(timeout=60)
                    
                    self._stats['batches_loaded'] += 1
                    self._stats['avg_batch_load_time'] = np.mean(self._load_times)
                    
                    # Adapt batch size periodically
                    if self._stats['batches_loaded'] % 10 == 0:
                        self._adapt_batch_size()
                    
                    yield batch_data, targets
                    
                except Empty:
                    # Check if preparation worker is done
                    if preparation_future.done():
                        break
                    else:
                        logger.warning("Batch queue timeout, continuing...")
                        continue
                        
        finally:
            # Cleanup
            self._stop_event.set()
            try:
                preparation_future.result(timeout=5)
            except:
                preparation_future.cancel()
    
    def __len__(self):
        """Return number of batches"""
        return len(self.sampler)
    
    def get_statistics(self) -> dict:
        """Get loader statistics"""
        total_requests = self._stats['cache_hits'] + self._stats['cache_misses']
        hit_rate = self._stats['cache_hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'batches_loaded': self._stats['batches_loaded'],
            'cache_hit_rate': hit_rate,
            'avg_batch_load_time': self._stats['avg_batch_load_time'],
            'current_batch_size': self._current_batch_size,
            'adaptive_changes': self._stats['adaptive_changes'],
            'total_cache_hits': self._stats['cache_hits'],
            'total_cache_misses': self._stats['cache_misses']
        }
    
    def reset_statistics(self):
        """Reset all statistics"""
        self._stats = {
            'batches_loaded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_batch_load_time': 0.0,
            'adaptive_changes': 0
        }
        self._load_times.clear()

class MultiDatasetLoader:
    """Loader for handling multiple datasets with intelligent switching"""
    
    def __init__(self, datasets: dict, batch_size: int = 32, 
                 switch_strategy: str = 'round_robin'):
        self.datasets = {name: StreamingDataLoader(dataset, batch_size) 
                        for name, dataset in datasets.items()}
        self.switch_strategy = switch_strategy
        self.current_dataset = None
        self.dataset_names = list(self.datasets.keys())
        self.dataset_index = 0
        
    def set_strategy(self, strategy: str):
        """Set dataset switching strategy"""
        valid_strategies = ['round_robin', 'random', 'weighted', 'performance']
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        self.switch_strategy = strategy
    
    def _get_next_dataset(self) -> str:
        """Get next dataset based on strategy"""
        if self.switch_strategy == 'round_robin':
            dataset_name = self.dataset_names[self.dataset_index]
            self.dataset_index = (self.dataset_index + 1) % len(self.dataset_names)
            return dataset_name
        
        elif self.switch_strategy == 'random':
            return random.choice(self.dataset_names)
        
        elif self.switch_strategy == 'weighted':
            # Weight by dataset size
            weights = [len(self.datasets[name].dataset) for name in self.dataset_names]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            return np.random.choice(self.dataset_names, p=weights)
        
        elif self.switch_strategy == 'performance':
            # Choose dataset with best cache performance
            best_dataset = None
            best_hit_rate = -1
            
            for name, loader in self.datasets.items():
                stats = loader.get_statistics()
                hit_rate = stats.get('cache_hit_rate', 0)
                if hit_rate > best_hit_rate:
                    best_hit_rate = hit_rate
                    best_dataset = name
            
            return best_dataset or self.dataset_names[0]
    
    def get_loader(self, dataset_name: str) -> StreamingDataLoader:
        """Get loader for specific dataset"""
        return self.datasets.get(dataset_name)
    
    def get_combined_statistics(self) -> dict:
        """Get combined statistics from all loaders"""
        combined_stats = {
            'total_batches_loaded': 0,
            'total_cache_hits': 0,
            'total_cache_misses': 0,
            'datasets': {}
        }
        
        for name, loader in self.datasets.items():
            stats = loader.get_statistics()
            combined_stats['datasets'][name] = stats
            combined_stats['total_batches_loaded'] += stats.get('batches_loaded', 0)
            combined_stats['total_cache_hits'] += stats.get('total_cache_hits', 0)
            combined_stats['total_cache_misses'] += stats.get('total_cache_misses', 0)
        
        # Calculate overall hit rate
        total_requests = combined_stats['total_cache_hits'] + combined_stats['total_cache_misses']
        combined_stats['overall_cache_hit_rate'] = (
            combined_stats['total_cache_hits'] / total_requests if total_requests > 0 else 0
        )
        
        return combined_stats
    
    def switch_to_dataset(self, dataset_name: str):
        """Manually switch to specific dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        self.current_dataset = dataset_name
    
    def get_current_loader(self) -> Optional[StreamingDataLoader]:
        """Get current dataset loader"""
        if not self.current_dataset:
            self.current_dataset = self._get_next_dataset()
        return self.datasets.get(self.current_dataset)