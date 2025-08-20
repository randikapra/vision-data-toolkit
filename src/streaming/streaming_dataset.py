"""Streaming Dataset module."""

import os
import io
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Empty
import hashlib

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2
from loguru import logger

from ..cloud.cloud_interface import CloudManager
from ..cache.lru_cache import LRUCache
from ..processors.image_processor import ImageProcessor
from ..utils.storage_monitor import StorageMonitor

class StreamingVisionDataset(Dataset):
    """
    Streaming dataset that loads images from cloud storage on-demand
    with intelligent caching and prefetching
    """
    
    def __init__(self, 
                 cloud_manager: CloudManager,
                 folder_id: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 cache_size_gb: float = 5.0,
                 prefetch_buffer: int = 10,
                 image_processor: Optional[ImageProcessor] = None,
                 file_extensions: List[str] = None,
                 enable_background_sync: bool = True):
        
        self.cloud_manager = cloud_manager
        self.folder_id = folder_id
        self.transform = transform
        self.target_transform = target_transform
        self.image_processor = image_processor or ImageProcessor()
        self.storage_monitor = StorageMonitor()
        
        # Cache configuration
        self.cache = LRUCache(max_size_gb=cache_size_gb)
        self.prefetch_buffer = prefetch_buffer
        self.enable_background_sync = enable_background_sync
        
        # File extensions to consider
        self.file_extensions = file_extensions or ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        
        # Threading and prefetching
        self._prefetch_queue = Queue(maxsize=prefetch_buffer * 2)
        self._prefetch_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="prefetch")
        self._background_sync_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sync")
        self._prefetch_futures: Dict[int, Future] = {}
        self._access_lock = threading.RLock()
        
        # Dataset initialization
        self._initialize_dataset()
        self._start_background_processes()
    
    def _initialize_dataset(self):
        """Initialize dataset by loading file list from cloud"""
        logger.info(f"Initializing streaming dataset from folder {self.folder_id}")
        
        # Get file list from cloud
        self.file_list = self.cloud_manager.list_dataset_files(
            self.folder_id, self.file_extensions
        )
        
        if not self.file_list:
            raise ValueError(f"No valid files found in folder {self.folder_id}")
        
        # Create file index mapping
        self.file_index = {i: file_info for i, file_info in enumerate(self.file_list)}
        
        logger.info(f"Dataset initialized with {len(self.file_list)} files")
        
        # Log dataset statistics
        stats = self._calculate_dataset_stats()
        logger.info(f"Dataset stats: {stats}")
    
    def _calculate_dataset_stats(self) -> Dict:
        """Calculate basic dataset statistics"""
        total_size = sum(int(f.get('size', 0)) for f in self.file_list)
        formats = {}
        
        for file_info in self.file_list:
            ext = Path(file_info['name']).suffix.lower().lstrip('.')
            formats[ext] = formats.get(ext, 0) + 1
        
        return {
            'total_files': len(self.file_list),
            'total_size_gb': total_size / (1024**3),
            'file_formats': formats,
            'avg_file_size_mb': (total_size / len(self.file_list) / (1024**2)) if self.file_list else 0
        }
    
    def _start_background_processes(self):
        """Start background prefetching and synchronization"""
        if self.enable_background_sync:
            self._background_sync_executor.submit(self._background_sync_worker)
    
    def _background_sync_worker(self):
        """Background worker for cache synchronization"""
        while True:
            try:
                # Check cache health and cleanup if needed
                if self.cache.get_usage_gb() > self.cache.max_size_gb * 0.8:
                    self.cache.cleanup_expired()
                
                # Monitor storage usage
                if self.storage_monitor.should_cleanup():
                    self.cache.clear_old_entries(hours=1)
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Background sync error: {e}")
                time.sleep(60)  # Wait before retry
    
    def _get_cache_key(self, index: int) -> str:
        """Generate cache key for file index"""
        file_info = self.file_index[index]
        # Use file ID and modification time for cache key
        cache_key = f"{file_info['id']}_{file_info.get('modifiedTime', '')}"
        return hashlib.md5(cache_key.encode()).hexdigest()
    
    def _load_image_from_cloud(self, index: int) -> Optional[np.ndarray]:
        """Load image directly from cloud storage"""
        file_info = self.file_index[index]
        file_id = file_info['id']
        filename = file_info['name']
        
        try:
            # Stream file content
            image_data = b''
            for chunk in self.cloud_manager._client.stream_file(file_id):
                image_data += chunk
            
            # Convert to image
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            
            logger.debug(f"Loaded image {filename} from cloud: {image_array.shape}")
            return image_array
            
        except Exception as e:
            logger.error(f"Failed to load image {filename} from cloud: {e}")
            return None
    
    def _prefetch_worker(self, indices: List[int]):
        """Worker function for prefetching images"""
        for idx in indices:
            try:
                cache_key = self._get_cache_key(idx)
                
                # Skip if already cached
                if self.cache.exists(cache_key):
                    continue
                
                # Load from cloud
                image_data = self._load_image_from_cloud(idx)
                if image_data is not None:
                    # Cache the image
                    self.cache.put(cache_key, image_data)
                    logger.debug(f"Prefetched and cached image {idx}")
                
            except Exception as e:
                logger.error(f"Prefetch error for index {idx}: {e}")
    
    def _start_prefetch(self, indices: List[int]):
        """Start prefetching for given indices"""
        with self._access_lock:
            # Cancel existing prefetch futures that are not needed
            current_indices = set(indices)
            futures_to_cancel = []
            
            for idx, future in self._prefetch_futures.items():
                if idx not in current_indices and not future.done():
                    futures_to_cancel.append(idx)
            
            for idx in futures_to_cancel:
                self._prefetch_futures[idx].cancel()
                del self._prefetch_futures[idx]
            
            # Start new prefetch tasks
            for idx in indices:
                if idx not in self._prefetch_futures or self._prefetch_futures[idx].done():
                    future = self._prefetch_executor.submit(self._prefetch_worker, [idx])
                    self._prefetch_futures[idx] = future
    
    def __len__(self) -> int:
        """Return dataset length"""
        return len(self.file_list)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        """Get item by index with caching and prefetching"""
        if index >= len(self.file_list):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.file_list)}")
        
        cache_key = self._get_cache_key(index)
        
        # Try to get from cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            image_data = cached_data
            logger.debug(f"Cache hit for index {index}")
        else:
            # Load from cloud
            logger.debug(f"Cache miss for index {index}, loading from cloud")
            image_data = self._load_image_from_cloud(index)
            
            if image_data is None:
                raise RuntimeError(f"Failed to load image at index {index}")
            
            # Cache the loaded data
            self.cache.put(cache_key, image_data)
        
        # Process image
        image = self.image_processor.process(image_data)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Prefetch nearby images
        self._schedule_prefetch(index)
        
        # Create dummy target (modify based on your needs)
        target = index  # Placeholder target
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, target
    
    def _schedule_prefetch(self, current_index: int):
        """Schedule prefetching of nearby images"""
        if not self.enable_background_sync:
            return
        
        # Calculate indices to prefetch
        prefetch_indices = []
        for offset in range(1, self.prefetch_buffer + 1):
            # Prefetch forward
            if current_index + offset < len(self.file_list):
                prefetch_indices.append(current_index + offset)
            # Prefetch backward
            if current_index - offset >= 0:
                prefetch_indices.append(current_index - offset)
        
        if prefetch_indices:
            self._start_prefetch(prefetch_indices)
    
    def get_file_info(self, index: int) -> Dict:
        """Get file information for given index"""
        return self.file_index.get(index, {})
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cache_size_gb': self.cache.get_usage_gb(),
            'cache_hit_rate': self.cache.get_hit_rate(),
            'cached_items': len(self.cache),
            'max_cache_size_gb': self.cache.max_size_gb
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Dataset cache cleared")
    
    def preload_subset(self, indices: List[int], show_progress: bool = True):
        """Preload a subset of images into cache"""
        if show_progress:
            from tqdm import tqdm
            indices = tqdm(indices, desc="Preloading images")
        
        for idx in indices:
            try:
                self.__getitem__(idx)  # This will cache the image
            except Exception as e:
                logger.error(f"Failed to preload index {idx}: {e}")
    
    def __del__(self):
        """Cleanup resources"""
        try:
            self._prefetch_executor.shutdown(wait=False)
            self._background_sync_executor.shutdown(wait=False)
        except:
            pass

class MultiDatasetStreamer:
    """Handle multiple streaming datasets simultaneously"""
    
    def __init__(self, cloud_manager: CloudManager, cache_size_per_dataset: float = 2.0):
        self.cloud_manager = cloud_manager
        self.cache_size_per_dataset = cache_size_per_dataset
        self.datasets: Dict[str, StreamingVisionDataset] = {}
        self.current_dataset: Optional[str] = None
    
    def add_dataset(self, name: str, folder_id: str, **kwargs) -> StreamingVisionDataset:
        """Add a new streaming dataset"""
        dataset = StreamingVisionDataset(
            cloud_manager=self.cloud_manager,
            folder_id=folder_id,
            cache_size_gb=self.cache_size_per_dataset,
            **kwargs
        )
        
        self.datasets[name] = dataset
        logger.info(f"Added dataset '{name}' with {len(dataset)} images")
        return dataset
    
    def switch_dataset(self, name: str) -> Optional[StreamingVisionDataset]:
        """Switch to a different dataset"""
        if name not in self.datasets:
            logger.error(f"Dataset '{name}' not found")
            return None
        
        self.current_dataset = name
        logger.info(f"Switched to dataset '{name}'")
        return self.datasets[name]
    
    def get_dataset(self, name: str) -> Optional[StreamingVisionDataset]:
        """Get dataset by name"""
        return self.datasets.get(name)
    
    def list_datasets(self) -> List[str]:
        """List all available datasets"""
        return list(self.datasets.keys())
    
    def get_combined_stats(self) -> Dict:
        """Get combined statistics for all datasets"""
        total_files = 0
        total_cache_size = 0.0
        dataset_stats = {}
        
        for name, dataset in self.datasets.items():
            stats = dataset.get_cache_stats()
            dataset_stats[name] = {
                'files': len(dataset),
                'cache_size_gb': stats['cache_size_gb'],
                'hit_rate': stats['cache_hit_rate']
            }
            total_files += len(dataset)
            total_cache_size += stats['cache_size_gb']
        
        return {
            'total_datasets': len(self.datasets),
            'total_files': total_files,
            'total_cache_size_gb': total_cache_size,
            'datasets': dataset_stats
        }
    
    def cleanup_all_caches(self):
        """Clear caches for all datasets"""
        for dataset in self.datasets.values():
            dataset.clear_cache()
        logger.info("Cleared all dataset caches")
    
    def preload_current_dataset(self, num_samples: int = 100):
        """Preload samples from current dataset"""
        if not self.current_dataset:
            logger.error("No current dataset selected")
            return
        
        dataset = self.datasets[self.current_dataset]
        indices = list(range(min(num_samples, len(dataset))))
        dataset.preload_subset(indices)