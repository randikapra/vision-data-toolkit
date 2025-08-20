"""Prefetch Manager module."""

import time
import threading
from typing import Dict, List, Optional, Set, Callable
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from queue import Queue, PriorityQueue, Empty
from dataclasses import dataclass
from enum import Enum, auto
import heapq
import random

import numpy as np
from loguru import logger

from ..cloud.cloud_interface import CloudManager
from ..cache.lru_cache import LRUCache
from ..utils.storage_monitor import StorageMonitor

class PrefetchPriority(Enum):
    """Prefetch priority levels"""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

@dataclass
class PrefetchTask:
    """Prefetch task representation"""
    file_id: str
    priority: PrefetchPriority
    timestamp: float
    retries: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        # Higher priority first, then earlier timestamp
        return (self.priority.value, -self.timestamp) < (other.priority.value, -other.timestamp)

class PrefetchStrategy(Enum):
    """Different prefetch strategies"""
    SEQUENTIAL = "sequential"  # Prefetch next items in sequence
    RANDOM = "random"  # Random prefetching
    PREDICTIVE = "predictive"  # ML-based prediction
    LOCALITY = "locality"  # Based on access patterns
    HYBRID = "hybrid"  # Combination of strategies

class AccessPattern:
    """Track access patterns for intelligent prefetching"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.access_history: List[int] = []
        self.access_frequency: Dict[int, int] = {}
        self.sequential_patterns: Dict[int, List[int]] = {}
        self.last_access_time: Dict[int, float] = {}
        self.lock = threading.RLock()
    
    def record_access(self, index: int):
        """Record an access to update patterns"""
        with self.lock:
            current_time = time.time()
            
            # Update access history
            self.access_history.append(index)
            if len(self.access_history) > self.history_size:
                old_index = self.access_history.pop(0)
                # Decrease frequency of old access
                if old_index in self.access_frequency:
                    self.access_frequency[old_index] = max(0, self.access_frequency[old_index] - 1)
            
            # Update frequency
            self.access_frequency[index] = self.access_frequency.get(index, 0) + 1
            self.last_access_time[index] = current_time
            
            # Update sequential patterns
            if len(self.access_history) >= 2:
                prev_index = self.access_history[-2]
                if prev_index not in self.sequential_patterns:
                    self.sequential_patterns[prev_index] = []
                self.sequential_patterns[prev_index].append(index)
                
                # Keep only recent patterns
                if len(self.sequential_patterns[prev_index]) > 10:
                    self.sequential_patterns[prev_index].pop(0)
    
    def predict_next_accesses(self, current_index: int, count: int = 5) -> List[int]:
        """Predict next likely accesses"""
        with self.lock:
            predictions = []
            
            # Sequential prediction
            if current_index in self.sequential_patterns:
                pattern = self.sequential_patterns[current_index]
                # Get most common next indices
                from collections import Counter
                common_next = Counter(pattern).most_common(count)
                predictions.extend([idx for idx, _ in common_next])
            
            # Frequency-based prediction
            if len(predictions) < count:
                sorted_by_freq = sorted(self.access_frequency.items(), 
                                      key=lambda x: x[1], reverse=True)
                for idx, _ in sorted_by_freq:
                    if idx not in predictions and len(predictions) < count:
                        predictions.append(idx)
            
            return predictions[:count]
    
    def get_hot_indices(self, threshold: int = 3) -> Set[int]:
        """Get frequently accessed indices"""
        with self.lock:
            return {idx for idx, freq in self.access_frequency.items() if freq >= threshold}

class IntelligentPrefetchManager:
    """Advanced prefetch manager with multiple strategies"""
    
    def __init__(self, 
                 cloud_manager: CloudManager,
                 cache: LRUCache,
                 max_workers: int = 4,
                 queue_size: int = 100,
                 strategy: PrefetchStrategy = PrefetchStrategy.HYBRID):
        
        self.cloud_manager = cloud_manager
        self.cache = cache
        self.max_workers = max_workers
        self.strategy = strategy
        self.storage_monitor = StorageMonitor()
        
        # Task management
        self.task_queue = PriorityQueue(maxsize=queue_size)
        self.active_tasks: Dict[str, Future] = {}
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        
        # Access pattern tracking
        self.access_patterns = AccessPattern()
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="prefetch")
        self.manager_thread = threading.Thread(target=self._prefetch_manager, daemon=True)
        self.running = threading.Event()
        self.running.set()
        
        # Statistics
        self.stats = {
            'tasks_queued': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'cache_hits_from_prefetch': 0,
            'bytes_prefetched': 0,
            'avg_prefetch_time': 0.0
        }
        
        # Start manager
        self.manager_thread.start()
        logger.info(f"Prefetch manager started with strategy: {strategy.value}")
    
    def _prefetch_manager(self):
        """Main prefetch management loop"""
        prefetch_times = []
        
        while self.running.is_set():
            try:
                # Check storage constraints
                if not self._can_prefetch():
                    time.sleep(1)
                    continue
                
                # Get next task
                try:
                    priority_task = self.task_queue.get(timeout=1.0)
                    task = priority_task
                except Empty:
                    continue
                
                # Skip if already completed or in progress
                if task.file_id in self.completed_tasks or task.file_id in self.active_tasks:
                    continue
                
                # Submit prefetch task
                future = self.executor.submit(self._prefetch_file, task)
                self.active_tasks[task.file_id] = future
                self.stats['tasks_queued'] += 1
                
                # Clean up completed tasks
                self._cleanup_completed_tasks()
                
                # Update average prefetch time
                if prefetch_times:
                    self.stats['avg_prefetch_time'] = np.mean(prefetch_times[-100:])
                
            except Exception as e:
                logger.error(f"Prefetch manager error: {e}")
                time.sleep(1)
    
    def _can_prefetch(self) -> bool:
        """Check if prefetching is allowed based on system resources"""
        # Check memory usage
        if self.storage_monitor.get_memory_usage_percent() > 85:
            return False
        
        # Check cache usage
        if self.cache.get_usage_gb() > self.cache.max_size_gb * 0.9:
            return False
        
        # Check disk space
        if self.storage_monitor.get_disk_usage_percent() > 90:
            return False
        
        # Check active tasks
        if len(self.active_tasks) >= self.max_workers:
            return False
        
        return True
    
    def _prefetch_file(self, task: PrefetchTask) -> bool:
        """Prefetch a single file"""
        start_time = time.time()
        
        try:
            # Check if already cached
            if self.cache.exists(task.file_id):
                self.stats['cache_hits_from_prefetch'] += 1
                return True
            
            # Get file info
            file_info = self.cloud_manager._client.get_file_info(task.file_id)
            if not file_info:
                logger.warning(f"File info not found for {task.file_id}")
                return False
            
            # Check available space
            file_size_gb = int(file_info.get('size', 0)) / (1024**3)
            if not self.storage_monitor.check_available_space(file_size_gb):
                logger.warning(f"Insufficient space for prefetching {task.file_id}")
                return False
            
            # Stream and cache the file
            file_data = b''
            for chunk in self.cloud_manager._client.stream_file(task.file_id):
                file_data += chunk
            
            # Store in cache
            self.cache.put(task.file_id, file_data)
            
            # Update statistics
            self.stats['tasks_completed'] += 1
            self.stats['bytes_prefetched'] += len(file_data)
            
            prefetch_time = time.time() - start_time
            logger.debug(f"Prefetched {file_info['name']} in {prefetch_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to prefetch {task.file_id}: {e}")
            task.retries += 1
            
            # Retry if within limits
            if task.retries < task.max_retries:
                # Re-queue with lower priority
                retry_task = PrefetchTask(
                    file_id=task.file_id,
                    priority=PrefetchPriority.LOW,
                    timestamp=time.time(),
                    retries=task.retries,
                    max_retries=task.max_retries
                )
                self.task_queue.put(retry_task)
            else:
                self.failed_tasks.add(task.file_id)
                self.stats['tasks_failed'] += 1
            
            return False
        
        finally:
            # Remove from active tasks
            if task.file_id in self.active_tasks:
                del self.active_tasks[task.file_id]
            self.completed_tasks.add(task.file_id)
    
    def _cleanup_completed_tasks(self):
        """Clean up completed task tracking"""
        completed = []
        for file_id, future in self.active_tasks.items():
            if future.done():
                completed.append(file_id)
        
        for file_id in completed:
            del self.active_tasks[file_id]
            self.completed_tasks.add(file_id)
        
        # Limit completed task tracking
        if len(self.completed_tasks) > 10000:
            # Remove oldest 20%
            to_remove = list(self.completed_tasks)[:2000]
            for task_id in to_remove:
                self.completed_tasks.discard(task_id)
    
    def request_prefetch(self, file_id: str, priority: PrefetchPriority = PrefetchPriority.MEDIUM):
        """Request prefetch for a specific file"""
        if file_id in self.completed_tasks or file_id in self.active_tasks:
            return  # Already handled
        
        task = PrefetchTask(
            file_id=file_id,
            priority=priority,
            timestamp=time.time()
        )
        
        try:
            self.task_queue.put(task, timeout=0.1)
        except:
            logger.warning(f"Prefetch queue full, dropping task for {file_id}")
    
    def prefetch_sequence(self, file_ids: List[str], 
                         priority: PrefetchPriority = PrefetchPriority.MEDIUM):
        """Request prefetch for a sequence of files"""
        for i, file_id in enumerate(file_ids):
            # Earlier items in sequence get higher priority
            seq_priority = priority
            if i == 0:
                seq_priority = PrefetchPriority.HIGH
            elif i < 3:
                seq_priority = PrefetchPriority.MEDIUM
            else:
                seq_priority = PrefetchPriority.LOW
            
            self.request_prefetch(file_id, seq_priority)
    
    def record_access(self, file_id: str, index: int):
        """Record file access for pattern learning"""
        self.access_patterns.record_access(index)
        
        # Trigger predictive prefetching
        if self.strategy in [PrefetchStrategy.PREDICTIVE, PrefetchStrategy.HYBRID]:
            self._trigger_predictive_prefetch(index)
    
    def _trigger_predictive_prefetch(self, current_index: int):
        """Trigger predictive prefetching based on access patterns"""
        predictions = self.access_patterns.predict_next_accesses(current_index, 5)
        
        # Convert indices to file IDs (this would need dataset context)
        # For now, we'll assume a mapping function exists
        for pred_index in predictions:
            # This would need to be implemented based on your dataset structure
            predicted_file_id = f"file_{pred_index}"  # Placeholder
            self.request_prefetch(predicted_file_id, PrefetchPriority.MEDIUM)
    
    def prefetch_hot_data(self):
        """Prefetch frequently accessed data"""
        hot_indices = self.access_patterns.get_hot_indices()
        
        for index in hot_indices:
            # Convert to file ID (placeholder implementation)
            file_id = f"file_{index}"
            self.request_prefetch(file_id, PrefetchPriority.HIGH)
    
    def get_statistics(self) -> Dict:
        """Get prefetch manager statistics"""
        return {
            **self.stats,
            'active_tasks': len(self.active_tasks),
            'queue_size': self.task_queue.qsize(),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'cache_usage_gb': self.cache.get_usage_gb(),
            'cache_hit_rate': self.cache.get_hit_rate()
        }
    
    def adjust_strategy(self, new_strategy: PrefetchStrategy):
        """Dynamically adjust prefetch strategy"""
        old_strategy = self.strategy
        self.strategy = new_strategy
        logger.info(f"Prefetch strategy changed: {old_strategy.value} -> {new_strategy.value}")
    
    def pause_prefetching(self):
        """Pause prefetch operations"""
        self.running.clear()
        logger.info("Prefetching paused")
    
    def resume_prefetching(self):
        """Resume prefetch operations"""
        self.running.set()
        if not self.manager_thread.is_alive():
            self.manager_thread = threading.Thread(target=self._prefetch_manager, daemon=True)
            self.manager_thread.start()
        logger.info("Prefetching resumed")
    
    def shutdown(self):
        """Shutdown prefetch manager"""
        logger.info("Shutting down prefetch manager...")
        self.running.clear()
        
        # Wait for current tasks to complete
        if self.manager_thread.is_alive():
            self.manager_thread.join(timeout=10)
        
        # Cancel remaining tasks
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except Empty:
                break
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        logger.info("Prefetch manager shutdown complete")

class AdaptivePrefetchManager(IntelligentPrefetchManager):
    """Adaptive prefetch manager that learns and adjusts strategies"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Performance tracking
        self.strategy_performance = {
            strategy: {'hits': 0, 'misses': 0, 'avg_time': 0.0}
            for strategy in PrefetchStrategy
        }
        
        # Adaptive parameters
        self.evaluation_interval = 100  # Evaluate every N accesses
        self.access_count = 0
        self.last_evaluation = 0
    
    def record_access(self, file_id: str, index: int):
        """Enhanced access recording with performance tracking"""
        super().record_access(file_id, index)
        
        self.access_count += 1
        
        # Check if file was prefetched
        if file_id in self.completed_tasks:
            self.strategy_performance[self.strategy]['hits'] += 1
        else:
            self.strategy_performance[self.strategy]['misses'] += 1
        
        # Evaluate and adapt strategy periodically
        if self.access_count - self.last_evaluation >= self.evaluation_interval:
            self._evaluate_and_adapt_strategy()
            self.last_evaluation = self.access_count
    
    def _evaluate_and_adapt_strategy(self):
        """Evaluate current strategy performance and adapt if needed"""
        current_performance = self.strategy_performance[self.strategy]
        total_attempts = current_performance['hits'] + current_performance['misses']
        
        if total_attempts == 0:
            return
        
        current_hit_rate = current_performance['hits'] / total_attempts
        
        # Find best performing strategy
        best_strategy = None
        best_hit_rate = 0
        
        for strategy, perf in self.strategy_performance.items():
            strategy_total = perf['hits'] + perf['misses']
            if strategy_total > 0:
                hit_rate = perf['hits'] / strategy_total
                if hit_rate > best_hit_rate:
                    best_hit_rate = hit_rate
                    best_strategy = strategy
        
        # Switch if significant improvement is possible
        if best_strategy and best_strategy != self.strategy:
            improvement = best_hit_rate - current_hit_rate
            if improvement > 0.1:  # 10% improvement threshold
                logger.info(f"Switching strategy: {self.strategy.value} -> {best_strategy.value} "
                          f"(improvement: {improvement:.2%})")
                self.adjust_strategy(best_strategy)