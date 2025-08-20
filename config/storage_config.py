"""Storage Config module."""

import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class CloudConfig:
    """Cloud storage configuration"""
    credentials_path: str
    token_path: str
    root_folder_id: str
    max_concurrent_downloads: int = 3
    chunk_size: int = 8192  # 8KB chunks
    timeout: int = 300  # 5 minutes
    retry_attempts: int = 3
    retry_delay: int = 2

@dataclass
class LocalStorageConfig:
    """Local storage configuration"""
    cache_path: str
    temp_download_path: str
    model_checkpoint_path: str
    log_path: str
    max_cache_size_gb: int = 10
    cache_retention_hours: int = 24
    disk_usage_threshold: int = 85
    auto_cleanup_enabled: bool = True
    cleanup_schedule_hours: int = 6

@dataclass
class StreamingConfig:
    """Streaming configuration"""
    prefetch_buffer_size: int = 5
    max_workers: int = 4
    batch_prefetch_enabled: bool = True
    background_sync_enabled: bool = True
    stream_chunk_size: int = 1024 * 1024  # 1MB
    
class StorageManager:
    """Central storage configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv('CONFIG_PATH', '')
        self._load_config()
        self._create_directories()
    
    def _load_config(self):
        """Load configuration from environment variables"""
        self.cloud = CloudConfig(
            credentials_path=os.getenv('GDRIVE_CREDENTIALS_PATH', ''),
            token_path=os.getenv('GDRIVE_TOKEN_PATH', ''),
            root_folder_id=os.getenv('GDRIVE_ROOT_FOLDER_ID', ''),
            max_concurrent_downloads=int(os.getenv('MAX_CONCURRENT_DOWNLOADS', '3'))
        )
        
        self.local = LocalStorageConfig(
            cache_path=os.getenv('LOCAL_CACHE_PATH', '/home/oshadi/research_workspace/cache'),
            temp_download_path=os.getenv('TEMP_DOWNLOAD_PATH', '/data/temp_downloads'),
            model_checkpoint_path=os.getenv('MODEL_CHECKPOINT_PATH', '/home/oshadi/research_workspace/checkpoints'),
            log_path=os.getenv('LOG_PATH', '/home/oshadi/research_workspace/logs'),
            max_cache_size_gb=int(os.getenv('MAX_CACHE_SIZE_GB', '10')),
            cache_retention_hours=int(os.getenv('CACHE_RETENTION_HOURS', '24')),
            disk_usage_threshold=int(os.getenv('DISK_USAGE_THRESHOLD', '85')),
            auto_cleanup_enabled=os.getenv('AUTO_CLEANUP_ENABLED', 'true').lower() == 'true',
            cleanup_schedule_hours=int(os.getenv('CLEANUP_SCHEDULE_HOURS', '6'))
        )
        
        self.streaming = StreamingConfig(
            prefetch_buffer_size=int(os.getenv('PREFETCH_BUFFER_SIZE', '5')),
            max_workers=min(4, os.cpu_count() or 4)
        )
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.local.cache_path,
            self.local.temp_download_path,
            self.local.model_checkpoint_path,
            self.local.log_path,
            os.path.dirname(self.cloud.credentials_path),
            os.path.dirname(self.cloud.token_path)
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_available_space_gb(self, path: str) -> float:
        """Get available space in GB for given path"""
        import shutil
        _, _, free_bytes = shutil.disk_usage(path)
        return free_bytes / (1024**3)
    
    def is_space_available(self, required_gb: float, path: Optional[str] = None) -> bool:
        """Check if required space is available"""
        check_path = path or self.local.cache_path
        available = self.get_available_space_gb(check_path)
        return available >= required_gb
    
    def get_cache_usage_gb(self) -> float:
        """Get current cache usage in GB"""
        cache_path = Path(self.local.cache_path)
        if not cache_path.exists():
            return 0.0
        
        total_size = 0
        for file_path in cache_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size / (1024**3)
    
    def should_cleanup(self) -> bool:
        """Determine if cleanup is needed"""
        if not self.local.auto_cleanup_enabled:
            return False
        
        # Check cache size
        if self.get_cache_usage_gb() > self.local.max_cache_size_gb:
            return True
        
        # Check disk usage
        cache_path_usage = (shutil.disk_usage(self.local.cache_path).used / 
                           shutil.disk_usage(self.local.cache_path).total) * 100
        
        return cache_path_usage > self.local.disk_usage_threshold
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'cloud': self.cloud.__dict__,
            'local': self.local.__dict__,
            'streaming': self.streaming.__dict__
        }
    
    def save_config(self, path: str):
        """Save current configuration to file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

# Global storage manager instance
storage_config = StorageManager()