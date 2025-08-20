"""Cloud Interface module."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Generator
from pathlib import Path
import threading
from loguru import logger

from .gdrive_manager import GDriveManager
from ..utils.storage_monitor import StorageMonitor

class CloudStorageInterface(ABC):
    """Abstract interface for cloud storage providers"""
    
    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with cloud provider"""
        pass
    
    @abstractmethod
    def list_files(self, folder_id: str) -> List[Dict]:
        """List files in a folder"""
        pass
    
    @abstractmethod
    def download_file(self, file_id: str, local_path: str) -> bool:
        """Download a single file"""
        pass
    
    @abstractmethod
    def upload_file(self, local_path: str, folder_id: str) -> Optional[str]:
        """Upload a single file"""
        pass
    
    @abstractmethod
    def stream_file(self, file_id: str) -> Generator[bytes, None, None]:
        """Stream file content"""
        pass

class CloudManager:
    """Unified cloud storage manager"""
    
    def __init__(self, provider: str = "gdrive", **kwargs):
        self.provider = provider
        self.storage_monitor = StorageMonitor()
        self._client = None
        self._lock = threading.Lock()
        self._initialize_client(**kwargs)
    
    def _initialize_client(self, **kwargs):
        """Initialize cloud storage client"""
        if self.provider == "gdrive":
            credentials_path = kwargs.get('credentials_path')
            token_path = kwargs.get('token_path')
            
            if not credentials_path or not token_path:
                raise ValueError("Google Drive requires credentials_path and token_path")
            
            self._client = GDriveManager(credentials_path, token_path)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def list_dataset_files(self, folder_id: str, 
                          file_extensions: Optional[List[str]] = None) -> List[Dict]:
        """List dataset files with optional filtering"""
        files = self._client.list_files(folder_id)
        
        if file_extensions:
            extensions = [ext.lower().lstrip('.') for ext in file_extensions]
            files = [f for f in files if any(f['name'].lower().endswith(f'.{ext}') 
                                           for ext in extensions)]
        
        return files
    
    def download_with_monitoring(self, file_id: str, local_path: str) -> bool:
        """Download file with storage monitoring"""
        # Check available space before download
        file_info = self._client.get_file_info(file_id)
        if file_info and 'size' in file_info:
            required_space_gb = int(file_info['size']) / (1024**3)
            if not self.storage_monitor.check_available_space(required_space_gb, 
                                                            str(Path(local_path).parent)):
                logger.error(f"Insufficient space for download: {required_space_gb:.2f}GB required")
                return False
        
        return self._client.download_file(file_id, local_path)
    
    def batch_download_with_retry(self, file_list: List[Dict], local_dir: str,
                                 max_retries: int = 3, max_workers: int = 3) -> Dict:
        """Batch download with retry mechanism"""
        results = {"success": [], "failed": [], "retried": []}
        failed_files = file_list.copy()
        
        for attempt in range(max_retries + 1):
            if not failed_files:
                break
            
            logger.info(f"Download attempt {attempt + 1}/{max_retries + 1} for {len(failed_files)} files")
            
            batch_results = self._client.batch_download(failed_files, local_dir, max_workers)
            
            # Update results
            current_failed = []
            for filename, success in batch_results.items():
                if success:
                    results["success"].append(filename)
                    if attempt > 0:
                        results["retried"].append(filename)
                else:
                    current_failed.append(next(f for f in failed_files if f['name'] == filename))
            
            failed_files = current_failed
            
            if failed_files and attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        results["failed"] = [f['name'] for f in failed_files]
        return results
    
    def upload_model_checkpoint(self, checkpoint_path: str, 
                               model_folder_id: str) -> Optional[str]:
        """Upload model checkpoint to cloud storage"""
        if not Path(checkpoint_path).exists():
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return None
        
        # Add timestamp to filename
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{Path(checkpoint_path).stem}_{timestamp}{Path(checkpoint_path).suffix}"
        
        return self._client.upload_file(checkpoint_path, model_folder_id, filename)
    
    def cleanup_old_checkpoints(self, model_folder_id: str, keep_latest: int = 5):
        """Clean up old model checkpoints, keeping only the latest ones"""
        files = self._client.list_files(model_folder_id)
        
        # Filter checkpoint files and sort by modification time
        checkpoint_files = [f for f in files if f['name'].endswith(('.pth', '.pt', '.ckpt'))]
        checkpoint_files.sort(key=lambda x: x.get('modifiedTime', ''), reverse=True)
        
        # Delete old checkpoints
        files_to_delete = checkpoint_files[keep_latest:]
        for file_info in files_to_delete:
            if self._client.delete_file(file_info['id']):
                logger.info(f"Deleted old checkpoint: {file_info['name']}")
    
    def get_dataset_statistics(self, folder_id: str) -> Dict:
        """Get statistics about a dataset folder"""
        files = self._client.list_files(folder_id)
        
        total_size = 0
        file_count = 0
        formats = {}
        
        for file in files:
            file_count += 1
            if 'size' in file:
                total_size += int(file['size'])
            
            ext = Path(file['name']).suffix.lower().lstrip('.')
            formats[ext] = formats.get(ext, 0) + 1
        
        return {
            'total_files': file_count,
            'total_size_gb': total_size / (1024**3),
            'file_formats': formats,
            'average_file_size_mb': (total_size / file_count / (1024**2)) if file_count > 0 else 0
        }
    
    def validate_dataset_integrity(self, folder_id: str, 
                                  sample_size: int = 10) -> Dict:
        """Validate dataset integrity by checking a sample of files"""
        files = self._client.list_files(folder_id)
        
        if not files:
            return {"valid": False, "reason": "No files found"}
        
        # Sample files for validation
        import random
        sample_files = random.sample(files, min(sample_size, len(files)))
        
        validation_results = {
            "total_sampled": len(sample_files),
            "valid_files": 0,
            "invalid_files": 0,
            "errors": []
        }
        
        for file_info in sample_files:
            # Basic validation: check if file has required fields
            if all(key in file_info for key in ['id', 'name', 'size']):
                validation_results["valid_files"] += 1
            else:
                validation_results["invalid_files"] += 1
                validation_results["errors"].append(f"Invalid file info: {file_info['name']}")
        
        validation_results["valid"] = validation_results["invalid_files"] == 0
        return validation_results