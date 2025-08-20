"""Dataset Config module."""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

class DatasetType(Enum):
    SUPER_RESOLUTION = "super_resolution"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"

@dataclass
class DatasetSpec:
    """Dataset specification"""
    name: str
    dataset_type: DatasetType
    gdrive_folder_id: str
    total_images: int
    estimated_size_gb: float
    supported_formats: List[str]
    default_resolution: Tuple[int, int]
    has_labels: bool = True
    has_annotations: bool = False
    preprocessing_required: bool = True
    description: str = ""
    
@dataclass
class ProcessingConfig:
    """Image processing configuration"""
    default_image_size: int = 256
    default_batch_size: int = 32
    supported_formats: List[str] = None
    compression_quality: int = 95
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    max_image_size: int = 2048
    min_image_size: int = 32
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']

class DatasetRegistry:
    """Registry for all supported datasets"""
    
    def __init__(self):
        self.datasets: Dict[str, DatasetSpec] = {}
        self.processing = ProcessingConfig()
        self._register_standard_datasets()
    
    def _register_standard_datasets(self):
        """Register standard computer vision datasets"""
        
        # Super Resolution Datasets
        self.register_dataset(DatasetSpec(
            name="SET5",
            dataset_type=DatasetType.SUPER_RESOLUTION,
            gdrive_folder_id="",  # To be filled
            total_images=5,
            estimated_size_gb=0.001,
            supported_formats=['png', 'bmp'],
            default_resolution=(256, 256),
            has_labels=False,
            description="Standard 5-image super-resolution test set"
        ))
        
        self.register_dataset(DatasetSpec(
            name="URBAN100",
            dataset_type=DatasetType.SUPER_RESOLUTION,
            gdrive_folder_id="",  # To be filled
            total_images=100,
            estimated_size_gb=0.1,
            supported_formats=['png'],
            default_resolution=(512, 512),
            has_labels=False,
            description="Urban scenes for super-resolution evaluation"
        ))
        
        self.register_dataset(DatasetSpec(
            name="MANGA109",
            dataset_type=DatasetType.SUPER_RESOLUTION,
            gdrive_folder_id="",  # To be filled
            total_images=109,
            estimated_size_gb=0.2,
            supported_formats=['png'],
            default_resolution=(1024, 1024),
            has_labels=False,
            description="Manga images for super-resolution"
        ))
        
        self.register_dataset(DatasetSpec(
            name="DIV2K",
            dataset_type=DatasetType.SUPER_RESOLUTION,
            gdrive_folder_id="",  # To be filled
            total_images=1000,
            estimated_size_gb=5.0,
            supported_formats=['png'],
            default_resolution=(2048, 1080),
            has_labels=False,
            description="High-quality images for super-resolution training"
        ))
        
        # Classification Datasets
        self.register_dataset(DatasetSpec(
            name="IMAGENET_SUBSET",
            dataset_type=DatasetType.CLASSIFICATION,
            gdrive_folder_id="",  # To be filled
            total_images=50000,
            estimated_size_gb=15.0,
            supported_formats=['jpg', 'jpeg'],
            default_resolution=(224, 224),
            has_labels=True,
            description="ImageNet subset for classification experiments"
        ))
        
        # Multimodal Datasets
        self.register_dataset(DatasetSpec(
            name="CUSTOM_MULTIMODAL",
            dataset_type=DatasetType.MULTIMODAL,
            gdrive_folder_id="",  # To be filled
            total_images=0,  # Variable
            estimated_size_gb=0.0,  # Variable
            supported_formats=['jpg', 'jpeg', 'png'],
            default_resolution=(512, 512),
            has_labels=True,
            has_annotations=True,
            description="Custom multimodal dataset for agent research"
        ))
    
    def register_dataset(self, dataset_spec: DatasetSpec):
        """Register a new dataset"""
        self.datasets[dataset_spec.name] = dataset_spec
    
    def get_dataset(self, name: str) -> Optional[DatasetSpec]:
        """Get dataset specification by name"""
        return self.datasets.get(name.upper())
    
    def list_datasets(self) -> List[str]:
        """List all registered datasets"""
        return list(self.datasets.keys())
    
    def get_datasets_by_type(self, dataset_type: DatasetType) -> List[DatasetSpec]:
        """Get all datasets of a specific type"""
        return [spec for spec in self.datasets.values() if spec.dataset_type == dataset_type]
    
    def estimate_total_size(self, dataset_names: List[str]) -> float:
        """Estimate total size for multiple datasets"""
        total_size = 0.0
        for name in dataset_names:
            dataset = self.get_dataset(name)
            if dataset:
                total_size += dataset.estimated_size_gb
        return total_size
    
    def get_processing_config(self, dataset_name: str) -> ProcessingConfig:
        """Get processing configuration for specific dataset"""
        dataset = self.get_dataset(dataset_name)
        if dataset:
            # Create custom processing config based on dataset
            config = ProcessingConfig()
            config.default_image_size = min(dataset.default_resolution)
            config.supported_formats = dataset.supported_formats
            return config
        return self.processing
    
    def update_gdrive_folder_id(self, dataset_name: str, folder_id: str):
        """Update Google Drive folder ID for a dataset"""
        dataset = self.get_dataset(dataset_name)
        if dataset:
            dataset.gdrive_folder_id = folder_id
    
    def validate_dataset_config(self, dataset_name: str) -> Dict[str, bool]:
        """Validate dataset configuration"""
        dataset = self.get_dataset(dataset_name)
        if not dataset:
            return {"exists": False}
        
        return {
            "exists": True,
            "has_gdrive_id": bool(dataset.gdrive_folder_id),
            "has_valid_formats": len(dataset.supported_formats) > 0,
            "has_valid_resolution": all(r > 0 for r in dataset.default_resolution),
            "size_reasonable": 0 < dataset.estimated_size_gb < 1000
        }

# Global dataset registry instance
dataset_registry = DatasetRegistry()

# Environment-based configuration
DATASET_CONFIG = {
    'DEFAULT_IMAGE_SIZE': int(os.getenv('DEFAULT_IMAGE_SIZE', '256')),
    'DEFAULT_BATCH_SIZE': int(os.getenv('DEFAULT_BATCH_SIZE', '32')),
    'SUPPORTED_FORMATS': os.getenv('SUPPORTED_FORMATS', 'jpg,jpeg,png,bmp,tiff').split(','),
    'COMPRESSION_QUALITY': int(os.getenv('COMPRESSION_QUALITY', '95')),
}