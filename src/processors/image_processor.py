"""Image Processor module."""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms as transforms
from typing import Union, Tuple, Optional, List, Dict, Any
from enum import Enum

from loguru import logger

class ImageFormat(Enum):
    """Supported image formats"""
    RGB = "RGB"
    BGR = "BGR"
    GRAY = "L"
    RGBA = "RGBA"

class ResizeMode(Enum):
    """Image resize modes"""
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"
    ANTIALIAS = "antialias"

class ImageProcessor:
    """Comprehensive image processing pipeline for computer vision tasks"""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (256, 256),
                 output_format: ImageFormat = ImageFormat.RGB,
                 normalize: bool = True,
                 resize_mode: ResizeMode = ResizeMode.BILINEAR,
                 quality_threshold: float = 0.1,
                 enable_enhancement: bool = False):
        
        self.target_size = target_size
        self.output_format = output_format
        self.normalize = normalize
        self.resize_mode = resize_mode
        self.quality_threshold = quality_threshold
        self.enable_enhancement = enable_enhancement
        
        # Standard normalization parameters
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        
        # Setup transforms
        self._setup_transforms()
        
        # Statistics
        self.stats = {
            'images_processed': 0,
            'total_processing_time': 0.0,
            'format_conversions': 0,
            'resizes': 0,
            'enhancements': 0,
            'errors': 0
        }
    
    def _setup_transforms(self):
        """Setup torchvision transforms"""
        transform_list = []
        
        # Resize
        if self.resize_mode == ResizeMode.NEAREST:
            interpolation = Image.NEAREST
        elif self.resize_mode == ResizeMode.BILINEAR:
            interpolation = Image.BILINEAR
        elif self.resize_mode == ResizeMode.BICUBIC:
            interpolation = Image.BICUBIC
        elif self.resize_mode == ResizeMode.LANCZOS:
            interpolation = Image.LANCZOS
        else:
            interpolation = Image.ANTIALIAS
        
        transform_list.append(transforms.Resize(self.target_size, interpolation=interpolation))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalization
        if self.normalize:
            transform_list.append(transforms.Normalize(mean=self.norm_mean, std=self.norm_std))
        
        self.transform = transforms.Compose(transform_list)
    
    def _validate_image(self, image: Union[np.ndarray, Image.Image]) -> bool:
        """Validate image quality and format"""
        if isinstance(image, np.ndarray):
            if image.size == 0:
                return False
            if len(image.shape) not in [2, 3]:
                return False
            if image.shape[0] < 32 or image.shape[1] < 32:  # Too small
                return False
        elif isinstance(image, Image.Image):
            if image.size[0] < 32 or image.size[1] < 32:  # Too small
                return False
        else:
            return False
        
        return True
    
    def _convert_format(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Convert image to PIL Image with correct format"""
        if isinstance(image, np.ndarray):
            # Handle different numpy array formats
            if len(image.shape) == 2:
                # Grayscale
                pil_image = Image.fromarray(image, mode='L')
            elif len(image.shape) == 3:
                if image.shape[2] == 3:
                    # RGB or BGR
                    if image.dtype == np.uint8:
                        pil_image = Image.fromarray(image, mode='RGB')
                    else:
                        # Normalize to 0-255 range
                        normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                        pil_image = Image.fromarray(normalized, mode='RGB')
                elif image.shape[2] == 4:
                    # RGBA
                    pil_image = Image.fromarray(image, mode='RGBA')
                else:
                    raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        else:
            pil_image = image
        
        # Convert to target format
        if self.output_format == ImageFormat.RGB and pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        elif self.output_format == ImageFormat.GRAY and pil_image.mode != 'L':
            pil_image = pil_image.convert('L')
        elif self.output_format == ImageFormat.RGBA and pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')
        
        self.stats['format_conversions'] += 1
        return pil_image
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply image enhancements"""
        if not self.enable_enhancement:
            return image
        
        try:
            # Auto-adjust contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
            
            # Auto-adjust sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.05)
            
            # Reduce noise with slight blur
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            self.stats['enhancements'] += 1
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
        
        return image
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Intelligent image resizing"""
        original_size = image.size
        
        if original_size == self.target_size:
            return image
        
        # Calculate aspect ratio
        aspect_ratio = original_size[0] / original_size[1]
        target_aspect = self.target_size[0] / self.target_size[1]
        
        # Smart resizing to maintain aspect ratio
        if abs(aspect_ratio - target_aspect) < 0.1:
            # Aspect ratios are similar, direct resize
            resized = image.resize(self.target_size, Image.ANTIALIAS)
        else:
            # Significant aspect ratio difference, use padding or cropping
            if aspect_ratio > target_aspect:
                # Image is wider, crop horizontally
                new_width = int(original_size[1] * target_aspect)
                left = (original_size[0] - new_width) // 2
                image = image.crop((left, 0, left + new_width, original_size[1]))
            else:
                # Image is taller, crop vertically
                new_height = int(original_size[0] / target_aspect)
                top = (original_size[1] - new_height) // 2
                image = image.crop((0, top, original_size[0], top + new_height))
            
            resized = image.resize(self.target_size, Image.ANTIALIAS)
        
        self.stats['resizes'] += 1
        return resized
    
    def process(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """Main processing pipeline"""
        import time
        start_time = time.time()
        
        try:
            # Validate input
            if not self._validate_image(image):
                raise ValueError("Invalid image format or quality")
            
            # Convert to PIL Image
            pil_image = self._convert_format(image)
            
            # Enhance if enabled
            if self.enable_enhancement:
                pil_image = self._enhance_image(pil_image)
            
            # Resize to target size
            pil_image = self._resize_image(pil_image)
            
            # Apply transforms (resize, normalize, convert to tensor)
            tensor = self.transform(pil_image)
            
            # Update statistics
            self.stats['images_processed'] += 1
            self.stats['total_processing_time'] += time.time() - start_time
            
            return tensor
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Image processing error: {e}")
            # Return zero tensor as fallback
            if self.output_format == ImageFormat.GRAY:
                return torch.zeros(1, *self.target_size)
            else:
                return torch.zeros(3, *self.target_size)
    
    def batch_process(self, images: List[Union[np.ndarray, Image.Image]]) -> torch.Tensor:
        """Process a batch of images"""
        processed_images = []
        
        for image in images:
            try:
                processed = self.process(image)
                processed_images.append(processed)
            except Exception as e:
                logger.error(f"Failed to process image in batch: {e}")
                # Add zero tensor for failed image
                if self.output_format == ImageFormat.GRAY:
                    processed_images.append(torch.zeros(1, *self.target_size))
                else:
                    processed_images.append(torch.zeros(3, *self.target_size))
        
        if processed_images:
            return torch.stack(processed_images)
        else:
            # Return empty batch
            if self.output_format == ImageFormat.GRAY:
                return torch.empty(0, 1, *self.target_size)
            else:
                return torch.empty(0, 3, *self.target_size)
    
    def preprocess_for_model(self, image: Union[np.ndarray, Image.Image], 
                           model_input_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Preprocess image specifically for model input"""
        original_target_size = self.target_size
        
        # Temporarily change target size if specified
        if model_input_size:
            self.target_size = model_input_size
            self._setup_transforms()
        
        try:
            tensor = self.process(image)
            return tensor.unsqueeze(0)  # Add batch dimension
        finally:
            # Restore original target size
            if model_input_size:
                self.target_size = original_target_size
                self._setup_transforms()
                
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.stats.copy()
        if stats['images_processed'] > 0:
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['images_processed']
        else:
            stats['avg_processing_time'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.stats = {
            'images_processed': 0,
            'total_processing_time': 0.0,
            'format_conversions': 0,
            'resizes': 0,
            'enhancements': 0,
            'errors': 0
        }
    
    def update_config(self, **kwargs):
        """Update processor configuration"""
        config_changed = False
        
        for key, value in kwargs.items():
            if hasattr(self, key) and getattr(self, key) != value:
                setattr(self, key, value)
                config_changed = True
        
        # Rebuild transforms if config changed
        if config_changed:
            self._setup_transforms()
    
    def save_processed_image(self, tensor: torch.Tensor, output_path: str):
        """Save processed tensor as image file"""
        try:
            # Convert tensor back to PIL Image
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)  # Remove batch dimension
            
            # Denormalize if normalized
            if self.normalize:
                for i in range(tensor.size(0)):
                    tensor[i] = tensor[i] * self.norm_std[i] + self.norm_mean[i]
            
            # Convert to numpy array
            if tensor.size(0) == 1:  # Grayscale
                array = tensor.squeeze(0).numpy()
                array = (array * 255).astype(np.uint8)
                image = Image.fromarray(array, mode='L')
            else:  # RGB
                array = tensor.permute(1, 2, 0).numpy()
                array = (array * 255).astype(np.uint8)
                image = Image.fromarray(array, mode='RGB')
            
            image.save(output_path)
            logger.info(f"Saved processed image to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
    
    def __repr__(self) -> str:
        return (f"ImageProcessor("
                f"target_size={self.target_size}, "
                f"output_format={self.output_format.value}, "
                f"resize_mode={self.resize_mode.value}, "
                f"normalize={self.normalize}, "
                f"enable_enhancement={self.enable_enhancement})")
    
    def __call__(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """Make processor callable"""
        return self.process(image)