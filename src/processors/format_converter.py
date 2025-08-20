"""Format Converter module."""

"""
Image format converter for vision-data-toolkit
"""

import os
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple
from enum import Enum
import numpy as np
from PIL import Image, ImageOps
import cv2
import io

from loguru import logger


class ImageFormat(Enum):
    """Supported image formats"""
    JPEG = "JPEG"
    PNG = "PNG"
    WEBP = "WEBP"
    TIFF = "TIFF"
    BMP = "BMP"
    GIF = "GIF"
    ICO = "ICO"


class ColorMode(Enum):
    """Color modes"""
    RGB = "RGB"
    RGBA = "RGBA"
    GRAYSCALE = "L"
    CMYK = "CMYK"
    LAB = "LAB"
    HSV = "HSV"


class CompressionLevel(Enum):
    """Compression quality levels"""
    LOW = 30
    MEDIUM = 60
    HIGH = 85
    MAXIMUM = 95


class FormatConverter:
    """Comprehensive image format converter"""
    
    def __init__(self):
        # Format-specific default settings
        self.format_settings = {
            ImageFormat.JPEG: {
                'quality': 90,
                'optimize': True,
                'progressive': False
            },
            ImageFormat.PNG: {
                'optimize': True,
                'compress_level': 6
            },
            ImageFormat.WEBP: {
                'quality': 90,
                'lossless': False,
                'method': 4
            },
            ImageFormat.TIFF: {
                'compression': 'lzw'
            },
            ImageFormat.BMP: {},
            ImageFormat.GIF: {
                'optimize': True
            },
            ImageFormat.ICO: {}
        }
        
        # Supported conversions
        self.supported_inputs = {'.jpg', '.jpeg', '.png', '.webp', '.tiff', '.tif', 
                               '.bmp', '.gif', '.ico', '.ppm', '.pgm', '.pbm'}
        
        logger.info("Format converter initialized")
    
    def detect_format(self, image_path: Union[str, Path]) -> Optional[ImageFormat]:
        """Detect image format from file"""
        try:
            with Image.open(image_path) as img:
                format_name = img.format
                if format_name:
                    try:
                        return ImageFormat(format_name.upper())
                    except ValueError:
                        logger.warning(f"Unsupported format detected: {format_name}")
                        return None
        except Exception as e:
            logger.error(f"Failed to detect format for {image_path}: {e}")
            return None
    
    def get_image_info(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Get comprehensive image information"""
        try:
            path = Path(image_path)
            with Image.open(path) as img:
                info = {
                    'filename': path.name,
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.width,
                    'height': img.height,
                    'file_size': path.stat().st_size,
                    'has_transparency': self._has_transparency(img),
                    'is_animated': getattr(img, 'is_animated', False),
                    'n_frames': getattr(img, 'n_frames', 1)
                }
                
                # Add EXIF data if available
                if hasattr(img, '_getexif') and img._getexif():
                    info['has_exif'] = True
                    try:
                        info['exif'] = dict(img._getexif().items())
                    except:
                        info['exif'] = {}
                else:
                    info['has_exif'] = False
                    info['exif'] = {}
                
                return info
                
        except Exception as e:
            logger.error(f"Failed to get image info for {image_path}: {e}")
            return {}
    
    def _has_transparency(self, image: Image.Image) -> bool:
        """Check if image has transparency"""
        return (
            image.mode in ('RGBA', 'LA') or 
            (image.mode == 'P' and 'transparency' in image.info)
        )
    
    def _prepare_image_for_format(self, image: Image.Image, 
                                 target_format: ImageFormat,
                                 preserve_transparency: bool = True) -> Image.Image:
        """Prepare image for specific format conversion"""
        
        if target_format == ImageFormat.JPEG:
            # JPEG doesn't support transparency
            if image.mode in ('RGBA', 'LA', 'P'):
                if preserve_transparency:
                    # Create white background
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    if image.mode == 'P':
                        image = image.convert('RGBA')
                    background.paste(image, mask=image.split()[-1] if