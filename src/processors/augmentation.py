"""Augmentation module."""

"""
Advanced image augmentation pipeline for vision-data-toolkit
"""

import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from enum import Enum
import math

from loguru import logger


class AugmentationType(Enum):
    """Types of augmentation"""
    GEOMETRIC = "geometric"
    PHOTOMETRIC = "photometric"
    NOISE = "noise"
    BLUR = "blur"
    DISTORTION = "distortion"
    CUTOUT = "cutout"
    MIXUP = "mixup"


class AugmentationSeverity(Enum):
    """Augmentation severity levels"""
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    EXTREME = "extreme"


class AugmentationConfig:
    """Configuration for augmentation pipeline"""
    
    def __init__(self,
                 probability: float = 0.5,
                 severity: AugmentationSeverity = AugmentationSeverity.MODERATE,
                 preserve_aspect_ratio: bool = True,
                 maintain_image_bounds: bool = True,
                 random_seed: Optional[int] = None):
        
        self.probability = probability
        self.severity = severity
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.maintain_image_bounds = maintain_image_bounds
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)


class GeometricAugmentation:
    """Geometric transformations"""
    
    @staticmethod
    def rotate(image: Image.Image, angle_range: Tuple[float, float] = (-30, 30)) -> Image.Image:
        """Random rotation"""
        angle = random.uniform(*angle_range)
        return image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=0)
    
    @staticmethod
    def horizontal_flip(image: Image.Image, probability: float = 0.5) -> Image.Image:
        """Random horizontal flip"""
        if random.random() < probability:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image
    
    @staticmethod
    def vertical_flip(image: Image.Image, probability: float = 0.5) -> Image.Image:
        """Random vertical flip"""
        if random.random() < probability:
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        return image
    
    @staticmethod
    def translate(image: Image.Image, 
                 translate_range: Tuple[float, float] = (-0.1, 0.1)) -> Image.Image:
        """Random translation"""
        width, height = image.size
        dx = int(random.uniform(*translate_range) * width)
        dy = int(random.uniform(*translate_range) * height)
        
        # Create translation matrix
        return image.transform(
            image.size, Image.AFFINE, (1, 0, dx, 0, 1, dy),
            resample=Image.BILINEAR, fillcolor=0
        )
    
    @staticmethod
    def scale(image: Image.Image, 
              scale_range: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
        """Random scaling"""
        scale_factor = random.uniform(*scale_range)
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Resize and then crop/pad to original size
        scaled = image.resize((new_width, new_height), Image.BILINEAR)
        
        if scale_factor > 1.0:
            # Crop center
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            return scaled.crop((left, top, left + width, top + height))
        else:
            # Pad
            new_image = Image.new(image.mode, (width, height), 0)
            paste_x = (width - new_width) // 2
            paste_y = (height - new_height) // 2
            new_image.paste(scaled, (paste_x, paste_y))
            return new_image
    
    @staticmethod
    def shear(image: Image.Image, 
              shear_range: Tuple[float, float] = (-0.2, 0.2)) -> Image.Image:
        """Random shear transformation"""
        shear_x = random.uniform(*shear_range)
        shear_y = random.uniform(*shear_range)
        
        return image.transform(
            image.size, Image.AFFINE, 
            (1, shear_x, 0, shear_y, 1, 0),
            resample=Image.BILINEAR, fillcolor=0
        )


class PhotometricAugmentation:
    """Photometric transformations"""
    
    @staticmethod
    def brightness(image: Image.Image, 
                   brightness_range: Tuple[float, float] = (0.7, 1.3)) -> Image.Image:
        """Random brightness adjustment"""
        factor = random.uniform(*brightness_range)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def contrast(image: Image.Image, 
                 contrast_range: Tuple[float, float] = (0.7, 1.3)) -> Image.Image:
        """Random contrast adjustment"""
        factor = random.uniform(*contrast_range)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def saturation(image: Image.Image, 
                   saturation_range: Tuple[float, float] = (0.7, 1.3)) -> Image.Image:
        """Random saturation adjustment"""
        if image.mode not in ['RGB', 'RGBA']:
            return image
        factor = random.uniform(*saturation_range)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def hue(image: Image.Image, 
            hue_range: Tuple[float, float] = (-0.1, 0.1)) -> Image.Image:
        """Random hue shift"""
        if image.mode not in ['RGB', 'RGBA']:
            return image
        
        hue_shift = random.uniform(*hue_range)
        
        # Convert to numpy for HSV manipulation
        img_array = np.array(image)
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Convert RGB to HSV
            from colorsys import rgb_to_hsv, hsv_to_rgb
            
            # Normalize to 0-1
            img_float = img_array.astype(np.float32) / 255.0
            
            # Apply hue shift to each pixel
            for i in range(img_float.shape[0]):
                for j in range(img_float.shape[1]):
                    r, g, b = img_float[i, j]
                    h, s, v = rgb_to_hsv(r, g, b)
                    h = (h + hue_shift) % 1.0  # Keep hue in [0, 1]
                    r, g, b = hsv_to_rgb(h, s, v)
                    img_float[i, j] = [r, g, b]
            
            # Convert back to PIL
            img_array = (img_float * 255).astype(np.uint8)
            return Image.fromarray(img_array, mode='RGB')
        
        return image
    
    @staticmethod
    def gamma_correction(image: Image.Image, 
                        gamma_range: Tuple[float, float] = (0.7, 1.3)) -> Image.Image:
        """Random gamma correction"""
        gamma = random.uniform(*gamma_range)
        
        # Create gamma lookup table
        gamma_table = [int(((i / 255.0) ** (1.0 / gamma)) * 255) for i in range(256)]
        
        if image.mode == 'RGB':
            return image.point(gamma_table * 3)
        elif image.mode == 'L':
            return image.point(gamma_table)
        
        return image


class NoiseAugmentation:
    """Noise-based augmentations"""
    
    @staticmethod
    def gaussian_noise(image: Image.Image, 
                      noise_std: float = 0.1) -> Image.Image:
        """Add Gaussian noise"""
        img_array = np.array(image).astype(np.float32)
        noise = np.random.normal(0, noise_std * 255, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array, mode=image.mode)
    
    @staticmethod
    def salt_pepper_noise(image: Image.Image, 
                         noise_ratio: float = 0.05) -> Image.Image:
        """Add salt and pepper noise"""
        img_array = np.array(image)
        
        # Salt noise
        salt_mask = np.random.random(img_array.shape[:2]) < noise_ratio / 2
        img_array[salt_mask] = 255
        
        # Pepper noise
        pepper_mask = np.random.random(img_array.shape[:2]) < noise_ratio / 2
        img_array[pepper_mask] = 0
        
        return Image.fromarray(img_array, mode=image.mode)
    
    @staticmethod
    def speckle_noise(image: Image.Image, 
                     noise_variance: float = 0.1) -> Image.Image:
        """Add speckle noise"""
        img_array = np.array(image).astype(np.float32)
        noise = np.random.normal(1, noise_variance, img_array.shape)
        noisy_array = np.clip(img_array * noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array, mode=image.mode)


class BlurAugmentation:
    """Blur-based augmentations"""
    
    @staticmethod
    def gaussian_blur(image: Image.Image, 
                     radius_range: Tuple[float, float] = (0.5, 2.0)) -> Image.Image:
        """Apply Gaussian blur"""
        radius = random.uniform(*radius_range)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    @staticmethod
    def motion_blur(image: Image.Image, 
                   kernel_size: int = 9) -> Image.Image:
        """Apply motion blur"""
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Apply kernel (simplified version)
        img_array = np.array(image).astype(np.float32)
        if len(img_array.shape) == 3:
            for c in range(img_array.shape[2]):
                img_array[:, :, c] = np.convolve(img_array[:, :, c].flatten(), 
                                               kernel.flatten(), mode='same').reshape(img_array[:, :, c].shape)
        else:
            img_array = np.convolve(img_array.flatten(), 
                                  kernel.flatten(), mode='same').reshape(img_array.shape)
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8), mode=image.mode)


class CutoutAugmentation:
    """Cutout augmentation"""
    
    @staticmethod
    def cutout(image: Image.Image, 
              num_holes: int = 1,
              hole_size_ratio: float = 0.1) -> Image.Image:
        """Apply cutout augmentation"""
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        hole_size = int(min(height, width) * hole_size_ratio)
        
        for _ in range(num_holes):
            # Random position for hole
            y = random.randint(0, height - hole_size)
            x = random.randint(0, width - hole_size)
            
            # Cut out the hole
            img_array[y:y+hole_size, x:x+hole_size] = 0
        
        return Image.fromarray(img_array, mode=image.mode)


class AugmentationPipeline:
    """Main augmentation pipeline"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        
        # Define severity parameters
        self.severity_params = {
            AugmentationSeverity.LIGHT: {
                'rotation_range': (-10, 10),
                'brightness_range': (0.9, 1.1),
                'contrast_range': (0.9, 1.1),
                'noise_std': 0.05,
                'blur_radius': (0.1, 0.5)
            },
            AugmentationSeverity.MODERATE: {
                'rotation_range': (-30, 30),
                'brightness_range': (0.7, 1.3),
                'contrast_range': (0.7, 1.3),
                'noise_std': 0.1,
                'blur_radius': (0.5, 2.0)
            },
            AugmentationSeverity.HEAVY: {
                'rotation_range': (-45, 45),
                'brightness_range': (0.5, 1.5),
                'contrast_range': (0.5, 1.5),
                'noise_std': 0.2,
                'blur_radius': (1.0, 3.0)
            },
            AugmentationSeverity.EXTREME: {
                'rotation_range': (-90, 90),
                'brightness_range': (0.3, 1.7),
                'contrast_range': (0.3, 1.7),
                'noise_std': 0.3,
                'blur_radius': (2.0, 5.0)
            }
        }
        
        # Initialize augmentation methods
        self.geometric = GeometricAugmentation()
        self.photometric = PhotometricAugmentation()
        self.noise = NoiseAugmentation()
        self.blur = BlurAugmentation()
        self.cutout = CutoutAugmentation()
    
    def _should_apply(self) -> bool:
        """Check if augmentation should be applied"""
        return random.random() < self.config.probability
    
    def _get_severity_params(self) -> Dict[str, Any]:
        """Get parameters for current severity level"""
        return self.severity_params[self.config.severity]
    
    def apply_geometric(self, image: Image.Image) -> Image.Image:
        """Apply geometric augmentations"""
        if not self._should_apply():
            return image
        
        params = self._get_severity_params()
        
        # Apply random geometric transformations
        augmentations = [
            lambda img: self.geometric.rotate(img, params['rotation_range']),
            lambda img: self.geometric.horizontal_flip(img, 0.5),
            lambda img: self.geometric.vertical_flip(img, 0.3),
            lambda img: self.geometric.translate(img, (-0.1, 0.1)),
            lambda img: self.geometric.scale(img, (0.8, 1.2)),
            lambda img: self.geometric.shear(img, (-0.1, 0.1))
        ]
        
        # Apply 1-3 random augmentations
        num_augs = random.randint(1, 3)
        selected_augs = random.sample(augmentations, num_augs)
        
        for aug in selected_augs:
            image = aug(image)
        
        return image
    
    def apply_photometric(self, image: Image.Image) -> Image.Image:
        """Apply photometric augmentations"""
        if not self._should_apply():
            return image
        
        params = self._get_severity_params()
        
        # Apply random photometric transformations
        augmentations = [
            lambda img: self.photometric.brightness(img, params['brightness_range']),
            lambda img: self.photometric.contrast(img, params['contrast_range']),
            lambda img: self.photometric.saturation(img, (0.7, 1.3)),
            lambda img: self.photometric.hue(img, (-0.05, 0.05)),
            lambda img: self.photometric.gamma_correction(img, (0.8, 1.2))
        ]
        
        # Apply 1-3 random augmentations
        num_augs = random.randint(1, 3)
        selected_augs = random.sample(augmentations, num_augs)
        
        for aug in selected_augs:
            image = aug(image)
        
        return image
    
    def apply_noise(self, image: Image.Image) -> Image.Image:
        """Apply noise augmentations"""
        if not self._should_apply():
            return image
        
        params = self._get_severity_params()
        
        # Choose random noise type
        noise_types = [
            lambda img: self.noise.gaussian_noise(img, params['noise_std']),
            lambda img: self.noise.salt_pepper_noise(img, 0.02),
            lambda img: self.noise.speckle_noise(img, params['noise_std'])
        ]
        
        noise_aug = random.choice(noise_types)
        return noise_aug(image)
    
    def apply_blur(self, image: Image.Image) -> Image.Image:
        """Apply blur augmentations"""
        if not self._should_apply():
            return image
        
        params = self._get_severity_params()
        
        # Choose random blur type
        blur_types = [
            lambda img: self.blur.gaussian_blur(img, params['blur_radius']),
            lambda img: self.blur.motion_blur(img, random.randint(3, 7))
        ]
        
        blur_aug = random.choice(blur_types)
        return blur_aug(image)
    
    def apply_cutout(self, image: Image.Image) -> Image.Image:
        """Apply cutout augmentation"""
        if not self._should_apply():
            return image
        
        return self.cutout.cutout(image, 
                                num_holes=random.randint(1, 3),
                                hole_size_ratio=random.uniform(0.05, 0.15))
    
    def __call__(self, image: Image.Image, 
                 augmentation_types: Optional[List[AugmentationType]] = None) -> Image.Image:
        """Apply augmentation pipeline"""
        if augmentation_types is None:
            augmentation_types = [
                AugmentationType.GEOMETRIC,
                AugmentationType.PHOTOMETRIC,
                AugmentationType.NOISE,
                AugmentationType.BLUR,
                AugmentationType.CUTOUT
            ]
        
        try:
            original_image = image.copy()
            
            for aug_type in augmentation_types:
                try:
                    if aug_type == AugmentationType.GEOMETRIC:
                        image = self.apply_geometric(image)
                    elif aug_type == AugmentationType.PHOTOMETRIC:
                        image = self.apply_photometric(image)
                    elif aug_type == AugmentationType.NOISE:
                        image = self.apply_noise(image)
                    elif aug_type == AugmentationType.BLUR:
                        image = self.apply_blur(image)
                    elif aug_type == AugmentationType.CUTOUT:
                        image = self.apply_cutout(image)
                except Exception as e:
                    logger.warning(f"Failed to apply {aug_type.value} augmentation: {e}")
                    continue
            
            return image
            
        except Exception as e:
            logger.error(f"Augmentation pipeline failed: {e}")
            return original_image


# Convenience function for quick augmentation
def augment_image(image: Union[Image.Image, np.ndarray],
                 severity: AugmentationSeverity = AugmentationSeverity.MODERATE,
                 probability: float = 0.5,
                 augmentation_types: Optional[List[AugmentationType]] = None) -> Image.Image:
    """Quick image augmentation function"""
    
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = Image.fromarray(image, mode='RGB')
        elif len(image.shape) == 2:
            image = Image.fromarray(image, mode='L')
        else:
            raise ValueError(f"Unsupported image array shape: {image.shape}")
    
    config = AugmentationConfig(probability=probability, severity=severity)
    pipeline = AugmentationPipeline(config)
    
    return pipeline(image, augmentation_types)