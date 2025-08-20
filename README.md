
# Vision Data Toolkit

A professional, reusable toolkit for handling large-scale image datasets with cloud storage integration, streaming capabilities, and intelligent caching. Designed for computer vision and multimodal AI research.

## 🚀 Features

- **☁️ Cloud Integration**: Seamless Google Drive and AWS S3 support
- **📡 Streaming Data Loading**: Load massive datasets without local storage
- **💾 Intelligent Caching**: LRU cache with automatic cleanup and size management
- **🔄 Data Processing**: Comprehensive image preprocessing and augmentation pipelines
- **📊 Dataset Support**: Built-in support for SET5, URBAN100, Manga109, and custom datasets
- **⚡ Performance Optimized**: Multi-threaded downloading and prefetching
- **🔧 Professional API**: Clean PyTorch integration for seamless workflow

## 📋 Requirements

- Python 3.11+
- PyTorch 1.13+
- Google Drive API credentials (for cloud storage)
- 5GB+ available disk space (for caching)

## 🛠️ Installation

### 1. Clone and Install
```bash
git clone https://github.com/randikapra/vision-data-toolkit.git
cd vision-data-toolkit
pip install -e .
```

### 2. Setup Environment
```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Configure Google Drive (Optional)
```bash
python scripts/setup_gdrive.py
```

## 🚀 Quick Start

### Basic Usage
```python
from vision_data_toolkit import StreamingDataset
from torch.utils.data import DataLoader

# Stream data from Google Drive
dataset = StreamingDataset(
    cloud_path="gdrive://your-folder-id/",
    cache_size="5GB",
    preprocessing=["resize", "normalize"]
)

# Use with PyTorch DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    # Your training code here
    pass
```

### Using Built-in Datasets
```python
from vision_data_toolkit.datasets import SET5Dataset, URBAN100Dataset

# Load standard datasets
set5 = SET5Dataset(
    cloud_path="gdrive://set5-folder-id/",
    preprocessing=["resize", "normalize"]
)

urban100 = URBAN100Dataset(
    cloud_path="gdrive://urban100-folder-id/",
    cache_size="2GB"
)
```

### Advanced Configuration
```python
from vision_data_toolkit import StreamingDataset
from vision_data_toolkit.cache import LRUCache
from vision_data_toolkit.processors import ImageProcessor

# Custom preprocessing pipeline
processor = ImageProcessor([
    ("resize", (224, 224)),
    ("normalize", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}),
    ("augment", {"rotation": 15, "flip": True})
])

# Advanced dataset configuration
dataset = StreamingDataset(
    cloud_path="gdrive://my-dataset/",
    cache_size="10GB",
    cache_dir="./custom_cache",
    processor=processor,
    prefetch_size=4,
    max_workers=8
)
```

## 📚 Supported Datasets

| Dataset | Images | Description | Size |
|---------|--------|-------------|------|
| SET5 | 5 | Super-resolution benchmark | ~10MB |
| URBAN100 | 100 | Urban scenes for SR | ~50MB |
| Manga109 | 109 | Japanese manga images | ~2GB |
| Custom | Variable | Your own datasets | Variable |

## 🏗️ Architecture

```
vision-data-toolkit/
├── src/
│   ├── cloud/          # Cloud storage managers
│   ├── streaming/      # Data streaming and loading
│   ├── cache/          # Caching mechanisms
│   ├── processors/     # Image processing pipelines
│   └── utils/          # Utilities and monitoring
├── datasets/           # Dataset-specific handlers
├── scripts/           # Setup and maintenance scripts
└── examples/          # Usage examples
```

## 💾 Storage Management

### Cache Configuration
```python
from vision_data_toolkit.cache import LRUCache

cache = LRUCache(
    max_size="5GB",              # Maximum cache size
    cache_dir="./cache",         # Cache directory
    cleanup_threshold=0.9,       # Cleanup when 90% full
    max_age_hours=168           # Remove files older than 1 week
)
```

### Storage Monitoring
```python
from vision_data_toolkit.utils import StorageMonitor

monitor = StorageMonitor()
monitor.start_monitoring()          # Background monitoring
monitor.get_usage_stats()          # Current usage statistics
monitor.cleanup_old_files()        # Manual cleanup
```

## 🌥️ Cloud Storage Setup

### Google Drive
1. Create Google Cloud Project
2. Enable Google Drive API
3. Create OAuth2 credentials
4. Run setup script:
```bash
python scripts/setup_gdrive.py
```

### AWS S3 (Future Support)
```python
from vision_data_toolkit.cloud import S3Manager

s3 = S3Manager(
    bucket_name="my-dataset-bucket",
    region="us-east-1"
)
```

## 📊 Performance Tips

### Optimization Strategies
- **Cache Size**: Set cache to 20-30% of dataset size for optimal performance
- **Prefetching**: Use 2-4 worker threads for downloading
- **Batch Size**: Larger batches reduce I/O overhead
- **Storage**: Use SSD for cache directory when possible

### Example Configuration for Large Datasets
```python
# For 100GB+ datasets
dataset = StreamingDataset(
    cloud_path="gdrive://large-dataset/",
    cache_size="20GB",           # 20% of dataset
    prefetch_size=8,             # Aggressive prefetching
    max_workers=6,               # More download threads
    cleanup_policy="aggressive"   # Frequent cleanup
)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/test_streaming/

# Run with coverage
pytest --cov=src tests/
```

## 📖 Documentation

- [Setup Guide](docs/setup_guide.md) - Detailed installation and configuration
- [API Reference](docs/api_reference.md) - Complete API documentation
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions
- [Examples](examples/) - Comprehensive usage examples

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Google Drive API for cloud storage capabilities
- Open source computer vision community

## 📞 Support

- 📧 Email: research@university.edu
- 🐛 Issues: [GitHub Issues](https://github.com/randikapra/vision-data-toolkit/issues)
- 📖 Documentation: [Project Wiki](https://github.com/randikapra/vision-data-toolkit/wiki)

---

**Made with ❤️ for the computer vision research community**
