# GeoGPT - Advanced Geospatial Analysis Assistant

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-API-green.svg)](https://earthengine.google.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🌍 Overview

GeoGPT is an advanced AI-powered geospatial analysis assistant that combines the power of Large Language Models (LLMs) with Google Earth Engine, providing intelligent geospatial data processing, analysis, and visualization capabilities. Built specifically for environmental analysis, flood risk assessment, site suitability analysis, and comprehensive geospatial workflows.

## ✨ Key Features

### 🤖 Dual Agent Architecture
- **Tool Agent**: Executes data extraction, processing, and analysis workflows
- **Reasoning Agent**: Provides conceptual explanations and technical guidance
- **Smart Router**: Automatically routes queries to the appropriate agent

### 🛠 Core Capabilities
- **Data Extraction**: SRTM DEM, NDVI, rainfall, landcover, temperature, precipitation
- **Analysis**: Flood risk assessment, site suitability analysis, environmental modeling
- **Visualization**: Interactive maps with Folium, customizable overlays and styling
- **Real-time Streaming**: Live Chain-of-Thought reasoning and step-by-step execution
- **Multi-format Support**: GeoJSON, GeoTIFF, HTML maps, PNG exports

### 🌐 Data Sources Integration
- Google Earth Engine (Landsat, Sentinel, MODIS, SRTM)
- OpenStreetMap via OSMnx
- ESA WorldCover Land Classification
- CHIRPS Rainfall Data
- ERA5 Climate Reanalysis

## 🏗 Architecture

```
ISRO/
├── agent/                      # Main application directory
│   ├── agent.py               # Tool agent with streaming capabilities
│   ├── workflow_agent.py      # Reasoning agent with LangGraph
│   ├── top_node.py           # Router agent for query classification
│   ├── site.py               # Streamlit web interface
│   ├── geetools.py           # Google Earth Engine tools and functions
│   ├── requirements.txt      # Python dependencies
│   ├── downloads/            # Downloaded geospatial data
│   ├── outputs/              # Generated maps and analysis results
│   └── cache/                # Cached API responses
├── testing_lib/              # Development and testing environment
├── docs/                     # Documentation and API references
├── extract_doc/              # Documentation extraction tools
└── README.md                # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Earth Engine account and authentication
- OpenAI API key (for LLM capabilities)
- Pinecone API key (for vector storage - optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ISRO
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   cd agent
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the `agent/` directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_KEY=your_pinecone_api_key  # Optional
   GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
   ```

5. **Authenticate Google Earth Engine**
   ```bash
   earthengine authenticate
   ```

6. **Run the application**
   ```bash
   streamlit run site.py
   ```

## 💻 Usage Examples

### 1. Flood Risk Analysis
```python
# Query: "Perform flood risk analysis for Uttarakhand with visualization"
# The system will:
# 1. Extract bounding box for Uttarakhand
# 2. Download DEM, rainfall, and landcover data
# 3. Perform weighted analysis (elevation + precipitation + land cover)
# 4. Generate interactive flood risk map
```

### 2. Site Suitability Analysis
```python
# Query: "Analyze solar farm suitability in Bihar using NDVI and slope"
# The system will:
# 1. Extract Bihar boundaries
# 2. Download NDVI and DEM data
# 3. Calculate slope from DEM
# 4. Perform multi-criteria analysis
# 5. Visualize suitable areas
```

### 3. Vegetation Health Assessment
```python
# Query: "Create vegetation health map for Karnataka"
# The system will:
# 1. Download NDVI time series data
# 2. Calculate vegetation indices
# 3. Generate health classification
# 4. Create interactive visualization
```

## 🔧 API Reference

### Core Functions

#### Data Extraction
```python
extract_bbox(location, distance, filepath)          # Extract bounding box
get_srtm_dem(bbox, filepath)                        # Download DEM data
get_ndvi_data(bbox, filepath)                       # Download NDVI data
get_rainfall(bbox, filepath)                        # Download rainfall data
get_landcover(bbox, filepath)                       # Download landcover data
```

#### Analysis Functions
```python
suitability_analysis(
    bbox,
    criteria,
    weights,
    inverse_flags,
    output_path
)                                                   # Multi-criteria analysis

visualise_map(
    raster_path,
    title,
    colormap,
    overlay_paths
)                                                   # Interactive visualization
```

### Agent System

#### Tool Agent
- Executes geospatial workflows
- Handles data processing and analysis
- Generates visualizations and reports

#### Reasoning Agent
- Provides conceptual explanations
- Technical guidance and best practices
- Educational content and methodology

#### Router Agent
- Automatically classifies queries
- Routes to appropriate agent
- Manages streaming responses

## 🎛 Configuration

### Analysis Parameters

#### Flood Risk Analysis
```python
{
    "factors": ["dem", "rainfall", "landcover"],
    "weights": [0.4, 0.4, 0.2],
    "inverse": [True, False, True],
    "description": "Higher elevation and vegetation = lower risk"
}
```

#### Site Suitability Analysis
```python
{
    "factors": ["ndvi", "slope", "distance_to_roads"],
    "weights": [0.5, 0.3, 0.2],
    "inverse": [False, True, True],
    "description": "Higher NDVI, gentle slopes = more suitable"
}
```

### Visualization Settings
```python
{
    "colormap": "RdYlBu_r",      # Matplotlib colormap
    "opacity": 0.7,              # Layer transparency
    "zoom_level": 8,             # Initial zoom
    "tile_layer": "OpenStreetMap" # Base map
}
```

## 📊 Features in Detail

### Real-time Streaming
- **Live Reasoning**: See AI thought process in real-time
- **Step-by-step Execution**: Monitor tool usage and data processing
- **Progress Tracking**: Visual indicators for long-running operations
- **Error Handling**: Graceful error recovery with user feedback

### Interactive Interface
- **Collapsible Sections**: Organized display of reasoning and actions
- **Map Preview**: Integrated map viewer with zoom and pan
- **Download Options**: Export maps and data in multiple formats
- **Session History**: Track previous queries and results

### Advanced Analytics
- **Multi-criteria Decision Analysis**: Weighted factor combination
- **Temporal Analysis**: Time series processing and change detection
- **Spatial Statistics**: Zonal statistics and area calculations
- **Custom Workflows**: Flexible analysis pipeline configuration

## 🧪 Testing and Development

### Testing Environment
The `testing_lib/` directory contains:
- Jupyter notebooks for development
- Test datasets and sample analyses
- Validation scripts and benchmarks
- Documentation generation tools

### Key Test Cases
- Flood risk analysis validation
- Site suitability benchmarking
- Performance optimization tests
- Error handling verification

## 📚 Documentation

### Available Documentation
- **API Reference**: Complete function documentation
- **User Guides**: Step-by-step tutorials
- **Best Practices**: Optimization and methodology guides
- **Examples**: Sample workflows and use cases

### External Resources
- [Google Earth Engine Documentation](https://developers.google.com/earth-engine)
- [Streamlit Documentation](https://docs.streamlit.io)
- [LangChain Documentation](https://python.langchain.com)
- [Folium Documentation](https://python-visualization.github.io/folium)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 🐛 Troubleshooting

### Common Issues

#### Google Earth Engine Authentication
```bash
# Re-authenticate if needed
earthengine authenticate --force
```

#### Memory Issues with Large Datasets
```python
# Adjust chunk size in geetools.py
CHUNK_SIZE = 512  # Reduce for lower memory usage
```

#### Streamlit Port Conflicts
```bash
# Use different port
streamlit run site.py --server.port 8502
```

### Performance Optimization
- Use appropriate bounding box sizes
- Cache frequently used data
- Optimize analysis weights and parameters
- Monitor memory usage for large datasets

## 📋 Roadmap

### Upcoming Features
- [ ] Sentinel-2 time series analysis
- [ ] Machine learning classification models
- [ ] Advanced change detection algorithms
- [ ] Multi-temporal analysis workflows
- [ ] Custom model training interface
- [ ] Export to GIS formats (Shapefile, KML)
- [ ] Integration with additional data sources
- [ ] Mobile-responsive interface

### Performance Improvements
- [ ] Parallel processing for large datasets
- [ ] Advanced caching mechanisms
- [ ] Optimized memory management
- [ ] GPU acceleration for computations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Earth Engine Team** for providing access to planetary-scale geospatial data
- **Streamlit Team** for the excellent web framework
- **LangChain Community** for LLM integration tools
- **Open Source GIS Community** for foundational libraries (GDAL, Rasterio, GeoPandas)

## 📞 Support

For support, questions, or feature requests:
- Open an issue on GitHub
- Check the documentation in the `docs/` directory
- Review existing discussions and solutions

## 🔗 Related Projects

- [Google Earth Engine Python API](https://github.com/google/earthengine-api)
- [Streamlit](https://github.com/streamlit/streamlit)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Folium](https://github.com/python-visualization/folium)

---

<div align="center">

**Built with ❤️ for the geospatial community**

[Documentation](docs/) • [Examples](testing_lib/) • [Contributing](CONTRIBUTING.md) • [License](LICENSE)

</div>
