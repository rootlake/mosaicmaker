# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Mosaic Maker application that converts images into numbered grid patterns for coloring. It processes images using K-means clustering to create mystery coloring grids where each number corresponds to a specific color.

## Architecture

### Core Components

1. **mysterygen.py** - Main image processing script with three classes:
   - `ImageProcessor` - Handles image loading, resizing, and K-means color quantization
   - `GridGenerator` - Creates grid data by sampling quantized image pixels
   - `SVGRenderer` - Renders various SVG outputs (grid, colored, combined, legend)

2. **simple_server.py** - HTTP server that serves the web interface and handles image uploads via POST to `/process`

3. **index.html** - Web interface with dark theme, file upload, parameter controls, and 5-panel grid display

### Data Flow

1. Image uploaded through web interface or CLI
2. `ImageProcessor` loads, resizes, and quantizes colors using scikit-learn K-means
3. `GridGenerator` samples grid cells and maps to color clusters
4. `SVGRenderer` creates multiple SVG views: mystery grid, colored grid, combined view, and legend
5. Web interface displays all outputs in responsive grid layout

## Commands

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start development server
python simple_server.py

# Run CLI version
python mysterygen.py input/image.jpg --colors 12 --grid-size 30

# Run tests
python test_mysterygen.py
```

### Common Parameters
- `--colors N`: Number of colors (2-26, default 12)
- `--grid-size N`: Grid cell size in pixels (10-100, default 30)

## Directory Structure

- `input/` - Source images
- `output/` - Generated SVG files
- `uploads/` - Web interface uploads
- `venv/` - Python virtual environment

## Key Dependencies

- **Pillow** - Image processing
- **scikit-learn** - K-means clustering for color quantization
- **svgwrite** - SVG file generation
- **numpy** - Array operations for image data

## Output Files

For each processed image, generates:
- `*_mystery_grid.svg` - Numbers-only coloring grid
- `*_mystery_colored.svg` - Fully colored reference
- `*_mystery_combined.svg` - Colors with numbers overlay
- `*_mystery_legend.svg` - Color legend with numbers and names