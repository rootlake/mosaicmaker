# Mosaic Maker

A simple tool that converts images into numbered grid patterns for coloring. It processes an image into a mystery coloring grid where each number corresponds to a specific color.

## Features

- Upload any image (JPG/PNG)
- Adjust number of colors (2-26)
- Adjust grid size (10-100)
- Generate:
  - Original image preview
  - Gridded color version
  - Mystery grid with numbers
  - Color legend
  - Combined view

## Setup

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/mosaicMaker.git
cd mosaicMaker
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
python simple_server.py
```

2. Open `http://localhost:8000` in your web browser

3. Use the interface to:
   - Choose an image file
   - Adjust colors (2-26) and grid size (10-100)
   - Click "Process Image"
   - View the generated patterns

## Files

- `mysterygen.py` - Core image processing script
- `simple_server.py` - Local web server
- `index.html` - Web interface
- `requirements.txt` - Python dependencies

## Development

- Python 3.6+ required
- Uses standard library `http.server` for local serving
- Core image processing uses:
  - scikit-learn for color quantization
  - Pillow for image processing
  - svgwrite for output generation

## License

MIT License 