#!/usr/bin/env python3
"""
Mystery Coloring Book Generator

Converts images into numbered grid patterns for coloring.
"""

import os
import argparse
import random
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import svgwrite
from collections import OrderedDict
from reportlab.graphics import renderPDF
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import svglib.svglib

# Constants
LETTER_WIDTH_INCHES = 8.5
LETTER_HEIGHT_INCHES = 11.0
# KDP artwork dimensions (8 x 10.5 inches)
KDP_ARTWORK_WIDTH = 8.0
KDP_ARTWORK_HEIGHT = 10.5
DPI = 300
MARGIN_PERCENT = 5
DEFAULT_COLORS = 12
MAX_COLORS = 26  # Limited by alphabet for simple labeling
INPUT_DIR = "input"  # Directory containing input images
OUTPUT_DIR = "output"  # Directory for output files
PDF_DIR = "PDFs"  # Directory for PDF files

# Define the colored pencil palette based on standard 48-color pencil sets
# Colors are chosen to be distinct and match common colored pencil names
COLORED_PENCIL_PALETTE = [
    {"name": "Red", "rgb": [237, 28, 36]},
    {"name": "Crimson Red", "rgb": [220, 20, 60]},
    {"name": "Scarlet", "rgb": [255, 36, 0]},
    {"name": "Orange", "rgb": [255, 127, 39]},
    {"name": "Yellow Orange", "rgb": [255, 201, 14]},
    {"name": "Yellow", "rgb": [255, 242, 0]},
    {"name": "Green Yellow", "rgb": [181, 230, 29]},
    {"name": "Yellow Green", "rgb": [140, 198, 63]},
    {"name": "Green", "rgb": [0, 162, 82]},
    {"name": "Forest Green", "rgb": [34, 139, 34]},
    {"name": "Blue Green", "rgb": [0, 150, 136]},
    {"name": "Turquoise", "rgb": [64, 224, 208]},
    {"name": "Sky Blue", "rgb": [135, 206, 235]},
    {"name": "Blue", "rgb": [0, 114, 188]},
    {"name": "Navy Blue", "rgb": [0, 0, 128]},
    {"name": "Violet", "rgb": [148, 0, 211]},
    {"name": "Purple", "rgb": [128, 0, 128]},
    {"name": "Magenta", "rgb": [236, 0, 140]},
    {"name": "Pink", "rgb": [255, 192, 203]},
    {"name": "Hot Pink", "rgb": [255, 105, 180]},
    {"name": "Brown", "rgb": [139, 69, 19]},
    {"name": "Tan", "rgb": [210, 180, 140]},
    {"name": "Peach", "rgb": [255, 218, 185]},
    {"name": "Black", "rgb": [0, 0, 0]},
    {"name": "Dark Gray", "rgb": [64, 64, 64]},
    {"name": "Gray", "rgb": [128, 128, 128]},
    {"name": "Light Gray", "rgb": [192, 192, 192]},
    {"name": "White", "rgb": [255, 255, 255]},
    {"name": "Maroon", "rgb": [128, 0, 0]},
    {"name": "Burnt Orange", "rgb": [204, 85, 0]},
    {"name": "Gold", "rgb": [255, 215, 0]},
    {"name": "Olive Green", "rgb": [128, 128, 0]},
    {"name": "Mint Green", "rgb": [152, 251, 152]},
    {"name": "Teal", "rgb": [0, 128, 128]},
    {"name": "Royal Blue", "rgb": [65, 105, 225]},
    {"name": "Indigo", "rgb": [75, 0, 130]},
    {"name": "Plum", "rgb": [221, 160, 221]},
    {"name": "Rose", "rgb": [255, 0, 127]},
    {"name": "Salmon", "rgb": [250, 128, 114]},
    {"name": "Chocolate", "rgb": [210, 105, 30]},
    {"name": "Beige", "rgb": [245, 245, 220]},
    {"name": "Mahogany", "rgb": [192, 64, 0]},
    {"name": "Sienna", "rgb": [160, 82, 45]},
    {"name": "Copper", "rgb": [184, 115, 51]},
    {"name": "Lavender", "rgb": [230, 230, 250]},
    {"name": "Aqua", "rgb": [0, 255, 255]},
    {"name": "Lime", "rgb": [0, 255, 0]},
    {"name": "Coral", "rgb": [255, 127, 80]}
]

class ImageProcessor:
    """Handles image loading, resizing, and color quantization."""
    
    def __init__(self, image_path, num_colors=DEFAULT_COLORS):
        """Initialize with image path and number of colors for quantization."""
        self.image_path = image_path
        self.num_colors = min(num_colors, MAX_COLORS)
        self.image = None
        self.quantized_image = None
        self.color_map = {}  # Maps cluster index to RGB color
        self.number_map = {}  # Maps cluster index to assigned number
        self.color_names = {}  # Maps cluster index to color name
        
    def load_and_resize(self):
        """Load image and resize to fit KDP artwork dimensions (8 x 10.5 inches)."""
        # Load image
        self.image = Image.open(self.image_path).convert('RGB')
        
        # Calculate dimensions for KDP artwork (8 x 10.5 inches)
        max_width = int(KDP_ARTWORK_WIDTH * DPI)
        max_height = int(KDP_ARTWORK_HEIGHT * DPI)
        
        # Resize maintaining aspect ratio
        width, height = self.image.size
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        self.image = self.image.resize((new_width, new_height), Image.LANCZOS)
        return self.image
        
    def quantize_colors(self):
        """Quantize image using fixed colored pencil palette."""
        # Convert image to numpy array
        img_array = np.array(self.image)
        pixels = img_array.reshape(-1, 3)
        
        # Create array of palette colors for vectorized distance calculation
        palette_colors = np.array([color["rgb"] for color in COLORED_PENCIL_PALETTE])
        
        # Find the most representative colors from the palette
        # Use K-means to find dominant colors in the image first
        kmeans = KMeans(n_clusters=min(self.num_colors * 3, len(COLORED_PENCIL_PALETTE)), random_state=42)
        kmeans.fit(pixels)
        image_centers = kmeans.cluster_centers_
        
        # For each K-means center, find the closest palette color
        selected_palette_indices = []
        used_colors = set()
        
        # Sort image centers by how much of the image they represent
        center_counts = np.bincount(kmeans.labels_)
        sorted_centers = sorted(enumerate(image_centers), key=lambda x: center_counts[x[0]], reverse=True)
        
        for _, center in sorted_centers:
            if len(selected_palette_indices) >= self.num_colors:
                break
                
            # Find closest palette color that hasn't been used
            distances = np.sum((palette_colors - center)**2, axis=1)
            sorted_palette_indices = np.argsort(distances)
            
            for palette_idx in sorted_palette_indices:
                if palette_idx not in used_colors:
                    selected_palette_indices.append(palette_idx)
                    used_colors.add(palette_idx)
                    break
        
        # Ensure we have exactly the requested number of colors
        while len(selected_palette_indices) < self.num_colors:
            for i, color in enumerate(COLORED_PENCIL_PALETTE):
                if i not in used_colors:
                    selected_palette_indices.append(i)
                    used_colors.add(i)
                    break
        
        # Truncate if we somehow have too many
        selected_palette_indices = selected_palette_indices[:self.num_colors]
        
        # Create color mappings using selected palette colors
        selected_colors = [COLORED_PENCIL_PALETTE[i] for i in selected_palette_indices]
        
        for i, color_info in enumerate(selected_colors):
            self.color_map[i] = tuple(color_info["rgb"])
            self.color_names[i] = color_info["name"]
        
        # Assign random numbers to colors
        numbers = list(range(1, self.num_colors + 1))
        random.shuffle(numbers)
        for i in range(self.num_colors):
            self.number_map[i] = numbers[i]
        
        # Quantize the image using the selected palette
        quantized_pixels = np.zeros_like(pixels)
        selected_palette_colors = np.array([color["rgb"] for color in selected_colors])
        
        # For each pixel, find the closest color in our selected palette
        for i in range(0, len(pixels), 10000):  # Process in chunks to avoid memory issues
            chunk = pixels[i:i+10000]
            distances = np.sum((chunk[:, np.newaxis] - selected_palette_colors)**2, axis=2)
            closest_indices = np.argmin(distances, axis=1)
            quantized_pixels[i:i+10000] = selected_palette_colors[closest_indices]
        
        # Reconstruct quantized image
        quantized_image_array = quantized_pixels.reshape(img_array.shape)
        self.quantized_image = Image.fromarray(quantized_image_array.astype('uint8'))
        
        return self.quantized_image, self.color_map, self.number_map, self.color_names


class GridGenerator:
    """Generates a grid from a quantized image."""
    
    def __init__(self, quantized_image, color_map, number_map, color_names, grid_size=None):
        """Initialize with quantized image and color mappings."""
        self.image = quantized_image
        self.color_map = color_map
        self.number_map = number_map
        self.color_names = color_names
        self.width, self.height = quantized_image.size
        
        # Determine grid size if not specified
        if grid_size is None:
            # Default to a reasonable grid size based on image dimensions
            self.grid_size = max(20, min(50, min(self.width, self.height) // 20))
        else:
            self.grid_size = grid_size
            
        # Calculate grid dimensions
        self.cols = self.width // self.grid_size
        self.rows = self.height // self.grid_size
        
        # Adjust image size to fit grid exactly
        self.width = self.cols * self.grid_size
        self.height = self.rows * self.grid_size
        self.image = self.image.resize((self.width, self.height), Image.LANCZOS)
        
        # Grid data will store the color index for each cell
        self.grid_data = np.zeros((self.rows, self.cols), dtype=int)
        
    def generate_grid(self):
        """Generate grid data by sampling the center of each grid cell."""
        img_array = np.array(self.image)
        
        for row in range(self.rows):
            for col in range(self.cols):
                # Sample center pixel of each grid cell
                center_x = col * self.grid_size + self.grid_size // 2
                center_y = row * self.grid_size + self.grid_size // 2
                
                pixel_color = tuple(img_array[center_y, center_x])
                
                # Find closest color in our color map
                closest_idx = min(self.color_map, 
                                 key=lambda idx: np.sum((np.array(self.color_map[idx]) - np.array(pixel_color))**2))
                
                self.grid_data[row, col] = closest_idx
                
        return self.grid_data


class SVGRenderer:
    """Renders grid and legend as SVG files."""
    
    def __init__(self, grid_generator, output_prefix, num_colors=None, grid_size=None):
        """Initialize with grid data and output path."""
        self.grid = grid_generator
        self.output_prefix = output_prefix
        self.num_colors = num_colors or len(grid_generator.color_map)
        self.grid_size = grid_size or grid_generator.grid_size
        
        # Create filename suffix with parameters
        self.param_suffix = f"_{self.num_colors}color_{self.grid_size}grid"
        
        # Ensure output directory exists
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
    def render_grid(self):
        """Render grid as SVG with numbered cells for KDP printing."""
        # Create SVG drawing with KDP dimensions
        output_path = os.path.join(OUTPUT_DIR, f"{self.output_prefix}_grid{self.param_suffix}.svg")
        
        # KDP page dimensions in pixels at 300 DPI
        page_width = int(LETTER_WIDTH_INCHES * DPI)
        page_height = int(LETTER_HEIGHT_INCHES * DPI)
        
        # Calculate centering offsets for 8x10.5 artwork on 8.5x11 page
        artwork_width = int(KDP_ARTWORK_WIDTH * DPI)
        artwork_height = int(KDP_ARTWORK_HEIGHT * DPI)
        offset_x = (page_width - artwork_width) // 2
        offset_y = (page_height - artwork_height) // 2
        
        # Scale grid to fit artwork area
        scale_x = artwork_width / self.grid.width
        scale_y = artwork_height / self.grid.height
        scale = min(scale_x, scale_y)
        
        scaled_width = int(self.grid.width * scale)
        scaled_height = int(self.grid.height * scale)
        
        # Center the scaled grid within the artwork area
        grid_offset_x = offset_x + (artwork_width - scaled_width) // 2
        grid_offset_y = offset_y + (artwork_height - scaled_height) // 2
        
        dwg = svgwrite.Drawing(
            output_path,
            size=(f"{page_width}px", f"{page_height}px")
        )
        
        # Add white background for entire page
        dwg.add(dwg.rect(
            insert=(0, 0),
            size=(page_width, page_height),
            fill="white"
        ))
        
        # Draw grid cells with scaling and centering
        scaled_cell_size = self.grid.grid_size * scale
        
        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                # Get scaled cell coordinates
                x = grid_offset_x + col * scaled_cell_size
                y = grid_offset_y + row * scaled_cell_size
                
                # Draw cell outline
                dwg.add(dwg.rect(
                    insert=(x, y),
                    size=(scaled_cell_size, scaled_cell_size),
                    fill="none",
                    stroke="black",
                    stroke_width=max(1, scale)
                ))
                
                # Get color index and corresponding number
                color_idx = self.grid.grid_data[row, col]
                number = self.grid.number_map[color_idx]
                
                # Add number to cell with scaled font
                dwg.add(dwg.text(
                    str(number),
                    insert=(x + scaled_cell_size//2, y + scaled_cell_size//2),
                    text_anchor="middle",
                    dominant_baseline="middle",
                    font_size=max(8, scaled_cell_size//3),
                    font_family="Arial, Helvetica, sans-serif",
                    font_weight="bold",
                    fill="gray",
                    fill_opacity=0.2
                ))
                
        # Save SVG
        dwg.save()
        return output_path
    
    def render_colored_grid(self):
        """Render colored grid without numbers for KDP printing."""
        # Create SVG drawing with KDP dimensions
        output_path = os.path.join(OUTPUT_DIR, f"{self.output_prefix}_colored{self.param_suffix}.svg")
        
        # KDP page dimensions in pixels at 300 DPI
        page_width = int(LETTER_WIDTH_INCHES * DPI)
        page_height = int(LETTER_HEIGHT_INCHES * DPI)
        
        # Calculate centering offsets for 8x10.5 artwork on 8.5x11 page
        artwork_width = int(KDP_ARTWORK_WIDTH * DPI)
        artwork_height = int(KDP_ARTWORK_HEIGHT * DPI)
        offset_x = (page_width - artwork_width) // 2
        offset_y = (page_height - artwork_height) // 2
        
        # Scale grid to fit artwork area
        scale_x = artwork_width / self.grid.width
        scale_y = artwork_height / self.grid.height
        scale = min(scale_x, scale_y)
        
        scaled_width = int(self.grid.width * scale)
        scaled_height = int(self.grid.height * scale)
        
        # Center the scaled grid within the artwork area
        grid_offset_x = offset_x + (artwork_width - scaled_width) // 2
        grid_offset_y = offset_y + (artwork_height - scaled_height) // 2
        
        dwg = svgwrite.Drawing(
            output_path,
            size=(f"{page_width}px", f"{page_height}px")
        )
        
        # Add white background for entire page
        dwg.add(dwg.rect(
            insert=(0, 0),
            size=(page_width, page_height),
            fill="white"
        ))
        
        # Draw grid cells with scaling and centering
        scaled_cell_size = self.grid.grid_size * scale
        
        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                # Get scaled cell coordinates
                x = grid_offset_x + col * scaled_cell_size
                y = grid_offset_y + row * scaled_cell_size
                
                # Get color index and RGB color
                color_idx = self.grid.grid_data[row, col]
                rgb = self.grid.color_map[color_idx]
                
                # Draw colored cell
                dwg.add(dwg.rect(
                    insert=(x, y),
                    size=(scaled_cell_size, scaled_cell_size),
                    fill=f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})",
                    stroke="black",
                    stroke_width=max(1, scale)
                ))
                
        # Save SVG
        dwg.save()
        return output_path
    
    def render_combined_grid(self):
        """Render colored grid with numbers for KDP printing."""
        # Create SVG drawing with KDP dimensions
        output_path = os.path.join(OUTPUT_DIR, f"{self.output_prefix}_combined{self.param_suffix}.svg")
        
        # KDP page dimensions in pixels at 300 DPI
        page_width = int(LETTER_WIDTH_INCHES * DPI)
        page_height = int(LETTER_HEIGHT_INCHES * DPI)
        
        # Calculate centering offsets for 8x10.5 artwork on 8.5x11 page
        artwork_width = int(KDP_ARTWORK_WIDTH * DPI)
        artwork_height = int(KDP_ARTWORK_HEIGHT * DPI)
        offset_x = (page_width - artwork_width) // 2
        offset_y = (page_height - artwork_height) // 2
        
        # Scale grid to fit artwork area
        scale_x = artwork_width / self.grid.width
        scale_y = artwork_height / self.grid.height
        scale = min(scale_x, scale_y)
        
        scaled_width = int(self.grid.width * scale)
        scaled_height = int(self.grid.height * scale)
        
        # Center the scaled grid within the artwork area
        grid_offset_x = offset_x + (artwork_width - scaled_width) // 2
        grid_offset_y = offset_y + (artwork_height - scaled_height) // 2
        
        dwg = svgwrite.Drawing(
            output_path,
            size=(f"{page_width}px", f"{page_height}px")
        )
        
        # Add white background for entire page
        dwg.add(dwg.rect(
            insert=(0, 0),
            size=(page_width, page_height),
            fill="white"
        ))
        
        # Draw grid cells with scaling and centering
        scaled_cell_size = self.grid.grid_size * scale
        
        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                # Get scaled cell coordinates
                x = grid_offset_x + col * scaled_cell_size
                y = grid_offset_y + row * scaled_cell_size
                
                # Get color index, RGB color, and number
                color_idx = self.grid.grid_data[row, col]
                rgb = self.grid.color_map[color_idx]
                number = self.grid.number_map[color_idx]
                
                # Draw colored cell
                dwg.add(dwg.rect(
                    insert=(x, y),
                    size=(scaled_cell_size, scaled_cell_size),
                    fill=f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})",
                    stroke="black",
                    stroke_width=max(1, scale)
                ))
                
                # Add number to cell (white or black depending on color brightness)
                brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
                text_color = "white" if brightness < 128 else "black"
                
                dwg.add(dwg.text(
                    str(number),
                    insert=(x + scaled_cell_size//2, y + scaled_cell_size//2),
                    text_anchor="middle",
                    dominant_baseline="middle",
                    font_size=max(8, scaled_cell_size//3),
                    font_family="Arial, Helvetica, sans-serif",
                    font_weight="bold",
                    fill=text_color
                ))
                
        # Save SVG
        dwg.save()
        return output_path
        
    def render_legend(self):
        """Render color legend as SVG."""
        # Create SVG drawing for legend
        output_path = os.path.join(OUTPUT_DIR, f"{self.output_prefix}_key{self.param_suffix}.svg")
        legend_width = 400
        legend_height = 50 * len(self.grid.color_map)
        dwg = svgwrite.Drawing(
            output_path,
            size=(f"{legend_width}px", f"{legend_height}px")
        )
        
        # Add white background
        dwg.add(dwg.rect(
            insert=(0, 0),
            size=(legend_width, legend_height),
            fill="white"
        ))
        
        # Add title
        dwg.add(dwg.text(
            "Color Legend",
            insert=(legend_width//2, 30),
            text_anchor="middle",
            font_size=20,
            font_weight="bold"
        ))
        
        # Draw color swatches with numbers and names
        y_offset = 60
        for color_idx, color in self.grid.color_map.items():
            number = self.grid.number_map[color_idx]
            color_name = self.grid.color_names[color_idx]
            rgb = f"rgb({color[0]}, {color[1]}, {color[2]})"
            
            # Draw color swatch
            dwg.add(dwg.rect(
                insert=(50, y_offset),
                size=(40, 40),
                fill=rgb,
                stroke="black",
                stroke_width=1
            ))
            
            # Add number
            dwg.add(dwg.text(
                str(number),
                insert=(30, y_offset + 25),
                text_anchor="end",
                dominant_baseline="middle",
                font_size=18,
                font_weight="bold"
            ))
            
            # Add color name
            dwg.add(dwg.text(
                color_name,
                insert=(100, y_offset + 25),
                dominant_baseline="middle",
                font_size=14
            ))
            
            y_offset += 50
            
        # Save legend SVG
        dwg.save()
        return output_path
    
    def render_bw_legend(self):
        """Render black and white legend with numbers and color names only."""
        # Create SVG drawing for B&W legend
        output_path = os.path.join(OUTPUT_DIR, f"{self.output_prefix}_bw_key{self.param_suffix}.svg")
        legend_width = 300
        legend_height = 50 + 30 * len(self.grid.color_map)
        dwg = svgwrite.Drawing(
            output_path,
            size=(f"{legend_width}px", f"{legend_height}px")
        )
        
        # Add white background
        dwg.add(dwg.rect(
            insert=(0, 0),
            size=(legend_width, legend_height),
            fill="white"
        ))
        
        # Add title
        dwg.add(dwg.text(
            "Color Key",
            insert=(legend_width//2, 30),
            text_anchor="middle",
            font_size=20,
            font_weight="bold",
            fill="black"
        ))
        
        # Draw numbers and color names only
        y_offset = 60
        for color_idx, color in self.grid.color_map.items():
            number = self.grid.number_map[color_idx]
            color_name = self.grid.color_names[color_idx]
            
            # Add number
            dwg.add(dwg.text(
                str(number),
                insert=(30, y_offset),
                text_anchor="end",
                dominant_baseline="middle",
                font_size=16,
                font_weight="bold",
                fill="black"
            ))
            
            # Add color name
            dwg.add(dwg.text(
                color_name,
                insert=(40, y_offset),
                dominant_baseline="middle",
                font_size=14,
                fill="black"
            ))
            
            y_offset += 30
            
        # Save B&W legend SVG
        dwg.save()
        return output_path
    
    def export_to_pdf(self, svg_type="grid"):
        """Export SVG to PDF format."""
        # Ensure PDF directory exists
        if not os.path.exists(PDF_DIR):
            os.makedirs(PDF_DIR)
        
        # Define SVG file paths
        svg_files = {
            "grid": os.path.join(OUTPUT_DIR, f"{self.output_prefix}_grid{self.param_suffix}.svg"),
            "key": os.path.join(OUTPUT_DIR, f"{self.output_prefix}_key{self.param_suffix}.svg"),
            "bw_key": os.path.join(OUTPUT_DIR, f"{self.output_prefix}_bw_key{self.param_suffix}.svg"),
            "colored": os.path.join(OUTPUT_DIR, f"{self.output_prefix}_colored{self.param_suffix}.svg"),
            "combined": os.path.join(OUTPUT_DIR, f"{self.output_prefix}_combined{self.param_suffix}.svg")
        }
        
        svg_path = svg_files.get(svg_type)
        if not svg_path or not os.path.exists(svg_path):
            raise FileNotFoundError(f"SVG file not found: {svg_path}")
        
        # Create PDF path
        pdf_path = os.path.join(PDF_DIR, f"{self.output_prefix}_{svg_type}{self.param_suffix}.pdf")
        
        try:
            # Convert SVG to PDF using svglib and reportlab
            drawing = svglib.svglib.svg2rlg(svg_path)
            renderPDF.drawToFile(drawing, pdf_path)
            return pdf_path
        except Exception as e:
            print(f"Error converting SVG to PDF: {str(e)}")
            raise
    
    def export_both_pdfs(self):
        """Export both grid and key PDFs."""
        try:
            grid_pdf = self.export_to_pdf("grid")
            bw_key_pdf = self.export_to_pdf("bw_key")
            return {"grid": grid_pdf, "bw_key": bw_key_pdf}
        except Exception as e:
            print(f"Error exporting PDFs: {str(e)}")
            raise


def get_available_images():
    """Get list of available images in the input directory."""
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Created {INPUT_DIR} directory. Please add images there.")
        return []
    
    image_files = []
    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_files.append(os.path.join(INPUT_DIR, file))
    
    return image_files


def select_image_interactive():
    """Allow user to select an image interactively."""
    images = get_available_images()
    
    if not images:
        print(f"No images found in {INPUT_DIR} directory.")
        return None
    
    print("\nAvailable images:")
    for i, img_path in enumerate(images):
        print(f"{i+1}. {os.path.basename(img_path)}")
    
    while True:
        try:
            choice = input("\nSelect image number (or 'q' to quit): ")
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(images):
                return images[idx]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number or 'q'.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate mystery coloring book from image")
    parser.add_argument("input_image", nargs='?', help="Path to input image or filename in input directory (optional)")
    parser.add_argument("-o", "--output", help="Output prefix for generated files")
    parser.add_argument("-c", "--colors", type=int, default=DEFAULT_COLORS,
                        help=f"Number of colors (max {MAX_COLORS}, default {DEFAULT_COLORS})")
    parser.add_argument("-g", "--grid-size", type=int, help="Grid cell size in pixels")
    parser.add_argument("-l", "--list", action="store_true", help="List available images and exit")
    return parser.parse_args()


def main():
    """Main function to run the generator."""
    # Parse arguments
    args = parse_args()
    
    # List available images if requested
    if args.list:
        images = get_available_images()
        if images:
            print("\nAvailable images:")
            for i, img_path in enumerate(images):
                print(f"{i+1}. {os.path.basename(img_path)}")
        else:
            print(f"No images found in {INPUT_DIR} directory.")
        return
    
    # Determine input image
    input_image = args.input_image
    if input_image:
        # Check if it's a filename in the input directory
        if not os.path.exists(input_image) and os.path.exists(os.path.join(INPUT_DIR, input_image)):
            input_image = os.path.join(INPUT_DIR, input_image)
    else:
        input_image = select_image_interactive()
        if not input_image:
            print("No image selected. Exiting.")
            return
    
    # Determine output prefix
    if args.output:
        output_prefix = args.output
    else:
        base_name = os.path.splitext(os.path.basename(input_image))[0]
        output_prefix = base_name
    
    # Process image
    print(f"Processing image: {input_image}")
    processor = ImageProcessor(input_image, args.colors)
    processor.load_and_resize()
    quantized_image, color_map, number_map, color_names = processor.quantize_colors()
    
    # Generate grid
    print("Generating grid...")
    grid_generator = GridGenerator(quantized_image, color_map, number_map, color_names, args.grid_size)
    grid_generator.generate_grid()
    
    # Render SVG files
    print("Rendering SVG files...")
    renderer = SVGRenderer(grid_generator, output_prefix, args.colors, args.grid_size)
    grid_file = renderer.render_grid()
    colored_file = renderer.render_colored_grid()
    combined_file = renderer.render_combined_grid()
    key_file = renderer.render_legend()
    bw_key_file = renderer.render_bw_legend()
    
    print(f"Mystery coloring grid saved to: {grid_file}")
    print(f"Colored grid saved to: {colored_file}")
    print(f"Combined grid saved to: {combined_file}")
    print(f"Color key saved to: {key_file}")
    print(f"B&W color key saved to: {bw_key_file}")


if __name__ == "__main__":
    main() 