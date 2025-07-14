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

# Constants
LETTER_WIDTH_INCHES = 8.5
LETTER_HEIGHT_INCHES = 11.0
DPI = 300
MARGIN_PERCENT = 5
DEFAULT_COLORS = 12
MAX_COLORS = 26  # Limited by alphabet for simple labeling
INPUT_DIR = "input"  # Directory containing input images
OUTPUT_DIR = "output"  # Directory for output files

# Define the colored pencil palette
pencil_palette = [
    {"name": "Crimson Red", "rgb": [220, 20, 60]},
    {"name": "Lemon Yellow", "rgb": [255, 247, 0]},
    {"name": "Yellow Orange", "rgb": [255, 140, 0]},
    {"name": "Orange", "rgb": [255, 165, 0]},
    {"name": "Gold", "rgb": [255, 215, 0]},
    {"name": "Peach", "rgb": [255, 218, 185]},
    {"name": "Light Orange", "rgb": [250, 240, 190]},
    {"name": "Mango", "rgb": [255, 196, 87]},
    {"name": "Yellow", "rgb": [255, 255, 0]},
    {"name": "Light Yellow", "rgb": [255, 255, 224]},
    {"name": "Lemon Chiffon", "rgb": [255, 245, 238]},
    {"name": "Cornsilk", "rgb": [255, 248, 220]},
    {"name": "Lavender", "rgb": [230, 230, 250]},
    {"name": "Lavender Blush", "rgb": [255, 240, 245]},
    {"name": "Misty Rose", "rgb": [255, 228, 225]},
    {"name": "Snow", "rgb": [255, 250, 250]},
    {"name": "Sand", "rgb": [240, 240, 230]},
    {"name": "Gray", "rgb": [128, 128, 128]},
    {"name": "Dim Gray", "rgb": [105, 105, 105]},
    {"name": "Slate Gray", "rgb": [112, 128, 144]},
    {"name": "Light Slate Gray", "rgb": [119, 136, 153]},
    {"name": "Cool Gray", "rgb": [70, 130, 180]},
    {"name": "Slate", "rgb": [90, 90, 90]},
    {"name": "White", "rgb": [255, 255, 255]},
    {"name": "Black", "rgb": [0, 0, 0]},
    {"name": "Blue", "rgb": [0, 0, 255]},
    {"name": "Medium Blue", "rgb": [0, 0, 205]},
    {"name": "Dark Blue", "rgb": [0, 0, 139]},
    {"name": "Navy Blue", "rgb": [0, 0, 128]},
    {"name": "Midnight Blue", "rgb": [25, 25, 112]},
    {"name": "Dark Slate Blue", "rgb": [72, 61, 139]},
    {"name": "Slate Blue", "rgb": [106, 90, 205]},
    {"name": "Medium Slate Blue", "rgb": [123, 104, 238]},
    {"name": "Medium Purple", "rgb": [147, 112, 219]},
    {"name": "Green Blue", "rgb": [0, 128, 128]},
    {"name": "Cerulean", "rgb": [0, 128, 255]},
    {"name": "Blue Violet", "rgb": [138, 43, 226]},
    {"name": "Dark Violet", "rgb": [148, 0, 211]},
    {"name": "Dark Orchid", "rgb": [153, 50, 204]},
    {"name": "Medium Orchid", "rgb": [186, 85, 211]},
    {"name": "Purple", "rgb": [128, 0, 128]},
    {"name": "Dark Magenta", "rgb": [139, 0, 139]},
    {"name": "Magenta", "rgb": [255, 0, 255]},
    {"name": "Orchid", "rgb": [218, 112, 214]},
    {"name": "Plum", "rgb": [221, 160, 221]},
    {"name": "Violet", "rgb": [238, 130, 238]},
    {"name": "Fuchsia", "rgb": [255, 0, 255]},
    {"name": "Thistle", "rgb": [216, 191, 216]},
    {"name": "Mauve", "rgb": [221, 160, 221]},
    {"name": "Pale Rose", "rgb": [204, 153, 204]},
    {"name": "Bubblegum", "rgb": [255, 105, 180]},
    {"name": "Brown", "rgb": [165, 42, 42]},
    {"name": "Saddle Brown", "rgb": [139, 69, 19]},
    {"name": "Sienna", "rgb": [160, 82, 45]},
    {"name": "Chocolate", "rgb": [210, 105, 30]},
    {"name": "Peru", "rgb": [205, 133, 63]},
    {"name": "Sandy Brown", "rgb": [244, 164, 96]},
    {"name": "Burlywood", "rgb": [222, 184, 135]},
    {"name": "Tan", "rgb": [210, 180, 140]},
    {"name": "Rosy Brown", "rgb": [188, 143, 143]},
    {"name": "Dark Brown", "rgb": [150, 75, 0]},
    {"name": "Mahogany", "rgb": [101, 67, 33]},
    {"name": "Light Brown", "rgb": [210, 180, 140]},
    {"name": "Taupe", "rgb": [150, 113, 23]},
    {"name": "Wheat", "rgb": [245, 222, 179]},
    {"name": "Beige", "rgb": [245, 245, 220]},
    {"name": "Antique White", "rgb": [250, 235, 215]},
    {"name": "Bisque", "rgb": [255, 228, 196]},
    {"name": "Blanched Almond", "rgb": [255, 235, 205]},
    {"name": "Navajo White", "rgb": [255, 222, 173]},
    {"name": "Peach Puff", "rgb": [255, 218, 185]},
    {"name": "Moccasin", "rgb": [255, 228, 181]},
    {"name": "Cornsilk", "rgb": [255, 248, 220]},
    {"name": "Lemon Chiffon", "rgb": [255, 245, 238]},
    {"name": "Seashell", "rgb": [245, 255, 250]},
    {"name": "Mint Cream", "rgb": [240, 255, 240]},
    {"name": "Honeydew", "rgb": [240, 255, 255]},
    {"name": "Azure", "rgb": [240, 255, 255]},
    {"name": "Ghost White", "rgb": [248, 248, 255]},
    {"name": "Alice Blue", "rgb": [240, 248, 255]},
    {"name": "Lavender", "rgb": [230, 230, 250]},
    {"name": "Lavender Blush", "rgb": [255, 240, 245]},
    {"name": "Misty Rose", "rgb": [255, 228, 225]},
    {"name": "Snow", "rgb": [255, 250, 250]},
    {"name": "Sand", "rgb": [240, 240, 230]},
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
        """Load image and resize to fit letter-sized page with margins."""
        # Load image
        self.image = Image.open(self.image_path).convert('RGB')
        
        # Calculate dimensions with margins
        margin_w = int(LETTER_WIDTH_INCHES * DPI * MARGIN_PERCENT / 100)
        margin_h = int(LETTER_HEIGHT_INCHES * DPI * MARGIN_PERCENT / 100)
        max_width = int(LETTER_WIDTH_INCHES * DPI) - 2 * margin_w
        max_height = int(LETTER_HEIGHT_INCHES * DPI) - 2 * margin_h
        
        # Resize maintaining aspect ratio
        width, height = self.image.size
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        self.image = self.image.resize((new_width, new_height), Image.LANCZOS)
        return self.image
        
    def quantize_colors(self):
        """Quantize image to specified number of colors using k-means clustering."""
        # Convert image to numpy array
        img_array = np.array(self.image)
        pixels = img_array.reshape(-1, 3)
        
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=self.num_colors, random_state=42)
        kmeans.fit(pixels)
        
        # Create quantized image
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_.astype(int)
        
        # Store color map and find closest color names
        for i, center in enumerate(centers):
            rgb = tuple(center)
            self.color_map[i] = rgb
            
            # Find closest named color
            closest_color = min(pencil_palette, 
                              key=lambda c: np.sum((np.array(c["rgb"]) - np.array(rgb))**2))
            self.color_names[i] = closest_color["name"]
        
        # Assign random numbers to colors
        numbers = list(range(1, self.num_colors + 1))
        random.shuffle(numbers)
        for i in range(self.num_colors):
            self.number_map[i] = numbers[i]
        
        # Reconstruct quantized image
        quantized_pixels = centers[labels].reshape(img_array.shape)
        self.quantized_image = Image.fromarray(quantized_pixels.astype('uint8'))
        
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
    
    def __init__(self, grid_generator, output_prefix):
        """Initialize with grid data and output path."""
        self.grid = grid_generator
        self.output_prefix = output_prefix
        
        # Ensure output directory exists
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
    def render_grid(self):
        """Render grid as SVG with numbered cells."""
        # Create SVG drawing
        output_path = os.path.join(OUTPUT_DIR, f"{self.output_prefix}_grid.svg")
        dwg = svgwrite.Drawing(
            output_path,
            size=(f"{self.grid.width}px", f"{self.grid.height}px")
        )
        
        # Draw grid cells
        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                # Get cell coordinates
                x = col * self.grid.grid_size
                y = row * self.grid.grid_size
                
                # Draw cell outline
                dwg.add(dwg.rect(
                    insert=(x, y),
                    size=(self.grid.grid_size, self.grid.grid_size),
                    fill="none",
                    stroke="black",
                    stroke_width=1
                ))
                
                # Get color index and corresponding number
                color_idx = self.grid.grid_data[row, col]
                number = self.grid.number_map[color_idx]
                
                # Add number to cell
                dwg.add(dwg.text(
                    str(number),
                    insert=(x + self.grid.grid_size//2, y + self.grid.grid_size//2),
                    text_anchor="middle",
                    dominant_baseline="middle",
                    font_size=self.grid.grid_size//3,
                    fill="gray",
                    fill_opacity=0.2
                ))
                
        # Save SVG
        dwg.save()
        return output_path
    
    def render_colored_grid(self):
        """Render colored grid without numbers."""
        # Create SVG drawing
        output_path = os.path.join(OUTPUT_DIR, f"{self.output_prefix}_colored.svg")
        dwg = svgwrite.Drawing(
            output_path,
            size=(f"{self.grid.width}px", f"{self.grid.height}px")
        )
        
        # Draw grid cells
        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                # Get cell coordinates
                x = col * self.grid.grid_size
                y = row * self.grid.grid_size
                
                # Get color index and RGB color
                color_idx = self.grid.grid_data[row, col]
                rgb = self.grid.color_map[color_idx]
                
                # Draw colored cell
                dwg.add(dwg.rect(
                    insert=(x, y),
                    size=(self.grid.grid_size, self.grid.grid_size),
                    fill=f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})",
                    stroke="black",
                    stroke_width=1
                ))
                
        # Save SVG
        dwg.save()
        return output_path
    
    def render_combined_grid(self):
        """Render colored grid with numbers."""
        # Create SVG drawing
        output_path = os.path.join(OUTPUT_DIR, f"{self.output_prefix}_combined.svg")
        dwg = svgwrite.Drawing(
            output_path,
            size=(f"{self.grid.width}px", f"{self.grid.height}px")
        )
        
        # Draw grid cells
        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                # Get cell coordinates
                x = col * self.grid.grid_size
                y = row * self.grid.grid_size
                
                # Get color index, RGB color, and number
                color_idx = self.grid.grid_data[row, col]
                rgb = self.grid.color_map[color_idx]
                number = self.grid.number_map[color_idx]
                
                # Draw colored cell
                dwg.add(dwg.rect(
                    insert=(x, y),
                    size=(self.grid.grid_size, self.grid.grid_size),
                    fill=f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})",
                    stroke="black",
                    stroke_width=1
                ))
                
                # Add number to cell (white or black depending on color brightness)
                brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
                text_color = "white" if brightness < 128 else "black"
                
                dwg.add(dwg.text(
                    str(number),
                    insert=(x + self.grid.grid_size//2, y + self.grid.grid_size//2),
                    text_anchor="middle",
                    dominant_baseline="middle",
                    font_size=self.grid.grid_size//3,
                    fill=text_color
                ))
                
        # Save SVG
        dwg.save()
        return output_path
        
    def render_legend(self):
        """Render color legend as SVG."""
        # Create SVG drawing for legend
        output_path = os.path.join(OUTPUT_DIR, f"{self.output_prefix}_legend.svg")
        legend_width = 400
        legend_height = 50 * len(self.grid.color_map)
        dwg = svgwrite.Drawing(
            output_path,
            size=(f"{legend_width}px", f"{legend_height}px")
        )
        
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
        output_prefix = f"{base_name}_mystery"
    
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
    renderer = SVGRenderer(grid_generator, output_prefix)
    grid_file = renderer.render_grid()
    colored_file = renderer.render_colored_grid()
    combined_file = renderer.render_combined_grid()
    legend_file = renderer.render_legend()
    
    print(f"Mystery coloring grid saved to: {grid_file}")
    print(f"Colored grid saved to: {colored_file}")
    print(f"Combined grid saved to: {combined_file}")
    print(f"Color legend saved to: {legend_file}")


if __name__ == "__main__":
    main() 