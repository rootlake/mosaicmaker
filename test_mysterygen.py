#!/usr/bin/env python3
"""
Test script for Mystery Coloring Book Generator.

This script demonstrates how to use the mystery coloring book generator
with a sample image from the input directory.
"""

import os
import sys
from mysterygen import ImageProcessor, GridGenerator, SVGRenderer, INPUT_DIR, OUTPUT_DIR

def main():
    """Run a test of the mystery coloring book generator."""
    # Check for images in the input directory
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory '{INPUT_DIR}' not found. Creating it...")
        os.makedirs(INPUT_DIR)
        print(f"Please add images to the '{INPUT_DIR}' directory and run this script again.")
        sys.exit(1)
    
    # Get first image from input directory
    image_files = [f for f in os.listdir(INPUT_DIR) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not image_files:
        print(f"No images found in '{INPUT_DIR}' directory. Please add some images and try again.")
        sys.exit(1)
    
    # Use the first image found
    sample_image = os.path.join(INPUT_DIR, image_files[0])
    print(f"Using sample image: {sample_image}")
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Process image with default settings
    print("Processing image...")
    processor = ImageProcessor(sample_image, num_colors=12)
    processor.load_and_resize()
    quantized_image, color_map, number_map, color_names = processor.quantize_colors()
    
    # Generate grid
    print("Generating grid...")
    grid_generator = GridGenerator(quantized_image, color_map, number_map, color_names)
    grid_generator.generate_grid()
    
    # Render SVG
    print("Rendering SVG files...")
    output_prefix = "sample_output"
    renderer = SVGRenderer(grid_generator, output_prefix)
    grid_file = renderer.render_grid()
    colored_file = renderer.render_colored_grid()
    combined_file = renderer.render_combined_grid()
    legend_file = renderer.render_legend()
    
    print("\nTest completed successfully!")
    print(f"Mystery coloring grid saved to: {grid_file}")
    print(f"Colored grid saved to: {colored_file}")
    print(f"Combined grid saved to: {combined_file}")
    print(f"Color legend saved to: {legend_file}")
    print("\nYou can now open these SVG files in a web browser or vector graphics editor.")

if __name__ == "__main__":
    main() 