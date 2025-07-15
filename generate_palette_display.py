#!/usr/bin/env python3
"""
One-time script to generate a visual display of the complete colored pencil palette.
"""

import svgwrite
import os
from reportlab.graphics import renderPDF
import svglib.svglib

# Define the colored pencil palette (same as in mysterygen.py)
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

def create_palette_svg():
    """Create an SVG showing all colors in the palette."""
    # Calculate dimensions
    colors_per_row = 4  # Reduced to give more space
    rows = (len(COLORED_PENCIL_PALETTE) + colors_per_row - 1) // colors_per_row
    
    box_size = 60
    margin = 20
    spacing = 20
    
    row_height = box_size + spacing
    col_width = 280  # Increased for text next to boxes
    
    svg_width = colors_per_row * col_width + margin * 2
    svg_height = rows * row_height + margin * 2 + 50  # Extra space for title
    
    # Create SVG
    dwg = svgwrite.Drawing('palette_display.svg', size=(svg_width, svg_height))
    
    # Add white background
    dwg.add(dwg.rect(insert=(0, 0), size=(svg_width, svg_height), fill='white'))
    
    # Add title
    dwg.add(dwg.text(
        f'Colored Pencil Palette ({len(COLORED_PENCIL_PALETTE)} Colors)',
        insert=(svg_width//2, 40),
        text_anchor='middle',
        font_size=24,
        font_weight='bold',
        fill='black'
    ))
    
    # Draw color swatches
    for i, color in enumerate(COLORED_PENCIL_PALETTE):
        row = i // colors_per_row
        col = i % colors_per_row
        
        x = margin + col * col_width
        y = margin + 60 + row * row_height  # 60 for title space
        
        rgb = color['rgb']
        rgb_string = f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})'
        
        # Color box
        dwg.add(dwg.rect(
            insert=(x, y),
            size=(box_size, box_size),
            fill=rgb_string,
            stroke='black',
            stroke_width=1
        ))
        
        # Color name (to the right of the box)
        dwg.add(dwg.text(
            color['name'],
            insert=(x + box_size + 15, y + 20),
            text_anchor='start',
            font_size=14,
            font_weight='bold',
            fill='black'
        ))
        
        # RGB values (smaller text, below the name)
        dwg.add(dwg.text(
            f'RGB({rgb[0]}, {rgb[1]}, {rgb[2]})',
            insert=(x + box_size + 15, y + 40),
            text_anchor='start',
            font_size=10,
            fill='gray'
        ))
    
    # Save SVG
    dwg.save()
    return 'palette_display.svg'

def convert_to_pdf(svg_path):
    """Convert SVG to PDF."""
    try:
        drawing = svglib.svglib.svg2rlg(svg_path)
        pdf_path = svg_path.replace('.svg', '.pdf')
        renderPDF.drawToFile(drawing, pdf_path)
        return pdf_path
    except Exception as e:
        print(f"Error converting to PDF: {e}")
        return None

def main():
    print("Generating colored pencil palette display...")
    
    # Create SVG
    svg_path = create_palette_svg()
    print(f"SVG created: {svg_path}")
    
    # Convert to PDF
    pdf_path = convert_to_pdf(svg_path)
    if pdf_path:
        print(f"PDF created: {pdf_path}")
    else:
        print("PDF conversion failed")
    
    print(f"Total colors in palette: {len(COLORED_PENCIL_PALETTE)}")

if __name__ == "__main__":
    main()