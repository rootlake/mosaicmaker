# Feature Options

This document outlines customization options and features available in the Mosaic Maker application.

## Color Palette Customization

The application uses a curated palette of 48 standard colored pencil colors. This palette is fully customizable to meet your specific needs.

### Current Palette
- **48 distinct colors** based on standard colored pencil sets
- **No duplicates** - each color has unique RGB values and names
- **Professional names** like "Forest Green", "Sky Blue", "Burnt Orange"
- **Full spectrum coverage** including primary, secondary, earth tones, and neutrals

### Customization Options

#### 1. Rename Existing Colors
You can easily rename any color in the palette without changing its RGB values:

```python
# Current:
{"name": "Crimson Red", "rgb": [220, 20, 60]},

# Could become:
{"name": "Cherry Red", "rgb": [220, 20, 60]},
{"name": "Fire Engine Red", "rgb": [220, 20, 60]},
```

#### 2. Replace Colors Entirely
Suggest new colors with different RGB values:

```python
# Replace with a new color:
{"name": "Sunset Orange", "rgb": [255, 94, 77]},
{"name": "Ocean Blue", "rgb": [0, 119, 190]},
```

#### 3. Add or Remove Colors
- **Expand palette**: Add more colors for greater variety
- **Reduce palette**: Remove colors that are rarely used
- **Maintain limits**: Keep within practical limits (typically 26-50 colors)

### How to Update Colors

1. **Edit the palette**: Modify the `COLORED_PENCIL_PALETTE` array in `mysterygen.py`
2. **Update display**: Run `generate_palette_display.py` to create new reference
3. **Test changes**: Process sample images to verify color selection works well
4. **Regenerate existing**: Reprocess any existing images with new palette

### Palette Display Generator

Use the included `generate_palette_display.py` script to create visual references:

```bash
./venv/bin/python generate_palette_display.py
```

This generates:
- **SVG version**: `palette_display.svg` - Scalable for web use
- **PDF version**: `palette_display.pdf` - Print-ready reference

### Color Selection Algorithm

The system uses a smart color selection process:

1. **Analyze image**: Find dominant colors using K-means clustering
2. **Map to palette**: Match each dominant color to closest unused palette color
3. **Prioritize usage**: Select colors based on how much of the image they represent
4. **Prevent duplicates**: Ensure no color is used twice in the same image
5. **Fill to target**: Add additional colors if needed to reach requested number

### Considerations for Color Changes

- **Distinctness**: Ensure colors are visually distinct when printed
- **Availability**: Use names that correspond to actual colored pencils people can buy
- **Balance**: Maintain good coverage across the color spectrum
- **Accessibility**: Consider colorblind-friendly options
- **Cultural context**: Use names that are widely recognized

### Future Enhancements

Potential color palette features:
- **Multiple palettes**: Different sets for different art styles
- **Seasonal palettes**: Spring, summer, fall, winter color sets
- **Brand-specific**: Match specific colored pencil brand sets
- **Custom upload**: Allow users to upload their own color palettes
- **Color suggestions**: AI-powered color recommendations based on image content

---

*To request color palette changes, provide feedback on the current palette or suggest specific alternatives. The development team can implement changes quickly and generate new reference materials.*