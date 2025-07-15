#!/usr/bin/env python3
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
import subprocess
from urllib.parse import parse_qs
import tempfile
import shutil
import io

class ImageHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        """Handle POST requests for image processing"""
        print(f"POST request received for path: {self.path}")
        print(f"Headers: {dict(self.headers)}")
        
        if self.path == '/process':
            # Get content type
            content_type = self.headers.get('Content-Type')
            if not content_type or not content_type.startswith('multipart/form-data'):
                self.send_error(400, "Bad Request: Must be multipart/form-data")
                return

            try:
                # Simple multipart parser
                content_length = int(self.headers.get('Content-Length', 0))
                print(f"Content-Length: {content_length}")
                post_data = self.rfile.read(content_length)
                print(f"Read {len(post_data)} bytes of data")
                
                # Extract boundary
                boundary = content_type.split('boundary=')[1]
                boundary_bytes = ('--' + boundary).encode()
                
                # Split by boundary
                parts = post_data.split(boundary_bytes)
                
                form_data = {}
                image_data = None
                filename = None
                
                for part in parts:
                    if not part.strip():
                        continue
                        
                    if b'Content-Disposition' in part:
                        # Split headers and data
                        header_end = part.find(b'\r\n\r\n')
                        if header_end == -1:
                            continue
                            
                        headers = part[:header_end].decode('utf-8', errors='ignore')
                        data = part[header_end + 4:]
                        
                        # Remove trailing boundary markers
                        if data.endswith(b'\r\n'):
                            data = data[:-2]
                        if data.endswith(b'--'):
                            data = data[:-2]
                        if data.endswith(b'\r\n'):
                            data = data[:-2]
                        
                        if 'name="image"' in headers:
                            # Extract filename
                            if 'filename=' in headers:
                                filename_start = headers.find('filename="') + 10
                                filename_end = headers.find('"', filename_start)
                                filename = headers[filename_start:filename_end]
                            image_data = data
                        elif 'name="colors"' in headers:
                            form_data['colors'] = data.decode().strip()
                        elif 'name="grid_size"' in headers:
                            form_data['grid_size'] = data.decode().strip()
                
                if not image_data or not filename:
                    print("ERROR: No image data or filename found")
                    print(f"image_data length: {len(image_data) if image_data else 0}")
                    print(f"filename: {filename}")
                    self.send_error(400, "No image file provided")
                    return
                    
                print(f"Successfully parsed: filename={filename}, image_data={len(image_data)} bytes")

                # Create directories if they don't exist
                for directory in ['uploads', 'output']:
                    os.makedirs(directory, exist_ok=True)

                # Save the file
                filepath = os.path.join('uploads', filename)
                with open(filepath, 'wb') as f:
                    f.write(image_data)

                # Get processing parameters
                colors = form_data.get('colors', '12')
                grid_size = form_data.get('grid_size', '30')

                # Run the script using virtual environment and get color data
                cmd = ['./venv/bin/python', '-c', f'''
import sys
sys.path.append('.')
from mysterygen import *
import json

# Process the image
processor = ImageProcessor("{filepath}", {colors})
processor.load_and_resize()
quantized_image, color_map, number_map, color_names = processor.quantize_colors()

# Generate grid
grid_generator = GridGenerator(quantized_image, color_map, number_map, color_names, {grid_size})
grid_generator.generate_grid()

# Render SVG files
base_name = "{os.path.splitext(filename)[0]}"
renderer = SVGRenderer(grid_generator, base_name, {colors}, {grid_size})
renderer.render_grid()
renderer.render_colored_grid()
renderer.render_combined_grid()
renderer.render_legend()
renderer.render_bw_legend()

# Output color data as JSON
color_data = []
for color_idx, color in color_map.items():
    color_data.append({{
        "number": int(number_map[color_idx]),
        "color": [int(c) for c in color],
        "name": color_names[color_idx]
    }})

print("COLOR_DATA:" + json.dumps(color_data))
print("Processing completed successfully")
''']
                print(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                print(f"Command exit code: {result.returncode}")
                print(f"Command stdout: {result.stdout}")
                print(f"Command stderr: {result.stderr}")

                if result.returncode != 0:
                    error_msg = f"Processing failed: {result.stderr}"
                    print(f"Sending error: {error_msg}")
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    error_response = {'error': error_msg}
                    self.wfile.write(json.dumps(error_response).encode())
                    return
                
                # Extract color data from stdout
                color_data = []
                for line in result.stdout.split('\n'):
                    if line.startswith('COLOR_DATA:'):
                        color_data = json.loads(line[11:])
                        break

                # Get output files
                base_name = os.path.splitext(filename)[0]
                param_suffix = f"_{colors}color_{grid_size}grid"
                expected_files = {
                    'grid': f'{base_name}_grid{param_suffix}.svg',
                    'colored': f'{base_name}_colored{param_suffix}.svg',
                    'combined': f'{base_name}_combined{param_suffix}.svg',
                    'key': f'{base_name}_key{param_suffix}.svg',
                    'bw_key': f'{base_name}_bw_key{param_suffix}.svg'
                }

                output_files = {}
                for key, filename in expected_files.items():
                    filepath = os.path.join('output', filename)
                    if os.path.exists(filepath):
                        output_files[key] = f'/output/{filename}'

                # Send response
                response = {
                    'message': 'Image processed successfully',
                    'files': output_files,
                    'colors': color_data,
                    'stdout': result.stdout
                }
                response_json = json.dumps(response)
                print(f"Sending response: {response_json}")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Content-Length', str(len(response_json.encode())))
                self.end_headers()
                self.wfile.write(response_json.encode())

            except Exception as e:
                print(f"Server error: {str(e)}")
                import traceback
                traceback.print_exc()
                self.send_error(500, f"Server error: {str(e)}")
                return

        elif self.path == '/export-both-pdfs':
            # Handle combined PDF export
            try:
                # Parse form data to get parameters
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                
                # Extract boundary
                content_type = self.headers.get('Content-Type')
                boundary = content_type.split('boundary=')[1]
                boundary_bytes = ('--' + boundary).encode()
                
                # Split by boundary
                parts = post_data.split(boundary_bytes)
                
                form_data = {}
                for part in parts:
                    if not part.strip():
                        continue
                        
                    if b'Content-Disposition' in part:
                        # Split headers and data
                        header_end = part.find(b'\r\n\r\n')
                        if header_end == -1:
                            continue
                            
                        headers = part[:header_end].decode('utf-8', errors='ignore')
                        data = part[header_end + 4:]
                        
                        # Remove trailing boundary markers
                        if data.endswith(b'\r\n'):
                            data = data[:-2]
                        if data.endswith(b'--'):
                            data = data[:-2]
                        if data.endswith(b'\r\n'):
                            data = data[:-2]
                        
                        if 'name="colors"' in headers:
                            form_data['colors'] = data.decode().strip()
                        elif 'name="grid_size"' in headers:
                            form_data['grid_size'] = data.decode().strip()
                
                colors = form_data.get('colors', '12')
                grid_size = form_data.get('grid_size', '30')
                
                # Get the latest file prefix from uploads
                uploads_dir = 'uploads'
                if os.path.exists(uploads_dir):
                    files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if files:
                        # Use the most recent file
                        latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(uploads_dir, f)))
                        base_name = os.path.splitext(latest_file)[0]
                        output_prefix = base_name
                        
                        # Create a dummy grid generator for PDF export
                        from mysterygen import SVGRenderer
                        class DummyGrid:
                            pass
                        
                        dummy_grid = DummyGrid()
                        renderer = SVGRenderer(dummy_grid, output_prefix, int(colors), int(grid_size))
                        
                        # Export both PDFs
                        pdf_paths = renderer.export_both_pdfs()
                        
                        # Return JSON response with both file paths
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        response = {
                            'message': 'PDFs exported successfully',
                            'files': pdf_paths
                        }
                        self.wfile.write(json.dumps(response).encode())
                        return
                
                self.send_error(404, "No files to export")
            except Exception as e:
                print(f"Combined PDF export error: {str(e)}")
                self.send_error(500, f"Combined PDF export failed: {str(e)}")
        
        elif self.path.startswith('/export-pdf'):
            # Handle PDF export requests
            try:
                # Parse the URL to get the file type
                parts = self.path.split('/')
                if len(parts) >= 3:
                    file_type = parts[2]  # grid, key, colored, or combined
                    
                    # Get the latest file prefix from uploads
                    uploads_dir = 'uploads'
                    if os.path.exists(uploads_dir):
                        files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if files:
                            # Use the most recent file
                            latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(uploads_dir, f)))
                            base_name = os.path.splitext(latest_file)[0]
                            output_prefix = base_name
                            
                            # Create a dummy grid generator for PDF export
                            from mysterygen import SVGRenderer
                            class DummyGrid:
                                pass
                            
                            dummy_grid = DummyGrid()
                            renderer = SVGRenderer(dummy_grid, output_prefix)
                            
                            # Export to PDF
                            pdf_path = renderer.export_to_pdf(file_type)
                            
                            # Send the PDF file
                            with open(pdf_path, 'rb') as f:
                                self.send_response(200)
                                self.send_header('Content-Type', 'application/pdf')
                                self.send_header('Content-Disposition', f'attachment; filename="{os.path.basename(pdf_path)}"')
                                self.end_headers()
                                self.wfile.write(f.read())
                            return
                
                self.send_error(404, "PDF not found")
            except Exception as e:
                print(f"PDF export error: {str(e)}")
                self.send_error(500, f"PDF export failed: {str(e)}")
        
        else:
            self.send_error(404, "Not Found")

    def do_GET(self):
        """Serve static files and handle other GET requests"""
        # Redirect root to index.html
        if self.path == '/':
            self.path = '/index.html'
        return SimpleHTTPRequestHandler.do_GET(self)

def run_server(port=8000):
    """Start the server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, ImageHandler)
    print(f"Server running at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server() 