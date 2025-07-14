#!/usr/bin/env python3
from http.server import HTTPServer, SimpleHTTPRequestHandler
import cgi
import json
import os
import subprocess
from urllib.parse import parse_qs

class ImageHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        """Handle POST requests for image processing"""
        if self.path == '/process':
            # Get content type
            content_type = self.headers.get('Content-Type')
            if not content_type or not content_type.startswith('multipart/form-data'):
                self.send_error(400, "Bad Request: Must be multipart/form-data")
                return

            # Parse the form data
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    'REQUEST_METHOD': 'POST',
                    'CONTENT_TYPE': self.headers['Content-Type'],
                }
            )

            try:
                # Get the file
                if 'image' not in form:
                    self.send_error(400, "No image file provided")
                    return

                fileitem = form['image']
                if not fileitem.filename:
                    self.send_error(400, "No filename provided")
                    return

                # Create directories if they don't exist
                for directory in ['uploads', 'output']:
                    os.makedirs(directory, exist_ok=True)

                # Save the file
                filename = os.path.join('uploads', fileitem.filename)
                with open(filename, 'wb') as f:
                    f.write(fileitem.file.read())

                # Get processing parameters
                colors = form.getvalue('colors', '12')
                grid_size = form.getvalue('grid_size', '30')

                # Run the script
                cmd = ['python', 'mysterygen.py', filename, '--colors', colors, '--grid-size', grid_size]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    self.send_error(500, f"Processing failed: {result.stderr}")
                    return

                # Get output files
                base_name = os.path.splitext(fileitem.filename)[0]
                expected_files = {
                    'grid': f'{base_name}_mystery_grid.svg',
                    'colored': f'{base_name}_mystery_colored.svg',
                    'combined': f'{base_name}_mystery_combined.svg',
                    'legend': f'{base_name}_mystery_legend.svg'
                }

                output_files = {}
                for key, filename in expected_files.items():
                    filepath = os.path.join('output', filename)
                    if os.path.exists(filepath):
                        output_files[key] = f'/output/{filename}'

                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    'message': 'Image processed successfully',
                    'files': output_files,
                    'stdout': result.stdout
                }
                self.wfile.write(json.dumps(response).encode())

            except Exception as e:
                self.send_error(500, f"Server error: {str(e)}")
                return

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