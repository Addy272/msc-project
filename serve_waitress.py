"""
Run the Flask app with Waitress for an easy Windows-friendly demo deployment.

Usage:
    python serve_waitress.py
    set PORT=8000 && python serve_waitress.py
"""

import os

from waitress import serve

from app import app


def main():
    """Start the application with production-style defaults."""
    host = os.environ.get('HOST', '0.0.0.0')
    #host = "110.44.10.114"
    port = int(os.environ.get('PORT', '5000'))
    threads = int(os.environ.get('WAITRESS_THREADS', '4'))

    print(f"Starting Waitress on {host}:{port}")
    print(f"Open this on your laptop: http://localhost:{port}")
    if host == '0.0.0.0':
        print(f"Open this on the same Wi-Fi using your laptop IP: http://YOUR_LAPTOP_IP:{port}")
    serve(app, host=host, port=port, threads=threads)


if __name__ == '__main__':
    main()
