"""
Main entry point for running the Streamlit OCR application.
"""
import subprocess
import sys
from pathlib import Path


def main():
    """Run the Streamlit server for the OCR application."""
    # Get the path to app.py in the same directory
    app_path = Path(__file__).parent / "app.py"
    
    if not app_path.exists():
        print(f"Error: {app_path} not found!")
        sys.exit(1)
    
    # Run streamlit with the app.py file
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path)
    ])


if __name__ == "__main__":
    main()
