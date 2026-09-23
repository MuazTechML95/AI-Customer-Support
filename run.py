"""
run.py
------
WHAT THIS FILE DOES:
Convenience entry point for starting the app from the project root without
remembering the full Streamlit command/path.

HOW TO USE:
    python run.py

WHAT IT ACTUALLY DOES UNDER THE HOOD:
Equivalent to running:
    streamlit run app/ui/streamlit_app.py

This file exists purely for developer convenience - all real logic lives in
app/ui/streamlit_app.py and the modules it calls.
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    ui_path = os.path.join(os.path.dirname(__file__), "app", "ui", "streamlit_app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", ui_path])
