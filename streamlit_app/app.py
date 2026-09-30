"""
FinTech AI - legacy entry point.

The dashboard lives in streamlit_app/streamlit_app.py (the Streamlit Cloud
entry file). ``streamlit run streamlit_app/app.py`` runs the same app.
"""

import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_app.py"), run_name="__main__")
