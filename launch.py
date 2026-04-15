"""
launch.py — Streamlit launcher with Windows registry permission patch.

Python 3.10 on Windows can raise PermissionError when mimetypes.init()
tries to read HKEY_CLASSES_ROOT. This is fixed in Python 3.11.1+.
This script patches MimeTypes.read_windows_registry() before Streamlit
initialises, so the app starts cleanly on restricted Windows environments.

Usage (from the sportvision/ folder):
    python launch.py
"""

import mimetypes
import sys

# ── Patch mimetypes before Streamlit touches it ───────────────────────────────
if hasattr(mimetypes.MimeTypes, "read_windows_registry"):
    _original_rwr = mimetypes.MimeTypes.read_windows_registry

    def _safe_read_windows_registry(self, strict=True):
        """Wrap the registry read in a try/except for restricted Windows users."""
        try:
            _original_rwr(self, strict)
        except (PermissionError, OSError):
            pass  # Skip registry MIME types if access is denied

    mimetypes.MimeTypes.read_windows_registry = _safe_read_windows_registry

# Pre-initialise mimetypes now so Streamlit's add_type() calls don't re-trigger it
mimetypes.init()

# ── Hand off to Streamlit CLI ─────────────────────────────────────────────────
sys.argv = ["streamlit", "run", "app.py"]

from streamlit.web import cli as stcli
stcli.main()

