from __future__ import annotations

import os


def setup_matplotlib_headless() -> None:
    """
    Ensure matplotlib works in headless / restricted environments by:
    - forcing Agg backend
    - routing caches to writable locations
    """
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    # Fontconfig cache often defaults to unwritable paths; XDG_CACHE_HOME helps.

    import matplotlib

    try:
        matplotlib.use("Agg", force=True)
    except Exception:
        # If backend already selected, we still try to proceed.
        pass

