import platform
import os
from typing import Dict


def get_device_info() -> Dict[str, str]:
    """Get information about available compute devices."""
    system = platform.system()
    processor = platform.processor()
    n_cores = os.cpu_count()

    if system == "Darwin" and processor == "arm":  # Apple Silicon
        try:
            import sklearn_apple

            return {"device": "Apple Silicon GPU", "type": "M-series", "backend": "Metal", "cores": str(n_cores)}
        except ImportError:
            return {"device": "Apple Silicon CPU", "type": "M-series", "backend": "OpenMP", "cores": str(n_cores)}
    else:
        return {"device": "CPU", "type": "Unknown", "backend": "OpenMP", "cores": str(n_cores)}
