"""
sogmm_py
========

Python-wrapper over Self-Organizing Gaussian Mixture Models (SOGMM) package.
"""

__version__ = "0.0.0"

from . import sogmm, utils

__all__ = ["sogmm", "utils"]

from .sogmm import (SOGMM)
from .utils import (ImageUtils)

__all__.extend(["SOGMM", "ImageUtils"])

__pdoc__ = {}
__pdoc__['liegroups'] = False
__pdoc__['run_sogmm'] = False
__pdoc__['vis_open3d'] = False