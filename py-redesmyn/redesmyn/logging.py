from pathlib import Path
from typing import Optional, Self

from redesmyn.py_redesmyn import LogConfig as _LogConfig


class LogConfig:
    """`LogConfig` configures logging output for the application.

    Redesmyn applications write to two logging output files: a primary logging output file,
    specified via the `path` initialization parameter, and an optional Amazon Web Services (AWS)
    Embedded Metrics Format (EMF) output file, specified via the `emf_path` initialization parameter.
    """

    def __init__(self, path: Path, emf_path: Optional[Path] = None):
        """Initialize a logging configuration for the present application."""
        self._log_config = _LogConfig(path=path, emf_path=emf_path)

    def init(self):
        """Initialize logging for the present application using this logging configuration."""
        self._log_config.init()
