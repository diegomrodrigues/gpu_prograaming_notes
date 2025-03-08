from typing import Any, Dict
from pathlib import Path

class WithReadAndWriteFilesMixin:
    def read_file(self, filename: str) -> str:
        # Implement file reading logic
        return Path(filename).read_text()

    def write_file(self, content: str, filename: str):
        # Implement file writing logic
        Path(filename).write_text(content)