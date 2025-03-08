from typing import Dict, Any

class Context:
    """Shared context for workflows and tasks."""

    def __init__(self, initial_data: Dict[str, Any] = None):
        self.data = initial_data if initial_data is not None else {}

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any):
        self.data[key] = value

    def __contains__(self, key: str):
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)