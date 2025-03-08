from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path
import yaml

class Task(ABC):
    """Abstract base class for all tasks."""

    def __init__(self, task_config: Dict[str, Any], task_name: str):
        self.task_config = task_config
        self.task_name = task_name
        self.model_name = task_config.get('model_name', 'gemini-2.0-flash-exp')  # Example
        self.temperature = task_config.get('temperature', 0.7)
        self.response_mime_type = task_config.get('response_mime_type', 'text/plain')
        # ... other common task configurations ...

    def config(self, ctx: Dict[str, Any], yaml_config: Dict[str, Any]):
        """
        Called with the yaml file in the same directory.
        This method can be used to validate the yaml file.
        """
        pass

    def load(self, ctx: Dict[str, Any], args: Dict[str, Any]):
        """
        Called in the workflows Workdlow().load_task(name).
        This method can be used to load resources.
        """
        print(f"Task load: {self.task_name}")

    def before_start(self, ctx: Dict[str, Any], args: Dict[str, Any]):
        """Hook called before the workflow starts."""
        print(f"Task before_start: {self.task_name}")
        pass

    def pre_run(self, ctx: Dict[str, Any], args: Dict[str, Any]):
        """Hook called before the task runs."""
        print(f"Task pre_run: {self.task_name}")
        pass

    @abstractmethod
    def run(self, ctx: Dict[str, Any], args: Dict[str, Any]) -> Any:
        """Executes the core task logic. Must be implemented by subclasses."""
        pass

    def post_run(self, ctx: Dict[str, Any], args: Dict[str, Any]):
        """Hook called after the task runs."""
        print(f"Task post_run: {self.task_name}")
        pass

    def after_start(self, ctx: Dict[str, Any], args: Dict[str, Any]):
        """Hook called after the workflow finish."""
        print(f"Task after_start: {self.task_name}")
        pass

    def _load_yaml_config(self, task_file: str) -> Dict[str, Any]:
        """Loads the YAML configuration for the task."""
        current_dir = Path(__file__).resolve().parent
        yaml_path = current_dir.parent / self.task_name / f"{task_file}.yaml"

        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)