from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pathlib import Path
import yaml
from ceviche.core.context import Context
from ceviche.core.task import Task

class Workflow(ABC):
    """Abstract base class for all workflows."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_configs: Dict[str, Dict[str, Any]] = {}
        self._load_all_task_configs()

    def _load_all_task_configs(self):
        """Loads configurations for all tasks in the tasks directory."""
        tasks_dir = Path(__file__).resolve().parent.parent / "tasks"
        for task_dir in tasks_dir.iterdir():
            if task_dir.is_dir():
                task_name = task_dir.name
                yaml_path = task_dir / f"{task_name}.yaml"
                if yaml_path.exists():
                    try:
                        with open(yaml_path, 'r', encoding='utf-8') as f:
                            config = yaml.safe_load(f)
                            # Ensure the config is a dictionary and contains the task name
                            if isinstance(config, dict) and task_name in config:
                                self.task_configs[task_name] = config[task_name]
                            else:
                                print(f"Warning: Invalid YAML configuration for task '{task_name}'")
                    except Exception as e:
                        print(f"Error loading YAML configuration for task '{task_name}': {e}")

    def load_task(self, task_name: str, ctx: Context, args: Dict[str, Any]) -> Task:
        """Loads a task instance by name."""

        if task_name not in self.task_configs:
            raise ValueError(f"Task '{task_name}' configuration not found.")

        task_config = self.task_configs[task_name]

        # Dynamically import and instantiate the task class
        module_name = f"ceviche.tasks.{task_name}.{task_name}"
        class_name = "".join(part.title() for part in task_name.split("_")) + "Task"
        try:
            module = __import__(module_name, fromlist=[class_name])
            task_class = getattr(module, class_name)
            task_instance = task_class(task_config, task_name)
            self.tasks[task_name] = task_instance
            
            # Wrap the run method with pre/post hooks
            original_run = task_instance.run
            def wrapped_run(ctx, args):
                task_instance.pre_run(ctx, args)
                result = original_run(ctx, args)
                task_instance.post_run(ctx, args)
                return result
            task_instance.run = wrapped_run
            
            # Call load hook with context and args
            task_instance.load(ctx, args)
            return task_instance
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import or instantiate task '{task_name}': {e}")

    def before_start(self, ctx: Dict[str, Any], args: Dict[str, Any]):
        """Hook called before the workflow starts."""
        print(f"Workflow before_start")

    @abstractmethod
    def run(self, ctx: Dict[str, Any], args: Dict[str, Any]) -> Any:
        """Executes the workflow logic. Must be implemented by subclasses."""
        pass

    def after_start(self, ctx: Dict[str, Any], args: Dict[str, Any]):
        """Hook called after the workflow finishes."""
        print("Workflow after_start")