from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type
from pathlib import Path
from ceviche.core.workflow import Workflow
from ceviche.core.context import Context
import importlib
import pkgutil

class Agent(ABC):
    """
    Abstract base class for agents.  Agents orchestrate workflows.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.workflows: Dict[str, Workflow] = {}
        self._load_workflows()

    def _load_workflows(self):
        """
        Dynamically loads workflow instances from the ceviche.workflows directory.
        This uses a meta-approach, avoiding hardcoded workflow references.
        """
        workflows_package = 'ceviche.workflows'
        try:
            package = importlib.import_module(workflows_package)
        except ImportError:
            print(f"Error: Could not import workflows package '{workflows_package}'.")
            return

        # Get all workflow classes from the package
        for _, workflow_name, _ in pkgutil.iter_modules(package.__path__):
            try:
                # Import the workflow module
                module = importlib.import_module(f"{workflows_package}.{workflow_name}")
                
                # Look for the main workflow class that matches the module name
                expected_class_name = ''.join(word.capitalize() for word in workflow_name.split('_')) + 'Workflow'
                workflow_class = getattr(module, expected_class_name, None)
                
                if workflow_class and issubclass(workflow_class, Workflow) and workflow_class != Workflow:
                    # Use the module name (without _workflow) as the key
                    workflow_key = workflow_name.replace('_workflow', '')
                    self.workflows[workflow_key] = workflow_class()
                    if self.debug:
                        print(f"Loaded workflow: {workflow_key} -> {workflow_class.__name__}")
            except ImportError as e:
                print(f"Error: Could not import workflow '{workflow_name}'. {e}")
            except Exception as e:
                print(f"Error loading workflow {workflow_name}: {e}")


    def get_workflow(self, workflow_name: str, ctx: Context, args: Dict[str, Any]) -> Workflow:
        """Retrieves a loaded workflow instance by name."""
        workflow = self.workflows.get(workflow_name)
        if not workflow:
            raise ValueError(f"Workflow '{workflow_name}' not found.")
        
        # Call workflow's before_start with context and args
        workflow.before_start(ctx, args)
        return workflow

    @abstractmethod
    def pre_execution(self, ctx: Context, args: Dict[str, Any]):
        """Hook called before workflow execution."""
        pass

    @abstractmethod
    def execute(self, ctx: Context, args: Dict[str, Any]) -> Any:
        """Executes the agent's main logic. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def post_execution(self, ctx: Context, args: Dict[str, Any], result: Any):
        """Hook called after workflow execution."""
        pass