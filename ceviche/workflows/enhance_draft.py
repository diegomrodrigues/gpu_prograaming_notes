from ceviche.core.context import Context
from ceviche.core.workflow import Workflow
from ceviche.core.utilities.model_utils import ModelUtilsMixin
from ceviche.core.utilities.file_utils import WithReadAndWriteFilesMixin
from typing import Dict, Any

class EnhanceDraftWorkflow(
    Workflow,
    ModelUtilsMixin,
    WithReadAndWriteFilesMixin
):
    def __init__(self):
        super().__init__()

    def before_start(self, ctx: Context, args: Dict[str, Any]):
        super().before_start(ctx, args)
        print("EnhanceDraftWorkflow before_start")

        default_tasks = [
            "cleanup",
            "generate_logical_steps",
            "generate_step_proofs",
            "generate_examples",
            "inject_images",
            "format_math"
        ]

        args_tasks = args.get("tasks", default_tasks)

        for task_name in args_tasks:
            self.load_task(task_name, ctx, args)
        
        self.load_task("content_verification")

    def run(self, ctx: Context, args: Dict[str, Any]) -> Any:
        print("EnhanceDraftWorkflow run")

        content = args.get("content")
        if not content:
            raise ValueError("Content is required in args for EnhanceDraftWorkflow.")
        base_directory = args.get("base_directory")
        directory = args.get("directory")
        tasks = args.get("tasks")

        # --- Iteration Loop ---
        for iteration in range(3):  # max_iterations=3
            print(f"Starting Enhancement Iteration {iteration + 1}")

            # --- Task Execution ---
            for task_name in tasks:
                task_instance = self.tasks[task_name]
                task_args = {
                    "content": content,
                    "base_directory": base_directory
                }
                content = task_instance.run(ctx, task_args)

            # --- Verification ---
            content_verification_task = self.tasks["content_verification"]
            verification_result = content_verification_task.run(ctx, {"content": content})
            if verification_result.strip().lower() == "yes":
                print("Content verification passed. Exiting enhancement loop.")
                break  # Exit loop if verification passes
            else:
                print("Content verification failed. Continuing to next iteration.")

        return content

    def after_start(self, ctx: Dict[str, Any], args: Dict[str, Any]):
        super().after_start(ctx, args)
        print("EnhanceDraftWorkflow after_start")
