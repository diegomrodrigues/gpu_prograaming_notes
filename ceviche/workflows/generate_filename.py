from ceviche.core.context import Context
from ceviche.core.workflow import Workflow
from ceviche.tasks.generate_filename.generate_filename import GenerateFilenameTask
from typing import Dict, Any, Tuple
from pathlib import Path
import re

class GenerateFilenameWorkflow(Workflow):
    def __init__(self):
        super().__init__()

    def before_start(self, ctx: Dict[str, Any], args: Dict[str, Any]):
        super().before_start(ctx, args)
        self.generate_filename_task = self.load_task("generate_filename", ctx, args)

    def _get_existing_numbers(self, directory: Path) -> set[int]:
        """Get set of existing file numbers in the directory."""
        existing_files = [f for f in directory.glob("*.md")]
        return {int(f.name.split('.')[0]) for f in existing_files if f.name.split('.')[0].isdigit()}


    def run(self, ctx: Context, args: Dict[str, Any]) -> Tuple[str, Path]:
        """Generates a numbered filename, handling existing files."""
        directory = Path(args["directory"])
        topic = args["topic"]
        section_name = args.get("section_name", "") # Optional section name

        if not directory.is_dir():
            raise ValueError(f"Directory does not exist: {directory}")

        suggested_name = self.generate_filename_task.run(ctx, {"topic": topic})

        # Get existing numbers
        existing_numbers = self._get_existing_numbers(directory)
        next_num = 1
        while next_num in existing_numbers:
            next_num += 1

        # Create new filename
        new_filename = f"{next_num:02d}. {suggested_name}.md"
        return new_filename, directory / new_filename
