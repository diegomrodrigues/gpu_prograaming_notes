from ceviche.core.task import Task
from typing import Any, Dict
from ceviche.core.utilities.model_utils import ModelUtilsMixin
from ceviche.core.utilities.file_utils import WithReadAndWriteFilesMixin
from ceviche.core.utilities.json_utils import JsonUtilitiesMixin  # Included for completeness
from pathlib import Path

class CleanupTask(
    Task,
    ModelUtilsMixin,
    WithReadAndWriteFilesMixin,
    JsonUtilitiesMixin  # Included for completeness
):

    def __init__(self, task_config: Dict[str, Any], task_name: str):
        super().__init__(task_config, task_name)

    def run(self, ctx: Dict[str, Any], args: Dict[str, Any]) -> Any:
        print(f"Running CleanupTask")

        self.model = self.init_model(ctx, args)  # Initialize the model

        # 1. Prepare the prompt:
        content = args.get("content", "")  # Get the content to clean up
        prompt = self.prepare_prompt(self.task_config, content)

        # 2. File Handling (Not directly applicable for cleanup, but included for consistency)
        #    Cleanup typically operates on the *output* of a previous task, not input files.
        #    If you *did* need file input, you'd add it here, similar to GenerateDraftTask.

        # 3. Interact with the model:
        chat = self.start_chat()  # No files needed for a simple cleanup
        response = self.send_message(chat=chat, user_content=prompt)
        result = response.text

        # 4. Process and return the result:
        #    Cleanup returns modified text, *not* JSON.
        if "<!-- END -->" not in result:
            raise Exception("Cleanup did not complete (missing <!-- END --> marker).")

        return result