from ceviche.core.context import Context
from ceviche.core.task import Task
from typing import Any, Dict
from ceviche.core.utilities.model_utils import ModelUtilsMixin
from ceviche.core.utilities.file_utils import WithReadAndWriteFilesMixin
from ceviche.core.utilities.json_utils import JsonUtilitiesMixin  # Included for completeness
from pathlib import Path

class FormatMathTask(
    Task,
    ModelUtilsMixin,
    WithReadAndWriteFilesMixin,
    JsonUtilitiesMixin  # Included for completeness
):

    def __init__(self, task_config: Dict[str, Any], task_name: str):
        super().__init__(task_config, task_name)

    def run(self, ctx: Context, args: Dict[str, Any]) -> Any:
        print(f"Running FormatMathTask with config")

        self.model = self.init_model(ctx, args)  # Initialize the model

        # 1. Prepare the prompt:
        content = args.get("content", "")  # Get the content to format
        prompt = self.prepare_prompt(self.task_config, content)

        # 2. File Handling (Not directly applicable for formatting, but included for consistency)
        #    Formatting typically operates on the *output* of a previous task, not input files.

        # 3. Interact with the model:
        chat = self.start_chat()  # No files needed
        response = self.send_message(chat=chat, user_content=prompt)
        result = response.text

        # 4. Process and return the result:
        #    Formatting returns modified text, *not* JSON.
        if "<!-- END -->" not in result:
            raise Exception("Math formatting did not complete (missing <!-- END --> marker).")

        return result