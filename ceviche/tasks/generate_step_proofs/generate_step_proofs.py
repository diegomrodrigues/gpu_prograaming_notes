from ceviche.core.context import Context
from ceviche.core.task import Task
from typing import Any, Dict
from ceviche.core.utilities.model_utils import ModelUtilsMixin
from ceviche.core.utilities.file_utils import WithReadAndWriteFilesMixin
from ceviche.core.utilities.json_utils import JsonUtilitiesMixin  # Included for completeness
from pathlib import Path

class GenerateStepProofsTask(
    Task,
    ModelUtilsMixin,
    WithReadAndWriteFilesMixin,
    JsonUtilitiesMixin  # Included for completeness
):
    def __init__(self, task_config: Dict[str, Any], task_name: str):
        super().__init__(task_config, task_name)

    def run(self, ctx: Context, args: Dict[str, Any]) -> Any:
        print(f"Running GenerateStepProofsTask")

        self.model = self.init_model(ctx, args)  # Initialize the model

        content = args.get("content", "")
        prompt = self.prepare_prompt(self.task_config, content)

        # Interact with the model:
        chat = self.start_chat()  # No files needed
        response = self.send_message(chat=chat, user_content=prompt)
        result = response.text

        # Process and return the result:
        if "<!-- END -->" not in result:
            raise Exception("Step proof generation did not complete (missing <!-- END --> marker).")

        return result 