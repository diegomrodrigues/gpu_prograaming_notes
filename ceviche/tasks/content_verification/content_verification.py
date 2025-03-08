from ceviche.core.context import Context
from ceviche.core.task import Task
from typing import Any, Dict
from ceviche.core.utilities.model_utils import ModelUtilsMixin
from ceviche.core.utilities.file_utils import WithReadAndWriteFilesMixin
from ceviche.core.utilities.json_utils import JsonUtilitiesMixin
from pathlib import Path

class ContentVerificationTask(
    Task,
    ModelUtilsMixin,
    WithReadAndWriteFilesMixin,
    JsonUtilitiesMixin
):
    def __init__(self, task_config: Dict[str, Any], task_name: str):
        super().__init__(task_config, task_name)

    def run(self, ctx: Context, args: Dict[str, Any]) -> Any:
        print(f"Running ContentVerificationTask")

        self.model = self.init_model(ctx, args)

        content = args.get("content", "")
        prompt = self.prepare_prompt(self.task_config, content)

        chat = self.start_chat()
        response = self.send_message(chat, prompt)
        result = response.text

        if not ctx.get("mock_api", False):
            return result 
        else:
            return "yes"