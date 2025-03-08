from ceviche.core.context import Context
from ceviche.core.task import Task
from typing import Any, Dict
from ceviche.core.utilities.model_utils import ModelUtilsMixin
from pathlib import Path
import re

class GenerateFilenameTask(Task, ModelUtilsMixin):
    def __init__(self, task_config: Dict[str, Any], task_name: str):
        super().__init__(task_config, task_name)

    def run(self, ctx: Context, args: Dict[str, Any]) -> str:
        """Generates a filename based on the provided topic."""

        self.model = self.init_model(ctx, args)

        topic = args.get("topic")
        if not topic:
            raise ValueError("A 'topic' argument must be provided.")

        prompt = self.prepare_prompt(self.task_config, content=topic)
        chat = self.start_chat()  # No files needed
        response = self.send_message(chat=chat, user_content=prompt)
        filename = response.text.strip()

        if ctx.get("mock_api"):
            return topic

        return filename 