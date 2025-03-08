from ceviche.core.context import Context
from ceviche.core.task import Task
from typing import Any, Dict
from ceviche.core.utilities.model_utils import ModelUtilsMixin
from ceviche.core.utilities.file_utils import WithReadAndWriteFilesMixin
from ceviche.core.utilities.json_utils import JsonUtilitiesMixin
from pathlib import Path

class InjectImagesTask(
    Task,
    WithReadAndWriteFilesMixin,
    JsonUtilitiesMixin,
    ModelUtilsMixin
):

    def __init__(self, task_config: Dict[str, Any], task_name: str):
        super().__init__(task_config, task_name)

    def run(self, ctx: Context, args: Dict[str, Any]) -> Any:
        print(f"Running InjectImagesTask")

        self.model = self.init_model(ctx, args)  # Initialize the model

        # 1. Prepare the prompt:
        content = args.get("content", "")
        base_directory = args.get("base_directory")

        # Skip on local executions
        if ctx.get("mock_api", False):
            return content

        user_message, success = self._prepare_content(content, base_directory, self.task_config)
        if not success:
            raise Exception("Failed to prepare user content for InjectImagesTask.")
        prompt = user_message

        # 2. Interact with the model:
        chat = self.start_chat()  # No files to upload
        response = self.send_message(chat=chat, user_content=prompt)
        result = response.text

        # 3. Process and return the result:
        if "<!-- END -->" not in result:
            raise Exception("Image injection did not complete (missing <!-- END --> marker).")

        return result
    
    def _prepare_content(self, content: str, directory: str, task_config: Dict[str, Any]) -> tuple[str, bool]:
        """Prepare the user content, handling image placeholders (moved from ModelUtilsMixin)."""
        user_message = task_config.get("user_message", "")
        
        try:
            images_file = Path(directory) / "images.md"

            if images_file.exists():
                images_content = images_file.read_text(encoding='utf-8')
            else:
                print(f"No images.md available in directory: {directory}")
                return "No images available", False

            user_message = user_message.replace("{images_content}", images_content)
        except (StopIteration, IndexError):
            user_message = user_message.replace("{images_content}", "No directory context available")

        return user_message.replace("{content}", content), True
