from ceviche.core.task import Task
from typing import Any, Dict
from ceviche.core.utilities.model_utils import ModelUtilsMixin
from ceviche.core.utilities.file_utils import WithReadAndWriteFilesMixin
from ceviche.core.utilities.json_utils import JsonUtilitiesMixin  # Although not directly used, good practice to include
from pathlib import Path

class GenerateDraftTask(
    Task,
    ModelUtilsMixin,
    WithReadAndWriteFilesMixin,
    JsonUtilitiesMixin # Included for completeness, even if not all methods are *directly* used in `run`
):

    def __init__(self, task_config: Dict[str, Any], task_name: str):
        super().__init__(task_config, task_name)

    def run(self, ctx: Dict[str, Any], args: Dict[str, Any]) -> Any:
        print(f"Running GenerateDraftTask with config")

        self.model = self.init_model(ctx, args)  # Initialize the model

        # 1. Prepare the prompt:
        content = args.get("content", "")
        prompt = self.prepare_prompt(self.task_config, content)

        # 2. File Handling (if needed, adapt as necessary):
        #    This example assumes you might *optionally* use files.  If files are
        #    *never* used, remove this section. If they are *always* required,
        #    remove the `if` check.
        if "directory" in args:  # Check if a directory is provided
            pdf_files = self.get_pdf_files(args.get("directory", "."))
            uploaded_files = self.upload_files(pdf_files)
        else:
            uploaded_files = None

        # 3. Interact with the model:
        chat = self.start_chat(files=uploaded_files)  # Pass files if they exist
        response = self.send_message(chat=chat, user_content=prompt, files=uploaded_files)
        result = response.text

        # 4. Process and return the result:
        #    The generate_draft task is expected to return raw text, *not* JSON.
        #    Therefore, we *don't* try to parse it as JSON.
        if "<!-- END -->" not in result:
            raise Exception("Draft generation did not complete (missing <!-- END --> marker).")

        return result
