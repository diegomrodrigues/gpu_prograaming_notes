from ceviche.core.context import Context
from ceviche.core.task import Task
from typing import Any, Dict
from ceviche.core.utilities.model_utils import ModelUtilsMixin
from ceviche.core.utilities.file_utils import WithReadAndWriteFilesMixin
from ceviche.core.utilities.json_utils import JsonUtilitiesMixin
import json
from pathlib import Path

class CreateTopicsTask(
    Task, 
    ModelUtilsMixin, 
    WithReadAndWriteFilesMixin, 
    JsonUtilitiesMixin
):
    
    def __init__(self, task_config: Dict[str, Any], task_name: str):
        super().__init__(task_config, task_name)

    def run(self, ctx: Context, args: Dict[str, Any]) -> Any:
        print(f"Running CreateTopicsTask")

        self.model = self.init_model(ctx, args)

        # 1. Prepare the prompt:
        content = args.get("perspective", "")  # Get content from arguments
        prompt = self.prepare_prompt(self.task_config, content)

        # 2. File Handling (using the mixin):
        pdf_files = self.get_pdf_files(args.get("directory", "."))
        uploaded_files = self.upload_files(pdf_files)

        print(f"Uploaded files for topic generation")

        # 3. Interact with the model (using mixin methods):
        chat = self.start_chat(files=uploaded_files)
        response = self.send_message(chat=chat, user_content=prompt, files=uploaded_files)
        result = response.text

        # 4. Process and return the result:
        if not ctx.get("mock_api", False):
            extracted_json = self.extract_json(result)
            if not extracted_json:
                raise Exception("Failed to extract valid JSON from response.")
        else:
            extracted_json = self._mock_json()

        return extracted_json
    
    def _mock_json(self): 
        return {"topics": [{"topic": "Section 1", "sub_topics": ["Concept A", "Concept B"]}]}