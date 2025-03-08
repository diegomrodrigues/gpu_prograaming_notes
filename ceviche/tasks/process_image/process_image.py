from ceviche.core.context import Context
from ceviche.core.task import Task
from typing import Any, Dict
from ceviche.core.utilities.model_utils import ModelUtilsMixin
from pathlib import Path

class ProcessImageTask(Task, ModelUtilsMixin):
    
    def run(self, ctx: Context, args: Dict[str, Any]) -> Dict[str, Any]:
        image_file = Path(args.get("image_file"))
        pdf_file = Path(args.get("pdf_file"))

        self.model = self.init_model(ctx, args)

        if not image_file.exists() or not pdf_file.exists():
            raise ValueError("Image and PDF files must exist.")

        prompt = self.prepare_prompt(self.task_config, content="")
        uploaded_files = self.upload_files([image_file, pdf_file])
        chat = self.start_chat(files=uploaded_files)
        response = self.send_message(chat=chat, user_content=prompt, files=uploaded_files)
        result = response.text

        extracted_json = self.extract_json(result)
        if not extracted_json:
            raise Exception("Failed to extract JSON from process_image response.")

        return extracted_json 