from ceviche.core.context import Context
from ceviche.core.task import Task
from typing import Any, Dict
from ceviche.core.utilities.model_utils import ModelUtilsMixin
from ceviche.core.utilities.json_utils import JsonUtilitiesMixin

class ConsolidateSubtopicsTask(Task, ModelUtilsMixin, JsonUtilitiesMixin):
    def __init__(self, task_config: Dict[str, Any], task_name: str):
        super().__init__(task_config, task_name)

    def run(self, ctx: Context, args: Dict[str, Any]) -> Any:
        print(f"Running ConsolidateSubTopicsTask with config")

        self.model = self.init_model(ctx, args)

        input_json = args.get("topics")
        if not input_json:
            raise ValueError("No input from topics found in context.")

        # Prepare prompt
        prompt = self.prepare_prompt(
            task_config=self.task_config,
            content=self.dump_json(input_json),  # Pass JSON as string
        )

        # Model interaction
        chat = self.start_chat()  # No files for this task
        response = self.send_message(chat=chat, user_content=prompt)
        result = response.text

        # Response processing
        if not ctx.get("mock_api", False):
            extracted_json = self.extract_json(result)
            if not extracted_json:
                raise Exception("Failed to extract JSON from consolidate_subtopics response.")
        else:
            extracted_json = self._mock_json()

        return extracted_json

    def _mock_json(self): 
        return {"topics": [{"topic": "Section 1", "sub_topics": ["Concept A", "Concept B"]}]}