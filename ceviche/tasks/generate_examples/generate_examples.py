from ceviche.core.context import Context
from ceviche.core.task import Task
from typing import Any, Dict
from ceviche.core.utilities.model_utils import ModelUtilsMixin
from pathlib import Path

from ceviche.core.utilities.plot_utils import PlotGenerationMixin

class GenerateExamplesTask(
    Task,
    ModelUtilsMixin,
    PlotGenerationMixin
):

    def __init__(self, task_config: Dict[str, Any], task_name: str):
        super().__init__(task_config, task_name)

    def run(self, ctx: Context, args: Dict[str, Any]) -> Any:
        print(f"Running GenerateExamplesTask")

        self.model = self.init_model(ctx, args)  # Initialize the model

        # 1. Prepare the prompt:
        content = args.get("content", "")  # Get the content to add examples to
        prompt = self.prepare_prompt(self.task_config, content)

        # 2. Interact with the model:
        chat = self.start_chat()  # No files needed
        response = self.send_message(chat=chat, user_content=prompt)
        result = response.text

        # 3. Process and return the result:
        #    This task returns modified text, *not* JSON.
        if "<!-- END -->" not in result:
            raise Exception("Example generation did not complete (missing <!-- END --> marker).")

        # 4. Generate plots if enabled in the task configuration
        if self.task_config.get("generate_plots", False):  # Check config
            base_directory = args.get("base_directory")
            if base_directory:
                base_directory = Path(base_directory)
            result = self.generate_plots_from_code(ctx, result, self.task_config, base_directory)

        return result 