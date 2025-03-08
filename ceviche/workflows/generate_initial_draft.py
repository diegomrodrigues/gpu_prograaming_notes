from ceviche.core.task import Task
from ceviche.core.workflow import Workflow
from ceviche.tasks.generate_draft.generate_draft import GenerateDraftTask
from ceviche.tasks.cleanup.cleanup import CleanupTask
from ceviche.tasks.content_verification.content_verification import ContentVerificationTask
from typing import Dict, Any, List, Optional
from pathlib import Path
from ceviche.core.utilities.model_utils import ModelUtilsMixin
from ceviche.core.utilities.file_utils import WithReadAndWriteFilesMixin

class GenerateInitialDraftWorkflow(
    Workflow,
    ModelUtilsMixin,
    WithReadAndWriteFilesMixin
):
    def __init__(self):
        super().__init__()
        self.max_iterations = 3

    def before_start(self, ctx: Dict[str, Any], args: Dict[str, Any]):
        super().before_start(ctx, args)
        print("Starting GenerateInitialDraftWorkflow")
        ctx["directory"] = args.get("directory", ".")

        self.generate_draft_task = self.load_task("generate_draft", ctx, args)
        self.cleanup_task = self.load_task("cleanup", ctx, args)
        self.content_verification_task = self.load_task("content_verification", ctx, args)


    def run(self, ctx: Dict[str, Any], args: Dict[str, Any]) -> Any:
        directory = Path(ctx.get("directory", "."))
        section_name = args.get("section_name", "DefaultSection")
        topic = args.get("topic", "DefaultTopic")
        pdf_files = self.get_pdf_files(str(directory))
        max_previous_topics = args.get("max_previous_topics", 5)
        self.max_iterations = args.get("max_iterations", 3)
        previous_topics = args.get("previous_topics", [])
        context_str = self._build_previous_topics_context(previous_topics, max_previous_topics)

        input_text = self._build_input_text(directory, section_name, topic, ctx.get("context", ""))

        draft_content = self._generate_with_retries(
            ctx,
            input_text,
            pdf_files,
            context_str,
            self.generate_draft_task,
            "generate_draft"
        )

        if not draft_content:
            raise Exception(f"Failed to generate draft for topic: {topic}")
        
        verification_result = self.content_verification_task.run(
            ctx, {"content": draft_content}
        )
        if verification_result.lower().strip() != "yes":
            raise Exception(f"Content verification failed for topic: {topic}")

        cleaned_content = self._generate_with_retries(
            ctx,
            draft_content,
            [],
            "",
            self.cleanup_task,
            "cleanup"
        )

        if not cleaned_content:
            raise Exception(f"Failed to cleanup draft for topic: {topic}")

        return cleaned_content

    def _generate_with_retries(
        self,
        ctx: Dict[str, Any],
        input_content: str,
        pdf_files: List[Path],
        context_str: str,
        task: Task,
        task_name: str
    ) -> Optional[str]:
        iterations = 0
        content = None
        while iterations < self.max_iterations:
            try:
                if task_name == "generate_draft":
                    args = {
                        "content": input_content,
                        "directory": ctx["directory"],
                        "additional_context": context_str,
                    }
                elif task_name == "cleanup":
                    args = {
                        "content": input_content
                    }
                else:
                    args = {
                        "content": input_content
                    }
                
                content = task.run(ctx, args)

                if content and "<!-- END -->" in content:
                    return content
                elif content and "<!-- END -->" not in content:
                    print(f"⚠️ <!-- END --> marker not found. Continue ({iterations + 1}/{self.max_iterations})...")
                    content = content + "\n<!-- END -->"
                    return content
                else:
                    print(f"⚠️ Task {task_name} returned no content. Retrying ({iterations + 1}/{self.max_iterations})...")

            except Exception as e:
                print(f"❌ Error during '{task_name}': {str(e)}")
                if "Content verification failed" in str(e):
                    return None

            iterations += 1
        return None

    def _build_previous_topics_context(self, previous_topics: List[Any], max_previous_topics: int) -> str:
        if not previous_topics:
            return ""

        context_parts = ["\n\nPrevious Topics:"]
        for topic_result in previous_topics[-max_previous_topics:]:
            context_parts.extend([
                f"--- START {topic_result.topic} ---",
                topic_result.content,
                f"--- END {topic_result.topic} ---"
            ])

        return "\n".join(context_parts)

    def _build_input_text(self, directory: Path, section_name: str, topic: str, context: str) -> str:
        return "\n".join([
            f"CONTEXT_PLACEHOLDER = {context}",
            f"DIRECTORY_PLACEHOLDER = {directory}",
            f"SECTION_PLACEHOLDER = {section_name}",
            f"SUBTOPIC_PLACEHOLDER = {topic}"
        ])

    def after_start(self, ctx: Dict[str, Any], args: Dict[str, Any]):
        super().after_start(ctx, args)
        print("GenerateInitialDraftWorkflow completed")
