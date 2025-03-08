from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from ceviche.core.agent import Agent
from ceviche.core.context import Context
import re

class DirectoryProcessorAgent(Agent):
    """Handles the processing of directories and their contents."""

    def __init__(self, debug: bool = False):
        super().__init__(debug)
        self.debug = debug

    def pre_execution(self, ctx: Context, args: Dict[str, Any]):
        """Prepare the context before processing."""
        if self.debug:
            print("DirectoryProcessor: pre_execution")
        ctx["context"] = args.get("context", "General Context")
        ctx["debug"] = self.debug

    def execute(self, ctx: Context, args: Dict[str, Any]) -> Any:
        """Process a directory: create topics, generate drafts, and enhance."""
        base_directory = Path(args["directory"])
        max_previous_topics = args.get("max_previous_topics", 3)
        target_folders = args.get("target_folders", [])
        excluded_folders = args.get("excluded_folders", [])

        try:
            # Determine which folders to process
            folders_to_process = []
            if target_folders:
                # Process only specified target folders
                folders_to_process = [base_directory / folder for folder in target_folders]
            else:
                # Process all folders except excluded ones
                folders_to_process = [
                    d for d in base_directory.iterdir() 
                    if d.is_dir() and d.name not in excluded_folders
                ]

            # Process each folder
            for directory in folders_to_process:
                if self.debug:
                    print(f"Processing directory: {directory}")
                
                # Create Topics
                topics = self._create_topics(directory, ctx, args)

                # Process Sections sequentially
                for section in topics["topics"]:
                    self._process_section(
                        ctx,
                        args,
                        directory,
                        section["topic"],
                        section["sub_topics"],
                        max_previous_topics,
                    )
            
        except Exception as e:
            print(f"❌ Failed to process directories")
            print(f"Error: {str(e)}")
            raise

    def _create_topics(self, directory: Path, ctx: Context, args: Dict[str, Any]):
        """Creates topics for the given directory."""
        if self.debug:
            print("Creating topics...")

        topics_file = directory / "topics.json"
        
        # Check for existing topics.json
        if topics_file.exists():
            if self.debug:
                print(f"Loading existing topics from {topics_file}")
            return json.loads(topics_file.read_text())

        # Create new topics if file doesn't exist
        create_topics_workflow = self.get_workflow("create_topics", ctx, args)
        create_topics_args = {
            "directory": str(directory),
            "content": ctx["context"],
            "perspectives": args.get("perspectives"),
            "json_per_perspective": args.get("json_per_perspective")
        }
        topics = create_topics_workflow.run(ctx, create_topics_args)

        # Save generated topics to JSON file
        topics_file.write_text(json.dumps(topics, indent=2))
        if self.debug:
            print(f"Saved topics to {topics_file}")
            print(f"Topics created: {topics}")
        
        return topics

    def _process_section(
        self,
        ctx: Context,
        args: Dict[str, Any],
        directory: Path,
        section_name: str,
        subtopics: List[str],
        max_previous_topics: int,
    ):
        """Processes a single section within the directory."""
        if self.debug:
            print(f"Processing section: {section_name}")

        # Get existing section numbers and create numbered directory
        existing_sections = self._get_existing_numbers(directory, is_directory=True)
        if existing_sections:
            section_num = max(existing_sections) + 1
        else:
            section_num = 1

        section_dir = directory / f"{section_num:02d}. {section_name}"
        section_dir.mkdir(parents=True, exist_ok=True)

        previous_topics = []

        # Process each subtopic with numbering
        for topic_idx, topic_name in enumerate(subtopics, start=1):
            enhanced_draft = self._process_topic(
                ctx,
                args,
                directory,
                section_dir,
                section_name,
                f"{topic_idx:02d}. {topic_name}",
                previous_topics,
                max_previous_topics,
            )
            
            # Add to previous_topics list (for context in subsequent iterations)
            if len(previous_topics) >= max_previous_topics:
                previous_topics.pop(0)  # Keep the list to max_previous_topics
            
            previous_topics.append({
                "topic": topic_name, 
                "content": enhanced_draft
            })

    def _process_topic(
        self,
        ctx: Context,
        args: Dict[str, Any],
        base_dir: Path,
        section_dir: Path,
        section_name: str,
        topic_name: str,
        previous_topics: List[Dict[str, Any]],
        max_previous_topics: int,
        max_retries: int = 3,
    ):
        """Generates and enhances a draft for a single topic."""
        if self.debug:
            print(f"Processing topic: {topic_name}")

        attempt = 0
        last_error = None

        while attempt < max_retries:
            try:
                # Generate Initial Draft
                draft_workflow = self.get_workflow("generate_initial_draft", ctx, args)
                draft_args = {
                    "directory": str(section_dir),
                    "section_name": section_name,
                    "topic": topic_name,
                    "context": self.build_context_string(previous_topics),
                }
                initial_draft = draft_workflow.run(ctx, draft_args)
                ctx["current_topic_content"] = initial_draft

                # Enhance Draft
                enhance_workflow = self.get_workflow("enhance_draft", ctx, args)
                enhance_args = {
                    "content": initial_draft,
                    "base_directory": str(base_dir), 
                    "directory": str(section_dir),
                    "tasks": args.get("tasks")
                }
                enhanced_draft = enhance_workflow.run(ctx, enhance_args)

                # Generate Filename and Save
                filename_workflow = self.get_workflow("generate_filename", ctx, args)
                filename_args = {
                    "topic": topic_name.split('. ')[1],
                    "content": topic_name.split('. ')[1],
                    "directory": str(section_dir)
                }
                filename, filepath = filename_workflow.run(ctx, filename_args)

                filepath.write_text(enhanced_draft, encoding="utf-8")
                print(f"✔️ Saved topic to: {filepath}")

                return enhanced_draft

            except Exception as e:
                attempt += 1
                last_error = e
                if attempt < max_retries:
                    print(f"⚠️ Attempt {attempt} failed for topic '{topic_name}'. Retrying... Error: {str(e)}")
                else:
                    print(f"❌ All {max_retries} attempts failed for topic '{topic_name}'")
                    raise last_error

    def build_context_string(self, previous_topics: List[Dict[str, Any]]) -> str:
        """Builds a context string from previous topics."""
        context_parts = []
        for topic_result in previous_topics:
            context_parts.extend([
                f"<PreviousTopicContent topic='{topic_result['topic']}'>",
                topic_result['content'],
                f"<PreviousTopicContent topic='{topic_result['topic']}'>",
            ])
        return "\n".join(context_parts)

    def _get_existing_numbers(self, directory: Path, is_directory: bool = False) -> set[int]:
        """Get set of existing numbers in the directory (either files or directories)."""
        items = [f for f in directory.iterdir() if f.is_dir()] if is_directory else [f for f in directory.glob("*.md")]
        numbers = set()
        for item in items:
            match = re.match(r'^(\d+)\.', item.name)
            if match:
                numbers.add(int(match.group(1)))
        return numbers

    def post_execution(self, ctx: Context, args: Dict[str, Any], result: Any):
        """Clean up or finalize after processing."""
        if self.debug:
            print("DirectoryProcessor: post_execution")
        # Add any post-processing steps here, if needed.