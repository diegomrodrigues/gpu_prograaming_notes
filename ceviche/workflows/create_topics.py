from ceviche.core.context import Context
from ceviche.core.workflow import Workflow
from typing import Dict, Any
from ceviche.core.utilities.file_utils import WithReadAndWriteFilesMixin
from ceviche.core.utilities.json_utils import JsonUtilitiesMixin
import json

class CreateTopicsWorkflow(
    Workflow, 
    WithReadAndWriteFilesMixin, 
    JsonUtilitiesMixin
):

    def __init__(self):
        super().__init__()

    def before_start(self, ctx: Context, args: Dict[str, Any]):
        super().before_start(ctx, args)  # Call super to ensure hooks are called
        print("Starting CreateTopicsWorkflow")

        self.create_topics = self.load_task("create_topics", ctx, args)
        self.consolidate_subtopics = self.load_task("consolidate_subtopics", ctx, args)

    def run(self, ctx: Context, args: Dict[str, Any]) -> Any:
        perspectives = args.get("perspectives")
        json_per_perspective = args.get("json_per_perspective", 3)
        directory = args.get("directory", ".")

        all_topics = []

        if perspectives and len(perspectives) > 0:
            for perspective in perspectives:
                for _ in range(json_per_perspective+1):
                    create_topics_args = {
                        'directory': directory,
                        'perspective': perspective
                    }
                    topics = self.create_topics.run(ctx, create_topics_args)
                    all_topics.append(topics)
        else:
            topics = self.create_topics.run(ctx, {})
            all_topics.append(topics)

        # Merge topics using the dedicated method
        merged_topics = self._merge_topics(all_topics)

        # --- Consolidate Subtopics Task ---
        consolidated_args = {
            "topics": merged_topics
        }
        consolidated_topics = self.consolidate_subtopics.run(ctx, consolidated_args)

        # --- Final Result Processing ---
        final_result_str = self.deduplicate_json_keys(
            json.dumps(consolidated_topics)
        )
        self.final_result = json.loads(final_result_str)

        return self.final_result

    def after_start(self, ctx: Context, args: Dict[str, Any]):
        super().after_start(ctx, args)  # Call super to ensure hooks are called
        json_str = self.dump_json(ctx["topics"])
        self.write_file(
            content=json_str,
            filename="topics.json"
        )
        print("CreateTopicsWorkflow completed")


    def _merge_topics(self, all_topics: list) -> dict:
        """
        Merges topics from different perspectives by combining subtopics of matching topic names.        
        """
        merged_topics = {"topics": []}
        topic_map = {}  # Dictionary to track topics by their names

        # Process all topics from different perspectives
        for topics_group in all_topics:
            for topic in topics_group["topics"]:
                topic_name = topic["topic"]
                if topic_name in topic_map:
                    # Append new subtopics to existing topic
                    topic_map[topic_name]["sub_topics"].extend(topic["sub_topics"])
                else:
                    # Create new topic entry
                    topic_map[topic_name] = {
                        "topic": topic_name,
                        "sub_topics": topic["sub_topics"].copy()
                    }

        # Convert map back to list format
        merged_topics["topics"] = list(topic_map.values())
        return merged_topics
