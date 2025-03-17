from typing import Annotated, Dict, List, Literal, Optional, Tuple, Type, TypedDict, Any
import json
from pathlib import Path
import os

from langchain_core.tools import BaseTool, tool
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda

from pollo.utils.base_tools import GeminiBaseTool

# Define state schemas
class TopicsState(TypedDict):
    directory: str
    perspectives: List[str]
    json_per_perspective: int
    current_perspective_index: int  
    current_json_index: int  
    all_topics: List[Dict]
    merged_topics: Optional[Dict]
    consolidated_topics: Optional[Dict]
    status: str

# Define Pydantic models for topic structure
class Topic(BaseModel):
    topic: str = Field(description="Name of the topic without numbers")
    sub_topics: List[str] = Field(description="List of subtopics as detailed text strings")

class TopicsOutput(BaseModel):
    topics: List[Topic] = Field(description="List of topics with their subtopics")

# Create the Pydantic output parser
topics_parser = PydanticOutputParser(pydantic_object=TopicsOutput)


# Tools for the agent
class PDFReaderTool(BaseTool):
    name: str = "pdf_reader"
    description: str = "Reads PDF files from a specified directory"
    
    def _run(self, directory: str) -> List[str]:
        """Read PDF files from a directory."""
        pdf_files = []
        for file in Path(directory).glob("*.pdf"):
            pdf_files.append(str(file))
        return pdf_files

# Modified TopicsGeneratorTool using the base class
class TopicsGeneratorTool(GeminiBaseTool):
    name: str = "topics_generator"
    description: str = "Generates topics and subtopics based on PDFs and a perspective"
    system_instruction: str = "Generate topics and subtopics based on the provided perspective."
    user_template: str = "Perspective of analysis: {perspective}"
    response_mime_type: str = "application/json"
    response_schema: Type[BaseModel] = TopicsOutput
    
    def __init__(self):
        prompt_file = Path(__file__).parent / "create_topics.yaml"
        super().__init__(prompt_file=prompt_file)

    def _build_chain(self):
        """Build the LCEL chain for topic generation."""
        def process_with_files(inputs):
            messages = self.create_messages(**inputs["model_inputs"])
            return self.gemini.invoke(
                messages, 
                files=inputs["files"]
            )

        self.chain = (
            RunnableLambda(lambda inputs: {
                "model_inputs": {"perspective": inputs["perspective"]},
                "files": inputs.get("files", [])
            }) |
            RunnableLambda(process_with_files) |
            topics_parser
        )

    def _run(self, directory: str, perspective: str) -> Dict:
        """Generate topics based on PDFs and a perspective."""
        # Read PDF files
        pdf_reader = PDFReaderTool()
        pdf_files = pdf_reader.invoke({"directory": directory})
        
        if not pdf_files:
            return {"topics": []}
        
        # Use the helper method from GeminiBaseTool to upload files
        uploaded_files = self.upload_files(pdf_files)
        
        # Invoke the chain with context
        return self.chain.invoke({
            "perspective": perspective,
            "files": uploaded_files
        })

# Modified SubtopicsConsolidatorTool using the base class
class SubtopicsConsolidatorTool(GeminiBaseTool):
    name: str = "subtopics_consolidator"
    description: str = "Consolidates similar subtopics from multiple topic sets"
    system_instruction: str = "Consolidate similar subtopics from the provided topics."
    user_template: str = "Consolidate the sub-topics in this JSON: {content}"
    response_mime_type: str = "application/json"
    response_schema: Type[BaseModel] = TopicsOutput

    def __init__(self):
        prompt_file = Path(__file__).parent / "consolidate_subtopics.yaml"
        super().__init__(prompt_file=prompt_file)

    def _build_chain(self):
        """Build the LCEL chain for subtopic consolidation."""
        def process_input(content):
            messages = self.create_messages(content=content)
            return self.gemini.invoke(messages)

        self.chain = (
            RunnableLambda(lambda x: {"content": json.dumps(x)}) |
            RunnableLambda(process_input) |
            topics_parser
        )

    def _run(self, topics: Dict) -> Dict:
        """Consolidate similar subtopics."""
        # Invoke the chain
        return self.chain.invoke(topics)

# Function to load topics from JSON file
def load_topics_from_json(directory: str) -> Optional[TopicsOutput]:
    """Load topics from topics.json if it exists in the directory."""
    json_path = Path(directory) / "topics.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                topics_data = json.load(f)
            
            # Convert plain dictionary to Pydantic model
            topics_list = []
            for topic_data in topics_data.get("topics", []):
                topics_list.append(Topic(
                    topic=topic_data.get("topic", ""),
                    sub_topics=topic_data.get("sub_topics", [])
                ))
            
            return TopicsOutput(topics=topics_list)
        except Exception as e:
            print(f"Error loading topics from JSON: {str(e)}")
            return None
    return None

# Function to save topics to JSON file
def save_topics_to_json(topics: TopicsOutput, directory: str) -> bool:
    """Save topics to topics.json in the specified directory."""
    json_path = Path(directory) / "topics.json"
    try:        
        # Convert Pydantic model to dictionary
        topics_dict = topics.dict()
        
        # Save to JSON file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(topics_dict, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving topics to JSON: {str(e)}")
        return False

# Graph nodes
def initialize(state: TopicsState) -> TopicsState:
    """Initialize the state."""
    return {
        **state,
        "current_perspective_index": 0,
        "current_json_index": 0,
        "all_topics": [],
        "status": "initialized"
    }

def generate_topics(state: TopicsState) -> TopicsState:
    """Generate topics for the current perspective and json index."""
    perspective_index = state["current_perspective_index"]
    json_index = state["current_json_index"]
    
    if perspective_index >= len(state["perspectives"]):
        return {
            **state,
            "status": "generation_complete"
        }
    
    perspective = state["perspectives"][perspective_index]
    
    generator = TopicsGeneratorTool()
    topics = generator.invoke({
        "directory": state["directory"], 
        "perspective": perspective
    })
    
    return {
        **state,
        "all_topics": [*state["all_topics"], topics],
        "current_json_index": json_index + 1,
        "status": "topics_generated"
    }

def check_generation_status(state: TopicsState) -> Literal["next_json", "next_perspective", "merge"]:
    """Check if we need to generate more JSONs for the current perspective or move to the next."""
    if state["current_json_index"] <= state["json_per_perspective"]:
        return "next_json"
    elif state["current_perspective_index"] < len(state["perspectives"]) - 1:
        return "next_perspective"
    else:
        return "merge"

def next_json(state: TopicsState) -> TopicsState:
    """Keep the same perspective, just update the status."""
    return {
        **state,
        "status": "ready_for_next_json"
    }

def next_perspective(state: TopicsState) -> TopicsState:
    """Move to the next perspective and reset json index."""
    return {
        **state,
        "current_perspective_index": state["current_perspective_index"] + 1,
        "current_json_index": 0,
        "status": "ready_for_next_perspective"
    }

def merge_topics(state: TopicsState) -> TopicsState:
    """Merge all generated topics."""
    all_topics = state["all_topics"]
    
    merged_topics = {"topics": []}
    topic_map = {}
    
    # Process all topics from different perspectives
    for topics_group in all_topics:
        for topic in topics_group.topics:
            topic_name = topic.topic
            if topic_name in topic_map:
                # Append new subtopics to existing topic
                topic_map[topic_name]["sub_topics"].extend(topic.sub_topics)
            else:
                # Create new topic entry
                topic_map[topic_name] = {
                    "topic": topic_name,
                    "sub_topics": topic.sub_topics.copy()
                }
    
    # Convert map back to list format
    merged_topics["topics"] = list(topic_map.values())
    
    return {
        **state,
        "merged_topics": merged_topics,
        "status": "topics_merged"
    }

def consolidate_subtopics(state: TopicsState) -> TopicsState:
    """Consolidate similar subtopics."""
    if not state.get("merged_topics"):
        return {
            **state,
            "status": "error",
            "consolidated_topics": {"topics": []}
        }
    
    consolidator = SubtopicsConsolidatorTool()
    consolidated = consolidator.invoke({"topics": state["merged_topics"]})
    
    # Save consolidated topics to JSON file
    save_topics_to_json(consolidated, state["directory"])
    
    return {
        **state,
        "consolidated_topics": consolidated,
        "status": "complete"
    }

# Create the LangGraph
def create_topic_generator() -> StateGraph:
    """Create the topic generator graph."""
    # Create the graph
    workflow = StateGraph(TopicsState)
    
    # Add nodes
    workflow.add_node("initialize", initialize)
    workflow.add_node("generate_topics", generate_topics)
    workflow.add_node("next_json", next_json)
    workflow.add_node("next_perspective", next_perspective)
    workflow.add_node("merge_topics", merge_topics)
    workflow.add_node("consolidate_subtopics", consolidate_subtopics)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "generate_topics")
    workflow.add_edge("next_json", "generate_topics")
    workflow.add_edge("next_perspective", "generate_topics")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "generate_topics",
        check_generation_status,
        {
            "next_json": "next_json",
            "next_perspective": "next_perspective",
            "merge": "merge_topics"
        }
    )
    
    workflow.add_edge("merge_topics", "consolidate_subtopics")
    workflow.add_edge("consolidate_subtopics", END)
    
    # Compile the graph
    return workflow.compile()

# Main function to use the generator
def generate_topics_from_pdfs(
    directory: str,
    perspectives: List[str],
    json_per_perspective: int = 3
) -> Dict:
    """Generate topics from PDFs based on multiple perspectives."""
    # Check if topics.json already exists in the directory
    existing_topics = load_topics_from_json(directory)
    if existing_topics:
        print(f"Loading topics from existing topics.json in {directory}")
        return existing_topics
    
    print(f"Generating new topics from PDFs in {directory}")
    # Create the graph
    topic_generator = create_topic_generator()
    
    # Prepare initial state
    initial_state = {
        "directory": directory,
        "perspectives": perspectives,
        "json_per_perspective": json_per_perspective,
        "current_perspective_index": 0,
        "current_json_index": 0,
        "all_topics": [],
        "merged_topics": None,
        "consolidated_topics": None,
        "status": "starting"
    }
    
    # Run the graph
    final_state = topic_generator.invoke(initial_state)
    
    # Return the consolidated topics
    return final_state.get("consolidated_topics")
