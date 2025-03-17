from typing import TypedDict, List, Optional, Dict, Literal, Any, Annotated
from langgraph.graph import StateGraph, END, START
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field
import yaml
from pathlib import Path
import os
import operator
import re

from pollo.agents.topics.generator import Topic, TopicsOutput
from pollo.agents.topics.generator import generate_topics_from_pdfs, load_topics_from_json
from pollo.utils.gemini import GeminiChatModel
from pollo.utils.base_tools import GeminiBaseTool

# Define state schemas
class DraftSubtaskState(TypedDict):
    topic: str
    subtopic: str
    draft: Optional[str]
    cleaned_draft: Optional[str]
    filename: Optional[str]
    topic_index: Optional[int]
    subtopic_index: Optional[int]
    directory: Optional[str]
    status: Literal["pending", "draft_generated", "cleaned", "filename_generated", "error"]

class DraftWritingState(TypedDict, total=False):
    directory: str
    perspectives: List[str]
    json_per_perspective: int
    topics: List[Dict]
    current_topic_index: int
    current_subtopic_index: int
    drafts: List[Dict]
    status: str
    current_batch: List[Dict]  # Batch of subtopics to process in parallel
    branching_factor: int      # Number of subtopics to process in parallel
    branch_results: Annotated[Dict[str, Dict], operator.or_]  # Use operator.or_ as reducer for concurrent updates
    uploaded_pdf_files: Dict[str, List[Dict]]  # Store uploaded PDF files per API key to reuse across all draft generations


# Update tool classes to inherit from GeminiBaseTool
class FilenameGeneratorTool(GeminiBaseTool):
    name: str = "filename_generator"
    description: str = "Generates an appropriate filename for a subtopic"
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.2
    mock_response: Optional[str] = None
    
    def __init__(self):
        prompt_file = Path(__file__).parent / "generate_filename.yaml"
        super().__init__(prompt_file=prompt_file)
    
    def _build_chain(self):
        """Build the LCEL chain for filename generation."""
        def process_input(inputs):
            messages = self.create_messages(**inputs)
            return self.gemini.invoke(messages)
            
        self.chain = RunnableLambda(process_input)
    
    def _run(self, topic: str, subtopic: str) -> str:
        """Generate a filename based on topic and subtopic."""
        return self.chain.invoke({
            "topic": topic,
            "subtopic": subtopic
        })

class DraftGeneratorTool(GeminiBaseTool):
    name: str = "draft_generator"
    description: str = "Generates an initial draft for a subtopic"
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.7
    
    def __init__(self):
        prompt_file = Path(__file__).parent / "generate_draft.yaml"
        super().__init__(prompt_file=prompt_file)
    
    def _build_chain(self):
        """Build the LCEL chain for draft generation."""
        def process_with_files(inputs):
            messages = self.create_messages(**inputs["model_inputs"])
            return self.gemini.invoke(
                messages, 
                files=inputs.get("files", [])
            )
            
        self.chain = RunnableLambda(lambda inputs: {
            "model_inputs": {
                "topic": inputs["topic"],
                "subtopic": inputs["subtopic"]
            },
            "files": inputs.get("files", [])
        }) | RunnableLambda(process_with_files)
    
    def _run(self, topic: str, subtopic: str, directory: Optional[str] = None, uploaded_files: Optional[Dict[str, List]] = None) -> str:
        """Generate a draft based on topic and subtopic."""
        # Prepare inputs for the chain
        inputs = {
            "topic": topic,
            "subtopic": subtopic
        }
        
        files = []
        
        # Use pre-uploaded files if provided
        if uploaded_files:
            # Get current API key to use appropriate uploaded files
            current_api_key = self.gemini._get_current_api_key() if hasattr(self.gemini, '_get_current_api_key') else None
            
            if current_api_key and current_api_key in uploaded_files:
                # Use files uploaded with the current API key
                files = uploaded_files[current_api_key]
            elif len(uploaded_files) > 0:
                # If current key not found but we have uploads for other keys,
                # get first available set of uploads
                first_key = list(uploaded_files.keys())[0]
                files = uploaded_files[first_key]
        
        # If no pre-uploaded files are found and directory is available, upload them
        if not files and directory:
            pdf_reader = PDFReaderTool()
            pdf_files = pdf_reader.invoke({"directory": directory})
            if pdf_files:
                files = self.upload_files(pdf_files)
        
        return self.chain.invoke({
            **inputs,
            "files": files
        })

class DraftCleanupTool(GeminiBaseTool):
    name: str = "draft_cleanup"
    description: str = "Cleans and improves a generated draft"
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.2
    
    def __init__(self):
        prompt_file = Path(__file__).parent / "cleanup_draft.yaml"
        super().__init__(prompt_file=prompt_file)
    
    def _build_chain(self):
        """Build the LCEL chain for draft cleanup."""
        def process_input(inputs):
            messages = self.create_messages(**inputs)
            return self.gemini.invoke(messages)
            
        self.chain = RunnableLambda(process_input)
    
    def _run(self, draft: str) -> str:
        """Clean and improve a draft."""
        return self.chain.invoke({"draft": draft})

# Add PDFReaderTool to read files (used by DraftGeneratorTool)
class PDFReaderTool(BaseTool):
    name: str = "pdf_reader"
    description: str = "Reads PDF files from a specified directory"
    
    def _run(self, directory: str) -> List[str]:
        """Read PDF files from a directory."""
        pdf_files = []
        for file in Path(directory).glob("*.pdf"):
            pdf_files.append(str(file))
        return pdf_files

# Subgraph for individual draft generation + cleanup
def create_draft_subgraph() -> StateGraph:
    """Subgraph for generating and cleaning a single draft"""
    builder = StateGraph(DraftSubtaskState)

    # Add nodes
    builder.add_node("generate_draft", generate_draft)
    builder.add_node("clean_draft", clean_draft)
    builder.add_node("generate_filename", generate_filename)
    builder.add_node("handle_error", handle_draft_error)

    # Set edges
    builder.add_edge(START, "generate_draft")
    builder.add_conditional_edges(
        "generate_draft",
        lambda s: "clean_draft" if s["draft"] else "handle_error"
    )
    builder.add_edge("clean_draft", "generate_filename")
    builder.add_edge("generate_filename", END)
    builder.add_edge("handle_error", END)

    return builder.compile()

# Modified parent graph implementation
def create_draft_writer(branching_factor: int = 3) -> StateGraph:
    """Main graph that coordinates topic generation and draft writing
    
    Args:
        branching_factor: Number of subtopics to process in parallel
    """
    builder = StateGraph(DraftWritingState)
    
    # Add main nodes
    builder.add_node("generate_topics", generate_topics)
    builder.add_node("initialize_processing", initialize_processing)
    builder.add_node("prepare_batch", prepare_subtopic_batch)
    builder.add_node("finalize_batch", finalize_batch)
    builder.add_node("finalize", finalize_output)

    # Set edges
    builder.add_edge(START, "generate_topics")
    builder.add_edge("generate_topics", "initialize_processing")
    builder.add_edge("initialize_processing", "prepare_batch")
    
    # Fan out for parallel processing
    def branch_out(state: DraftWritingState):
        return [f"subtopic_{i}" for i in range(len(state["current_batch"]))]
    
    # Dynamic branching based on batch
    builder.add_conditional_edges(
        "prepare_batch",
        branch_out,
        [f"subtopic_{i}" for i in range(branching_factor)]
    )
    
    # Add parallel processing nodes
    for i in range(branching_factor):
        # Create a node for each potential parallel branch
        node_name = f"subtopic_{i}"
        builder.add_node(node_name, lambda state, i=i: process_subtopic_parallel(state, i))
        builder.add_edge(node_name, "finalize_batch")
    
    # Add conditional edges for processing loop
    builder.add_conditional_edges(
        "finalize_batch",
        lambda s: "prepare_batch" if has_more_subtopics(s) else "finalize"
    )
    
    builder.add_edge("finalize", END)

    return builder.compile()

# Node implementations
def generate_topics(state: DraftWritingState) -> DraftWritingState:
    """Generate topics structure using existing topic generator"""
    # Check if topics.json already exists in the directory
    existing_topics = load_topics_from_json(state["directory"])
    if existing_topics:
        print(f"Loading topics from existing topics.json in {state['directory']}")
        return {**state, "topics": existing_topics.topics}
    
    # If topics.json doesn't exist, generate topics using the topic generator
    print(f"Generating new topics from PDFs in {state['directory']}")
    topics_output: TopicsOutput = generate_topics_from_pdfs(
        directory=state["directory"],
        perspectives=state.get("perspectives", ["technical_depth"]),
        json_per_perspective=state.get("json_per_perspective", 3)
    )
    
    return {**state, "topics": topics_output.topics}

def initialize_processing(state: DraftWritingState) -> DraftWritingState:
    """Initialize processing state"""
    return {
        **state,
        "current_topic_index": 0,
        "current_subtopic_index": 0,
        "drafts": [],
        "branch_results": {},  # Initialize branch_results
        "branching_factor": state.get("branching_factor", 3),
        "status": "processing"
    }

def process_subtopic(state: DraftWritingState) -> DraftWritingState:
    """Process current subtopic using subgraph"""
    topic: Topic = state["topics"][state["current_topic_index"]]
    subtopic = topic.sub_topics[state["current_subtopic_index"]]
    
    # Prepare subgraph input
    subtask_state = {
        "topic": topic.topic,
        "subtopic": subtopic,
        "draft": None,
        "cleaned_draft": None,
        "status": "pending",
        "subtopic_index": state["current_subtopic_index"],
        "topic_index": state["current_topic_index"],
        "directory": state["directory"]
    }
    
    # Execute subgraph
    result = create_draft_subgraph().invoke(subtask_state)
    
    # Update main state
    new_drafts = state["drafts"] + [{
        "topic": result["topic"],
        "subtopic": result["subtopic"],
        "draft": result.get("draft"),
        "cleaned_draft": result.get("cleaned_draft"),
        "filename": result.get("filename"),
        "topic_index": result["topic_index"],
        "subtopic_index": result["subtopic_index"],
        "status": result["status"]
    }]
    
    # Move to next subtopic
    new_state = {**state, "drafts": new_drafts}
    return advance_indices(new_state)

# Helper functions
def has_more_subtopics(state: DraftWritingState) -> bool:
    """Check if more subtopics need processing"""
    # Check if the current topic index is valid before accessing
    if state["current_topic_index"] >= len(state["topics"]):
        return False
        
    current_topic = state["topics"][state["current_topic_index"]]
    has_more_subtopics = (state["current_subtopic_index"] + 1) < len(current_topic.sub_topics)
    has_more_topics = (state["current_topic_index"] + 1) < len(state["topics"])
    
    return has_more_subtopics or has_more_topics

def advance_indices(state: DraftWritingState) -> DraftWritingState:
    """Advance topic/subtopic indices"""
    current_topic: Topic = state["topics"][state["current_topic_index"]]
    
    if (state["current_subtopic_index"] + 1) < len(current_topic.sub_topics):
        return {
            **state,
            "current_subtopic_index": state["current_subtopic_index"] + 1
        }
    elif (state["current_topic_index"] + 1) < len(state["topics"]):
        return {
            **state,
            "current_topic_index": state["current_topic_index"] + 1,
            "current_subtopic_index": 0
        }
    return state

def finalize_output(state: DraftWritingState) -> DraftWritingState:
    """Finalize output structure - files are already written to disk"""
    # Filter completed drafts
    completed_drafts = [d for d in state["drafts"] if d["status"] == "filename_generated"]
    
    # Group drafts by topic for reporting
    drafts_by_topic = {}
    for draft in completed_drafts:
        topic_index = draft.get("topic_index", 0)
        topic = draft["topic"]
        if topic_index not in drafts_by_topic:
            drafts_by_topic[topic_index] = {"topic": topic, "drafts": []}
        drafts_by_topic[topic_index]["drafts"].append(draft)
    
    # Report summary
    output_dir = state["directory"]
    print(f"\nGeneration completed: {len(completed_drafts)} files saved to {output_dir}")
    for topic_index, topic_data in sorted(drafts_by_topic.items()):
        topic_dir_name = f"{topic_index+1:02d}. {topic_data['topic']}"
        print(f"- {topic_dir_name}: {len(topic_data['drafts'])} files")
    
    return {
        **state,
        "status": "completed",
        "drafts": completed_drafts,
        "output_directory": output_dir
    }

# Subgraph node implementations
def generate_draft(state: DraftSubtaskState) -> DraftSubtaskState:
    """Generate draft for a subtopic using the DraftGeneratorTool"""
    try:
        generator = DraftGeneratorTool()
        
        # Get directory from the parent state or use a default path
        directory = state.get("directory", None)
        
        # Use pre-uploaded files if available in the state
        pre_uploaded_files = state.get("uploaded_pdf_files", None)
        
        draft = generator.invoke({
            "topic": state["topic"],
            "subtopic": state["subtopic"],
            "directory": directory,
            "uploaded_files": pre_uploaded_files  # Pass the pre-uploaded files
        })
        return {**state, "draft": draft, "status": "draft_generated"}
    except Exception as e:
        print(f"Error generating draft: {str(e)}")
        return {**state, "status": "error"}

def clean_draft(state: DraftSubtaskState) -> DraftSubtaskState:
    """Clean generated draft using the DraftCleanupTool"""
    try:
        cleaner = DraftCleanupTool()
        cleaned_draft = cleaner.invoke({"draft": state["draft"]})
        return {**state, "cleaned_draft": cleaned_draft, "status": "cleaned"}
    except Exception as e:
        print(f"Error cleaning draft: {str(e)}")
        return {**state, "status": "error"}

def handle_draft_error(state: DraftSubtaskState) -> DraftSubtaskState:
    """Handle draft generation errors"""
    return {**state, "status": "error"}

def generate_filename(state: DraftSubtaskState) -> DraftSubtaskState:
    """Generate a filename for the draft and save the file"""
    try:
        generator = FilenameGeneratorTool()
        response = generator.invoke({
            "topic": state["topic"],
            "subtopic": state["subtopic"]
        })
        
        # Extract clean filename from potentially complex response
        if isinstance(response, str):
            # Simple string response
            base_filename = response
        elif hasattr(response, 'content') and isinstance(response.content, str):
            # Handle object with content attribute
            base_filename = response.content.strip()
                
        # Format filename with numbering prefix based on subtopic index
        subtopic_index = state.get("subtopic_index", 0)
        formatted_filename = f"{subtopic_index+1:02d}. {base_filename}"
        
        # Ensure it has .md extension if not already present
        if not formatted_filename.lower().endswith('.md'):
            formatted_filename += '.md'
        
        print(f"Generated filename: {formatted_filename}")
        
        # Save the file immediately
        if state.get("directory") and state.get("cleaned_draft") and state.get("topic_index") is not None:
            # Create topic directory structure
            topic_index = state.get("topic_index", 0)
            topic_dir_name = f"{topic_index+1:02d}. {state['topic']}"
            topic_path = os.path.join(state["directory"], topic_dir_name)
            os.makedirs(topic_path, exist_ok=True)
            
            # Write file
            file_path = os.path.join(topic_path, formatted_filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(state["cleaned_draft"].content)
            print(f"File saved: {file_path}")
            
        return {**state, "filename": formatted_filename, "status": "filename_generated"}
    except Exception as e:
        print(f"Error generating filename: {str(e)}")
        print(f"Response from generator: {str(response) if 'response' in locals() else 'No response'}")
        return {**state, "status": "error"}

# New function definitions for parallel processing
def prepare_subtopic_batch(state: DraftWritingState) -> DraftWritingState:
    """Prepare a batch of subtopics for parallel processing"""
    # Get current topic
    if state["current_topic_index"] >= len(state["topics"]):
        return {**state, "current_batch": []}
    
    topic = state["topics"][state["current_topic_index"]]
    
    # Calculate how many subtopics remain
    remaining_subtopics = len(topic.sub_topics) - state["current_subtopic_index"]
    
    # Create batch of subtopic indices to process
    batch = []
    for i in range(min(remaining_subtopics, state.get("branching_factor", 3))):
        batch.append({
            "topic_index": state["current_topic_index"],
            "subtopic_index": state["current_subtopic_index"] + i
        })
    
    return {**state, "current_batch": batch}

def process_subtopic_parallel(state: DraftWritingState, branch_id: int = 0) -> DraftWritingState:
    """Process a single subtopic in parallel"""
    # Get batch of subtopics
    batch = state.get("current_batch", [])
    
    # If batch index out of range, return unchanged state
    if branch_id >= len(batch):
        return {}  # Return empty dict instead of full state
    
    # Get topic and subtopic indices from batch
    subtopic_data = batch[branch_id]
    topic_index = subtopic_data["topic_index"]
    subtopic_index = subtopic_data["subtopic_index"]
    
    # Get topic and subtopic
    topic = state["topics"][topic_index]
    subtopic = topic.sub_topics[subtopic_index]
    
    # Prepare subgraph input
    subtask_state = {
        "topic": topic.topic,
        "subtopic": subtopic,
        "draft": None,
        "cleaned_draft": None,
        "status": "pending",
        "subtopic_index": subtopic_index,
        "topic_index": topic_index,
        "directory": state["directory"],
        "uploaded_pdf_files": state.get("uploaded_pdf_files", None)  # Pass the pre-uploaded files
    }
    
    # Execute subgraph
    result = create_draft_subgraph().invoke(subtask_state)
    
    # Create a unique key for this branch result
    branch_key = f"branch_{branch_id}"
    
    # Return branch_results with just this branch's result
    return {
        "branch_results": {
            branch_key: {
                "topic": result["topic"],
                "subtopic": result["subtopic"],
                "draft": result.get("draft"),
                "cleaned_draft": result.get("cleaned_draft"),
                "filename": result.get("filename"),
                "topic_index": result["topic_index"],
                "subtopic_index": result["subtopic_index"],
                "status": result["status"]
            }
        }
    }

def finalize_batch(state: DraftWritingState) -> DraftWritingState:
    """Collect results from parallel branches and update state"""
    # Extract results from branch_results
    new_drafts = state.get("drafts", [])
    branch_results = state.get("branch_results", {})
    
    # Add all branch results to drafts
    for result_key, result in branch_results.items():
        if result:
            new_drafts.append(result)
            
    # Calculate how many subtopics were processed
    batch_size = len(state.get("current_batch", []))
    
    # Update indices based on batch size
    topic_index = state["current_topic_index"]
    subtopic_index = state["current_subtopic_index"] + batch_size
    
    # Check if we need to move to the next topic
    if topic_index < len(state["topics"]):
        topic = state["topics"][topic_index]
        if subtopic_index >= len(topic.sub_topics):
            topic_index += 1
            subtopic_index = 0
    
    # Ensure indices stay within bounds
    topic_index = min(topic_index, len(state["topics"]))
    
    # Create new state with updated indices and drafts
    new_state = {
        **state,
        "current_topic_index": topic_index,
        "current_subtopic_index": subtopic_index,
        "drafts": new_drafts,
        "branch_results": {}  # Clear branch results for next batch
    }
    
    return new_state

# Modified function to generate drafts with branching factor
def generate_drafts_from_topics(
    directory: str,
    perspectives: List[str] = ["technical_depth"],
    json_per_perspective: int = 3,
    branching_factor: int = 3
) -> Dict:
    """Generate drafts from topics extracted from PDFs
    
    Args:
        directory: Directory containing PDFs and for output
        perspectives: List of perspectives to use for topic generation
        json_per_perspective: Number of JSON files to generate per perspective
        branching_factor: Number of subtopics to process in parallel
    """
    # Create a draft generator tool to access API keys and upload_files method
    tool = DraftGeneratorTool()
    
    # Get list of API keys from the tool's gemini client
    api_keys = tool.gemini._api_keys if hasattr(tool.gemini, '_api_keys') else []
    
    # Upload PDF files once per API key at the beginning
    uploaded_pdf_files = {}
    pdf_reader = PDFReaderTool()
    pdf_files = pdf_reader.invoke({"directory": directory})
    
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF files, uploading for each API key...")
        
        for i, api_key in enumerate(api_keys):
            print(f"Uploading PDFs for API key {i+1}/{len(api_keys)}...")
            
            # Temporarily set current API key for tool's gemini client
            original_key_index = tool.gemini._current_key_index
            tool.gemini._current_key_index = i
            tool.gemini._initialize_client()
            
            try:
                # Upload files for this specific API key
                key_uploads = tool.upload_files(pdf_files)
                uploaded_pdf_files[api_key] = key_uploads
                print(f"Successfully uploaded {len(key_uploads)} files for API key {i+1}.")
            except Exception as e:
                print(f"Error uploading files for API key {i+1}: {str(e)}")
                uploaded_pdf_files[api_key] = []
            
            # Restore original API key
            tool.gemini._current_key_index = original_key_index
            tool.gemini._initialize_client()
    
    # Create the graph with specified branching factor
    draft_writer = create_draft_writer(branching_factor)
        
    # Prepare initial state
    initial_state = {
        "directory": directory,
        "perspectives": perspectives,
        "json_per_perspective": json_per_perspective,
        "branching_factor": branching_factor,
        "topics": [],
        "current_topic_index": 0,
        "current_subtopic_index": 0,
        "drafts": [],
        "status": "starting",
        "uploaded_pdf_files": uploaded_pdf_files  # Include the uploaded files in the state
    }
    
    # Run the graph
    final_state = draft_writer.invoke(initial_state, {"recursion_limit": 500})
    
    # Return the drafts and output location
    return {
        "drafts": final_state.get("drafts", []),
        "status": final_state.get("status", "unknown"),
        "output_directory": directory
    }