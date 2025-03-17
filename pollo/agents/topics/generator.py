from typing import Annotated, Dict, List, Literal, Optional, Tuple, TypedDict, Any
import json
from pathlib import Path

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

TOPICS_GENERATOR_MOCK = """
{
  "topics": [
    {
      "topic": "Introduction to Machine Learning",
      "sub_topics": [
        "Machine Learning fundamentals encompass the core principles and theoretical foundations that underpin the field. These include the concept of learning from data, the distinction between supervised and unsupervised learning, and the fundamental trade-offs between model complexity and generalization ability. The mathematical frameworks of statistical learning theory provide the theoretical basis for understanding how algorithms can extract patterns from data and make predictions on unseen examples.",
        "Supervised learning algorithms are designed to learn patterns from labeled training data, enabling them to make predictions on new, unseen data. This paradigm includes classification algorithms that assign discrete categories to inputs and regression methods that predict continuous values. Key techniques include linear and logistic regression, decision trees, support vector machines, and neural networks, each with distinct mathematical foundations and appropriate use cases depending on data characteristics and problem requirements.",
        "Unsupervised learning approaches extract patterns and structures from unlabeled data without explicit guidance. Clustering algorithms like K-means and hierarchical clustering group similar data points together, while dimensionality reduction techniques such as Principal Component Analysis (PCA) and t-SNE transform high-dimensional data into lower-dimensional representations while preserving important relationships. These methods are essential for exploratory data analysis, feature extraction, and discovering hidden patterns in complex datasets."
      ]
    },
    {
      "topic": "Neural Networks Architecture",
      "sub_topics": [
        "Feedforward neural networks form the foundational architecture in deep learning, consisting of input, hidden, and output layers with neurons connected in a directed acyclic graph. Each neuron applies a non-linear activation function to the weighted sum of its inputs, enabling the network to learn complex, non-linear relationships in data. The universal approximation theorem establishes that even a single hidden layer network with sufficient neurons can approximate any continuous function, though deeper architectures often learn more efficiently for complex tasks.",
        "Convolutional Neural Networks (CNNs) are specialized architectures designed primarily for processing grid-like data such as images. Their distinctive components include convolutional layers that apply learnable filters across the input, pooling layers that reduce spatial dimensions while retaining important features, and fully connected layers for final predictions. The hierarchical feature extraction capability of CNNs—from low-level edges and textures to high-level semantic concepts—makes them exceptionally effective for computer vision tasks including image classification, object detection, and segmentation.",
        "Recurrent Neural Networks (RNNs) and their variants are designed to process sequential data by maintaining an internal state that captures information from previous inputs. Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) architectures address the vanishing gradient problem in traditional RNNs through gating mechanisms that control information flow. These architectures excel in tasks requiring temporal context understanding, such as natural language processing, time series analysis, and speech recognition, where dependencies between elements in the sequence are crucial for accurate predictions."
      ]
    }
  ]
}
"""

# Modified TopicsGeneratorTool using the base class
class TopicsGeneratorTool(GeminiBaseTool):
    name: str = "topics_generator"
    description: str = "Generates topics and subtopics based on PDFs and a perspective"
    mock_response: str = TOPICS_GENERATOR_MOCK
    system_instruction: str = "Generate topics and subtopics based on the provided perspective."
    user_template: str = "Perspective of analysis: {perspective}"
    
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

SUBTOPICS_CONSOLIDATOR_MOCK = """```json
{
  "topics": [
    {
      "topic": "Machine Learning Fundamentals",
      "sub_topics": [
        "Machine Learning paradigms encompass supervised, unsupervised, and reinforcement learning approaches, each with distinct methodologies and applications. Supervised learning uses labeled data to train models that can predict outcomes for new inputs, while unsupervised learning discovers patterns and structures in unlabeled data. Reinforcement learning enables agents to learn optimal behaviors through interaction with an environment and feedback in the form of rewards or penalties. These fundamental approaches form the backbone of modern AI systems across diverse domains.",
        "Statistical learning theory provides the mathematical foundation for machine learning, establishing principles that govern how algorithms generalize from training data to unseen examples. Key concepts include the bias-variance tradeoff, which balances model complexity against overfitting risk; empirical risk minimization, which guides the optimization of model parameters; and regularization techniques that constrain models to improve generalization. These theoretical frameworks enable practitioners to design algorithms with provable guarantees and understand their limitations under different data conditions.",
        "Feature engineering and selection techniques transform raw data into meaningful representations that enhance model performance. This process involves creating new features through mathematical transformations, domain knowledge application, or automated methods; selecting relevant features using filter, wrapper, or embedded methods; and reducing dimensionality to mitigate the curse of dimensionality. Effective feature engineering often requires both technical expertise and domain understanding, making it a crucial step that significantly impacts model accuracy, interpretability, and computational efficiency."
      ]
    },
    {
      "topic": "Deep Learning Architectures",
      "sub_topics": [
        "Neural network architectures have evolved from simple perceptrons to complex, specialized structures optimized for specific data types and tasks. Feedforward networks process information in one direction from input to output, while recurrent architectures incorporate feedback connections to maintain memory of previous inputs. Convolutional networks leverage spatial hierarchies through local connectivity patterns, and transformer models use attention mechanisms to process sequences in parallel. Each architecture class offers distinct advantages for different problem domains, from computer vision and natural language processing to reinforcement learning and generative modeling.",
        "Attention mechanisms and transformers have revolutionized sequence modeling by enabling models to focus selectively on relevant parts of the input. Unlike recurrent architectures that process sequences step-by-step, transformers use self-attention to compute representations of all positions simultaneously, capturing long-range dependencies more effectively. This parallel computation enables efficient training on massive datasets, while multi-head attention allows the model to attend to information from different representation subspaces. These innovations have led to state-of-the-art performance in natural language processing, computer vision, and multimodal learning tasks.",
        "Training methodologies for deep neural networks encompass specialized optimization algorithms, regularization techniques, and initialization strategies that address the challenges of high-dimensional, non-convex optimization. Adaptive learning rate methods like Adam and RMSprop automatically adjust parameter updates based on gradient history, while techniques such as batch normalization and layer normalization stabilize training by controlling the distribution of activations. Dropout prevents co-adaptation of neurons, and residual connections enable training of very deep networks by providing gradient shortcuts. These approaches collectively enable the successful training of increasingly complex architectures on diverse datasets."
      ]
    }
  ]
}
```"""

# Modified SubtopicsConsolidatorTool using the base class
class SubtopicsConsolidatorTool(GeminiBaseTool):
    name: str = "subtopics_consolidator"
    description: str = "Consolidates similar subtopics from multiple topic sets"
    mock_response: str = SUBTOPICS_CONSOLIDATOR_MOCK
    system_instruction: str = "Consolidate similar subtopics from the provided topics."
    user_template: str = "Consolidate the sub-topics in this JSON: {content}"
    
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
    return final_state.get("consolidated_topics", {"topics": []})
