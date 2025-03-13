from typing import TypedDict, List, Optional, Dict, Literal, Any
from langgraph.graph import StateGraph, END, START
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field
import yaml
from pathlib import Path
import os

from pollo.agents.topics.generator import Topic
from pollo.utils.gemini import GeminiChatModel
from pollo.utils.prompts import load_chat_prompt_from_yaml

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

# Define mock responses for testing
DRAFT_GENERATOR_MOCK = """
# Aprendizado Supervisionado: Fundamentos e Aplicações

### Introdução

O aprendizado supervisionado representa um dos paradigmas fundamentais no campo da aprendizagem de máquina, caracterizado pela utilização de dados rotulados para o desenvolvimento de modelos preditivos [^1]. Esta abordagem metodológica possibilita aos algoritmos estabelecer mapeamentos entre características de entrada e saídas desejadas, viabilizando previsões sobre dados não vistos anteriormente [^2]. Conforme indicado no contexto, os modelos supervisionados formam a base para diversas aplicações em análise preditiva e classificação automática.

Tenha cuidado para não se desviar do tema principal, mantendo o foco nas metodologias supervisionadas conforme especificado.

### Conceitos Fundamentais

O aprendizado supervisionado opera sobre uma premissa essencial: aprender a partir de exemplos onde as respostas corretas são fornecidas [^3]. O algoritmo analisa dados de treinamento compostos por pares de entrada-saída, identificando padrões que relacionam as entradas às suas respectivas saídas. Este processo estruturado de aprendizagem envolve:

1. **Fase de Treinamento**: O algoritmo processa exemplos rotulados, ajustando parâmetros internos para minimizar erros de predição [^4].
2. **Fase de Validação**: O desempenho do modelo é avaliado em dados reservados para garantir capacidade de generalização.
3. **Fase de Teste**: A avaliação final ocorre em dados completamente novos para medir o desempenho no mundo real.

A representação matemática do problema de aprendizado supervisionado pode ser expressa como:

$$f: X \rightarrow Y$$

Onde $f$ representa a função que mapeamos da entrada $X$ para a saída $Y$ [^5].

Baseie seu capítulo exclusivamente nas informações fornecidas no contexto e nos tópicos anteriores quando disponíveis.

### Algoritmos Principais

O panorama do aprendizado supervisionado engloba diversos algoritmos, cada um com fundamentos matemáticos distintos e domínios específicos de aplicabilidade:

#### Métodos Lineares
- **Regressão Linear**: Modela relações entre variáveis usando equações lineares, sendo ótimo para variáveis-alvo contínuas [^6].
- **Regressão Logística**: Um algoritmo de classificação que modela probabilidades utilizando a função logística, particularmente eficaz para resultados binários [^7].

A função logística pode ser representada como:

$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)}}$$

#### Métodos Baseados em Árvores
- **Árvores de Decisão**: Estruturas hierárquicas que particionam dados com base em valores de características, criando regras de decisão interpretáveis [^8].
- **Random Forests**: Métodos ensemble que combinam múltiplas árvores de decisão para melhorar a precisão e reduzir overfitting [^9].
- **Gradient Boosting Machines**: Técnicas ensemble sequenciais que constroem árvores para corrigir erros das anteriores [^10].

### Support Vector Machines
Estes algoritmos identificam hiperplanos ótimos que maximizam a margem entre classes, lidando com problemas lineares e não-lineares através de funções kernel [^11].

O problema de otimização para SVMs pode ser expresso como:

$$\\min_{w,b} \\frac{1}{2}||w||^2$$
$$\\text{sujeito a } y_i(w^Tx_i + b) \\geq 1, \\forall i$$

Organize o conteúdo logicamente com introdução, desenvolvimento e conclusão.

### Considerações Práticas

A implementação do aprendizado supervisionado requer atenção cuidadosa a:

- **Engenharia de Características**: Transformar dados brutos em representações significativas que melhorem o desempenho do modelo [^12].
- **Compromisso Viés-Variância**: Equilibrar a complexidade do modelo para evitar tanto underfitting quanto overfitting [^13].
- **Métricas de Avaliação**: Selecionar métricas apropriadas (acurácia, precisão, recall, F1-score, RMSE) com base no contexto do problema [^14].
- **Validação Cruzada**: Usar técnicas como validação cruzada k-fold para obter estimativas confiáveis de desempenho [^15].

Ao compreender os princípios matemáticos, opções algorítmicas e considerações de implementação do aprendizado supervisionado, os praticantes podem aplicar efetivamente essas técnicas para extrair insights valiosos e previsões a partir de dados.

### Referências
[^1]: Definição fundamental de aprendizado supervisionado.
[^2]: Capacidade de generalização em modelos supervisionados.
[^3]: Princípio básico do aprendizado a partir de exemplos rotulados.
[^4]: Processo de ajuste de parâmetros durante o treinamento.
[^5]: Formalização matemática do problema de aprendizado.
[^6]: Características e aplicações da regressão linear.
[^7]: Função e aplicabilidade da regressão logística.

Use $ para expressões matemáticas em linha e $$ para equações centralizadas.

Lembre-se de usar $ em vez de \$ para delimitar expressões matemáticas.

<!-- END -->
"""

DRAFT_CLEANUP_MOCK = """
# Aprendizado Supervisionado: Fundamentos e Aplicações

### Introdução

O aprendizado supervisionado representa um dos paradigmas fundamentais no campo da aprendizagem de máquina, caracterizado pela utilização de dados rotulados para o desenvolvimento de modelos preditivos [^1]. Esta abordagem metodológica possibilita aos algoritmos estabelecer mapeamentos entre características de entrada e saídas desejadas, viabilizando previsões sobre dados não vistos anteriormente [^2]. Os modelos supervisionados formam a base para diversas aplicações em análise preditiva e classificação automática.

### Conceitos Fundamentais

O aprendizado supervisionado opera sobre uma premissa essencial: aprender a partir de exemplos onde as respostas corretas são fornecidas [^3]. O algoritmo analisa dados de treinamento compostos por pares de entrada-saída, identificando padrões que relacionam as entradas às suas respectivas saídas. Este processo estruturado de aprendizagem envolve:

1. **Fase de Treinamento**: O algoritmo processa exemplos rotulados, ajustando parâmetros internos para minimizar erros de predição [^4].
2. **Fase de Validação**: O desempenho do modelo é avaliado em dados reservados para garantir capacidade de generalização.
3. **Fase de Teste**: A avaliação final ocorre em dados completamente novos para medir o desempenho no mundo real.

A representação matemática do problema de aprendizado supervisionado pode ser expressa como:

$$f: X \rightarrow Y$$

Onde $f$ representa a função que mapeamos da entrada $X$ para a saída $Y$ [^5].

### Algoritmos Principais

O panorama do aprendizado supervisionado engloba diversos algoritmos, cada um com fundamentos matemáticos distintos e domínios específicos de aplicabilidade:

#### Métodos Lineares
- **Regressão Linear**: Modela relações entre variáveis usando equações lineares, sendo ótimo para variáveis-alvo contínuas [^6].
- **Regressão Logística**: Um algoritmo de classificação que modela probabilidades utilizando a função logística, particularmente eficaz para resultados binários [^7].

A função logística pode ser representada como:

$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)}}$$

#### Métodos Baseados em Árvores
- **Árvores de Decisão**: Estruturas hierárquicas que particionam dados com base em valores de características, criando regras de decisão interpretáveis [^8].
- **Random Forests**: Métodos ensemble que combinam múltiplas árvores de decisão para melhorar a precisão e reduzir overfitting [^9].
- **Gradient Boosting Machines**: Técnicas ensemble sequenciais que constroem árvores para corrigir erros das anteriores [^10].

Estes métodos apresentam diferentes compromissos entre viés e variância.

#### Support Vector Machines
Estes algoritmos identificam hiperplanos ótimos que maximizam a margem entre classes, lidando com problemas lineares e não-lineares através de funções kernel [^11].

O problema de otimização para SVMs pode ser expresso como:

$$\\min_{w,b} \\frac{1}{2}||w||^2$$
$$\\text{sujeito a } y_i(w^Tx_i + b) \\geq 1, \\forall i$$

### Considerações Práticas

A implementação do aprendizado supervisionado requer atenção cuidadosa a:

- **Engenharia de Características**: Transformar dados brutos em representações significativas que melhorem o desempenho do modelo [^12].
- **Compromisso Viés-Variância**: Equilibrar a complexidade do modelo para evitar tanto underfitting quanto overfitting [^13].
- **Métricas de Avaliação**: Selecionar métricas apropriadas (acurácia, precisão, recall, F1-score, RMSE) com base no contexto do problema [^14].
- **Validação Cruzada**: Usar técnicas como validação cruzada k-fold para obter estimativas confiáveis de desempenho [^15].

Ao compreender os princípios matemáticos, opções algorítmicas e considerações de implementação do aprendizado supervisionado, os praticantes podem aplicar efetivamente essas técnicas para extrair insights valiosos e previsões a partir de dados.

### Referências
[^1]: Definição fundamental de aprendizado supervisionado.
[^2]: Capacidade de generalização em modelos supervisionados.
[^3]: Princípio básico do aprendizado a partir de exemplos rotulados.
[^4]: Processo de ajuste de parâmetros durante o treinamento.
[^5]: Formalização matemática do problema de aprendizado.
[^6]: Características e aplicações da regressão linear.
[^7]: Função e aplicabilidade da regressão logística.

<!-- END -->
"""

# Create Tool classes for draft generation and cleanup
class FilenameGeneratorTool(BaseTool):
    name: str = "filename_generator"
    description: str = "Generates an appropriate filename for a subtopic"
    gemini: Optional[GeminiChatModel] = None
    generate_filename_prompt: Optional[ChatPromptTemplate] = None
    chain: Optional[Runnable] = None

    def __init__(self):
        super().__init__()
        self.gemini = GeminiChatModel(
            model_name="gemini-2.0-flash",
            temperature=0.2,
            mock_response="Default Filename"
        )
        
        # Create a prompt template similar to the reference implementation
        self.generate_filename_prompt =  load_chat_prompt_from_yaml(
            Path(__file__).parent / "generate_filename.yaml",
            default_system="You are an expert at organizing academic content. Your job is to create appropriate filenames for technical document sections.",
            default_user="Generate an appropriate filename for a section about: {subtopic}. This section belongs to the chapter on {topic}. Return only the filename with an appropriate extension."
        )
        
        # Create the LCEL chain
        self._build_chain()
    
    def _build_chain(self):
        """Build the LCEL chain for filename generation."""
        # Format input for the chain
        def format_input(inputs):
            return {
                "topic": inputs["topic"],
                "subtopic": inputs["subtopic"]
            }
        
        # Build the chain
        self.chain = (
            RunnableLambda(format_input) | 
            self.generate_filename_prompt | 
            self.gemini
        )

    def _run(self, topic: str, subtopic: str) -> str:
        """Generate a filename for the given topic/subtopic."""
        response = self.chain.invoke({
            "topic": topic,
            "subtopic": subtopic
        })
        return response.content.strip()

class DraftGeneratorTool(BaseTool):
    name: str = "draft_generator"
    description: str = "Generates an initial draft for a subtopic"
    gemini: Optional[GeminiChatModel] = None
    generate_draft_prompt: Optional[ChatPromptTemplate] = None
    chain: Optional[Runnable] = None

    def __init__(self):
        super().__init__()
        self.gemini = GeminiChatModel(
            model_name="gemini-2.0-flash",
            temperature=0.7,
            mock_response=DRAFT_GENERATOR_MOCK
        )
        
        self.generate_draft_prompt = load_chat_prompt_from_yaml(
            Path(__file__).parent / "generate_draft.yaml",
            default_system="You are an expert academic writer. Generate a detailed, well-structured section draft for a technical document.",
            default_user="Generate a detailed chapter section about: {subtopic}. The section belongs to the chapter on {topic}."
        )
        
        # Create the LCEL chain
        self._build_chain()
    
    def _build_chain(self):
        """Build the LCEL chain for draft generation."""
        from langchain_core.runnables import RunnableLambda
        
        # Format input for the chain
        def format_input(inputs):
            return {
                "topic": inputs["topic"],
                "subtopic": inputs["subtopic"]
            }
        
        # Process with files
        def process_with_files(inputs):
            prompt_args = inputs["prompt_args"]
            model_inputs = inputs["model_inputs"]
            files = inputs.get("files", [])
            
            # Call the model with files
            return self.gemini.invoke(
                model_inputs, 
                files=files
            )
            
        # Build the chain
        self.chain = (
            RunnableLambda(lambda inputs: {
                "prompt_args": inputs,
                "model_inputs": format_input(inputs),
                "files": inputs.get("files", [])
            }) |
            {
                "prompt_args": lambda x: x["prompt_args"],
                "model_inputs": lambda x: self.generate_draft_prompt.invoke(x["model_inputs"]),
                "files": lambda x: x["files"]
            } |
            RunnableLambda(process_with_files)
        )

    def _run(self, topic: str, subtopic: str, directory: str = None) -> str:
        """Generate a draft for the given subtopic using PDFs if available."""
        # If no directory provided, generate without files
        if not directory:
            response = self.chain.invoke({
                "topic": topic,
                "subtopic": subtopic
            })
            return response.content
        
        # Read PDF files
        pdf_files = []
        for file in Path(directory).glob("*.pdf"):
            pdf_files.append(str(file))
        
        if not pdf_files:
            # No PDFs found, generate without files
            response = self.chain.invoke({
                "topic": topic,
                "subtopic": subtopic
            })
            return response.content
        
        # Upload the PDF files
        uploaded_files = []
        for pdf_file in pdf_files:
            uploaded_file = self.gemini.upload_file(pdf_file, mime_type="application/pdf")
            uploaded_files.append(uploaded_file)
        
        # Invoke the chain with files
        response = self.chain.invoke({
            "topic": topic,
            "subtopic": subtopic,
            "files": uploaded_files
        })
        
        return response.content

class DraftCleanupTool(BaseTool):
    name: str = "draft_cleanup"
    description: str = "Cleans and improves a generated draft"
    gemini: Optional[GeminiChatModel] = None
    cleanup_draft_prompt: Optional[ChatPromptTemplate] = None
    chain: Optional[Runnable] = None

    def __init__(self):
        super().__init__()
        self.gemini = GeminiChatModel(
            model_name="gemini-2.0-flash",
            temperature=0.2,
            mock_response=DRAFT_CLEANUP_MOCK
        )
        
        self.cleanup_draft_prompt = load_chat_prompt_from_yaml(
            Path(__file__).parent / "cleanup_draft.yaml",
            default_system="You are an expert editor. Refine and improve the given draft to ensure it is clear, concise, and well-structured.",
            default_user="Clean and improve this draft to make it more coherent and professional:\n\n{draft}"
        )
        
        # Create the LCEL chain
        self._build_chain()
    
    def _build_chain(self):
        """Build the LCEL chain for draft cleanup."""
        # Format input for the chain
        def format_input(inputs):
            return {"draft": inputs["draft"]}
        
        # Build the chain
        self.chain = (
            RunnableLambda(format_input) | 
            self.cleanup_draft_prompt | 
            self.gemini
        )

    def _run(self, draft: str) -> str:
        """Clean and improve the given draft."""
        response = self.chain.invoke({"draft": draft})
        return response.content

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

    # Add subgraph
    draft_subgraph = create_draft_subgraph()
    builder.add_node("draft_subgraph", draft_subgraph)

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
        builder.add_node(node_name, process_subtopic_parallel)
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
    from pollo.agents.topics.generator import create_topic_generator

    topic_generator = create_topic_generator()
    topics = topic_generator.invoke({
        "directory": state["directory"],
        "perspectives": state.get("perspectives", ["technical_depth"]),
        "json_per_perspective": state.get("json_per_perspective", 3)
    })
    return {**state, "topics": topics["consolidated_topics"].topics}

def initialize_processing(state: DraftWritingState) -> DraftWritingState:
    """Initialize processing state"""
    return {
        **state,
        "current_topic_index": 0,
        "current_subtopic_index": 0,
        "drafts": [],
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
    current_topic: Topic = state["topics"][state["current_topic_index"]]
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
    """Finalize output structure and write files to disk"""
    # Filter completed drafts
    completed_drafts = [d for d in state["drafts"] if d["status"] == "filename_generated"]
    
    # Group drafts by topic
    drafts_by_topic = {}
    for draft in completed_drafts:
        topic_index = draft.get("topic_index", 0)
        topic = draft["topic"]
        if topic_index not in drafts_by_topic:
            drafts_by_topic[topic_index] = {"topic": topic, "drafts": []}
        drafts_by_topic[topic_index]["drafts"].append(draft)
    
    # Create directories and write files
    output_dir = state["directory"]
    for topic_index, topic_data in sorted(drafts_by_topic.items()):
        # Create numbered topic directory
        topic_dir_name = f"{topic_index+1:02d}. {topic_data['topic']}"
        topic_path = os.path.join(output_dir, topic_dir_name)
        os.makedirs(topic_path, exist_ok=True)
        
        # Write each draft to a file
        for draft in sorted(topic_data["drafts"], key=lambda x: x.get("subtopic_index", 0)):
            if draft.get("cleaned_draft") and draft.get("filename"):
                file_path = os.path.join(topic_path, draft["filename"])
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(draft["cleaned_draft"])
    
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
        
        draft = generator.invoke({
            "topic": state["topic"],
            "subtopic": state["subtopic"],
            "directory": directory
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
    """Generate a filename for the draft"""
    try:
        generator = FilenameGeneratorTool()
        base_filename = generator.invoke({
            "topic": state["topic"],
            "subtopic": state["subtopic"]
        })
        
        # Format filename with numbering prefix based on subtopic index
        subtopic_index = state.get("subtopic_index", 0)
        formatted_filename = f"{subtopic_index+1:02d}. {base_filename}"
        
        # Ensure it has .md extension if not already present
        if not formatted_filename.lower().endswith('.md'):
            formatted_filename += '.md'
            
        return {**state, "filename": formatted_filename, "status": "filename_generated"}
    except Exception as e:
        print(f"Error generating filename: {str(e)}")
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
        return state
    
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
        "directory": state["directory"]
    }
    
    # Execute subgraph
    result = create_draft_subgraph().invoke(subtask_state)
    
    # Return result for this specific branch
    return {
        **state,
        f"branch_result_{branch_id}": {
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

def finalize_batch(state: DraftWritingState) -> DraftWritingState:
    """Collect results from parallel branches and update state"""
    # Extract results from all branches
    new_drafts = state.get("drafts", [])
    
    # Gather results from all branches
    for i in range(len(state.get("current_batch", []))):
        result_key = f"branch_result_{i}"
        if result_key in state:
            new_drafts.append(state[result_key])
            
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
    
    # Create new state with updated indices and drafts
    new_state = {
        **state,
        "current_topic_index": topic_index,
        "current_subtopic_index": subtopic_index,
        "drafts": new_drafts
    }
    
    # Clean up temporary branch results from state
    for i in range(len(state.get("current_batch", []))):
        result_key = f"branch_result_{i}"
        if result_key in new_state:
            del new_state[result_key]
    
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
        "status": "starting"
    }
    
    # Run the graph
    final_state = draft_writer.invoke(initial_state)
    
    # Return the drafts and output location
    return {
        "drafts": final_state.get("drafts", []),
        "status": final_state.get("status", "unknown"),
        "output_directory": directory
    }