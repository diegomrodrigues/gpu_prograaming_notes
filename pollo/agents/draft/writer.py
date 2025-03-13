from typing import TypedDict, List, Optional, Dict, Literal, Any
from langgraph.graph import StateGraph, END, START
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field
import yaml
from pathlib import Path

from pollo.agents.topics.generator import Topic
from pollo.utils.gemini import GeminiChatModel
from pollo.utils.prompts import load_chat_prompt_from_yaml

# Define state schemas
class DraftSubtaskState(TypedDict):
    topic: str
    subtopic: str
    draft: Optional[str]
    cleaned_draft: Optional[str]
    status: Literal["pending", "draft_generated", "cleaned", "error"]

class DraftWritingState(TypedDict):
    directory: str
    perspectives: List[str]
    json_per_perspective: int
    topics: List[Dict]
    current_topic_index: int
    current_subtopic_index: int
    drafts: List[Dict]
    status: str

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
        # Format input for the chain
        def format_input(inputs):
            return {
                "topic": inputs["topic"],
                "subtopic": inputs["subtopic"]
            }
        
        # Build the chain
        self.chain = (
            RunnableLambda(format_input) | 
            self.generate_draft_prompt | 
            self.gemini
        )

    def _run(self, topic: str, subtopic: str) -> str:
        """Generate a draft for the given subtopic."""
        response = self.chain.invoke({
            "topic": topic,
            "subtopic": subtopic
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
    builder.add_node("handle_error", handle_draft_error)

    # Set edges
    builder.add_edge(START, "generate_draft")
    builder.add_conditional_edges(
        "generate_draft",
        lambda s: "clean_draft" if s["draft"] else "handle_error"
    )
    builder.add_edge("clean_draft", END)
    builder.add_edge("handle_error", END)

    return builder.compile()

# Parent graph implementation
def create_draft_writer() -> StateGraph:
    """Main graph that coordinates topic generation and draft writing"""
    builder = StateGraph(DraftWritingState)
    
    # Add main nodes
    builder.add_node("generate_topics", generate_topics)
    builder.add_node("initialize_processing", initialize_processing)
    builder.add_node("process_subtopic", process_subtopic)
    builder.add_node("finalize", finalize_output)

    # Add subgraph
    draft_subgraph = create_draft_subgraph()
    builder.add_node("draft_subgraph", draft_subgraph)

    # Set edges
    builder.add_edge(START, "generate_topics")
    builder.add_edge("generate_topics", "initialize_processing")
    builder.add_edge("initialize_processing", "process_subtopic")
    
    # Conditional edges for processing loop
    builder.add_conditional_edges(
        "process_subtopic",
        lambda s: "process_subtopic" if has_more_subtopics(s) else "finalize"
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
        "status": "pending"
    }
    
    # Execute subgraph
    result = create_draft_subgraph().invoke(subtask_state)
    
    # Update main state
    new_drafts = state["drafts"] + [{
        "topic": result["topic"],
        "subtopic": result["subtopic"],
        "draft": result.get("draft"),
        "cleaned_draft": result.get("cleaned_draft"),
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
    """Finalize output structure"""
    return {
        **state,
        "status": "completed",
        "drafts": [d for d in state["drafts"] if d["status"] == "cleaned"]
    }

# Subgraph node implementations
def generate_draft(state: DraftSubtaskState) -> DraftSubtaskState:
    """Generate draft for a subtopic using the DraftGeneratorTool"""
    try:
        generator = DraftGeneratorTool()
        draft = generator.invoke({
            "topic": state["topic"],
            "subtopic": state["subtopic"]
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

# Main function to use the draft writer
def generate_drafts_from_topics(
    directory: str,
    perspectives: List[str] = ["technical_depth"],
    json_per_perspective: int = 3
) -> Dict:
    """Generate drafts from topics extracted from PDFs"""
    # Create the graph
    draft_writer = create_draft_writer()
    
    # Prepare initial state
    initial_state = {
        "directory": directory,
        "perspectives": perspectives,
        "json_per_perspective": json_per_perspective,
        "topics": [],
        "current_topic_index": 0,
        "current_subtopic_index": 0,
        "drafts": [],
        "status": "starting"
    }
    
    # Run the graph
    final_state = draft_writer.invoke(initial_state)
    
    # Return the drafts
    return {
        "drafts": final_state.get("drafts", []),
        "status": final_state.get("status", "unknown")
    } 