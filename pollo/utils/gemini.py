import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Dict, Any, List, Optional, Iterator
import os
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    AIMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from pydantic import Field, PrivateAttr

class GeminiChatModel(BaseChatModel):
    """Handles direct interaction with the Gemini API including mock operations."""

    # Model name to use.
    model_name: str = Field(default="gemini-2.0-flash-exp")
        
    # Private attributes
    _api_key: str = PrivateAttr()
    _model: Optional[Any] = PrivateAttr(default=None)
    _safety_settings: Dict[Any, Any] = PrivateAttr(default={})
    _system_instruction: Optional[str] = PrivateAttr(default=None)
    _mock: bool = PrivateAttr(default=False)
    _mock_response: Optional[str] = PrivateAttr(default=None)

    def __init__(
        self, 
        model_name: str = "gemini-2.0-flash-exp",
        system_instructions: Optional[str] = None,
        mock_response: Optional[str] = None,
        **kwargs: Any
    ):
        """Initialize the Gemini chat model.
        
        Args:
            api_key: API key for the Gemini API
            model_name: Name of the Gemini model to use
            mock: Whether to mock API calls (for testing)
            mock_response: Specific response to return when in mock mode
        """
        super().__init__(model_name=model_name, **kwargs)
        self._api_key = os.environ.get("GEMINI_API_KEY", "fake-key")
        self._mock = os.environ.get("MOCK_API", "false") == "true"
        self._system_instruction = system_instructions
        self._mock_response = mock_response
        
        if not self._mock:
            genai.configure(api_key=self._api_key)
        
        self._configure_safety_settings()
        self._model = self._create_model() if not self._mock else None

    def _configure_safety_settings(self) -> None:
        """Configure default safety settings for the model."""
        self._safety_settings = {
            category: HarmBlockThreshold.BLOCK_NONE
            for category in [
                HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                HarmCategory.HARM_CATEGORY_HARASSMENT,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            ]
        }

    def _create_model(self) -> genai.GenerativeModel:
        """Create a Gemini model with specific configuration."""
        return genai.GenerativeModel(
            model_name=self.model_name,
            safety_settings=self._safety_settings,
            system_instruction=self._system_instruction
        )

    def configure(self, generation_config: Dict[str, Any]) -> None:
        """Update the model's generation configuration."""
        if not self._mock:
            if self._model:
                self._model = genai.GenerativeModel(
                    model_name=self.model_name,
                    safety_settings=self._safety_settings,
                    system_instruction=self._system_instruction,
                    generation_config=generation_config
                )

    def _get_human_message_content(self, messages: List[BaseMessage]) -> str:
        """Extract the most recent human message content.
        
        Args:
            messages: List of LangChain messages
            
        Returns:
            The content of the most recent human message, or a default prompt
        """
        # Look for the most recent human message
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message.content
        
        # If no human message is found, use a default prompt
        return "Please respond to the conversation."

    def _gemini_messages_from_langchain(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain messages to Gemini format messages.
        
        Args:
            messages: List of LangChain messages
            
        Returns:
            List of messages in Gemini format
        """
        gemini_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                # Gemini doesn't have a system message type, so we'll handle it differently
                # Usually added as system_instruction during model creation
                continue
            elif isinstance(message, HumanMessage):
                gemini_messages.append({"role": "user", "parts": [{"text": message.content}]})
            elif isinstance(message, AIMessage):
                gemini_messages.append({"role": "model", "parts": [{"text": message.content}]})
            elif isinstance(message, ChatMessage):
                # Map custom roles to Gemini's expected format
                role = "user" if message.role == "human" else "model"
                gemini_messages.append({"role": role, "parts": [{"text": message.content}]})
        
        return gemini_messages

    def upload_file(self, file_path: str, mime_type: Optional[str] = None) -> Any:
        """Upload a file to the Gemini API.
        
        Args:
            file_path: Path to the file to upload
            mime_type: Optional MIME type of the file
            
        Returns:
            The uploaded file object
        """
        if self._mock:
            # Mock file upload
            class MockFileState:
                def __init__(self, name):
                    self.name = name

            class MockFile:
                def __init__(self, name, display_name, uri, state, mime_type=None):
                    self.name = name
                    self.display_name = display_name
                    self.uri = uri
                    self.state = state
                    self.mime_type = mime_type
                    import datetime, uuid
                    self.created_time = datetime.datetime.now().isoformat()
                    self.updated_time = datetime.datetime.now().isoformat()
                    self.size_bytes = 1024

            mock_file_name = str(uuid.uuid4())
            mock_display_name = file_path.split('/')[-1]
            mock_uri = f"mock://files/{mock_file_name}"
            mock_state = MockFileState("ACTIVE")

            return MockFile(mock_file_name, mock_display_name, mock_uri, mock_state, mime_type)
        else:
            # Actually upload the file
            return genai.upload_file(file_path, mime_type=mime_type)

    def _generate_with_files(
        self,
        messages: List[BaseMessage],
        files: List[Any],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response using the Gemini model with files.
        
        Args:
            messages: The messages to send to the model
            files: List of uploaded file objects
            stop: Optional list of strings to stop generation
            
        Returns:
            The generated text response
        """
        if self._mock:
            # If a specific mock response is provided, use it
            if self._mock_response:
                return self._mock_response
                
            # Otherwise, return a generic mock response
            file_names = [getattr(f, 'display_name', 'unknown') for f in files]
            return f"This is a mock response from Gemini model analyzing files: {', '.join(file_names)}"
            
        # Get the latest human message content
        prompt = self._get_human_message_content(messages)
        
        # Create parts with files and prompt
        parts = list(files)  # Add files as parts
        parts.append(prompt)  # Add the text prompt
        
        # Generate content with files
        response = self._model.generate_content(
            parts,
            generation_config={"stop_sequences": stop} if stop else None,
            **kwargs
        )
        
        return response.text

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the Gemini model.
        
        Args:
            messages: The messages to send to the model
            stop: Optional list of strings to stop generation
            run_manager: Optional callback manager
            
        Returns:
            ChatResult containing the generated response
        """
        # Check if files are provided in kwargs
        files = kwargs.pop("files", None)
        
        if files:
            # If files are provided, use the file-specific generation method
            response_text = self._generate_with_files(messages, files, stop, **kwargs)
            
            # Create the AI message and generation
            ai_message = AIMessage(content=response_text)
            generation = ChatGeneration(message=ai_message)
            
            return ChatResult(generations=[generation])
        
        if self._mock:
            # If a specific mock response is provided, use it
            if self._mock_response:
                mock_response = self._mock_response
            else:
                # Otherwise, use a generic mock response
                mock_response = "This is a mock response from Gemini model."
                
            message = AIMessage(content=mock_response)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
            
        # Convert LangChain messages to Gemini format
        gemini_messages = self._gemini_messages_from_langchain(messages)
        
        # Get the latest human message content
        prompt = self._get_human_message_content(messages)
        
        # Two options for handling messages:
        if len(gemini_messages) <= 1:
            # If there's only one or no messages, use direct generation
            response = self._model.generate_content(
                prompt,
                generation_config={"stop_sequences": stop} if stop else None,
                **kwargs
            )
        else:
            # If there's a conversation history, use a chat session
            chat = self._model.start_chat(history=gemini_messages[:-1])
            
            # Send the most recent message
            response = chat.send_message(
                prompt,
                stream=False,
                generation_config={"stop_sequences": stop} if stop else None,
                **kwargs
            )
        
        # Extract token usage if available
        token_usage = {}
        if hasattr(response, "usage_metadata"):
            token_usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
            }
        
        # Create the AI message and generation
        ai_message = AIMessage(
            content=response.text,
            additional_kwargs={},
            response_metadata={"token_usage": token_usage}
        )
        generation = ChatGeneration(message=ai_message)
        
        return ChatResult(generations=[generation], llm_output={"token_usage": token_usage})

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the response from the Gemini model.
        
        Args:
            messages: The messages to send to the model
            stop: Optional list of strings to stop generation
            run_manager: Optional callback manager
            
        Returns:
            Iterator of ChatGenerationChunk
        """
        if self._mock:
            # If a specific mock response is provided, use it
            if self._mock_response:
                mock_response = self._mock_response
            else:
                # Otherwise, use a generic mock response
                mock_response = "This is a mock response from Gemini model."
                
            # Yield the mock response in chunks
            for char in mock_response:
                chunk = AIMessageChunk(content=char)
                yield ChatGenerationChunk(message=chunk)
            return
            
        # Convert LangChain messages to Gemini format
        gemini_messages = self._gemini_messages_from_langchain(messages)
        
        # Get the latest human message content
        prompt = self._get_human_message_content(messages)
        
        # Two options for handling messages:
        if len(gemini_messages) <= 1:
            # If there's only one or no messages, use direct generation
            response_stream = self._model.generate_content(
                prompt,
                stream=True,
                generation_config={"stop_sequences": stop} if stop else None,
                **kwargs
            )
        else:
            # If there's a conversation history, use a chat session
            chat = self._model.start_chat(history=gemini_messages[:-1])
            
            # Generate the response in streaming mode with the most recent message
            response_stream = chat.send_message(
                prompt, 
                stream=True,
                generation_config={"stop_sequences": stop} if stop else None,
                **kwargs
            )
        
        # Stream the chunks
        for chunk in response_stream:
            if chunk.text:
                ai_chunk = AIMessageChunk(content=chunk.text)
                yield ChatGenerationChunk(message=ai_chunk)

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "gemini"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {
            "model_name": self.model_name,
            "system_instruction": self._system_instruction,
        }