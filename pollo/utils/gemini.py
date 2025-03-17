from google import genai
from google.genai import types
from typing import Dict, Any, List, Optional, Iterator, Type
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
from pydantic import Field, PrivateAttr, BaseModel

class GeminiChatModel(BaseChatModel):
    """Handles direct interaction with the Gemini API including mock operations."""

    # Model name to use
    model_name: str = Field(default="gemini-2.0-flash-exp")
    
    # API configuration
    api_key: Optional[str] = Field(default=None, description="Single API key or comma-separated list of API keys")
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.95)
    top_k: int = Field(default=40)
    max_output_tokens: Optional[int] = Field(default=None)
    mock_response: Optional[str] = Field(default=None)
    response_mime_type: Optional[str] = Field(default=None)
    response_schema: Optional[Type[BaseModel]] = Field(default=None)
    
    # Gemini client
    _client: Any = PrivateAttr()
    _api_keys: List[str] = PrivateAttr(default_factory=list)
    _current_key_index: int = PrivateAttr(default=0)
    _max_retries: int = PrivateAttr(default=3)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize API keys list
        api_key = self.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided or set as GOOGLE_API_KEY environment variable")
            
        # Parse comma-separated API keys into a list
        self._api_keys = [key.strip() for key in api_key.split(",")]
        self._current_key_index = 0
        
        # Initialize the Gemini client with the first API key
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize or reinitialize the client with the current API key."""
        self._client = genai.Client(api_key=self._get_current_api_key())
    
    def _get_current_api_key(self) -> str:
        """Get the currently active API key."""
        return self._api_keys[self._current_key_index]
    
    def _rotate_api_key(self) -> str:
        """Rotate to the next available API key and reinitialize the client."""
        self._current_key_index = (self._current_key_index + 1) % len(self._api_keys)
        self._initialize_client()
        return self._get_current_api_key()
    
    def _convert_messages_to_gemini_format(self, messages: List[BaseMessage], files=None):
        contents = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                # System messages will be handled separately in _generate
                continue
            elif isinstance(message, HumanMessage):
                parts = []
                
                # Add the text content
                parts.append(types.Part.from_text(text=message.content))
                
                # Add files if they exist and this is a human message
                if files and isinstance(message, HumanMessage):
                    for file in files:
                        parts.append(types.Part.from_uri(
                            file_uri=file.uri,
                            mime_type=file.mime_type
                        ))
                
                contents.append(types.Content(
                    role="user",
                    parts=parts
                ))
            elif isinstance(message, AIMessage):
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=message.content)]
                ))
            elif isinstance(message, ChatMessage):
                role = "user" if message.role == "human" else "model"
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=message.content)]
                ))
        
        return contents
    
    def _get_system_message(self, messages: List[BaseMessage]) -> Optional[str]:
        """Extract system message if present."""
        for message in messages:
            if isinstance(message, SystemMessage):
                return message.content
        return None
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        """Generate a chat response using the Gemini API."""
        # For testing - return mock response if specified
        if os.environ.get("MOCK_API") == "true" and self.mock_response:
            message = AIMessage(content=self.mock_response)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        
        # Get files from kwargs if available
        files = kwargs.get("files", None)
        
        # Convert messages to Gemini format
        system_instruction = self._get_system_message(messages)
        contents = self._convert_messages_to_gemini_format(messages, files=files)
        
        # Prepare generation config
        generation_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_output_tokens,
        )
        
        # Set system instruction if available
        if system_instruction:
            generation_config.system_instruction = [
                types.Part.from_text(text=system_instruction)
            ]
        
        if stop:
            generation_config.stop_sequences = stop
            
        if self.response_mime_type:
            generation_config.response_mime_type = self.response_mime_type
            
        if self.response_schema:
            generation_config.response_schema = self.response_schema
        
        # Try with retries and API key rotation
        attempts = 0
        last_error = None
        
        while attempts < len(self._api_keys) * self._max_retries:
            try:
                # Send the request to Gemini
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=generation_config
                )
                
                # Format the response
                message = AIMessage(content=response.text)
                generation = ChatGeneration(message=message)
                
                return ChatResult(generations=[generation])
            
            except Exception as e:
                last_error = e
                attempts += 1
                
                # Log the error
                key_info = f"API Key {self._current_key_index + 1}/{len(self._api_keys)}"
                print(f"Error with {key_info}: {str(e)}. Rotating to next key...")
                
                # Rotate to the next API key
                self._rotate_api_key()
        
        # If we've exhausted all retries and API keys, raise the last error
        raise ValueError(f"Failed to generate after trying all API keys: {str(last_error)}")
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream a chat response using the Gemini API."""
        # Get files from kwargs if available
        files = kwargs.get("files", None)
        
        # Convert messages to Gemini format
        system_instruction = self._get_system_message(messages)
        contents = self._convert_messages_to_gemini_format(messages, files=files)
        
        # Prepare generation config
        generation_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_output_tokens,
        )
        
        # Set system instruction if available
        if system_instruction:
            generation_config.system_instruction = types.SystemInstruction(
                parts=[types.Part.from_text(text=system_instruction)]
            )
        
        if stop:
            generation_config.stop_sequences = stop
            
        if self.response_mime_type:
            generation_config.response_mime_type = self.response_mime_type
            
        if self.response_schema:
            generation_config.response_schema = self.response_schema
        
        # Try with retries and API key rotation
        attempts = 0
        last_error = None
        
        while attempts < len(self._api_keys) * self._max_retries:
            try:
                # Stream the response
                response_stream = self._client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=generation_config
                )
                
                for chunk in response_stream:
                    if not chunk.text:
                        continue
                    message_chunk = AIMessageChunk(content=chunk.text)
                    yield ChatGenerationChunk(message=message_chunk)
                
                # If we successfully yielded all chunks, we're done
                return
            
            except Exception as e:
                last_error = e
                attempts += 1
                
                # Log the error
                key_info = f"API Key {self._current_key_index + 1}/{len(self._api_keys)}"
                print(f"Error with {key_info}: {str(e)}. Rotating to next key...")
                
                # Rotate to the next API key
                self._rotate_api_key()
        
        # If we've exhausted all retries and API keys, raise the last error
        raise ValueError(f"Failed to stream after trying all API keys: {str(last_error)}")
    
    def upload_file(self, file_path: str, mime_type: str = "application/pdf"):
        """Upload a file to be used with Gemini API."""
        return self._client.files.upload(
            file=file_path
        )
    
    @property
    def _llm_type(self) -> str:
        return "gemini-chat"