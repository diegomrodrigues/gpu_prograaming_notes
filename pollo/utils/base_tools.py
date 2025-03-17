from typing import Optional, Dict, Any, List, Type
from pathlib import Path
from pydantic import BaseModel
import yaml
from langchain_core.tools import BaseTool
from langchain_core.runnables.base import Runnable
from langchain_core.messages import SystemMessage, HumanMessage

from pollo.utils.gemini import GeminiChatModel

class GeminiBaseTool(BaseTool):
    """Base class for tools that use Gemini API."""
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.7
    mock_response: Optional[str] = None
    gemini: Optional[GeminiChatModel] = None
    system_instruction: Optional[str] = None
    user_template: Optional[str] = None
    prompt_file: Optional[Path] = None
    chain: Optional[Runnable] = None
    response_mime_type: Optional[str] = None
    response_schema: Optional[Type[BaseModel]] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Load prompt from YAML if specified
        if self.prompt_file:
            self._load_prompt_from_yaml(self.prompt_file)
            
        self.gemini = GeminiChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            mock_response=self.mock_response,
            response_mime_type=self.response_mime_type,
            response_schema=self.response_schema
        )
        
        # Build the chain if needed
        self._build_chain()
    
    def _load_prompt_from_yaml(self, file_path: Path):
        """Load system instruction and user template from a YAML file."""
        try:
            file_path_str = str(file_path)
            with open(file_path_str, encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.system_instruction = config["prompt"]["system_instruction"]
                self.user_template = config["prompt"]["user_message"]
        except (FileNotFoundError, KeyError) as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    def _build_chain(self):
        """Build the LCEL chain for the tool. Override in subclasses."""
        pass
    
    def create_messages(self, **kwargs):
        """Create a list of messages using the system instruction and user template."""
        messages = []
        
        # Add system message if available
        if self.system_instruction:
            messages.append(SystemMessage(content=self.system_instruction))
        
        # Add user message with template formatting if available
        if self.user_template:
            formatted_user_content = self.user_template.format(**kwargs)
            messages.append(HumanMessage(content=formatted_user_content))
        
        return messages

    def upload_files(self, file_paths, mime_type="application/pdf"):
        """Helper method to upload multiple files."""
        uploaded_files = []
        for file_path in file_paths:
            uploaded_file = self.gemini.upload_file(
                file_path=file_path,
                mime_type=mime_type
            )
            uploaded_files.append(uploaded_file)
        return uploaded_files 