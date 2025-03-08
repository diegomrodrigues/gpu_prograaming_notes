import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Dict, Any, List, Optional
import json
import re
import textwrap
import subprocess
import tempfile
from pathlib import Path
import time

from ceviche.core.context import Context
from .file_utils import WithReadAndWriteFilesMixin  # Import for plot saving
from .json_utils import JsonUtilitiesMixin
from ceviche.core.models.gemini import GeminiModel # Import Gemini

class ModelUtilsMixin:
    """Provides utility methods for interacting with generative models."""

    def init_model(self, ctx: Context, args: Dict[str, Any]) -> GeminiModel:
        """Initializes the Gemini model based on task configuration."""
        if not hasattr(self, 'task_config'):
            raise AttributeError("task_config must be set before calling init_model")

        return GeminiModel(
            api_key=ctx.get("api_key"),  # Get API key from config
            model_name=self.task_config.get("model_name", "gemini-2.0-flash-exp"),
            system_instruction=self.task_config.get("system_instruction"),
            mock=ctx.get("mock_api", False)  # Pass mock flag
        )

    def prepare_prompt(self, task_config: Dict[str, Any], content: str, **kwargs) -> str:
        """Prepares the user message prompt using the task configuration."""
        user_message = task_config.get("user_message", "")
        # Use kwargs for additional replacements
        replacements = {"content": content, **kwargs}
        for key, value in replacements.items():
            user_message = user_message.replace("{" + key + "}", str(value))
        return user_message

    def start_chat(self, files: Optional[List[Any]] = None) -> Any:
        """Starts a chat session with the model, optionally with file history."""
        if not hasattr(self, 'model'):
            raise AttributeError("Model not initialized. Call init_model() first.")
        return self.model.start_chat(history=self._build_file_history(files) if files else None)

    def _build_file_history(self, files: List[Any]) -> List[Dict[str, Any]]:
        """Builds the chat history for file uploads."""
        return [{"role": "user", "parts": [file]} for file in files]

    def send_message(self, chat: Any, user_content: str, files: Optional[List[Any]] = None) -> Any:
        """Sends a message to the model within a chat session."""
        if not hasattr(self, 'model'):
            raise AttributeError("Model not initialized. Call init_model() first.")

        return self.model.send_message(chat, user_content, files)

    def extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extracts JSON content from the given text, handling common issues."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Remove any text outside of JSON code blocks
        match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = text  # Try the whole text if no code block

        json_str = self._sanitize_json(json_str)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def _sanitize_json(self, json_str: str) -> str:
        """Clean up common JSON issues before parsing."""
        # Remove JSONP wrapper, if present
        json_str = re.sub(r'^[^{[]*', '', json_str)
        json_str = re.sub(r'[^}\]]*$', '', json_str)

        # Fix trailing commas
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

        # Remove comments (simple cases)
        json_str = re.sub(r'//.*?$|/\*.*?\*/', '', json_str, flags=re.MULTILINE | re.DOTALL)

        return json_str.strip()

    def get_pdf_files(self, directory: str) -> List[Path]:
        """Get all PDF files in the directory."""
        return list(Path(directory).glob("*.pdf"))

    def upload_files(self, files: List[Path]) -> List[Any]:
        """Uploads a list of files using the GeminiModel instance."""
        if not hasattr(self, 'model'):
            raise AttributeError("Model not initialized. Call init_model() first.")

        uploaded_files = []
        for file_path in files:
            print(f"Uploading file: {str(file_path)}")
            uploaded_file = self.model.upload_file(str(file_path))
            uploaded_files.append(uploaded_file)
        return uploaded_files

    def wait_for_files_active(self, files: List[Any]):
        """Waits for files to be in the ACTIVE state using GeminiModel instance."""
        if not hasattr(self, 'model'):
            raise AttributeError("Model not initialized. Call init_model() first.")

        # Skip waiting if using mock API
        if self.model.mock:
            print("Mock API: Skipping file activation wait")
            return

        for file in files:
            while True:
                retrieved_file = self.model.get_file(file.name)
                if retrieved_file.state.name == "ACTIVE":
                    break
                print(".", end="", flush=True)
                time.sleep(10)
        print("...all files ready")

    def get_model_config(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get model configuration from task config with defaults."""
        return {
            "temperature": task_config.get("temperature", 1),
            "top_p": task_config.get("top_p", 0.95),
            "top_k": task_config.get("top_k", 40),
            "max_output_tokens": task_config.get("max_output_tokens", 8192),
            "response_mime_type": task_config.get("response_mime_type", "text/plain"),
        }

