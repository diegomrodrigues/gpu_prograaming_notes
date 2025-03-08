import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime, timezone

class GeminiModel:
    """Handles direct interaction with the Gemini API including mock operations."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp", system_instruction: Optional[str] = None, mock: bool = False):
        self.api_key = api_key
        self.mock = mock  # Controls all mock behavior
        if not self.mock:
            genai.configure(api_key=api_key)
        self.model_name = model_name
        self.system_instruction = system_instruction
        self._configure_safety_settings()
        self.model = self._create_model() if not self.mock else None

    def _configure_safety_settings(self):
        """Configure default safety settings for the model."""
        self.SAFETY_SETTINGS = {
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
            safety_settings=self.SAFETY_SETTINGS,
            system_instruction=self.system_instruction
        )

    def configure(self, generation_config: Dict[str, Any]):
        """Update the model's generation configuration."""
        if not self.mock:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=self.SAFETY_SETTINGS,
                system_instruction=self.system_instruction
            )

    def start_chat(self, history: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Starts a chat session with the model."""
        if self.mock:
            class MockChat:
                def send_message(self, *args, **kwargs):
                    return self._mock_send_message(*args, **kwargs)
            return MockChat()
        else:
            return self.model.start_chat(history=history)

    def send_message(self, chat: Any, user_content: str, files: Optional[List[Any]] = None) -> Any:
        """Sends a message to the model within a chat session."""
        if self.mock:
            return self._mock_send_message(user_content, files)
        else:
            if files:
                parts = [file for file in files]
                parts.append(user_content)
                response = chat.send_message(parts)
            else:
                response = chat.send_message(user_content)
            return response

    def upload_file(self, file_path: str, mime_type: Optional[str] = None) -> Any:
        """Upload a file to Gemini."""
        if self.mock:
            return self._mock_upload_file(file_path, mime_type)
        return genai.upload_file(file_path, mime_type=mime_type)

    def get_file(self, name: str) -> Any:
        """Retrieves a file by its name."""
        if self.mock:
            return self._mock_get_file(name)
        return genai.get_file(name)

    def _mock_send_message(self, user_content: str, files: Optional[List[Any]] = None) -> Any:
        """Generates a mock response for send_message."""
        class MockCandidate:
            def __init__(self, content):
                self.content = content

        class MockResponse:
            def __init__(self, text):
                self.text = text
                self.prompt_feedback = None
                self.candidates = [MockCandidate(text)]

            def __str__(self):
                return self.text

        mock_text = f"Mock response to: {user_content}"
        if files:
            file_details = []
            for file in files:
                if hasattr(file, 'display_name') and hasattr(file, 'mime_type'):
                    file_details.append(f"{file.display_name} ({file.mime_type or 'unknown type'})")
                else:
                    file_details.append("Unknown file")
            mock_text += f"\nAnalyzed files: {', '.join(file_details)}"

        return MockResponse(mock_text)

    def _mock_upload_file(self, file_path: str, mime_type: Optional[str] = None) -> Any:
        """Generates a mock response for upload_file."""
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
                self.created_time = datetime.now(timezone.utc).isoformat()
                self.updated_time = datetime.now(timezone.utc).isoformat()
                self.size_bytes = 1024

        mock_file_name = str(uuid.uuid4())
        mock_display_name = file_path.split('/')[-1]
        mock_uri = f"mock://files/{mock_file_name}"
        mock_state = MockFileState("ACTIVE")

        return MockFile(mock_file_name, mock_display_name, mock_uri, mock_state, mime_type)

    def _mock_get_file(self, name: str) -> Any:
        """Generates a mock response for get_file, simulating file retrieval."""
        class MockFileState:
            def __init__(self, name):
                self.name = name

        class MockFile:
            def __init__(self, name, display_name, uri, state):
                self.name = name
                self.display_name = display_name
                self.uri = uri
                self.state = MockFileState(state)
                self.created_time = datetime.now(timezone.utc).isoformat()
                self.updated_time = datetime.now(timezone.utc).isoformat()
                self.size_bytes = 1024

        # Simulate different states based on the requested file name
        if "active" in name:
            state = "ACTIVE"
        elif "processing" in name:
            state = "PROCESSING"
        else:
            state = "ACTIVE"  # Default to active

        mock_display_name = f"mock_file_{state.lower()}.pdf"
        mock_uri = f"mock://files/{name}"
        return MockFile(name, mock_display_name, mock_uri, state)
