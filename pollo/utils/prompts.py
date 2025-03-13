import yaml
from langchain_core.prompts import ChatPromptTemplate

def load_chat_prompt_from_yaml(file_path, default_system="", default_user="", variable_name="content"):
    """Load a chat prompt template from a YAML file with fallback defaults."""
    try:
        with open(file_path) as f:
            config = yaml.safe_load(f)
            system_instruction = config["prompt"]["system_instruction"]
            user_template = config["prompt"]["user_message"]
            
            # Create and return a ChatPromptTemplate
            return ChatPromptTemplate.from_messages([
                ("system", system_instruction),
                ("user", user_template)
            ])
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not load {file_path}: {e}")
        # Create a fallback template
        return ChatPromptTemplate.from_messages([
            ("system", default_system),
            ("user", default_user)
        ])
