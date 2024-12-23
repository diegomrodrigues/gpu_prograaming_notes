import os
import time
import json
import google.generativeai as genai

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "02. Data Parallel Execution Model")
PROMPT_FILE = os.path.join(BASE_DIR, "00. prompts", "Resumo.md")
GEMINI_API_KEY = "AIzaSyCDzmg1GM54f05KqSxzx7266kzuEnFGCPs"  #os.environ["GEMINI_API_KEY"]

# Model configuration
GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

def init_gemini():
    """Initialize Gemini API with environment credentials."""
    genai.configure(api_key=GEMINI_API_KEY)

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    print(f"\nUploading file: {path}")
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"✓ Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
  """Waits for the given files to be active.

  Some files uploaded to the Gemini API need to be processed before they can be
  used as prompt inputs. The status can be seen by querying the file's "state"
  field.

  This implementation uses a simple blocking polling loop. Production code
  should probably employ a more sophisticated approach.
  """
  print("Waiting for file processing...")
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  print("...all files ready")
  print()

def create_model(prompt_file):
    """Create and configure the Gemini model."""
    print(f"\nCreating model with prompt from: {prompt_file}")
    with open(prompt_file, 'r') as system_prompt:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=GENERATION_CONFIG,
            system_instruction="\n".join(system_prompt.readlines()),
        )
    print("✓ Model created successfully")
    return model

def initialize_chat_session(model, files):
    """Initialize a chat session with multiple files."""
    print("\nInitializing chat session...")
    # Create initial history with all files
    history = []
    for file in files:
        history.append({
            "role": "user",
            "parts": [file],
        })
    
    chat_session = model.start_chat(history=history)
    print("✓ Chat session initialized")
    return chat_session

def process_topics(chat_session, topics):
    """Process each topic and collect responses."""
    sections = []
    total_topics = len(topics)
    print(f"\nProcessing {total_topics} topics:")
    
    for i, topic in enumerate(topics, 1):
        print(f"\nTopic {i}/{total_topics}:")
        print(f"- {topic[:100]}...")  # Print first 100 chars of topic
        response = chat_session.send_message(topic)
        sections.append(response.text)
        print("✓ Response received")
    
    print("\n✓ All topics processed successfully")
    return sections

def save_output(content, output_file):
    """Save the generated content to a file."""
    print(f"\nSaving output to: {output_file}")
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(content)
    print("✓ Output saved successfully")

def get_pdf_files_in_dir(directory):
    """Get all PDF files in the specified directory."""
    pdf_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory, file))
    return pdf_files

def upload_multiple_files(file_paths):
    """Upload multiple files to Gemini."""
    uploaded_files = []
    for path in file_paths:
        uploaded_files.append(upload_to_gemini(path, mime_type="application/pdf"))
    return uploaded_files

def read_topics_file(directory):
    """Read and find the topics.md file in the given directory."""
    for file in os.listdir(directory):
        if file.lower() == 'topics.md':
            file_path = os.path.join(directory, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    raise FileNotFoundError("topics.md not found in the specified directory")

def get_topics_dict(topics_content):
    """Convert topics content to dictionary using Gemini."""
    # Configure model for JSON parsing
    json_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }
    
    topics_model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=json_config,
        system_instruction="""given the topics, return it in a json format with the following template:
        {
        "topic_name": [ "topic 1", "topic 2", ... ],
        "topic_name": [ "topic 1", "topic 2", ... ],
        ...
        }"""
    )
    
    # Get JSON response
    chat = topics_model.start_chat()
    response = chat.send_message(topics_content)
    return json.loads(response.text)

def create_diagram_model():
    """Create a model specifically for generating Mermaid diagrams."""
    diagram_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=diagram_config,
        system_instruction="""You are a technical documentation expert specializing in Mermaid diagrams. Your task is to:

1. Analyze the given text
2. Identify concepts that would benefit from visual representation
3. Add Mermaid diagram code blocks where appropriate, using this format:
   ```mermaid
   [diagram code here]
   ```

Guidelines for diagrams:
- Focus on architecture and system relationships
- Use clear, descriptive labels in quotes
- Prefer flowcharts and sequence diagrams over mindmaps
- Keep diagrams focused and not too complex
- Always use proper Mermaid syntax
- Add diagrams inline where they best support the text

Do not modify the original text - only add Mermaid diagram blocks where helpful."""
    )

def create_section_directory(base_dir, section_name):
    """Create and return the path to a section directory."""
    safe_section_name = "".join(c for c in section_name if c.isalnum() or c in (' ', '-', '_')).strip()
    section_dir = os.path.join(base_dir, f"01. {safe_section_name}")
    os.makedirs(section_dir, exist_ok=True)
    return section_dir

def generate_topic_content(chat_session, topic):
    """Generate initial content for a topic."""
    return chat_session.send_message(topic)

def add_diagrams_to_content(diagram_model, content):
    """Process content to add Mermaid diagrams."""
    diagram_chat = diagram_model.start_chat()
    return diagram_chat.send_message(
        f"""Please enhance this text by adding appropriate Mermaid diagrams:

{content.text}

Remember to:
1. Keep the original text unchanged
2. Add Mermaid diagram code blocks where they would help explain concepts
3. Place diagrams in logical positions within the text"""
    )

def create_filename_model():
    """Create a model specifically for generating filenames."""
    filename_config = {
        "temperature": 0.7,  # Reduced for more consistent naming
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1024,  # Reduced since we only need short responses
        "response_mime_type": "text/plain",
    }
    
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=filename_config,
        system_instruction="""Generate a concise, descriptive filename (without extension) for the given topic. Requirements:
- Maximum 50 characters
- Use only letters, numbers, and spaces
- Start with a capital letter
- Be descriptive but concise
- Remove any special characters
- Return ONLY the filename, nothing else

Example input:
"Hierarchical Thread Structure: Concepts of grids, blocks, and threads within CUDA"

Example output:
"Hierarchical Thread Structure"
""")

def save_topic_file(section_dir, topic, index, content):
    """Save topic content to a file."""
    # Create filename model and get suggested name
    filename_model = create_filename_model()
    chat = filename_model.start_chat()
    suggested_name = chat.send_message(topic).text.strip()
    
    # Create filename with index and suggested name
    topic_filename = f"{index:02d}. {suggested_name}.md"
    topic_path = os.path.join(section_dir, topic_filename)
    
    with open(topic_path, "w", encoding='utf-8') as f:
        f.write(content.text)
    return topic_filename

def process_topic_section(chat_session, topics, section_name):
    """Process topics for a specific section and save individual topic files."""
    total_topics = len(topics)
    print(f"\nProcessing {total_topics} topics for section: {section_name}")
    
    # Create models and directory
    diagram_model = create_diagram_model()
    section_dir = create_section_directory(INPUT_DIR, section_name)
    
    for i, topic in enumerate(topics, 1):
        print(f"\nTopic {i}/{total_topics}:")
        print(f"- {topic[:100]}...")
        
        # Generate and enhance content
        initial_response = generate_topic_content(chat_session, topic)
        print("Adding diagrams...")
        final_response = add_diagrams_to_content(diagram_model, initial_response)
        
        # Save result
        filename = save_topic_file(section_dir, topic, i, final_response)
        print(f"✓ Saved topic with diagrams to: {filename}")

def main():
    """Main execution flow."""
    # Initialize Gemini
    init_gemini()

    # Get input directory and find PDFs
    pdf_files = get_pdf_files_in_dir(INPUT_DIR)
    
    if not pdf_files:
        print("No PDF files found in the specified directory!")
        return

    # Upload and process files
    files = upload_multiple_files(pdf_files)
    wait_for_files_active(files)

    # Create model and chat session
    model = create_model(PROMPT_FILE)
    chat_session = initialize_chat_session(model, files)

    # Get topics dictionary
    topics_content = read_topics_file(INPUT_DIR)
    topics_dict = get_topics_dict(topics_content)

    # Process each section separately
    for section_name, topics in topics_dict.items():
        # Process topics for this section
        process_topic_section(chat_session, topics, section_name)

if __name__ == "__main__":
    main()