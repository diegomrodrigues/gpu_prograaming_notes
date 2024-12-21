import os
import time
import google.generativeai as genai

# Constants
INPUT_FILE = "01. Data Parallelism\Data Parallelism_64-85.pdf"
OUTPUT_FILE = "01. Data Parallelism\02. CUDA Program Structure.md"
PROMPT_FILE = "00. prompts\Resumo.md"
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
    """Initialize a chat session with the given files."""
    print("\nInitializing chat session...")
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [files[0]],
            },
        ]
    )
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

def main():
    """Main execution flow."""
    # Initialize Gemini
    init_gemini()

    # Upload and process files
    files = [upload_to_gemini(INPUT_FILE, mime_type="application/pdf")]
    wait_for_files_active(files)

    # Create model and chat session
    model = create_model(PROMPT_FILE)
    chat_session = initialize_chat_session(model, files)

    # Process topics
    sections = process_topics(chat_session, TOPICS)
    
    # Save results
    full_markdown = "\n".join(sections)
    save_output(full_markdown, OUTPUT_FILE)

# Move topics list to top-level constant
TOPICS = [
    "Host-Device Model in CUDA: The fundamental architectural separation between the host (CPU) and one or more devices (GPUs) in a CUDA environment.",
    "Mixed Host and Device Code: The ability to integrate both CPU and GPU code within a single CUDA source file.",
    "Default Host Code: The interpretation of standard C code within a CUDA program as code intended for CPU execution.",
    "CUDA Keywords for Device Constructs: The use of specific keywords to identify functions and data structures intended for GPU execution.",
    "NVCC Compilation Process: The role of the NVIDIA CUDA Compiler (NVCC) in separating and compiling host and device code.",
    "Host Code Compilation Flow: Compilation of host code using standard C/C++ compilers and execution as a traditional CPU process.",
    "Device Code Compilation Flow: Marking of data-parallel functions (kernels) and data structures using CUDA keywords, and subsequent compilation by the NVCC runtime.",
    "Kernel Functions: The fundamental units of parallel execution on the CUDA device, encapsulating data-parallel operations.",
    "Threads in Modern Computing: Conceptual understanding of a thread as a unit of program execution within a processor.",
    "CUDA Kernel Launch and Thread Generation: The mechanism by which a CUDA program initiates parallel execution by launching kernel functions, leading to the creation of numerous threads.",
    "Efficiency of CUDA Thread Management: The hardware-level optimizations that enable rapid generation and scheduling of CUDA threads compared to traditional CPU threads.",
    "Grids of Threads: The collective term for all threads launched by a single kernel invocation.",
    "Kernel Execution Lifecycle: The sequence of host CPU execution initiating kernel launches, parallel GPU execution of threads, and the eventual termination of the grid, followed by continued host execution.",
    "Overlapping CPU and GPU Execution: Advanced techniques for concurrently executing code on the CPU and GPU to maximize resource utilization."
]

if __name__ == "__main__":
    main()