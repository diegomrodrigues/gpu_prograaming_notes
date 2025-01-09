from IPython.core.magic import register_cell_magic

@register_cell_magic
def run_cuda(line, cell):
    """
    Custom magic command to:
    1. Write the cell's content to a file.
    2. Compile the file using nvcc.
    3. Run the compiled program.
    
    Usage:
    %%write_and_run_cuda filename.cu
    """
    import os

    # Extract the filename from the first argument
    filename = line.strip()
    if not ".cu" in filename:
        raise ValueError("File must have a .cu extension.")
    
    # Write the content of the cell to the specified file
    with open(filename, "w") as f:
        f.write(cell)
    
    # Compile the CUDA file
    compile_cmd = f"nvcc {filename} -o {os.path.splitext(filename)[0]}"
    compile_result = os.system(compile_cmd)
    if compile_result != 0:
        raise RuntimeError("Compilation failed.")
    
    # Execute the compiled file using `!`
    executable = os.path.splitext(filename)[0]
    print(f"Running the executable: {executable}")
    get_ipython().system(f"./{executable}")
