from pathlib import Path
from typing import Dict, Any
from ceviche.core.agent import Agent
from ceviche.core.context import Context

class ImageProcessorAgent(Agent):
    """Processes images within directory structures and generates metadata."""
    
    def __init__(self, debug: bool = False):
        super().__init__(debug)
        self.debug = debug

    def pre_execution(self, ctx: Context, args: Dict[str, Any]):
        """Prepare context for image processing."""
        if self.debug:
            print("ImageProcessor: pre_execution")
        ctx["base_dir"] = args.get("directory", ".")
        ctx["excluded_dirs"] = args.get("excluded_folders", [])
        ctx["debug"] = self.debug

    def execute(self, ctx: Context, args: Dict[str, Any]) -> Any:
        """Main execution method for image processing."""
        base_dir = Path(ctx["base_dir"])
        excluded_dirs = ctx["excluded_dirs"]
        target_folders = args.get("target_folders", [])
        
        try:
            if self.debug:
                print(f"Starting image processing with base directory: {base_dir}")
            
            # Find directories to process
            dirs_to_process = self._find_directories_to_process(base_dir, excluded_dirs, target_folders)
            
            if not dirs_to_process:
                print(f"No valid directories with PDFs found in {base_dir}")
                return
                
            # Process each directory
            for directory in dirs_to_process:
                if self.debug:
                    print(f"Processing directory: {directory}")
                
                # Check if images directory exists and create if needed
                images_dir = directory / "images"
                images_dir.mkdir(exist_ok=True)
                
                # Get the process_images workflow
                process_images_workflow = self.get_workflow("process_images", ctx, args)
                
                # Run the workflow with directory context
                workflow_args = {
                    "base_dir": str(directory),
                    "excluded_dirs": excluded_dirs,
                    "pdf_file": self._find_pdf_file(directory)
                }
                
                process_images_workflow.run(ctx, workflow_args)
            
            if self.debug:
                print("All image processing completed successfully")
                
        except Exception as e:
            print(f"❌ Image processing failed: {str(e)}")
            raise

    def _find_directories_to_process(self, base_dir: Path, excluded_dirs: list, target_folders: list) -> list:
        """Find directories that should be processed based on configuration.
        
        Args:
            base_dir: The base directory to start from
            excluded_dirs: List of directory names to exclude
            target_folders: List of specific directory names to target (if empty, process all)
            
        Returns:
            List of Path objects representing directories to process
        """
        dirs_to_process = []
        
        # If target folders specified, only look at those
        if target_folders:
            for folder in target_folders:
                folder_path = base_dir / folder
                if folder_path.exists() and folder_path.is_dir() and self._has_pdf(folder_path):
                    dirs_to_process.append(folder_path)
            return dirs_to_process
        
        # Process the base directory if it has a PDF
        if self._has_pdf(base_dir):
            dirs_to_process.append(base_dir)
        
        # Find all subdirectories with PDFs
        for subdir in base_dir.iterdir():
            if subdir.is_dir() and subdir.name not in excluded_dirs:
                if self._has_pdf(subdir):
                    dirs_to_process.append(subdir)
                    
        return dirs_to_process
        
    def _has_pdf(self, directory: Path) -> bool:
        """Check if directory contains at least one PDF file."""
        return any(directory.glob("*.pdf"))

    def _find_pdf_file(self, directory: Path) -> str:
        """Find first PDF file in directory for context."""
        pdf_files = list(directory.glob("*.pdf"))
        if pdf_files:
            return str(pdf_files[0])
        return ""

    def post_execution(self, ctx: Context, args: Dict[str, Any], result: Any):
        """Cleanup after processing."""
        if self.debug:
            print("ImageProcessor: post_execution") 