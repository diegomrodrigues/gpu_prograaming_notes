from ceviche.core.context import Context
from ceviche.core.workflow import Workflow
from ceviche.core.utilities.model_utils import ModelUtilsMixin
from ceviche.core.utilities.file_utils import WithReadAndWriteFilesMixin
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

class ProcessImagesWorkflow(
    Workflow,
    ModelUtilsMixin,
    WithReadAndWriteFilesMixin
):
    def __init__(self):
        super().__init__()

    def before_start(self, ctx: Dict[str, Any], args: Dict[str, Any]):
        super().before_start(ctx, args)
        self.process_image_task = self.load_task("process_image", ctx, args)

    def run(self, ctx: Context, args: Dict[str, Any]) -> Any:
        base_dir = Path(args.get("base_dir"))
        excluded_dirs = args.get("excluded_dirs", ["exclude_dir"])
        pdf_file = Path(args.get("pdf_file"))

        if not base_dir.exists():
            raise ValueError(f"Base directory does not exist: {base_dir}")
        if not pdf_file.exists():
            raise ValueError(f"PDF file does not exist: {pdf_file}")

        target_dirs = self._find_image_directories(base_dir, excluded_dirs)

        for directory in target_dirs:
            self._process_directory_images(ctx, directory, pdf_file)

        print("ProcessImagesWorkflow completed")
        return None  # Workflows generally don't return the processed content directly.

    def _find_image_directories(self, base_dir: Path, excluded_dirs: List[str]) -> List[Path]:
        """Finds directories containing 'images' subdirectories, excluding specified ones."""
        target_dirs = []
        for path in base_dir.rglob("images"):
            if any(excluded in path.parent.parts for excluded in excluded_dirs):
                continue
            target_dirs.append(path.parent)
        return target_dirs

    def _process_directory_images(self, ctx: Context, directory: Path, pdf_file: Path) -> None:
        """Process all images in a directory's images folder."""
        images_dir = directory / "images"
        if not images_dir.exists():
            print(f"  - No images directory found in {directory}")
            return

        image_entries = []
        # Keep track of the image index
        idx = 1
        for image_file in images_dir.iterdir():
            if image_file.is_file() and image_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                result = self._process_single_image(ctx, image_file, pdf_file, idx)
                if result:
                    image_entries.append(result)
                    idx += 1

        if image_entries:
            self._generate_json_summary(directory, image_entries)
            self._generate_markdown_summary(directory, image_entries)

    def _process_single_image(self, ctx: Context, image_file: Path, pdf_file: Path, idx: int) -> Optional[Dict]:
        """Process a single image with its associated PDF."""
        print(f"Processing image: {image_file.name}")

        try:
            result = self.process_image_task.run(
                ctx,
                args={
                    "image_file": str(image_file),
                    "pdf_file": str(pdf_file),
                }
            )
            if not result:
                print(f"  - Image processing returned no result: {image_file.name}")
                return None

            # Create new filename with index
            new_file_name = f"image{idx}{image_file.suffix}"
            new_file_path = image_file.parent / new_file_name
            image_file.rename(new_file_path)
            result['filename'] = new_file_name  # Add filename to the result

            return result

        except Exception as e:
            print(f"❌ Failed to process image {image_file.name}: {str(e)}")
            return None

    def _generate_json_summary(self, directory: Path, entries: List[Dict]) -> None:
        """Generates a JSON summary of image metadata."""
        json_summary = {"images": entries}
        json_file = directory / "images.json"
        json_file.write_text(json.dumps(json_summary, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"✔️ Generated JSON summary at: {json_file}")

    def _generate_markdown_summary(self, directory: Path, entries: List[Dict]) -> None:
        """Generates a Markdown summary of image metadata."""
        markdown_content = []
        for entry in entries:
            figure_name = entry.get('legenda', "Unnamed Figure")
            markdown_content.extend([
                f"![{figure_name}](./images/{entry['filename']})",
                "",
                entry['description'],
                ""
            ])

        # Write Markdown file
        markdown_file = directory / "images.md"
        markdown_file.write_text('\n'.join(markdown_content), encoding='utf-8')
        print(f"✔️ Generated markdown documentation at: {markdown_file}")
