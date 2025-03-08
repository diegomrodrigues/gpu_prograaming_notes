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
from ceviche.core.models.gemini import GeminiModel


class PlotGenerationMixin:
    """Mixin class for generating plots from code blocks in content."""
    
    def generate_plots_from_code(self, ctx: Context, content: str, task_config: Dict[str, Any], directory: Optional[Path] = None) -> str:
        """Orchestrates plot generation from code blocks in content."""
        print("🔍 Plot generation from code blocks")

        base_dir = self._get_base_directory(directory, task_config)
        images_dir = base_dir / "images"
        self._prepare_images_directory(images_dir)
        
        code_blocks = self._find_plot_code_blocks(content)
        
        return self._process_code_blocks(ctx, content, code_blocks, images_dir)

    def _get_base_directory(self, directory: Optional[Path], task_config: Dict[str, Any]) -> Path:
        """Determine the base directory for plot generation."""
        if directory:
            return directory
        return Path(task_config.get('directory', '.'))

    def _prepare_images_directory(self, images_dir: Path) -> None:
        """Ensure images directory exists."""
        images_dir.mkdir(exist_ok=True)

    def _find_plot_code_blocks(self, content: str) -> List[tuple[str, str]]:
        """Identify Python code blocks containing plotting code."""
        lines = content.split('\n')
        code_blocks = []
        current_block = []
        original_block = []
        in_code_block = False
        
        for line in lines:
            stripped = line.strip()
            
            # Start of Python code block
            if '```py' in stripped:
                in_code_block = True
                original_block = [line]
                continue
                
            # End of code block
            if in_code_block and stripped.startswith('```'):
                in_code_block = False
                code = '\n'.join(current_block)
                original = '\n'.join(original_block + [line])
                
                # Only add if it contains plotting-related code
                if any(kw in code for kw in ['plt', 'matplotlib', 'seaborn', 'sns']):
                    code_blocks.append((original, code))
                    
                current_block = []
                original_block = []
                continue
                
            # Collect lines while in code block
            if in_code_block:
                current_block.append(line)
                original_block.append(line)
        
        print(f"  - Found {len(code_blocks)} potential plotting code blocks")
        return code_blocks

    def _process_code_blocks(self, ctx: Context, content: str, code_blocks: List[tuple[str, str]], images_dir: Path) -> str:
        """Process all identified code blocks to generate plots."""
        updated_content = content
        
        model = GeminiModel(
            api_key=ctx.get("api_key"),  # Get API key from config
            model_name="gemini-2.0-flash-exp",
            system_instruction="Please correct the indentation and syntax of this Python code to make it runnable.",
            mock=ctx.get("mock_api", False)  # Pass mock flag
        )
        
        for i, (original_block, code) in enumerate(code_blocks, start=1):
            try:
                # Get corrected code from model
                corrected_code = self._get_corrected_code(model, code)
                
                temp_path = self._create_temp_script(corrected_code, images_dir, i)
                success = self._execute_plot_script(temp_path, images_dir, i)
                
                if success:
                    plot_markdown = f'![Generated plot](./images/plot_{i}.png)'
                    updated_content = updated_content.replace(original_block, plot_markdown, 1)
                    
            except Exception as e:
                print(f"  ❌ Error processing code block: {str(e)}")
            finally:
                Path(temp_path).unlink(missing_ok=True)
                
        return updated_content

    def _get_corrected_code(self, model: GeminiModel, raw_code: str) -> str:
        """Use Gemini to correct and format Python code."""
        chat = model.start_chat()
        response = model.send_message(chat, raw_code)
        return response.text.strip().strip('`').replace('python\n', '')

    def _create_temp_script(self, code: str, images_dir: Path, index: int) -> str:
        """Create temporary Python script with necessary imports and savefig."""
        imports = []
        if 'matplotlib' not in code:
            imports.append('import matplotlib.pyplot as plt')
        if 'numpy' in code and 'import numpy' not in code:
            imports.append('import numpy as np')
        if 'seaborn' in code or 'sns.' in code:
            imports.append('import seaborn as sns')

        final_code = '\n'.join([
            *imports,
            code.replace('plt.show()', '').strip(),
            f'plt.savefig("plot_{index}.png", bbox_inches="tight")',
            'plt.close()'
        ])

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write(final_code)
            print(f"  - Created temporary file: {f.name}")
            return f.name

    def _execute_plot_script(self, script_path: str, images_dir: Path, index: int) -> bool:
        """Execute Python script and validate plot generation."""
        try:
            result = subprocess.run(
                ['python', script_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=images_dir
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Script failed with exit code {result.returncode}")
                
            if not (images_dir / f"plot_{index}.png").exists():
                raise FileNotFoundError(f"plot_{index}.png not generated")
                
            print(f"  ✓ Successfully generated plot_{index}.png")
            return True
            
        except subprocess.TimeoutExpired:
            print("  ⚠️ Plot generation timed out")
            return False
        except Exception as e:
            print(f"  ⚠️ Plot generation failed: {str(e)}")
            return False