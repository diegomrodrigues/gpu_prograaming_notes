from typing import Any, Dict, Optional
import json
import re

class JsonUtilitiesMixin:
    def deduplicate_json_keys(self, json_data: str) -> str:
        """Deduplicate keys in JSON structure while preserving order."""
        parsed = json.loads(json_data)
        seen = set()
        result = []
        
        for item in parsed.get('topics', []):
            key = item['topic']
            if key not in seen:
                seen.add(key)
                result.append(item)
        
        return json.dumps({'topics': result}, indent=2)

    def dump_json(self, data: Dict[str, Any]) -> str:
        return json.dumps(data, indent=2, ensure_ascii=False)

    def extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from text using multiple strategies."""
        strategies = [
            self._extract_json_via_code_block,
            self._extract_json_via_bracket_matching,
            self._extract_json_via_repair
        ]
        
        for strategy in strategies:
            result = strategy(text)
            if result and self.validate_json(result):
                return result
        return None

    def validate_json(self, json_str: str) -> bool:
        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError:
            return False

    def _extract_json_via_code_block(self, text: str) -> Optional[str]:
        code_blocks = re.findall(r'```json\n(.*?)\n```', text, re.DOTALL)
        return max(code_blocks, key=len).strip() if code_blocks else None

    def _extract_json_via_bracket_matching(self, text: str) -> Optional[str]:
        start = max(text.find('{'), text.find('['))
        end = max(text.rfind('}'), text.rfind(']')) + 1
        return text[start:end] if start != -1 and end != -1 else None

    def _extract_json_via_repair(self, text: str) -> Optional[str]:
        balanced = self._balance_json_brackets(text)
        return balanced if self.validate_json(balanced) else None

    def _balance_json_brackets(self, json_str: str) -> str:
        stack = []
        balanced = []
        for char in json_str:
            if char in ['{', '[']:
                stack.append(char)
            elif char in ['}', ']']:
                if stack and ((char == '}' and stack[-1] == '{') or (char == ']' and stack[-1] == '[')):
                    stack.pop()
            balanced.append(char)
        while stack:
            balanced.append('}' if stack.pop() == '{' else ']')
        return ''.join(balanced)