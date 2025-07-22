import ollama
import re
from typing import Dict, Any, Optional
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.llm.types import LLMResponse
from config.settings import Config

config = Config()

class OllamaLLM(LLMInterface):
    """Custom LLM implementation for Ollama models using ollama-python library"""
    
    def __init__(
        self,
        model_name: str = None,
        model_params: Optional[Dict[str, Any]] = None,
        base_url: str = None,
        hide_thinking: bool = None
    ):
        self.model_name = model_name or config.LLM_MODEL_NAME_OLLAMA
        self._model_params = model_params or config.LLM_MODEL_PARAMS
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.hide_thinking = hide_thinking if hide_thinking is not None else getattr(config, 'OLLAMA_HIDE_THINKING', True)
        
        # Configure ollama client
        if self.base_url != "http://localhost:11434":
            ollama._client._host = self.base_url

    def _hide_thinking_tags(self, text: str) -> str:
        """Remove <think>...</think> tags and their content from the response"""
        if not self.hide_thinking:
            return text
        
        # Remove thinking tags and their content using regex
        # This handles both single-line and multi-line thinking blocks
        pattern = r'<think>.*?</think>'
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up any extra whitespace that might be left
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text.strip())
        
        return cleaned_text

    def invoke(self, input_, *args, system_instruction=None, **kwargs) -> LLMResponse:
        """Send request to Ollama and return response"""
        try:
            prompt = input_
            if system_instruction:
                prompt = f"{system_instruction}\n\n{input_}"
            
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': self._model_params.get("temperature", 0.1),
                    'top_p': self._model_params.get("top_p", 0.9),
                    'num_predict': self._model_params.get("max_tokens", 10000),
                    'stop': self._model_params.get("stop", [])
                }
            )
            
            content = response['response']
            if not content:
                raise Exception("Empty response from Ollama")
            
            # hide thinking tags if disabled
            content = self._hide_thinking_tags(content)
            
            return LLMResponse(content=content)
        except ollama.ResponseError as e:
            raise Exception(f"Ollama API error: {e}")
        except Exception as e:
            raise Exception(f"Ollama LLM error: {e}")

    def ainvoke(self, input_: str, **kwargs) -> LLMResponse:
        return self.invoke(input_, **kwargs)

    @property
    def model_params(self) -> Dict[str, Any]:
        return self._model_params
    
    @model_params.setter
    def model_params(self, value: Dict[str, Any]):
        self._model_params = value or {}