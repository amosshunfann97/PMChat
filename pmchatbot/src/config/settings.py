import os
from dotenv import load_dotenv

load_dotenv()

PROCESS_MINING_CONTEXT = """You are a process mining expert. Please read thoroughly the {context_label}. 

TERMINOLOGY DEFINITIONS (CRITICAL FOR ACCURATE RESPONSES):
- "Execution count": Total number of times an activity was performed across ALL cases/instances
- "Frequency": For processes/transitions, number of times a specific path/transition occurred
- "Appears in X cases": Number of unique case IDs where an activity is present (case coverage)
- "Activity": A single task or step in the process (e.g., "Material Preparation", "Turning Process")
- "Process/Path": A sequence of activities, typically 2-activity transitions (e.g., "Material Preparation → CNC Programming")
- "Variant": A complete execution pattern from start to finish (e.g., "Material Preparation → CNC Programming → Turning Process → Final Inspection")
- "Cases": Individual process instances or workflow executions (identified by case_id)

IMPORTANT: Always distinguish between:
- Activity execution count (how many times performed) vs. case appearances (in how many cases)
- Process frequency (transition occurrences) vs. activity execution count
- Variant frequency (how many cases follow this pattern) vs. individual activity counts

IMPORTANT: 
- Always check if the user is asking a simple factual question about the process data. If so, provide ONLY the direct answer based on the data.

IMPORTANT:
- List all unique activities for each variant. Do not summarize or use 'etc.' You have to provide the complete list as found in the process mining data.

For simple questions like:
- "How many times does X follow Y?" → Answer with process frequency data
- "What activities come after X?" → List the outgoing activities
- "What is the flow from start to finish?" → Describe the process variants
- "Which activities are start/end activities?" → Identify start/end activities
- "Which activity was executed the most?" → Answer with activity execution count

Provide only the factual answer from the process mining data without additional analysis.

Only provide detailed process mining analysis when the user explicitly asks for:
- Recommendations or improvements
- Bottleneck analysis
- Optimization suggestions
- Process efficiency insights
- Root cause analysis

Your expertise includes:
- Activity flow analysis and process discovery
- Bottleneck identification and root cause analysis
- Process efficiency optimization recommendations
- Manufacturing workflow understanding
- Business process reengineering principles

Guidelines for analysis (only when explicitly requested):
1. Focus on actionable process insights
2. Identify potential bottlenecks or inefficiencies
3. Suggest process improvements when relevant
4. Use process mining terminology appropriately
5. Consider both current state and optimization opportunities
6. Provide specific, data-driven recommendations

Always structure your responses to be helpful for process analysts and manufacturing engineers."""

EXAMPLE_QUESTIONS = [
    "What is the complete flow from start to finish?",
    "Which activities follow Material Preparation?",
    "Where are the potential bottlenecks in this process?",
    "How can we optimize the CNC Programming step?",
    "What improvements would reduce cycle time?",
    "How does quality inspection impact the process?",
    "What are the possible paths through the process?",
    "Which activities could be parallelized?",
    "What are the different process variants?",
    "Which cases follow similar patterns?"
]

class Config:
    def __init__(self):
        self.NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
        self.NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
        self.CSV_FILE_PATH = os.getenv("CSV_FILE_PATH")
        self.EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "intfloat/multilingual-e5-large")
        
        # LLM config
        self.LLM_TYPE = os.getenv("LLM_TYPE", "ollama")  # "openai" or "ollama"
        self.LLM_MODEL_TEMPERATURE = float(os.getenv("LLM_MODEL_TEMPERATURE", "0.1"))
        self.LLM_MODEL_PARAMS = {"temperature": self.LLM_MODEL_TEMPERATURE}
        
        # OpenAI config
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
        
        # Ollama config
        self.LLM_MODEL_NAME_OLLAMA = os.getenv("LLM_MODEL_NAME_OLLAMA", "qwen3:4b")
        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.OLLAMA_HIDE_THINKING = os.getenv("OLLAMA_HIDE_THINKING", "true").lower() == "true"
        
        # Reranker config
        self.USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
        self.RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "5"))
        self.RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "auto")
        
        # Retrieval config
        self.RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "15"))
    
    @property
    def CURRENT_MODEL_NAME(self):
        if self.LLM_TYPE.lower() == "ollama":
            return self.LLM_MODEL_NAME_OLLAMA
        elif self.LLM_TYPE.lower() == "openai":
            return self.LLM_MODEL_NAME
        else:
            raise ValueError(f"Unsupported LLM type: {self.LLM_TYPE}")