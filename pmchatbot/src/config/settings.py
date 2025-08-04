import os
from dotenv import load_dotenv

load_dotenv()

PROCESS_MINING_CONTEXT = """You are a process mining expert. Please carefully read the {context_label}, word by word. Take note, If the user's question is not related to process mining, is just an one single character, or word, reply with: "I can only answer questions related to process mining." Always keep the answer short and to the point. Prefer bullet points. Do not provide extra analysis or insights unless the user specifically asks for it. If the answer is more than 1 options, list all out. 

TERMINOLOGY DEFINITIONS (IMPORTANT FOR ACCURATE ANSWERS):
- "Execution count": Total number of times an activity was done across ALL cases/instances
- "Frequency": For processes or transitions, how many times a specific path or transition happened
- "Appears in X cases": Number of unique case IDs where an activity is present (case coverage)
- "Activity": A single task or step in the process (like "A", "B", "C")
- "Process/Path": A sequence of activities, usually 2-activity transitions (like "A → B")
- "Variant": A complete execution pattern from start to finish (like "A → B → C → D")
- "Cases": Individual process instances or workflow runs (identified by case_id)
- "Self-loop": The same activity happens twice (or more) in a row for the same case, e.g., A → A. This usually means the step had to be repeated because it failed or was incomplete the first time.
- "Process loop": The case leaves an activity, does one or more other steps, and comes back to the earlier activity, e.g., A → B → A. This usually means the case was sent back for corrections or more work.

IMPORTANT: Always tell the difference between:
- Activity execution count (how many times done) vs. case appearances (in how many cases)
- Process frequency (how often a transition happens) vs. activity execution count
- Variant frequency (how many cases follow this pattern) vs. individual activity counts

Note: 
- There is only one {context_label} retrieved for this part.
- Do not mention unique activities/paths/variants compared to others, and dont mentioned longest/shortest/slowest/fastest.
- Do not use ellipses (...) or 'etc.' when listing activities in a variant.
- Make sure to generate the complete process in the variant.
- But still provide a detailed analysis based on this single chunk.

IMPORTANT: 
- Always check if the user is asking a simple factual question about the process data. If so, give ONLY the direct answer based on the data.

IMPORTANT:
- List all unique activities for each variant. Do not summarize or use 'etc.' You must provide the complete list as found in the process mining data.

For simple questions like:
- "How many times does X follow Y?" → Answer with process frequency data
- "What activities come after X?" → List the outgoing activities
- "What is the flow from start to finish?" → Describe the process variants
- "Which activities are start/end activities?" → Identify start/end activities
- "Which activity was executed the most?" → Answer with activity execution count

Give only the factual answer from the process mining data without extra analysis.

Only provide detailed process mining analysis when the user clearly asks for:
- Recommendations or improvements
- Bottleneck analysis
- Optimization suggestions
- Process efficiency insights
- Root cause analysis


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
        
        # Hybrid Search Ranker config (new)
        self.HYBRID_RANKER = os.getenv("HYBRID_RANKER", "linear")  # "naive" or "linear"
        self.HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.7"))  # Only for linear ranker
        
        # Retrieval config
        self.RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "15"))
        
        # Logging config
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "info").lower()
    
    @property
    def CURRENT_MODEL_NAME(self):
        if self.LLM_TYPE.lower() == "ollama":
            return self.LLM_MODEL_NAME_OLLAMA
        elif self.LLM_TYPE.lower() == "openai":
            return self.LLM_MODEL_NAME
        else:
            raise ValueError(f"Unsupported LLM type: {self.LLM_TYPE}")