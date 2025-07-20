import os
from dotenv import load_dotenv

load_dotenv()

PROCESS_MINING_CONTEXT = """You are a process mining expert. 

IMPORTANT: Always check if the user is asking a simple factual question about the process data. If so, provide ONLY the direct answer based on the data.

For simple questions like:
- "How many times does X follow Y?"
- "What activities come after X?"
- "What is the flow from start to finish?"
- "Which activities are start/end activities?"

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
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    CSV_FILE_PATH = os.getenv("CSV_FILE_PATH")
    EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "intfloat/multilingual-e5-large")
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    LLM_MODEL_TEMPERATURE = float(os.getenv("LLM_MODEL_TEMPERATURE", "0.1"))
    LLM_MODEL_PARAMS = {"temperature": LLM_MODEL_TEMPERATURE}