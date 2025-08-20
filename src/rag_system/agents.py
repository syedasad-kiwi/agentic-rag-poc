import os
from crewai import Agent, LLM
from .tools import document_retrieval_tool

# Initialize the Ollama LLM for the agents - using gemma3:4b for reliability
# Use environment variable for base URL to support Docker deployment
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_llm = LLM(
    model="ollama/gemma3:4b",  # Using a larger model for better context understanding
    base_url=ollama_base_url,
    temperature=1,
    timeout=300,
    verbose=True,  # Enable verbose logging for debugging
)

# --- AGENT 1: The Specialist Retriever ---
# This agent's only job is to call the retrieval tool correctly.
document_researcher = Agent(
    role='Document Researcher',
    goal='Use the Document Retrieval Tool to find information relevant to a user\'s query from the knowledge base.',
    backstory=(
   "You are an information retrieval specialist. Your role is strictly limited to:, "
   "1) Analyze the user's query to understand intent, "
   "2) Retrieve relevant text chunks using the Document Retrieval Tool, "
   "3) Return only the raw retrieved context - no interpretation or answers. "
   "DO NOT answer questions using your general knowledge. "
   "DO NOT provide explanations, summaries, or interpretations. "
   "ONLY return the exact text chunks retrieved from the tool for the next agent to use."
),
    tools=[document_retrieval_tool],
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,  # Limit iterations to prevent infinite loops
)

# --- AGENT 2: The Specialist Synthesizer ---
# This agent's only job is to write the final answer based on the context it receives.
insight_synthesizer = Agent(
    role='Insight Synthesizer',
    goal='Formulate a comprehensive and accurate answer to the user\'s question based ONLY on the provided context.',
    backstory=(
   "You are an expert analyst. You receive context from a document_researcher and the user's original question. "
   "Your job is to extract and synthesize relevant information from the provided context into a clear, concise answer. "
   "CRITICAL RULES: "
   "- Use ONLY the provided context - never add outside knowledge "
   "- Pay special attention to requested facts and figures "
   "- If context is empty or no documents found, state you cannot answer the question "
   "- Do not begin answers with 'document does not specify' or similar phrases "
   "- Cherry-pick the most relevant information from all parts of the provided context"
),
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,  # Limit iterations to prevent infinite loops
    # This agent does not need tools; it only processes text.
    tools=[]
)
