from crewai import Agent, LLM
from .tools import document_retrieval_tool
from .hybrid_tools import hybrid_search_tool 

# Initialize the Ollama LLM for the agents - using gemma3:1b for reliability
# Change to gpt-oss:latest if you want to use the larger model
ollama_llm = LLM(
    model="ollama/gemma3:4b",  # Using a larger model for better context understanding
    base_url="http://localhost:11434",
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
        "You are a specialist in information retrieval. Your sole purpose is to take a user's query, "
        "understand its intent, and use the Document Retrieval Tool to find the most relevant text chunks. "
        "You do not answer the question yourself; you only provide the raw, retrieved context for the next agent."
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
        "You are an expert analyst. You receive context from a {document_researcher} and a user's original question. "
        "Your job is to cherry pick the correct information from various parts of the provided information into a clear and concise answer. "
        "You must adhere strictly to the provided text and never use outside knowledge. You must pay special attention to any facts and figures being asked in the question. "
        "If the context is empty or states that no documents were found, you must state that you cannot answer the question."
    ),
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,  # Limit iterations to prevent infinite loops
    # This agent does not need tools; it only processes text.
    tools=[]
)
