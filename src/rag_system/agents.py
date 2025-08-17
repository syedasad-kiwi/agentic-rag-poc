from crewai import Agent, LLM
from .tools import document_retrieval_tool

# Initialize the Ollama LLM for the agents
ollama_llm = LLM(model="ollama/gemma3:1b", base_url="http://localhost:11434")

# Define the Document Researcher Agent
document_researcher = Agent(
    role='Document Researcher and Analyst',
    goal='Retrieve and analyze information from policy and standards documents to answer user questions.',
    backstory=(
        "You are a skilled document analyst who retrieves information from knowledge bases "
        "and provides comprehensive answers. Your process is: 1) Use the document retrieval "
        "tool to find relevant content, 2) Analyze the retrieved content to answer the user's "
        "question, 3) Provide a clear, comprehensive answer based solely on the retrieved "
        "information. CRITICAL: You must base your answers STRICTLY on the document content "
        "retrieved by the tool. If the tool returns 'No relevant documents found' or no useful "
        "content, you MUST respond with 'Based on the available documents, I cannot find "
        "relevant information to answer this question.' Do NOT use general knowledge. "
        "IMPORTANT: When providing your final answer, always include the source file information "
        "from the retrieved documents. At the end of your response, add a 'Sources:' section "
        "listing the source files mentioned in the retrieved content."
    ),
    tools=[document_retrieval_tool],
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False
)

# Define the Insight Synthesizer Agent
insight_synthesizer = Agent(
    role='Insight Synthesizer',
    goal='Analyze the retrieved document context and provide a clear, concise, and accurate answer to the user\'s question.',
    backstory=(
        "You are a skilled analyst with a talent for synthesizing information. "
        "You take raw text retrieved from documents and transform it into a "
        "comprehensive and easy-to-understand answer. CRITICAL: You must base your answers "
        "STRICTLY on the provided context from the Document Researcher. "
        "IMPORTANT: If you receive ANY document content, summaries, or analysis in the context "
        "(including organized breakdowns, key sections, takeaways, or document descriptions), "
        "you MUST use that information to provide a comprehensive answer. "
        "Only respond with 'Based on the available documents, I cannot provide an answer to this question as no relevant "
        "information was found' if the context explicitly states 'No relevant documents found' or similar. "
        "Do NOT ignore any document content, summaries, or analysis provided in the context."
    ),
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False
)
