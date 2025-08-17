from crewai import Crew, Process, Task
from .agents import document_researcher, insight_synthesizer

def create_rag_crew(query: str):
    """Creates and configures the RAG crew to process a query."""

    # Define a single comprehensive task that does both retrieval and analysis
    comprehensive_task = Task(
        description=f"Answer the question: '{query}' by following these steps: 1) Use the document retrieval tool to search for relevant information in the policy and standards documents. 2) Based on the retrieved content, provide a comprehensive answer to the question. 3) IMPORTANT: Include source file names from the retrieved documents in your final answer. Add a 'Sources:' section at the end listing all source files used. 4) If no relevant information is found, clearly state this. Use ONLY information from the retrieved documents - do not use general knowledge.",
        expected_output="A comprehensive answer to the user's question based solely on the retrieved document content, with a 'Sources:' section listing source file names, or a clear statement that no relevant information was found.",
        agent=document_researcher
    )

    # Use hierarchical process as it handles context passing correctly
    # Use sequential process with a single task
    rag_crew = Crew(
        agents=[document_researcher],
        tasks=[comprehensive_task],
        process=Process.sequential,
        verbose=True
    )

    return rag_crew
