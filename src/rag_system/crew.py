from crewai import Crew, Process, Task
from .agents import document_researcher, insight_synthesizer

def create_rag_crew(query: str):
    """
    Creates and configures a two-agent RAG crew to process a query.
    - The Document Researcher finds relevant information.
    - The Insight Synthesizer formulates the final answer based on the retrieved context.
    """

    # Task for the Document Researcher agent
    # This task focuses exclusively on using the tool to find information.
    research_task = Task(
        description=f"Find relevant information in the policy and standards documents for the query: '{query}'.",
        expected_output="A block of text containing chunks of the most relevant document sections and their source file names.",
        agent=document_researcher
    )

    # Task for the Insight Synthesizer agent
    # This task takes the context from the first task and focuses on crafting the answer.
    synthesis_task = Task(
        description=f"Analyze the provided document context from {research_task} and formulate a comprehensive and accurate answer to the user's original question: '{query}'.",
        expected_output="A clear, concise, and complete answer based solely on the provided context.",
        agent=insight_synthesizer,
        context=[research_task] # This ensures it uses the output from the research_task
    )

    # Create the crew with a sequential process
    rag_crew = Crew(
        agents=[document_researcher, insight_synthesizer],
        tasks=[research_task, synthesis_task],
        process=Process.sequential, # The tasks will be executed one after the other
        verbose=True
    )

    return rag_crew
