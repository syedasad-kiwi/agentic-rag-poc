import sys
from src.rag_system.crew import create_rag_crew

def main():
    """
    Main function to run the Agentic RAG pipeline.
    Accepts a query from the command line or prompts the user.
    """
    # Check if a query was passed as a command-line argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Prompt the user for a query if none was provided
        query = input("Please enter your question about the documents: ")

    if not query:
        print("No query provided. Exiting.")
        return

    print(f"\n🚀 Kicking off the RAG Crew for your query: '{query}'")
    print("--------------------------------------------------")

    # Create and run the RAG crew
    rag_crew = create_rag_crew(query)
    result = rag_crew.kickoff()

    print("\n--------------------------------------------------")
    print("✅ Crew execution finished. Here is the final answer:")
    print(result)

if __name__ == "__main__":
    main()
