#!/usr/bin/env python3

# Test script to understand CrewAI tool creation in current version

from crewai import Agent, Task, Crew

def document_retrieval_tool(query: str) -> str:
    """
    Retrieves relevant context from a collection of policy and standards documents.
    
    Args:
        query: The search query
        
    Returns:
        str: Retrieved context
    """
    return f"Mock retrieval result for query: {query}"

# Test if function works as tool
try:
    agent = Agent(
        role='Document Researcher',
        goal='Find relevant information',
        backstory='I help find information from documents',
        tools=[document_retrieval_tool],
        verbose=True
    )
    print("✅ Function-based tool creation successful!")
except Exception as e:
    print(f"❌ Error: {e}")

# Test imports to see what's available
try:
    from crewai.tools.base_tool import BaseTool
    print("✅ BaseTool import successful!")
except ImportError as e:
    print(f"❌ BaseTool import failed: {e}")
    
try:
    from crewai_tools import BaseTool
    print("✅ crewai_tools BaseTool import successful!")
except ImportError as e:
    print(f"❌ crewai_tools BaseTool import failed: {e}")
