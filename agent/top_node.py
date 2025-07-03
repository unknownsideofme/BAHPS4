from typing import Literal, Union, Generator, Dict, Any
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import time

# Load both existing agents
from agent import agent as tool_agent, run_agent
from workflow_agent import run_agent_with_streaming, StreamingEvent

# LLM for router
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ✳ Classification logic
def classify_query(query: str) -> Literal["tools", "reasoning"]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a router that classifies geospatial queries. Respond with only one word: tools or reasoning."),

        ("user", "Query: I want to download the NDVI for Karnataka."),
        ("assistant", "tools"),

        ("user", "Query: Explain how slope and rainfall influence flood risk."),
        ("assistant", "reasoning"),

        ("user", "Query: Perform site suitability analysis using NDVI and rainfall."),
        ("assistant", "tools"),

        ("user", "Query: What is the difference between SRTM and ASTER DEM?"),
        ("assistant", "reasoning"),

        ("user", "Query: Perform flood analysis of UP."),
        ("assistant", "tools"),

        ("user", "Query: {query}")
    ])
    
    messages = prompt.format_messages(query=query)
    response = llm.invoke(messages)
    return response.content.strip().lower()

# ✅ Streaming Router function
def router_agent_with_streaming(query: str, callback=None) -> str:
    """
    Router agent with streaming support for both tool and reasoning queries
    
    Args:
        query: User query
        callback: Function to handle streaming events
    
    Returns:
        Final response
    """
    route = classify_query(query)
    
    if route == "tools":
        print("[Router] Using Tool Agent")
        if callback:
            callback(StreamingEvent("status", "🛠 Using Tool Agent"))
        
        # Use the streaming-enabled tool agent
        return run_agent(query, callback)
    
    elif route == "reasoning":
        print("[Router] Using Reasoning Agent with streaming")
        if callback:
            callback(StreamingEvent("status", "🧠 Using Reasoning Agent"))
        
        result = run_agent_with_streaming(query, callback)
        return result["answer"]
    
    else:
        raise ValueError(f"Unknown route: {route}")

# ✅ Legacy Router function (for backward compatibility)
def router_agent(query: str) -> str:
    return router_agent_with_streaming(query)

# ✅ Generator version for Streamlit streaming
def router_agent_stream(query: str) -> Generator[Dict[str, Any], None, str]:
    """
    Generator version that yields streaming events
    
    Yields:
        Dict with keys: event_type, content, metadata
    
    Returns:
        Final response string
    """
    route = classify_query(query)
    
    if route == "tools":
        yield {"event_type": "status", "content": "🛠 Using Tool Agent", "metadata": {"route": "tools"}}
        
        final_result = ""
        
        def streaming_callback(event: StreamingEvent):
            nonlocal final_result
            event_dict = {
                "event_type": event.event_type,
                "content": event.content,
                "metadata": event.metadata
            }
            
            # Store final result
            if event.event_type == "result":
                final_result = event.content
        
        result = run_agent(query, streaming_callback)
        final_result = result
        
        yield {"event_type": "result", "content": final_result, "metadata": {"route": "tools"}}
        return final_result
    
    elif route == "reasoning":
        yield {"event_type": "status", "content": "🧠 Using Reasoning Agent", "metadata": {"route": "reasoning"}}
        
        final_result = ""
        
        def streaming_callback(event: StreamingEvent):
            nonlocal final_result
            event_dict = {
                "event_type": event.event_type,
                "content": event.content,
                "metadata": event.metadata
            }
            
            # Store final result
            if event.event_type == "result":
                final_result = event.content
        
        # Run the reasoning agent with streaming
        result = run_agent_with_streaming(query, streaming_callback)
        final_result = result["answer"]
        
        yield {"event_type": "result", "content": final_result, "metadata": {"route": "reasoning"}}
        return final_result
    
    else:
        error_msg = f"Unknown route: {route}"
        yield {"event_type": "error", "content": error_msg, "metadata": {"route": "unknown"}}
        raise ValueError(error_msg)

if __name__ == "__main__":
    # Test streaming functionality
    def test_callback(event):
        print(f"[{event.event_type.upper()}] {event.content}")
    
    query = input("Enter your geospatial query: ")
    answer = router_agent_with_streaming(query, test_callback)
    print("\n[FINAL ANSWER]\n", answer)