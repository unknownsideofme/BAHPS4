from typing import TypedDict, Optional, Literal, Generator, Dict, Any
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os
import json
import re

# ----------- ENV SETUP -----------
load_dotenv()
PINECONE_API_KEY = os.environ["PINECONE_KEY"]
PINECONE_INDEX_NAME = "documentations"

# ----------- LLM & VECTOR STORE SETUP -----------
token_llm = ChatOpenAI(model="gpt-4o", streaming=True, temperature=0.1)
embeddings = OllamaEmbeddings(model="llama3.2")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
vstore = PineconeVectorStore(index=index, embedding=embeddings)
retriever = vstore.as_retriever(search_type="similarity", k=8)

# ----------- STATE DEFINITION -----------
class GraphState(TypedDict):
    query: str
    decision: Literal["code", "qgis", "concept"]
    documents: Optional[str]
    chain_of_thought: Optional[str]
    answer: Optional[str]
    streaming_callback: Optional[callable]

# ----------- STREAMING EVENT TYPES -----------
class StreamingEvent:
    def __init__(self, event_type: str, content: str, metadata: Dict[str, Any] = None):
        self.event_type = event_type  # 'thought', 'action', 'result', 'status'
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = None

# ----------- PLANNER NODE -----------
def planner_node(state: GraphState) -> GraphState:
    query = state["query"]
    callback = state.get("streaming_callback")
    
    if callback:
        callback(StreamingEvent("status", "🔍 Classifying query..."))
    
    prompt = f"""
You are a reasoning assistant for geospatial tasks.
Classify this query into one of:
- 'code' for Python or GEE scripting tasks,
- 'qgis' for GUI-based QGIS workflows,
- 'concept' for general geospatial questions.


Query: {query}
Respond with only one of: code, qgis, concept.
"""
    decision = token_llm.invoke(prompt).content.strip().lower()
    
    if callback:
        callback(StreamingEvent("status", f"✅ Query classified as: {decision}"))
    
    print(f"🔍 [Planner] Decision: {decision}")
    return {**state, "decision": decision}

# ----------- RETRIEVER NODE -----------
def retriever_node(state: GraphState) -> GraphState:
    callback = state.get("streaming_callback")
    
    if callback:
        callback(StreamingEvent("status", "📚 Retrieving relevant documents..."))
    
    docs = retriever.invoke(state["query"])
    combined = "\n\n".join([d.page_content for d in docs])
    
    if callback:
        callback(StreamingEvent("status", f"✅ Retrieved {len(docs)} documents"))
    
    print(f"📚 [Retriever] Retrieved {len(docs)} documents.")
    return {**state, "documents": combined}

# ----------- ENHANCED REASONING NODE WITH STRUCTURED STREAMING -----------
def reasoning_node(state: GraphState) -> GraphState:
    query = state["query"]
    context = state.get("documents", "")[:4000]
    callback = state.get("streaming_callback")
    
    if callback:
        callback(StreamingEvent("status", "🧠 Starting reasoning process..."))
    
    system_prompt = """
You are GeoGPT, a scientific geospatial reasoning agent assisting professional users such as ISRO scientists.

Structure your response with clear sections:
**Specifically very important instructions :** 
-  Provide a first 'structured workkflow first before providing any code etc
-  After deciding the workflow then according to the workflow provide the code or qgis steps
- Always ensure to provide the source of the various raster files or the raw data the user can download it from
- Also provide the reasoning or chain of thoughts you indergo after fetching and retrieving the documentation from the rerievers
- Do not halliculate or make up any data, always provide the source of the data you are using

**THOUGHT:** Your step-by-step reasoning and analysis
- Break down the problem
- Consider the geospatial concepts involved
- Plan your approach

**ACTION:** The concrete solution or code implementation
- Provide exact Python/GEE code if applicable
- Include specific parameters and methods
- Explain implementation details

Use the provided documentation and scientific principles to explain with deep technical clarity.
"""
    
    full_prompt = f"{system_prompt}\n--- USER QUERY ---\n{query}\n\n--- CONTEXT ---\n{context}"

    print("🧠 [Reasoning Agent] Beginning structured CoT streaming:")
    
    full_response = ""
    current_section = None
    section_buffer = ""
    
    # Stream token-by-token with section detection
    for chunk in token_llm.stream(full_prompt):
        token = chunk.content
        full_response += token
        section_buffer += token
        
        # Detect section headers
        if "**THOUGHT:**" in section_buffer:
            current_section = "thought"
            section_buffer = section_buffer.replace("**THOUGHT:**", "").strip()
            if callback:
                callback(StreamingEvent("status", "💭 Generating thoughts..."))
        elif "**ACTION:**" in section_buffer:
            # Flush any remaining thought content
            if current_section == "thought" and section_buffer.strip():
                if callback:
                    callback(StreamingEvent("thought", section_buffer.strip()))
            current_section = "action"
            section_buffer = section_buffer.replace("**ACTION:**", "").strip()
            if callback:
                callback(StreamingEvent("status", "⚡ Generating actions..."))
        
        # Stream content based on current section
        if current_section and token and callback:
            if current_section == "thought":
                callback(StreamingEvent("thought", token))
            elif current_section == "action":
                callback(StreamingEvent("action", token))
        
        print(token, end="", flush=True)
    
    # Flush any remaining content
    if section_buffer.strip() and callback:
        if current_section == "thought":
            callback(StreamingEvent("thought", section_buffer.strip()))
        elif current_section == "action":
            callback(StreamingEvent("action", section_buffer.strip()))
    
    if callback:
        callback(StreamingEvent("status", "✅ Reasoning complete"))
    
    print("\n✅ [Reasoning Agent] Completed structured response.")
    
    return {**state, "chain_of_thought": full_response, "answer": full_response}

# ----------- BUILD GRAPH -----------
graph = StateGraph(GraphState)
graph.add_node("planner", planner_node)
graph.add_node("retrieve", retriever_node)
graph.add_node("reason", reasoning_node)
graph.add_node("output", lambda s: s)

graph.set_entry_point("planner")
graph.add_edge("planner", "retrieve")
graph.add_edge("retrieve", "reason")
graph.add_edge("reason", "output")
graph.set_finish_point("output")

chain = graph.compile()

# ----------- STREAMING INTERFACE -----------
def run_agent_with_streaming(query: str, callback: callable = None):
    """
    Run the agent with streaming callback support
    
    Args:
        query: The user query
        callback: Function to handle streaming events
    
    Returns:
        Final state of the agent
    """
    print(f"\n🚀 Running GeoGPT Agent with streaming...\nQuery: {query}\n")
    
    initial_state = {
        "query": query,
        "streaming_callback": callback
    }
    
    final_state = None
    for step in chain.stream(initial_state):
        final_state = step.get("output")
        if final_state:
            break
    
    return final_state

def run_agent(query: str):
    """Legacy interface for backward compatibility"""
    final_state = run_agent_with_streaming(query)
    return final_state

if __name__ == "__main__":
    def print_callback(event: StreamingEvent):
        print(f"[{event.event_type.upper()}] {event.content}")
    
    result = run_agent_with_streaming("Perform flood risk analysis using SRTM and NDVI", print_callback)
    print("\n--- RESULT SUMMARY ---")
    print("Decision:", result["decision"])
    print("Answer snippet:", result["answer"][:200])