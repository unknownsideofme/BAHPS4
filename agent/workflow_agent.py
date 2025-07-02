from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.tools.retriever import create_retriever_tool
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import OllamaEmbeddings




embeddings = OllamaEmbeddings(
    model="llama3.2",
)

PINECONE_API_KEY = os.environ["PINECONE_KEY"]
PINECONE_INDEX_NAME = "documentations"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)



vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
      # optional if you're not using namespaces
)

retriever = vectorstore.as_retriever(search_type="similarity", k=8)
llm = ChatOpenAI(model="gpt-4o")


# ----------- STATE DEFINITION -----------
class GraphState(TypedDict):
    query: str
    decision: Literal["code", "qgis", "concept"]
    documents: Optional[str]
    answer: Optional[str]

# ----------- PLANNER NODE -----------
def planner_node(state: GraphState) -> GraphState:
    query = state["query"]
    prompt = f"""
You are a reasoning assistant for geospatial tasks.
Classify this query into one of:
- 'code' for Python or GEE scripting tasks,
- 'qgis' for GUI-based QGIS workflows,
- 'concept' for general geospatial questions.

Query: {query}
Respond with only one of: code, qgis, concept.
"""
    decision = llm.invoke(prompt).content.strip().lower()
    return {**state, "decision": decision}

# ----------- RETRIEVER NODE -----------
def retriever_node(state: GraphState) -> GraphState:
    docs = retriever.get_relevant_documents(state["query"])
    combined = "\n\n".join([d.page_content for d in docs])
    return {**state, "documents": combined}


# ----------- REASONING NODE (Deep Reasoning) -----------
def reasoning_node(state: GraphState) -> GraphState:
    query = state["query"]
    context = state.get("documents", "")[:4000]

    system_prompt = """
You are GeoGPT, a scientific geospatial reasoning agent assisting professional users such as ISRO scientists.

Use the provided documentation and scientific principles to explain the solution with deep technical clarity.
You MUST:
1. Identify the user's end goal and break it into scientifically justified sub-tasks.
2. For each sub-task, explain the rationale (why this step matters).
3. Provide detailed, exact Python (or other GIS tool) code — parameter by parameter — and explain each choice.
4. Discuss possible alternatives or optimizations where relevant (e.g., D∞ vs D8, fill sinks vs not, flow threshold calibration).
5. Always integrate multiple raster layers logically, describing how each influences the final outcome.
6. Clearly mention assumptions (e.g., spatial resolution, CRS, reclassification strategies).
7. If data is missing (e.g., no rainfall input), suggest how to acquire or simulate it.

Think step-by-step like a scientific assistant guiding a geospatial research workflow.

DO NOT summarize. Instead, build a full logical execution plan. Your audience includes senior researchers.
"""

    prompt = f"""{system_prompt}

--- USER QUERY ---
{query}

--- RELEVANT DOCS ---
{context}

Respond now with deep technical reasoning and a complete solution.
"""

    answer = llm.invoke(prompt).content.strip()
    return {**state, "answer": answer}


# ----------- LANGGRAPH FLOW -----------
graph = StateGraph(GraphState)

graph.add_node("planner", planner_node)
graph.add_node("retrieve", retriever_node)
graph.add_node("reason", reasoning_node)
graph.add_node("output", lambda state: state)

graph.set_entry_point("planner")
graph.add_edge("planner", "retrieve")
graph.add_edge("retrieve", "reason")
graph.add_edge("reason", "output")
graph.set_finish_point("output")

# ----------- COMPILE GRAPH -----------
chain = graph.compile()


# ----------- SAMPLE RUN -----------
# query = input("Enter your geospatial query: ")
# response = chain.invoke({"query": query})
# print(response["answer"])