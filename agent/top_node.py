from typing import Literal, Union
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI  # ✅ Updated import

# Load both existing agents
from agent import agent as tool_agent
from workflow_agent import chain as reasoning_graph_chain

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


# ✅ Router function
def router_agent(query: str) -> str:
    route = classify_query(query)
    if route == "tools":
        print("[Router] Using Tool Agent")
        return tool_agent.run(query)  # Chat-style agent
    elif route == "reasoning":
        print("[Router] Using Reasoning Agent")
        result = reasoning_graph_chain.invoke({"query": query})
        return result["answer"]
    else:
        raise ValueError(f"Unknown route: {route}")

# ✅ Final: Run
# if _name_ == "_main_":
#     query = input("Enter your geospatial query: ")
#     answer = router_agent(query)
#     print("\n[ANSWER]\n", answer)