import os
import json
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferWindowMemory

# ✅ Your geetools functions and Pydantic models
from geetools import (
    ExtractBox, extract_bbox, GetData, get_srtm_dem, get_ndvi_data,
    get_rainfall, get_landcover, GenerateFloodRiskMapInput, generate_flood_risk_map,
    VisualizeFloodRiskInput, visualize_flood_risk_folium
)

# 🌍 Load environment variables
load_dotenv()

# ✅ Wrap each function using StructuredTool
tools = [
    StructuredTool.from_function(
        name="extract_bbox",
        description="Extract bounding box and save GeoJSON for a given location",
        func=extract_bbox,
        args_schema=ExtractBox
    ),
    StructuredTool.from_function(
        name="get_srtm_dem",
        description="Fetch and download SRTM DEM clipped to bounding box",
        func=get_srtm_dem,
        args_schema=GetData
    ),
    StructuredTool.from_function(
        name="get_ndvi_data",
        description="Fetch and download NDVI data for the bounding box",
        func=get_ndvi_data,
        args_schema=GetData
    ),
    StructuredTool.from_function(
        name="get_rainfall",
        description="Fetch and download total annual rainfall raster",
        func=get_rainfall,
        args_schema=GetData
    ),
    StructuredTool.from_function(
        name="get_landcover",
        description="Fetch and download ESA WorldCover land cover classification",
        func=get_landcover,
        args_schema=GetData
    ),
    StructuredTool.from_function(
        name="generate_flood_risk_map",
        description="Generate classified flood risk map from raster layers",
        func=generate_flood_risk_map,
        args_schema=GenerateFloodRiskMapInput
    ),
    StructuredTool.from_function(
        name="visualize_flood_risk_folium",
        description="Visualize flood risk GeoTIFF on interactive Folium map",
        func=visualize_flood_risk_folium,
        args_schema=VisualizeFloodRiskInput
    )
]

# 🧠 System prompt
system_prompt = """
You are a geospatial analysis assistant.
-Do not pass .tif inside the filename as an argument only provide the name
...
"""

# 🧠 Memory
memory = ConversationBufferWindowMemory(
    k=3,
    memory_key="chat_history",
    return_messages=True
)

# 🔮 LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    streaming=True,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ⚙️ Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs={"system_message": system_prompt},
    memory=memory
)

# 🧪 Debugging run
def debug_run(user_input: str):
    print(f"\n🔍 Prompt: {user_input}\n")
    output = agent.invoke({"input": user_input})
    print("\n✅ Final Answer:\n", output["output"])

# 🚀 CLI
if __name__ == "__main__":
    while True:
        try:
            user_input = input("\n🧑 User: ")
            if user_input.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break
            debug_run(user_input)
        except Exception as e:
            print("❌ Error:", e)
