import os
import json
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from langchain.agents.agent import AgentExecutor
from langchain.schema.agent import AgentAction, AgentFinish
from geetools import( ExtractBox , extract_bbox , GetData, get_srtm_dem, get_ndvi_data, 
                     get_rainfall, get_landcover, GenerateFloodRiskMapInput, generate_flood_risk_map, 
                     VisualizeFloodRiskInput, visualize_flood_risk_folium, OverlayMapInput, overlay_map)

# ------------------------- Tools -------------------------
Tools = [
    {
        "name": "extract_bbox",
        "description": "Extract bounding box and save GeoJSON for a given location",
        "parameters": ExtractBox
    },
    {
        "name": "get_srtm_dem",
        "description": "Fetch and download SRTM DEM clipped to bounding box",
        "parameters": GetData
    },
    {
        "name": "get_ndvi_data",
        "description": "Fetch and download NDVI data for the bounding box",
        "parameters": GetData
    },
    {
        "name": "get_rainfall",
        "description": "Fetch and download total annual rainfall raster",
        "parameters": GetData
    },
    {
        "name": "get_landcover",
        "description": "Fetch and download ESA WorldCover land cover classification",
        "parameters": GetData
    },
    {
        "name": "generate_flood_risk_map",
        "description": "Generate classified flood risk map from raster layers",
        "parameters": GenerateFloodRiskMapInput
    },
    {
        "name": "visualize_flood_risk_folium",
        "description": "Visualize flood risk GeoTIFF on interactive Folium map",
        "parameters": VisualizeFloodRiskInput
    }
]


# 🧭 System message to control agent behavior
system_prompt = """
You are a geospatial analysis assistant.

Your goal is to autonomously complete geospatial tasks using the available tools. Always use tools in the correct order to achieve the final result.

🔁 Reuse of previous outputs:
- When a tool returns a payload, bounding box, or file path, store it mentally.
- Reuse a previous output **only** if the next tool requires it (e.g., payload → process_request, filepath → clip_raster).
- Do not pass previous outputs unless they are clearly required by the next tool.
- Do not include unrelated previous tool outputs in arguments.

📁 Filenames:
- Do not ask the user for filenames.
- Always generate internal filenames like 'output.tif', 'ndvi.tif', 'flood_map.geojson'.

✅ Dataset types:
- When creating a Sentinel Hub payload, only use dataset types: 'dem', 'ndvi', 'ndwi', 'landcover', 'soil_saturation', 'aod'.
- Do not use unsupported types like 'S1_GRD'.

🧠 Behavior:
- Do not stop midway.
- Do not explain tool calls unless asked.
- Always continue the analysis pipeline until the task is fully complete.

**SPECIAL DIRECTION**
-Donot provide only the filepath
-While calling process_request, you must always provide the payload, filepath.
-Always assume that the last known location remains the context for follow-up queries unless user explicitly changes it.
-Never switch to a different geographic location unless clearly stated in the user query.
-Remember all the files and paths to the files created by the tools in the current session.
"""

from langchain.memory import ConversationBufferWindowMemory

# 🧠 Memory that keeps only last 3 exchanges to stay within token limits
memory = ConversationBufferWindowMemory(
    k=3,
    memory_key="chat_history",
    return_messages=True
)


# 🔮 LLM setup
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    streaming=True,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ⚙️ Agent with system prompt
agent = initialize_agent(
    tools=Tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs={"system_message": system_prompt},
    memory = memory
)

# 🧪 Run + Show internal steps
def debug_run(user_input: str):
    print(f"\n🔍 Prompt: {user_input}\n")
    output = agent.invoke({"input": user_input})

    # LangChain shows tool usage in verbose=True but doesn't separate thoughts
    # So we just wrap this around agent.invoke to track flow

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