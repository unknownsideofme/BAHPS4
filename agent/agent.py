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
    get_rainfall, get_landcover,VisualiseMap , visualise_map ,
    Analysis, suitability_analysis
    
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
        name="visualise_map",
        description="Visualize a raster map with optional overlays",
        func=visualise_map,
        args_schema=VisualiseMap
    ),
    StructuredTool.from_function(
        name="analysis_function",
        description="Perform analysis based on multiple criteria like flood risk , site suitability",
        func=suitability_analysis,
        args_schema=Analysis
    )
]

# 🧠 System prompt
system_prompt = """
You are a geospatial analysis assistant.

You help select raster layers for environmental suitability or flood risk analysis. Each raster has a `weight` (importance) and an `inverse` flag that indicates whether high values are good or bad for the target.

Use reasoning to decide `inverse`:
- Set `inverse = true` if higher raster values are bad for the target (e.g., high elevation may mean low flood risk).
- Set `inverse = false` if higher values are good (e.g., vegetation or NDVI — more is better).
- Think based on the analysis goal (e.g., flood risk vs site suitability).

Do not ask users to supply `.tif` paths — use previously returned results.
In the analysis function, filenames must include `.tif`.

**IMPORTANT**
✅ ALWAYS think step-by-step before using a tool.
✅ First, explain your plan in detail.
✅ Then choose the appropriate tool.

Example:

User: I want to analyze flood risk for Delhi.

Assistant:
Step 1: I will extract the bounding box for Delhi.
Step 2: I will get the DEM, because elevation affects flood risk.
Step 3: I will also fetch rainfall and landcover data.
Step 4: Then I will run the analysis with DEM (.tif), rainfall (.tif), and landcover (.tif), with appropriate weights and inverse values.

Let's begin.
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
# def debug_run(user_input: str):
#     print(f"\n🔍 Prompt: {user_input}\n")
#     output = agent.invoke({"input": user_input})
#     print("\n✅ Final Answer:\n", output["output"])

# # 🚀 CLI
# if __name__ == "__main__":
#     while True:
#         try:
#             user_input = input("\n🧑 User: ")
#             if user_input.lower() in {"exit", "quit"}:
#                 print("👋 Goodbye!")
#                 break
#             debug_run(user_input)
#         except Exception as e:
#             print("❌ Error:", e)
