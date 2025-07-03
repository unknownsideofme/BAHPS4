import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, Any, List, Optional
import re

# Import StreamingEvent if available
try:
    from workflow_agent import StreamingEvent
except ImportError:
    # Fallback StreamingEvent definition
    class StreamingEvent:
        def __init__(self, event_type: str, content: str, metadata: Dict = None):
            self.event_type = event_type
            self.content = content
            self.metadata = metadata or {}

# ✅ Your geetools functions and Pydantic models
from geetools import (
    ExtractBox, extract_bbox, GetData, get_srtm_dem, get_ndvi_data,
    get_rainfall, get_landcover, VisualiseMap, visualise_map,
    Analysis, suitability_analysis
)

# 🌍 Load environment variables
load_dotenv()

# Enhanced callback handler for detailed streaming
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, callback=None):
        self.callback = callback
        self.current_step = 0
        self.total_steps = 0
        self.step_history = []
        self.reasoning_buffer = ""
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """Called when LLM starts - capture reasoning"""
        if self.callback:
            self.callback(StreamingEvent(
                "thought",
                "🤔 Analyzing your request and planning the approach...",
                {"step": "planning", "prompts": prompts}
            ))
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM ends - capture the reasoning/planning"""
        if self.callback and response.generations:
            llm_output = response.generations[0][0].text
            
            # Extract reasoning from the LLM output
            if "Thought:" in llm_output or "I need to" in llm_output:
                reasoning = self._extract_reasoning(llm_output)
                if reasoning:
                    self.callback(StreamingEvent(
                        "thought",
                        f"💭 **Agent Reasoning:**\n{reasoning}",
                        {"step": "reasoning", "full_output": llm_output}
                    ))
    
    def on_agent_action(self, action, **kwargs):
        """Called when agent decides to take an action"""
        self.current_step += 1
        
        if self.callback:
            # Format the step information
            step_info = {
                "step_number": self.current_step,
                "tool_name": action.tool,
                "tool_input": action.tool_input,
                "log": action.log
            }
            
            self.step_history.append(step_info)
            
            # Extract reasoning from the action log
            reasoning = self._extract_reasoning_from_log(action.log)
            
            # Send detailed step information
            step_content = self._format_step_content(step_info, reasoning)
            
            self.callback(StreamingEvent(
                "action",
                step_content,
                step_info
            ))
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Called when tool starts execution"""
        if self.callback:
            tool_name = serialized.get("name", "Unknown")
            
            # Parse input if it's JSON
            try:
                if input_str.startswith('{') and input_str.endswith('}'):
                    input_dict = json.loads(input_str)
                    formatted_input = json.dumps(input_dict, indent=2)
                else:
                    formatted_input = input_str
            except:
                formatted_input = input_str
            
            self.callback(StreamingEvent(
                "action",
                f"⚙️ **Executing Tool: {tool_name}**\n\n```json\n{formatted_input}\n```\n\n🔄 Processing...",
                {
                    "tool": tool_name, 
                    "input": input_str,
                    "status": "starting",
                    "step": self.current_step
                }
            ))
    
    def on_tool_end(self, output: str, **kwargs):
        """Called when tool completes execution"""
        if self.callback:
            # Parse and format the output
            formatted_output = self._format_tool_output(output)
            
            self.callback(StreamingEvent(
                "action",
                f"✅ **Tool Execution Complete**\n\n{formatted_output}",
                {
                    "output": output,
                    "status": "completed",
                    "step": self.current_step
                }
            ))
    
    def on_tool_error(self, error: Exception, **kwargs):
        """Called when tool encounters an error"""
        if self.callback:
            self.callback(StreamingEvent(
                "action",
                f"❌ **Tool Error**\n\nError: {str(error)}\n\n🔄 Attempting to continue...",
                {
                    "error": str(error),
                    "status": "error",
                    "step": self.current_step
                }
            ))
    
    def on_agent_finish(self, finish, **kwargs):
        """Called when agent completes the task"""
        if self.callback:
            # Send summary of all steps
            summary = self._create_execution_summary()
            
            self.callback(StreamingEvent(
                "thought",
                f"🎯 **Analysis Complete!**\n\n{summary}",
                {
                    "final_output": finish.return_values,
                    "total_steps": self.current_step,
                    "step_history": self.step_history
                }
            ))
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Called when agent chain starts"""
        if self.callback:
            query = inputs.get('input', inputs.get('query', 'Unknown query'))
            
            self.callback(StreamingEvent(
                "thought",
                f"🚀 **Starting Geospatial Analysis**\n\n**Query:** {query}\n\n🔍 Initializing workflow...",
                {
                    "inputs": inputs,
                    "chain": serialized.get("name", "unknown")
                }
            ))
    
    def on_chain_error(self, error: Exception, **kwargs):
        """Called when chain encounters an error"""
        if self.callback:
            self.callback(StreamingEvent(
                "thought",
                f"❌ **Error in Analysis Chain**\n\nError: {str(error)}\n\n🔄 Attempting recovery...",
                {"error": str(error)}
            ))
    
    def _extract_reasoning(self, text: str) -> str:
        """Extract reasoning from LLM output"""
        # Look for common reasoning patterns
        patterns = [
            r"Thought:\s*(.*?)(?=Action:|$)",
            r"I need to\s*(.*?)(?=\n|$)",
            r"Let me\s*(.*?)(?=\n|$)",
            r"First,\s*(.*?)(?=\n|$)",
            r"To\s*(.*?)(?=\n|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning = match.group(1).strip()
                if len(reasoning) > 10:  # Only return substantial reasoning
                    return reasoning
        
        return ""
    
    def _extract_reasoning_from_log(self, log: str) -> str:
        """Extract reasoning from action log"""
        if not log:
            return ""
        
        # Clean up the log and extract meaningful reasoning
        lines = log.split('\n')
        reasoning_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Action:') and not line.startswith('Action Input:'):
                if any(word in line.lower() for word in ['need', 'should', 'will', 'because', 'since', 'to', 'for']):
                    reasoning_lines.append(line)
        
        return '\n'.join(reasoning_lines) if reasoning_lines else ""
    
    def _format_step_content(self, step_info: Dict, reasoning: str) -> str:
        """Format step content with reasoning"""
        content = f"📍 **Step {step_info['step_number']}: {step_info['tool_name']}**\n\n"
        
        if reasoning:
            content += f"💭 **Why this step:**\n{reasoning}\n\n"
        
        content += f"🔧 **Tool:** {step_info['tool_name']}\n"
        
        # Format input nicely
        if isinstance(step_info['tool_input'], dict):
            content += f"📝 **Parameters:**\n```json\n{json.dumps(step_info['tool_input'], indent=2)}\n```"
        else:
            content += f"📝 **Input:** {step_info['tool_input']}"
        
        return content
    
    def _format_tool_output(self, output: str) -> str:
        """Format tool output for better readability"""
        if not output:
            return "No output received"
        
        # Try to parse as JSON first
        try:
            if output.startswith('{') and output.endswith('}'):
                parsed = json.loads(output)
                if 'filepath' in parsed:
                    filename = os.path.basename(parsed['filepath'])
                    return f"📁 **Generated File:** `{filename}`\n📂 **Path:** `{parsed['filepath']}`"
                else:
                    return f"```json\n{json.dumps(parsed, indent=2)}\n```"
        except:
            pass
        
        # Look for common patterns
        if 'saved' in output.lower() or 'generated' in output.lower():
            return f"✅ **Success:** {output}"
        elif 'error' in output.lower():
            return f"❌ **Error:** {output}"
        else:
            # Truncate very long outputs
            if len(output) > 200:
                return f"```\n{output[:200]}...\n```"
            else:
                return f"```\n{output}\n```"
    
    def _create_execution_summary(self) -> str:
        """Create a summary of all execution steps"""
        if not self.step_history:
            return "No steps executed."
        
        summary = f"**Execution Summary ({len(self.step_history)} steps):**\n\n"
        
        for i, step in enumerate(self.step_history, 1):
            summary += f"{i}. **{step['tool_name']}** - Completed successfully\n"
        
        return summary

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
        description="Perform analysis based on multiple criteria like flood risk, site suitability",
        func=suitability_analysis,
        args_schema=Analysis
    )
]

# 🧠 Enhanced system prompt with explicit step-by-step instructions
system_prompt = """
You are a geospatial analysis assistant specialized in environmental analysis and data processing.

Your role is to help users with:
1. **Data Collection**: Extract bounding boxes, download DEM, NDVI, rainfall, and landcover data
2. **Analysis**: Perform flood risk analysis, site suitability analysis, and other geospatial assessments
3. **Visualization**: Create interactive maps and visualizations of results

**IMPORTANT: Always explain your reasoning before taking each action.**

**Analysis Guidelines:**
- For **flood risk analysis**:
  - DEM (inverse=true, higher elevation = lower risk)
  - Rainfall (inverse=false, more rain = higher risk)
  - Landcover (inverse=true, urban/impermeable = higher risk)
- For **site suitability**:
  - NDVI (inverse=false, more vegetation = better)
  - DEM and rainfall depend on the context and should be reasoned based on the goal
- Always assign appropriate weights:
  - Primary factors (0.4–0.5)
  - Secondary factors (0.2–0.3)
  - Minor factors (0.1–0.2)
- Always visualize the final results using the `visualise_map()` function

**Step-by-step process with reasoning:**

1. **Planning Phase**
   - Clearly state what you understand from the user's request
   - Identify what data you need and why
   - Outline the sequence of steps you'll take

2. **Extract Bounding Box**
   - Explain why you're choosing this specific location/region
   - Justify the bounding box size relative to the analysis objective
   - Show coordinates and explain their significance

3. **Download Required Data**
   - For each dataset, explain:
     - Why this specific data is needed for the analysis
     - How it contributes to the final goal
     - What the data represents and its source
   - Mention the file paths and what they contain

4. **Perform Analysis**
   - Before running analysis, explain:
     - The methodology you're using
     - Why you chose specific weights for each factor
     - Whether factors are inverted and why
     - Expected outcomes
   - Show your reasoning in a clear table format

5. **Visualize Results**
   - Explain the visualization approach
   - Justify color scheme choices
   - Describe what high/low values mean in context
   - Identify key findings from the visualization

6. **Provide Interpretation**
   - Explain what the results mean practically
   - Highlight areas of interest or concern
   - Suggest next steps or recommendations

**Critical Instructions:**
- **Always explain your reasoning before each action**
- **Use clear, numbered steps**
- **Justify every decision you make**
- **Show your thought process explicitly**
- **Provide context for technical choices**
- **Use meaningful, descriptive filenames**
- **Complete the full workflow including visualization**

Remember: Users want to understand not just what you're doing, but WHY you're doing it. Be educational and transparent in your approach.
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

# Create the base agent
base_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs={"system_message": system_prompt},
    memory=memory,
    max_iterations=10,  # Ensure we don't get stuck in loops
    early_stopping_method="generate"  # Generate final response if max iterations reached
)

# Enhanced agent wrapper class
class StreamingAgent:
    def __init__(self, base_agent):
        self.base_agent = base_agent
    
    def run(self, query: str, callback=None):
        """Run the agent with optional streaming callback"""
        # Create callback handler
        callback_handler = StreamingCallbackHandler(callback)
        
        # Add callback to the agent's callbacks
        if callback:
            callback(StreamingEvent(
                "status",
                "🚀 Starting geospatial analysis...",
                {"query": query}
            ))
        
        try:
            # Run the agent with callback
            result = self.base_agent.run(
                query, 
                callbacks=[callback_handler] if callback else None
            )
            
            if callback:
                callback(StreamingEvent(
                    "status",
                    "🎉 Analysis completed successfully!",
                    {"result": result}
                ))
            
            return result
            
        except Exception as e:
            if callback:
                callback(StreamingEvent(
                    "status",
                    f"❌ Error occurred: {str(e)}",
                    {"error": str(e)}
                ))
            raise e

# Create the streaming agent instance
agent = StreamingAgent(base_agent)

# Backward compatibility - expose the run method directly
def run_agent(query: str, callback=None):
    """Run the agent with optional streaming callback"""
    return agent.run(query, callback)

# For backward compatibility with existing code
if not hasattr(agent, 'run'):
    agent.run = lambda query: run_agent(query)