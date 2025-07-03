import streamlit as st
import sys
import os
from typing import Dict, Any
import time
import json
from datetime import datetime
import re
from threading import Thread
import queue

# Add the agent directory to Python path
agent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent')
if agent_path not in sys.path:
    sys.path.append(agent_path)

# Import the router agent with streaming support
try:
    from top_node import router_agent_with_streaming, classify_query, StreamingEvent
    from workflow_agent import StreamingEvent
except ImportError as e:
    st.error(f"Failed to import router agent: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="GeoGPT - Geospatial Analysis Assistant",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with fixed color schemes
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Main styling */
    .main-header {
        text-align: center;
        padding: 2.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header h1 {
        margin: 0;
        font-weight: 700;
        font-size: 2.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 400;
    }

    .query-container {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        color: #e2e8f0;
        padding: 2rem;
        border-radius: 16px;
        border-left: 6px solid #667eea;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        font-family: 'Inter', sans-serif;
    }
    
    .query-container strong {
        color: #f7fafc;
        font-weight: 600;
    }

    .result-container {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        color: #e2e8f0;
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid #4a5568;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        font-family: 'Inter', sans-serif;
    }

    /* Collapsible thoughts section - Darker Red */
    .thoughts-section {
        background: linear-gradient(135deg, #2d1b1b 0%, #3c2626 100%);
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        overflow: hidden;
    }
    
    .thoughts-header {
        background: linear-gradient(135deg, #c53030 0%, #9b2c2c 100%);
        color: white;
        padding: 1rem 1.5rem;
        cursor: pointer;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .thoughts-header:hover {
        background: linear-gradient(135deg, #9b2c2c 0%, #822727 100%);
    }
    
    .thoughts-content {
        padding: 1.5rem;
        color: #e2e8f0;
        background: #2d1b1b;
        font-family: 'Inter', sans-serif;
        line-height: 1.7;
        border-top: 1px solid #4a5568;
    }
    
    .collapse-icon {
        transition: transform 0.3s ease;
        font-size: 1.2rem;
    }
    
    .collapse-icon.collapsed {
        transform: rotate(-90deg);
    }

    /* Action container - Darker Green */
    .action-container {
        background: linear-gradient(135deg, #1a2e1a 0%, #2d4a2d 100%);
        color: #e2e8f0;
        padding: 2rem;
        border-radius: 12px;
        border-left: 6px solid #38a169;
        margin: 1rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        line-height: 1.6;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }

    .tool-output {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        color: #e2e8f0;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #4a5568;
        margin: 1rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        overflow-x: auto;
    }

    .streaming-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        animation: pulse 1.5s infinite;
        margin-right: 10px;
    }

    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.2); opacity: 0.6; }
        100% { transform: scale(1); opacity: 1; }
    }

    .agent-badge {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-family: 'Inter', sans-serif;
    }

    .tool-agent {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }

    .reasoning-agent {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        box-shadow: 0 4px 16px rgba(240, 147, 251, 0.3);
    }

    .status-success {
        color: #68d391;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }

    .status-error {
        color: #fc8181;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }

    .status-processing {
        color: #f6e05e;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }

    .section-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-family: 'Inter', sans-serif;
    }

    .code-block {
        background: #0d1117;
        color: #f0f6fc;
        padding: 1.5rem;
        border-radius: 12px;
        overflow-x: auto;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        line-height: 1.5;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        border: 1px solid #30363d;
    }

    .workflow-step {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: #e2e8f0;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #3182ce;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }

    .data-source {
        background: linear-gradient(135deg, #2d1b33 0%, #44337a 100%);
        color: #e2e8f0;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #9f7aea;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }

    .chat-timestamp {
        color: #a0aec0;
        font-size: 0.85rem;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #4a5568;
        font-family: 'Inter', sans-serif;
    }

    .metric-container {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        color: #e2e8f0;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
        border: 1px solid #4a5568;
    }
    
    .metric-container h4 {
        color: #f7fafc;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .metric-container small {
        color: #a0aec0;
        line-height: 1.6;
    }

    /* Streaming section styling */
    .streaming-section {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        color: #e2e8f0;
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        border: 1px solid #4a5568;
    }

    .streaming-container {
        margin: 1rem 0;
    }

    /* Fix Streamlit's default white backgrounds */
    .stApp {
        background-color: #0f172a;
    }
    
    .main .block-container {
        background-color: #0f172a;
    }
    
    /* Fix sidebar background */
    .css-1d391kg {
        background-color: #1e293b;
    }
    
    /* Fix text areas and inputs */
    .stTextArea textarea {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #4a5568 !important;
    }
    
    .stTextInput input {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #4a5568 !important;
    }
    
    /* Fix expander backgrounds */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #0f172a !important;
        border: 1px solid #4a5568 !important;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .query-container,
        .result-container,
        .action-container {
            padding: 1.5rem;
        }
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #1e293b;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: #4a5568;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #718096;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'current_thoughts' not in st.session_state:
    st.session_state.current_thoughts = ""
if 'current_actions' not in st.session_state:
    st.session_state.current_actions = ""
if 'thoughts_expanded' not in st.session_state:
    st.session_state.thoughts_expanded = True

# Helper functions for better formatting
def format_tool_output(text):
    """Format tool output with better structure"""
    if not text:
        return ""
    
    # Clean up the text
    text = text.strip()
    
    # Extract tool usage patterns
    tool_pattern = r'🔧 Using tool: (\w+).*?📝 Input: ({.*?})'
    tool_matches = re.findall(tool_pattern, text, re.DOTALL)
    
    formatted_output = ""
    
    for tool_name, tool_input in tool_matches:
        try:
            input_dict = eval(tool_input)
            formatted_output += f"""
**🔧 Tool: {tool_name}**
```json
{json.dumps(input_dict, indent=2)}
```
"""
        except:
            formatted_output += f"""
**🔧 Tool: {tool_name}**
Input: {tool_input}
"""
    
    # Extract success messages
    success_pattern = r'✅ Tool completed.*?📁 Generated: (.*?)(?=🔧|\n|$)'
    success_matches = re.findall(success_pattern, text)
    
    if success_matches:
        formatted_output += "\n**📁 Generated Files:**\n"
        for file in success_matches:
            formatted_output += f"- {file.strip()}\n"
    
    return formatted_output

def format_reasoning_output(text):
    """Format reasoning output with better structure"""
    if not text:
        return ""
    
    # Split by sections
    sections = re.split(r'\*\*(THOUGHT|ACTION):\*\*', text)
    
    formatted_output = ""
    current_section = None
    
    for i, section in enumerate(sections):
        if section.strip() in ['THOUGHT', 'ACTION']:
            current_section = section.strip()
            continue
        
        if current_section and section.strip():
            if current_section == 'THOUGHT':
                formatted_output += f"""
<div class="section-badge">💭 REASONING</div>
<div class="thoughts-content">
{section.strip()}
</div>
"""
            elif current_section == 'ACTION':
                # Check if it contains code
                if '```' in section or 'import ' in section or 'def ' in section:
                    code_content = section.strip()
                    formatted_output += f"""
<div class="section-badge">⚡ IMPLEMENTATION</div>
<div class="action-container">
<pre class="code-block">{code_content}</pre>
</div>
"""
                else:
                    formatted_output += f"""
<div class="section-badge">⚡ ACTIONS</div>
<div class="action-container">
{section.strip()}
</div>
"""
    
    return formatted_output

def parse_structured_content(text):
    """Parse and structure the content for better display"""
    if not text:
        return {"type": "plain", "content": text}
    
    # Check for workflow patterns
    if "workflow" in text.lower() or "step" in text.lower():
        return {"type": "workflow", "content": text}
    
    # Check for code patterns
    if any(keyword in text.lower() for keyword in ["import", "def ", "class ", "```", "python"]):
        return {"type": "code", "content": text}
    
    # Check for data source patterns
    if any(keyword in text.lower() for keyword in ["download", "data", "source", "dataset"]):
        return {"type": "data", "content": text}
    
    return {"type": "plain", "content": text}

# JavaScript for collapsible thoughts
st.markdown("""
<script>
function toggleThoughts(id) {
    const content = document.getElementById('thoughts-content-' + id);
    const icon = document.getElementById('thoughts-icon-' + id);
    
    if (content.style.display === 'none') {
        content.style.display = 'block';
        icon.className = 'collapse-icon';
        icon.innerHTML = '▼';
    } else {
        content.style.display = 'none';
        icon.className = 'collapse-icon collapsed';
        icon.innerHTML = '▶';
    }
}
</script>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🌍 GeoGPT - Geospatial Analysis Assistant</h1>
    <p>Advanced AI-powered geospatial analysis with real-time reasoning and tool execution</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🛠 System Overview")
    
    # Agent information
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h4>🔧 Tool Agent</h4>
            <small>
            • Data extraction<br>
            • SRTM DEM, NDVI<br>
            • Rainfall data<br>
            • Analysis & visualization
            </small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h4>🧠 Reasoning Agent</h4>
            <small>
            • Conceptual explanations<br>
            • Technical guidance<br>
            • Scientific reasoning<br>
            • Real-time CoT streaming
            </small>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Quick examples
    st.subheader("📋 Quick Examples")
    
    tool_examples = [
        "Download NDVI data for Karnataka",
        "Flood risk analysis for Delhi",
        "Site suitability analysis for solar farms",
        "Extract DEM for Himachal Pradesh"
    ]
    
    reasoning_examples = [
        "Explain flood risk methodology",
        "Compare SRTM vs ASTER DEM",
        "NDVI and vegetation health relationship",
        "Watershed delineation principles"
    ]
    
    with st.expander("🔧 Tool Examples"):
        for example in tool_examples:
            if st.button(example, key=f"tool_{hash(example)}", use_container_width=True):
                st.session_state.example_query = example
    
    with st.expander("🧠 Reasoning Examples"):
        for example in reasoning_examples:
            if st.button(example, key=f"reason_{hash(example)}", use_container_width=True):
                st.session_state.example_query = example
    
    st.divider()
    
    # Settings
    st.subheader("⚙ Display Settings")
    show_route_info = st.checkbox("Show routing information", value=True)
    show_timestamps = st.checkbox("Show timestamps", value=True)
    stream_thoughts = st.checkbox("Real-time streaming", value=True)
    auto_scroll = st.checkbox("Auto-scroll during streaming", value=True)
    
    # Statistics
    st.subheader("📊 Session Statistics")
    total_queries = len(st.session_state.chat_history)
    tool_queries = sum(1 for item in st.session_state.chat_history if item.get('agent_type') == 'tools')
    reasoning_queries = sum(1 for item in st.session_state.chat_history if item.get('agent_type') == 'reasoning')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total", total_queries)
    with col2:
        st.metric("Tools", tool_queries)
    with col3:
        st.metric("Reasoning", reasoning_queries)
    
    # Clear chat
    if st.button("🗑 Clear History", type="secondary", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.current_thoughts = ""
        st.session_state.current_actions = ""
        st.rerun()

# Main content area
st.subheader("💬 Query Interface")

# Query input
default_query = st.session_state.get('example_query', "")
if 'example_query' in st.session_state:
    del st.session_state.example_query

query = st.text_area(
    "Enter your geospatial query:",
    value=default_query,
    height=100,
    placeholder="e.g., 'Perform flood risk analysis for Uttarakhand' or 'Explain how DEM resolution affects analysis accuracy'",
    help="Ask about data extraction, analysis, or geospatial concepts"
)

# Submit button
col1, col2 = st.columns([1, 3])
with col1:
    submit_button = st.button("🚀 Analyze", type="primary", disabled=st.session_state.processing)

with col2:
    if st.session_state.processing:
        st.markdown("""
        <div style="display: flex; align-items: center;">
            <span class="streaming-indicator"></span>
            <span class="status-processing">Processing your query...</span>
        </div>
        """, unsafe_allow_html=True)

# Real-time streaming section (vertical layout)
if stream_thoughts:
    st.markdown("""
    <div class="streaming-section">
        <h3>🔄 Real-time Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Thoughts section (collapsible)
    st.markdown('<div class="section-badge">💭 REASONING</div>', unsafe_allow_html=True)
    thought_container = st.empty()
    
    # Actions section (below thoughts)
    st.markdown('<div class="section-badge">⚡ ACTIONS</div>', unsafe_allow_html=True)
    action_container = st.empty()

# Process query
if submit_button and query.strip():
    st.session_state.processing = True
    st.session_state.current_thoughts = ""
    st.session_state.current_actions = ""
    
    # Status and result containers
    status_container = st.empty()
    result_container = st.empty()
    
    # Clear streaming containers
    if stream_thoughts:
        thought_container.empty()
        action_container.empty()
    
    try:
        # Step 1: Classify query
        with st.spinner("🔍 Analyzing query..."):
            route = classify_query(query)
        
        if show_route_info:
            agent_type = "Tool Agent" if route == 'tools' else "Reasoning Agent"
            status_container.success(f"🎯 Routed to: {agent_type}")
        
        # Step 2: Execute with streaming
        start_time = time.time()
        
        def streaming_callback(event: StreamingEvent):
            """Enhanced streaming callback with better formatting"""
            if event.event_type == "status":
                status_container.info(f"📋 {event.content}")
            
            elif event.event_type == "thought" and stream_thoughts:
                st.session_state.current_thoughts += event.content
                
                # Create collapsible thoughts section
                thought_container.markdown(f"""
                <div class="thoughts-section">
                    <div class="thoughts-header" onclick="toggleThoughts('current')">
                        <span>💭 Reasoning Process</span>
                        <span class="collapse-icon" id="thoughts-icon-current">▼</span>
                    </div>
                    <div class="thoughts-content" id="thoughts-content-current">
                        {st.session_state.current_thoughts}
                        <span class="streaming-indicator"></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            elif event.event_type == "action" and stream_thoughts:
                st.session_state.current_actions += event.content
                
                # Format actions with code highlighting
                parsed_content = parse_structured_content(st.session_state.current_actions)
                
                if parsed_content["type"] == "code":
                    action_container.markdown(f"""
                    <div class="action-container">
                        <pre class="code-block">{st.session_state.current_actions}</pre>
                        <span class="streaming-indicator"></span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    action_container.markdown(f"""
                    <div class="action-container">
                        {st.session_state.current_actions}
                        <span class="streaming-indicator"></span>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Execute the query
        response = router_agent_with_streaming(query, streaming_callback)
        
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        
        # Final status
        status_container.success(f"✅ Analysis completed in {processing_time}s")
        
        # Remove streaming indicators
        if stream_thoughts:
            if st.session_state.current_thoughts:
                thought_container.markdown(f"""
                <div class="thoughts-section">
                    <div class="thoughts-header" onclick="toggleThoughts('current')">
                        <span>💭 Reasoning Process</span>
                        <span class="collapse-icon" id="thoughts-icon-current">▼</span>
                    </div>
                    <div class="thoughts-content" id="thoughts-content-current">
                        {st.session_state.current_thoughts}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.session_state.current_actions:
                parsed_content = parse_structured_content(st.session_state.current_actions)
                if parsed_content["type"] == "code":
                    action_container.markdown(f"""
                    <div class="action-container">
                        <pre class="code-block">{st.session_state.current_actions}</pre>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    action_container.markdown(f"""
                    <div class="action-container">
                        {st.session_state.current_actions}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Store in chat history
        chat_entry = {
            'timestamp': datetime.now(),
            'query': query,
            'response': response,
            'agent_type': route,
            'processing_time': processing_time,
            'thoughts': st.session_state.current_thoughts,
            'actions': st.session_state.current_actions
        }
        st.session_state.chat_history.append(chat_entry)
        
        # Display formatted result
        agent_badge_class = "tool-agent" if route == 'tools' else "reasoning-agent"
        agent_name = "Tool Agent" if route == 'tools' else "Reasoning Agent"
        
        # Format the response based on agent type
        if route == 'tools':
            formatted_response = format_tool_output(response)
            if not formatted_response.strip():
                formatted_response = response
        else:
            formatted_response = format_reasoning_output(response)
            if not formatted_response.strip():
                formatted_response = response
        
        result_container.markdown(f"""
        <div class="result-container">
            <div class="agent-badge {agent_badge_class}">
                {agent_name}
            </div>
            <div class="response-content">
                {formatted_response if formatted_response.strip() else response}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        status_container.error(f"❌ Error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
    
    finally:
        st.session_state.processing = False


# Chat history with enhanced formatting
if st.session_state.chat_history:
    st.subheader("📝 Analysis History")
    
    for i, chat_entry in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Query {len(st.session_state.chat_history) - i}: {chat_entry['query'][:60]}..."):
            
            # Query display
            st.markdown(f"""
            <div class="query-container">
                <strong>🔍 Query:</strong><br>
                {chat_entry['query']}
            </div>
            """, unsafe_allow_html=True)
            
            # Agent info
            agent_badge_class = "tool-agent" if chat_entry['agent_type'] == 'tools' else "reasoning-agent"
            agent_name = "Tool Agent" if chat_entry['agent_type'] == 'tools' else "Reasoning Agent"
            
            # Show streaming content for reasoning agent (vertical layout)
            if chat_entry['agent_type'] == 'reasoning' and (chat_entry.get('thoughts') or chat_entry.get('actions')):
                
                # Thoughts section (collapsible)
                if chat_entry.get('thoughts'):
                    st.markdown(f"""
                    <div class="thoughts-section">
                        <div class="thoughts-header" onclick="toggleThoughts('{i}')">
                            <span>💭 Reasoning Process</span>
                            <span class="collapse-icon" id="thoughts-icon-{i}">▼</span>
                        </div>
                        <div class="thoughts-content" id="thoughts-content-{i}">
                            {chat_entry['thoughts']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Actions section (below thoughts)
                if chat_entry.get('actions'):
                    parsed_content = parse_structured_content(chat_entry['actions'])
                    if parsed_content["type"] == "code":
                        st.markdown(f"""
                        <div class="section-badge">⚡ IMPLEMENTATION</div>
                        <div class="action-container">
                            <pre class="code-block">{chat_entry['actions']}</pre>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="section-badge">⚡ ACTIONS</div>
                        <div class="action-container">
                            {chat_entry['actions']}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Final response
            if chat_entry['agent_type'] == 'tools':
                formatted_response = format_tool_output(chat_entry['response'])
                if not formatted_response.strip():
                    formatted_response = chat_entry['response']
            else:
                formatted_response = format_reasoning_output(chat_entry['response'])
                if not formatted_response.strip():
                    formatted_response = chat_entry['response']
            
            st.markdown(f"""
            <div class="result-container">
                <div class="agent-badge {agent_badge_class}">
                    {agent_name}
                </div>
                <div class="response-content">
                    {formatted_response if formatted_response.strip() else chat_entry['response']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Metadata
            if show_timestamps:
                st.markdown(f"""
                <div class="chat-timestamp">
                    ⏱ {chat_entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | 
                    ⚡ {chat_entry['processing_time']}s | 
                    🤖 {chat_entry['agent_type'].title()}
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem; font-family: 'Inter', sans-serif;">
    <h4>🌍 GeoGPT - Advanced Geospatial Analysis Assistant</h4>
    <p>Powered by LangChain, Google Earth Engine, and AI with Real-time Chain-of-Thought Streaming</p>
    <small>Built with ❤️ for geospatial professionals and researchers</small>
</div>
""", unsafe_allow_html=True)