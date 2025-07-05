import streamlit as st
import streamlit.components.v1
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

    /* Code blocks within action container */
    .action-container pre {
        background: #0f1419 !important;
        color: #e6e1dc !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid #2d3748 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        margin: 0.5rem 0 !important;
        overflow-x: auto !important;
        white-space: pre-wrap !important;
    }

    .action-container code {
        background: #2d3748 !important;
        color: #e6e1dc !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
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
    
    /* Map display styling */
    .map-container {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        color: #e2e8f0;
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        border: 1px solid #4a5568;
    }
    
    .map-info-header {
        background: linear-gradient(135deg, #2b6cb0 0%, #3182ce 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    .map-selector {
        margin-bottom: 1.5rem;
    }
    
    .map-iframe-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        border: 2px solid #4a5568;
        margin: 1rem 0;
    }
    
    /* Section separators */
    .section-separator {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #4a5568 20%, #667eea 50%, #4a5568 80%, transparent 100%);
        margin: 3rem 0;
        border-radius: 1px;
    }
    
    /* Map section specific styling */
    .map-section {
        margin-top: 3rem;
        margin-bottom: 3rem;
    }
    
    .map-section h2 {
        color: #f1f5f9;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Fix Streamlit's default white backgrounds */
    .stApp {
        background-color: #0f172a;
    }
    
    .main .block-container {
        background-color: #0f172a;
    }
    
    /* Fix sidebar background and styling */
    .css-1d391kg {
        background-color: #1e293b !important;
    }
    
    .css-1d391kg .css-10trblm {
        background-color: #1e293b !important;
    }
    
    .css-1d391kg .css-16huue1 {
        background-color: #1e293b !important;
    }
    
    /* Fix sidebar text color */
    .css-1d391kg .css-10trblm {
        color: #e2e8f0 !important;
    }
    
    /* Fix sidebar headers */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #f1f5f9 !important;
    }
    
    /* Fix sidebar expander */
    .css-1d391kg .streamlit-expanderHeader {
        background-color: #334155 !important;
        color: #e2e8f0 !important;
        border: 1px solid #475569 !important;
    }
    
    .css-1d391kg .streamlit-expanderContent {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
    }
    
    /* Fix sidebar buttons */
    .css-1d391kg .stButton > button {
        background-color: #374151 !important;
        color: #e5e7eb !important;
        border: 1px solid #6b7280 !important;
    }
    
    .css-1d391kg .stButton > button:hover {
        background-color: #4b5563 !important;
        border-color: #9ca3af !important;
    }
    
    /* Fix sidebar checkboxes */
    .css-1d391kg .stCheckbox > label {
        color: #e2e8f0 !important;
    }
    
    /* Fix sidebar metrics */
    .css-1d391kg .metric-container {
        background-color: #334155 !important;
        border-color: #475569 !important;
    }
    
    /* Sidebar specific styling */
    .sidebar-agent-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: #e2e8f0;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #475569;
        margin-bottom: 1rem;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .sidebar-agent-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .sidebar-agent-card h4 {
        color: #f1f5f9;
        margin-bottom: 0.8rem;
        font-weight: 600;
        font-size: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .sidebar-agent-card small {
        color: #cbd5e1;
        line-height: 1.5;
        font-size: 0.85rem;
    }
    
    .sidebar-agent-card .agent-icon {
        font-size: 1.2rem;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .tool-agent-card {
        border-left: 4px solid #3b82f6;
    }
    
    .reasoning-agent-card {
        border-left: 4px solid #ec4899;
    }
    
    /* Sidebar metrics styling */
    .sidebar-metrics {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.8rem;
        margin-top: 1rem;
    }
    
    .sidebar-metric {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: #e2e8f0;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #475569;
        font-family: 'Inter', sans-serif;
    }
    
    .sidebar-metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.2rem;
    }
    
    .sidebar-metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar button styling */
    .sidebar-example-btn {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
        color: #e5e7eb;
        border: 1px solid #6b7280;
        padding: 0.7rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .sidebar-example-btn:hover {
        background: linear-gradient(135deg, #4b5563 0%, #6b7280 100%);
        transform: translateX(4px);
    }
    
    /* Sidebar section headers */
    .sidebar-section-header {
        color: #f1f5f9;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #475569;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar divider */
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, #475569 50%, transparent 100%);
        margin: 1.5rem 0;
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
def apply_text_formatting(text):
    """Apply consistent text formatting for markdown, bullets, and headers"""
    if not text:
        return ""
    
    # Split into lines and process each line
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        original_line = line
        line = line.strip()
        
        if not line:
            formatted_lines.append("")
            continue
        
        # Skip lines that are already in code blocks or HTML
        if (line.startswith('```') or '```' in original_line or 
            line.startswith('<') or line.startswith('**🔧') or 
            line.startswith('**📁')):
            formatted_lines.append(original_line)
            continue
        
        # Convert hash symbols to HTML bold headers
        if line.startswith('###'):
            line = f"<strong>{line[3:].strip()}</strong>"
        elif line.startswith('##'):
            line = f"<strong>{line[2:].strip()}</strong>"
        elif line.startswith('#'):
            line = f"<strong>{line[1:].strip()}</strong>"
        # Convert dashes to bullet points
        elif line.startswith('- '):
            line = f"• {line[2:]}"
        elif re.match(r'^-+\s+', line):
            content = re.sub(r'^-+\s+', '', line)
            line = f"• {content}"
        
        # Convert **text** to HTML bold
        line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
        
        # Preserve original indentation
        indent = len(original_line) - len(original_line.lstrip())
        if indent > 0:
            line = ' ' * indent + line
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def format_tool_output(text):
    """Format tool output with better structure"""
    if not text:
        return ""
    
    # Apply text formatting first
    text = apply_text_formatting(text)
    
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
            formatted_output += f"• {file.strip()}\n"
    
    # If no specific patterns found, return the formatted text
    if not formatted_output.strip():
        return text
    
    return formatted_output

def format_reasoning_output(text):
    """Format reasoning output with better structure"""
    if not text:
        return ""
    
    # Apply text formatting first
    text = apply_text_formatting(text)
    
    # Split by sections
    sections = re.split(r'\*\*(THOUGHT|ACTION):\*\*', text)
    
    formatted_output = ""
    current_section = None
    
    for i, section in enumerate(sections):
        if section.strip() in ['THOUGHT', 'ACTION']:
            current_section = section.strip()
            continue
        
        if current_section and section.strip():
            formatted_section = apply_text_formatting(section.strip())
            
            if current_section == 'THOUGHT':
                formatted_output += f"""
<div class="section-badge">💭 REASONING</div>
<div class="thoughts-content">
{formatted_section}
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
{formatted_section}
</div>
"""
    
    # If no specific sections found, return the formatted text
    if not formatted_output.strip():
        return text
    
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
    st.markdown('<h2 class="sidebar-section-header">🛠 System Overview</h2>', unsafe_allow_html=True)
    
    # Agent information with enhanced styling
    st.markdown("""
    <div class="sidebar-agent-card tool-agent-card">
        <h4><span class="agent-icon">🔧</span> Tool Agent</h4>
        <small>
        • Data extraction & processing<br>
        • SRTM DEM, NDVI analysis<br>
        • Rainfall & climate data<br>
        • Map generation & visualization<br>
        • Geospatial computations
        </small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-agent-card reasoning-agent-card">
        <h4><span class="agent-icon">🧠</span> Reasoning Agent</h4>
        <small>
        • Conceptual explanations<br>
        • Technical guidance & best practices<br>
        • Scientific reasoning & methodology<br>
        • Real-time Chain-of-Thought<br>
        • Educational content
        </small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Quick examples
    st.markdown('<h3 class="sidebar-section-header">📋 Quick Examples</h3>', unsafe_allow_html=True)
    
    tool_examples = [
        "Download NDVI data for Karnataka",
        "Flood risk analysis for Delhi with map",
        "Site suitability analysis for solar farms",
        "Create vegetation health map for Punjab",
        "Generate DEM visualization for Himachal Pradesh",
        "Rainfall pattern analysis with map for Kerala"
    ]
    
    reasoning_examples = [
        "Explain flood risk methodology",
        "Compare SRTM vs ASTER DEM",
        "NDVI and vegetation health relationship",
        "Watershed delineation principles"
    ]
    
    with st.expander("🔧 Tool Examples", expanded=False):
        for example in tool_examples:
            if st.button(example, key=f"tool_{hash(example)}", use_container_width=True):
                st.session_state.example_query = example
    
    with st.expander("🧠 Reasoning Examples", expanded=False):
        for example in reasoning_examples:
            if st.button(example, key=f"reason_{hash(example)}", use_container_width=True):
                st.session_state.example_query = example
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Settings
    st.markdown('<h3 class="sidebar-section-header">⚙ Display Settings</h3>', unsafe_allow_html=True)
    show_route_info = st.checkbox("Show routing information", value=True)
    show_timestamps = st.checkbox("Show timestamps", value=True)
    stream_thoughts = st.checkbox("Real-time streaming", value=True)
    auto_scroll = st.checkbox("Auto-scroll during streaming", value=True)
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Statistics with enhanced styling
    st.markdown('<h3 class="sidebar-section-header">📊 Session Statistics</h3>', unsafe_allow_html=True)
    total_queries = len(st.session_state.chat_history)
    tool_queries = sum(1 for item in st.session_state.chat_history if item.get('agent_type') == 'tools')
    reasoning_queries = sum(1 for item in st.session_state.chat_history if item.get('agent_type') == 'reasoning')
    
    # Count available maps
    outputs_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    total_maps = 0
    if os.path.exists(outputs_dir):
        total_maps = len([f for f in os.listdir(outputs_dir) if f.endswith('.html')])
    
    # Display metrics with custom styling
    st.markdown(f"""
    <div class="sidebar-metrics">
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">{total_queries}</div>
            <div class="sidebar-metric-label">Queries</div>
        </div>
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">{reasoning_queries}</div>
            <div class="sidebar-metric-label">Reasoning</div>
        </div>
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">{tool_queries}</div>
            <div class="sidebar-metric-label">Tools</div>
        </div>
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">{total_maps}</div>
            <div class="sidebar-metric-label">Maps</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # Clear chat with enhanced styling
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
    st.markdown('<div class="section-badge">� REASONING</div>', unsafe_allow_html=True)
    thought_container = st.empty()
    
    # Actions section (below thoughts)
    st.markdown('<div class="section-badge">⚡ ACTIONS</div>', unsafe_allow_html=True)
    action_container = st.empty()

# Helper functions for map display
def get_html_maps():
    """Get list of HTML map files from outputs directory"""
    outputs_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    if not os.path.exists(outputs_dir):
        return []
    
    html_files = []
    for file in os.listdir(outputs_dir):
        if file.endswith('.html'):
            file_path = os.path.join(outputs_dir, file)
            # Get file modification time for sorting
            mod_time = os.path.getmtime(file_path)
            html_files.append({
                'name': file,
                'path': file_path,
                'mod_time': mod_time,
                'size': os.path.getsize(file_path)
            })
    
    # Sort by modification time (newest first)
    html_files.sort(key=lambda x: x['mod_time'], reverse=True)
    return html_files

def extract_map_info_from_filename(filename):
    """Extract region and analysis type from filename"""
    # Remove .html extension
    name = filename.replace('.html', '')
    
    # Common patterns in the filenames
    if 'flood_risk' in name.lower():
        region = name.split('_flood_risk')[0]
        analysis_type = 'Flood Risk Analysis'
    elif 'suitability' in name.lower():
        region = name.split('_suitability')[0]
        analysis_type = 'Suitability Analysis'
    else:
        # Try to extract region from first part
        parts = name.split('_')
        region = parts[0] if parts else 'Unknown'
        analysis_type = 'Geospatial Analysis'
    
    return region, analysis_type

def check_for_new_maps():
    """Check if new maps have been generated"""
    current_maps = set(map_info['name'] for map_info in get_html_maps())
    
    if current_maps != st.session_state.known_maps:
        new_maps = current_maps - st.session_state.known_maps
        if new_maps:
            for new_map in new_maps:
                region, analysis_type = extract_map_info_from_filename(new_map)
                st.success(f"🗺️ New map generated: {region} - {analysis_type}")
        
        st.session_state.known_maps = current_maps
        return len(new_maps) > 0
    
    return False

# Initialize session state for map auto-refresh
if 'last_map_check' not in st.session_state:
    st.session_state.last_map_check = 0
if 'known_maps' not in st.session_state:
    st.session_state.known_maps = set()

# Auto-refresh maps every 5 seconds during processing
if st.session_state.processing:
    current_time = time.time()
    if current_time - st.session_state.last_map_check > 5:
        if check_for_new_maps():
            st.rerun()
        st.session_state.last_map_check = current_time

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
            """Enhanced streaming callback with better content accumulation"""
            if event.event_type == "status":
                status_container.info(f"📋 {event.content}")
            
            elif event.event_type == "thought" and stream_thoughts:
                # Properly accumulate thoughts content
                if hasattr(st.session_state, 'current_thoughts'):
                    st.session_state.current_thoughts += event.content
                else:
                    st.session_state.current_thoughts = event.content
                
                # Apply text formatting and display with better markdown handling
                formatted_thoughts = apply_text_formatting(st.session_state.current_thoughts)
                formatted_thoughts = formatted_thoughts.replace('\n', '<br>')
                
                # Create collapsible thoughts section with improved formatting
                thought_container.markdown(f"""
                <div class="thoughts-section">
                    <div class="thoughts-header" onclick="toggleThoughts('current')">
                        <span>💭 Reasoning Process</span>
                        <span class="collapse-icon" id="thoughts-icon-current">▼</span>
                    </div>
                    <div class="thoughts-content" id="thoughts-content-current">
                        {formatted_thoughts}
                        <span class="streaming-indicator"></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            elif event.event_type == "action" and stream_thoughts:
                # Properly accumulate actions content
                if hasattr(st.session_state, 'current_actions'):
                    st.session_state.current_actions += event.content
                else:
                    st.session_state.current_actions = event.content
                
                # Apply text formatting for actions
                formatted_actions = apply_text_formatting(st.session_state.current_actions)
                
                # Display actions with proper background styling and code block preservation
                action_container.markdown(f"""
                <div class="action-container">
                    {formatted_actions}
                    <br><span class="streaming-indicator"></span>
                </div>
                """, unsafe_allow_html=True)
        
        # Execute the query
        response = router_agent_with_streaming(query, streaming_callback)
        
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        
        # Final status
        status_container.success(f"✅ Analysis completed in {processing_time}s")
        
        # Check for newly generated maps
        if check_for_new_maps():
            st.balloons()  # Celebrate new map generation!
            st.info("🗺️ **New interactive map generated!** Check the Maps section above to view it.")
        
        # Remove streaming indicators and display final content
        if stream_thoughts:
            if st.session_state.current_thoughts:
                # Apply text formatting and clean up the final thoughts content
                final_thoughts = apply_text_formatting(st.session_state.current_thoughts)
                final_thoughts = final_thoughts.replace('\n', '<br>')
                
                thought_container.markdown(f"""
                <div class="thoughts-section">
                    <div class="thoughts-header" onclick="toggleThoughts('current')">
                        <span>💭 Reasoning Process</span>
                        <span class="collapse-icon" id="thoughts-icon-current">▼</span>
                    </div>
                    <div class="thoughts-content" id="thoughts-content-current">
                        {final_thoughts}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.session_state.current_actions:
                # Apply text formatting and display final actions content with proper styling
                final_actions = apply_text_formatting(st.session_state.current_actions)
                action_container.markdown(f"""
                <div class="action-container">
                    {final_actions}
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


# Map Display Section - Now positioned after analysis completion
st.markdown('<div class="section-separator"></div>', unsafe_allow_html=True)
st.markdown('<div class="map-section">', unsafe_allow_html=True)
st.subheader("🗺️ Generated Maps")

# Get available maps
html_maps = get_html_maps()

if html_maps:
    # Map selector with improved layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create options with map info
        map_options = []
        for map_file in html_maps:
            region, analysis_type = extract_map_info_from_filename(map_file['name'])
            mod_time_str = datetime.fromtimestamp(map_file['mod_time']).strftime('%Y-%m-%d %H:%M')
            size_kb = round(map_file['size'] / 1024, 1)
            
            display_name = f"🗺️ {region} - {analysis_type} ({mod_time_str}, {size_kb}KB)"
            map_options.append(display_name)
        
        selected_map = st.selectbox(
            "Select a map to display:",
            options=map_options,
            help="Choose from recently generated maps"
        )
    
    with col2:
        # Add some spacing to align with selectbox
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Maps", help="Scan for new map files", use_container_width=True):
            st.rerun()
    
    # Display selected map
    if selected_map:
        selected_index = map_options.index(selected_map)
        selected_map_info = html_maps[selected_index]
        
        # Map info header
        region, analysis_type = extract_map_info_from_filename(selected_map_info['name'])
        mod_time_str = datetime.fromtimestamp(selected_map_info['mod_time']).strftime('%Y-%m-%d %H:%M:%S')
        
        st.markdown(f"""
        <div class="result-container">
            <div class="agent-badge tool-agent">🗺️ Interactive Map</div>
            <div style="margin-bottom: 1rem;">
                <strong>📍 Region:</strong> {region}<br>
                <strong>🔍 Analysis:</strong> {analysis_type}<br>
                <strong>⏰ Generated:</strong> {mod_time_str}<br>
                <strong>📁 File:</strong> {selected_map_info['name']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the HTML map in a container with better styling
        st.markdown('<div class="map-iframe-container">', unsafe_allow_html=True)
        try:
            with open(selected_map_info['path'], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Display the map using Streamlit's HTML component
            st.components.v1.html(html_content, height=600, scrolling=True)
            
        except Exception as e:
            st.error(f"❌ Error loading map: {str(e)}")
            st.info("The map file might be corrupted or still being generated.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Download button
        try:
            with open(selected_map_info['path'], 'rb') as f:
                map_data = f.read()
            
            st.download_button(
                label="📥 Download Map HTML",
                data=map_data,
                file_name=selected_map_info['name'],
                mime="text/html",
                help="Download the interactive map file"
            )
        except Exception as e:
            st.warning(f"Download unavailable: {str(e)}")

else:
    st.info("""
    🗺️ **No maps available yet**
    
    Maps will appear here automatically after running geospatial analysis queries that generate visualizations.
    
    **Try queries like:**
    - "Perform flood risk analysis for Uttarakhand"
    - "Create suitability map for solar farms in Bihar"
    - "Analyze vegetation health in Karnataka"
    - "Generate DEM visualization for Himachal Pradesh"
    """)

st.markdown('</div>', unsafe_allow_html=True)  # Close map-section div

# Chat history with enhanced formatting
st.markdown('<div class="section-separator"></div>', unsafe_allow_html=True)
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