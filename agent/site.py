import streamlit as st
import sys
import os
from typing import Dict, Any
import time
import json
from datetime import datetime

# Add the agent directory to Python path
agent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent')
if agent_path not in sys.path:
    sys.path.append(agent_path)

# Import the router agent
try:
    from top_node import router_agent, classify_query
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

# Custom CSS for better styling
st.markdown("""
<style>
    body {
        background-color: #121212;
        color: #e0e0e0;
    }

    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    .query-container {
        background: #1e1e1e;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
        color: #f5f5f5;
    }

    .result-container {
        background: #1e1e1e;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #333;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(255,255,255,0.05);
        color: #f5f5f5;
    }

    .agent-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }

    .tool-agent {
        background: #003c8f;
        color: #ffffff;
    }

    .reasoning-agent {
        background: #6a1b9a;
        color: #ffffff;
    }

    .status-success {
        color: #81c784;
    }

    .status-error {
        color: #ef5350;
    }

    .status-processing {
        color: #ffb74d;
    }

    .stTextArea textarea {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }

    .stButton>button {
        background-color: #e53935;
        color: white;
    }

    .stMetric {
        color: #ffffff;
    }
</style>

""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Header
st.markdown("""
<div class="main-header">
    <h1>🌍 GeoGPT - Geospatial Analysis Assistant</h1>
    <p>Advanced AI-powered geospatial analysis with intelligent routing</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🛠 Agent Information")
    
    st.subheader("Tool Agent")
    st.write("Handles:")
    st.write("• Data extraction and download")
    st.write("• SRTM DEM, NDVI, rainfall data")
    st.write("• Suitability analysis")
    st.write("• Map visualization")
    
    st.subheader("Reasoning Agent")
    st.write("Handles:")
    st.write("• Conceptual explanations")
    st.write("• Technical guidance")
    st.write("• Methodology discussions")
    st.write("• Scientific reasoning")
    
    st.divider()
    
    # Query examples
    st.subheader("📋 Example Queries")
    
    tool_examples = [
        "Download NDVI data for Karnataka",
        "Perform flood risk analysis for Delhi",
        "Site suitability analysis using DEM and rainfall",
        "Extract bounding box for Mumbai"
    ]
    
    reasoning_examples = [
        "Explain how slope affects flood risk",
        "What is the difference between SRTM and ASTER DEM?",
        "How does NDVI relate to vegetation health?",
        "Describe watershed delineation methodology"
    ]
    
    st.write("*Tool Agent Examples:*")
    for example in tool_examples:
        if st.button(f"• {example}", key=f"tool_{example}"):
            st.session_state.example_query = example
    
    st.write("*Reasoning Agent Examples:*")
    for example in reasoning_examples:
        if st.button(f"• {example}", key=f"reason_{example}"):
            st.session_state.example_query = example
    
    st.divider()
    
    # Settings
    st.subheader("⚙ Settings")
    show_route_info = st.checkbox("Show routing information", value=True)
    show_timestamps = st.checkbox("Show timestamps", value=True)
    
    # Clear chat history
    if st.button("🗑 Clear Chat History", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    # Query input
    st.subheader("💬 Ask Your Geospatial Question")
    
    # Use example query if selected
    default_query = ""
    if 'example_query' in st.session_state:
        default_query = st.session_state.example_query
        del st.session_state.example_query
    
    query = st.text_area(
        "Enter your geospatial query:",
        value=default_query,
        height=100,
        placeholder="e.g., Download NDVI data for Karnataka or Explain how slope affects flood risk"
    )
    
    # Submit button
    submit_col1, submit_col2 = st.columns([1, 4])
    with submit_col1:
        submit_button = st.button("🚀 Submit", type="primary", disabled=st.session_state.processing)
    
    with submit_col2:
        if st.session_state.processing:
            st.markdown('<p class="status-processing">🔄 Processing your query...</p>', unsafe_allow_html=True)

with col2:
    # Query statistics
    st.subheader("📊 Session Stats")
    total_queries = len(st.session_state.chat_history)
    st.metric("Total Queries", total_queries)
    
    if st.session_state.chat_history:
        tool_queries = sum(1 for item in st.session_state.chat_history if item.get('agent_type') == 'tools')
        reasoning_queries = sum(1 for item in st.session_state.chat_history if item.get('agent_type') == 'reasoning')
        
        st.metric("Tool Agent", tool_queries)
        st.metric("Reasoning Agent", reasoning_queries)

# Process query
if submit_button and query.strip():
    st.session_state.processing = True
    
    # Create placeholder for real-time updates
    status_placeholder = st.empty()
    result_placeholder = st.empty()
    
    try:
        # Step 1: Classify query
        status_placeholder.markdown('<p class="status-processing">🔍 Classifying query...</p>', unsafe_allow_html=True)
        
        route = classify_query(query)
        
        if show_route_info:
            if route == 'tools':
                status_placeholder.markdown('<p class="status-processing">🛠 Routing to Tool Agent...</p>', unsafe_allow_html=True)
            else:
                status_placeholder.markdown('<p class="status-processing">🧠 Routing to Reasoning Agent...</p>', unsafe_allow_html=True)
        
        # Step 2: Get response
        start_time = time.time()
        response = router_agent(query)
        end_time = time.time()
        
        # Success status
        status_placeholder.markdown('<p class="status-success">✅ Query processed successfully!</p>', unsafe_allow_html=True)
        
        # Store in chat history
        chat_entry = {
            'timestamp': datetime.now(),
            'query': query,
            'response': response,
            'agent_type': route,
            'processing_time': round(end_time - start_time, 2)
        }
        st.session_state.chat_history.append(chat_entry)
        
        # Display result
        agent_badge_class = "tool-agent" if route == 'tools' else "reasoning-agent"
        agent_name = "Tool Agent" if route == 'tools' else "Reasoning Agent"
        
        result_placeholder.markdown(f"""
        <div class="result-container">
            <div class="agent-badge {agent_badge_class}">
                {agent_name}
            </div>
            <div style="white-space: pre-wrap;">{response}</div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        status_placeholder.markdown(f'<p class="status-error">❌ Error: {str(e)}</p>', unsafe_allow_html=True)
        st.error(f"An error occurred: {str(e)}")
    
    finally:
        st.session_state.processing = False

# Display chat history
if st.session_state.chat_history:
    st.subheader("📝 Chat History")
    
    # Reverse order to show most recent first
    for i, chat_entry in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Query {len(st.session_state.chat_history) - i}: {chat_entry['query'][:50]}..."):
            
            # Query info
            st.markdown(f"""
            <div class="query-container">
                <strong>Query:</strong> {chat_entry['query']}
            </div>
            """, unsafe_allow_html=True)
            
            # Agent info and response
            agent_badge_class = "tool-agent" if chat_entry['agent_type'] == 'tools' else "reasoning-agent"
            agent_name = "Tool Agent" if chat_entry['agent_type'] == 'tools' else "Reasoning Agent"
            
            st.markdown(f"""
            <div class="result-container">
                <div class="agent-badge {agent_badge_class}">
                    {agent_name}
                </div>
                <div style="white-space: pre-wrap;">{chat_entry['response']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Metadata
            if show_timestamps:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.text(f"Time: {chat_entry['timestamp'].strftime('%H:%M:%S')}")
                with col2:
                    st.text(f"Processing: {chat_entry['processing_time']}s")
                with col3:
                    st.text(f"Agent: {chat_entry['agent_type'].title()}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🌍 GeoGPT - Advanced Geospatial Analysis Assistant<br>
    Powered by LangChain, Google Earth Engine, and AI
</div>
""", unsafe_allow_html=True)