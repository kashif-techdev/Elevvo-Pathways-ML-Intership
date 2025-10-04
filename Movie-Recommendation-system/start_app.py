"""
Professional Startup Script for Movie Recommendation System
Enhanced UI with beautiful animations and professional styling
"""
import streamlit as st
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'scikit-learn', 
        'matplotlib', 'seaborn', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        st.error(f"❌ Missing packages: {', '.join(missing_packages)}")
        st.info("💡 Run: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_dataset():
    """Check if dataset exists"""
    data_path = Path("data/ml-100k")
    if not data_path.exists():
        st.warning("⚠️ Dataset not found. Downloading...")
        try:
            from download_dataset import download_movielens_100k
            download_movielens_100k()
            st.success("✅ Dataset downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download dataset: {e}")
            return False
    return True

def main():
    """Main startup function with enhanced checks"""
    
    # Page configuration
    st.set_page_config(
        page_title="🎬 Movie Recommendation System",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Startup checks
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h1 style='color: #6366f1; margin-bottom: 1rem;'>🎬 Movie Recommendation System</h1>
        <p style='color: #94a3b8; font-size: 1.2rem;'>Professional AI-Powered Movie Recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System checks
    with st.expander("🔍 System Status", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if check_dependencies():
                st.success("✅ Dependencies OK")
            else:
                st.error("❌ Dependencies Missing")
        
        with col2:
            if check_dataset():
                st.success("✅ Dataset Ready")
            else:
                st.error("❌ Dataset Missing")
        
        with col3:
            if Path("app.py").exists():
                st.success("✅ App Files Ready")
            else:
                st.error("❌ App Files Missing")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    st.markdown("""
    <div style='background: #1e293b; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #6366f1;'>
        <h4 style='color: #6366f1; margin-bottom: 1rem;'>🎯 How to Use:</h4>
        <ol style='color: #94a3b8; line-height: 1.6;'>
            <li><strong>Select Algorithm:</strong> Choose between User-Based CF, Item-Based CF, or SVD</li>
            <li><strong>Pick a User:</strong> Enter any user ID from 1-943</li>
            <li><strong>Get Recommendations:</strong> Click the button to generate personalized movie suggestions</li>
            <li><strong>Explore Data:</strong> View visualizations and algorithm performance metrics</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Features overview
    st.markdown("### ✨ Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: #1e293b; padding: 1rem; border-radius: 8px; text-align: center;'>
            <h4 style='color: #6366f1;'>🧠 AI Algorithms</h4>
            <p style='color: #94a3b8; font-size: 0.9rem;'>User-Based CF, Item-Based CF, SVD Matrix Factorization</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: #1e293b; padding: 1rem; border-radius: 8px; text-align: center;'>
            <h4 style='color: #8b5cf6;'>📊 Data Visualization</h4>
            <p style='color: #94a3b8; font-size: 0.9rem;'>Rating distributions, genre analysis, performance metrics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: #1e293b; padding: 1rem; border-radius: 8px; text-align: center;'>
            <h4 style='color: #f59e0b;'>🎨 Professional UI</h4>
            <p style='color: #94a3b8; font-size: 0.9rem;'>Modern design with animations and responsive layout</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Launch button
    if st.button("🚀 Launch Movie Recommendation System", type="primary", use_container_width=True):
        st.success("🎉 Redirecting to the main application...")
        st.balloons()
        
        # Import and run the main app
        try:
            from app import main as app_main
            app_main()
        except Exception as e:
            st.error(f"❌ Error launching app: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #94a3b8; padding: 1rem;'>
        <p style='font-size: 0.9rem;'>Built with ❤️ for the Elevvo Pathways Machine Learning Internship</p>
        <p style='font-size: 0.8rem; opacity: 0.7;'>Powered by Streamlit, Scikit-learn, and Python</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
