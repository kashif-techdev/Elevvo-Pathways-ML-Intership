"""
Streamlit Demo App for Movie Recommendation System
Step 8: Final touches with interactive demo
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import MovieDataProcessor
from collaborative_filtering import CollaborativeFiltering
from matrix_factorization import MatrixFactorization
from evaluation import RecommendationEvaluator
import time

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary-color: #6366f1;
        --primary-dark: #4f46e5;
        --secondary-color: #8b5cf6;
        --accent-color: #f59e0b;
        --success-color: #10b981;
        --danger-color: #ef4444;
        --warning-color: #f59e0b;
        --info-color: #3b82f6;
        --background-dark: #0f172a;
        --background-card: #1e293b;
        --background-glass: rgba(255, 255, 255, 0.05);
        --text-primary: #ffffff;
        --text-secondary: #94a3b8;
        --border-color: rgba(255, 255, 255, 0.1);
        --shadow-glow: 0 0 20px rgba(99, 102, 241, 0.3);
        --shadow-card: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--background-dark);
        color: var(--text-primary);
        overflow-x: hidden;
    }

    /* Animated Background */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 80%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(245, 158, 11, 0.1) 0%, transparent 50%);
        animation: backgroundShift 20s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }

    @keyframes backgroundShift {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        33% { transform: translate(-10px, -20px) rotate(1deg); }
        66% { transform: translate(20px, 10px) rotate(-1deg); }
    }

    /* Enhanced Header */
    .main-header {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 50%, var(--accent-color) 100%);
        color: white;
        border-radius: 20px;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-glow);
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        animation: grain 8s steps(10) infinite;
    }

    @keyframes grain {
        0%, 100% { transform: translate(0, 0); }
        10% { transform: translate(-5%, -10%); }
        20% { transform: translate(-15%, 5%); }
        30% { transform: translate(7%, -25%); }
        40% { transform: translate(-5%, 25%); }
        50% { transform: translate(-15%, 10%); }
        60% { transform: translate(15%, 0%); }
        70% { transform: translate(0%, 15%); }
        80% { transform: translate(3%, 35%); }
        90% { transform: translate(-10%, 10%); }
    }

    .main-title {
        font-size: clamp(2.5rem, 5vw, 4rem);
        font-weight: 800;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
        animation: titleGlow 3s ease-in-out infinite alternate;
        position: relative;
        z-index: 1;
    }

    @keyframes titleGlow {
        0% { filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.5)); }
        100% { filter: drop-shadow(0 0 20px rgba(255, 255, 255, 0.8)); }
    }

    .main-subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
        animation: slideUp 1s ease-out 0.5s both;
        line-height: 1.6;
        position: relative;
        z-index: 1;
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Enhanced Metric Cards */
    .metric-card {
        background: var(--background-card);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid var(--border-color);
        margin: 0.5rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        border-color: var(--primary-color);
    }

    .metric-card h4 {
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .metric-card p {
        color: var(--text-secondary);
        font-size: 1.5rem;
        font-weight: 700;
    }

    /* Enhanced Recommendation Cards */
    .recommendation-card {
        background: var(--background-card);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid var(--border-color);
        margin: 1rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        animation: cardSlideIn 0.8s ease-out;
    }

    .recommendation-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(135deg, var(--success-color), var(--accent-color));
    }

    .recommendation-card:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        border-color: var(--success-color);
    }

    @keyframes cardSlideIn {
        from { 
            opacity: 0; 
            transform: translateY(40px) scale(0.95); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0) scale(1); 
        }
    }

    .recommendation-card h4 {
        color: var(--text-primary);
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        line-height: 1.4;
    }

    .recommendation-card p {
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }

    .recommendation-card .rating {
        color: var(--accent-color);
        font-weight: 600;
        font-size: 1.1rem;
    }

    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 16px 32px;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: var(--shadow-glow);
        background: linear-gradient(135deg, var(--primary-dark), var(--secondary-color));
    }

    .stButton > button:active {
        transform: translateY(0) scale(0.98);
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    /* Enhanced Sidebar */
    .sidebar .sidebar-content {
        background: var(--background-glass);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--border-color);
    }

    .sidebar .element-container {
        animation: fadeInRight 0.8s ease-out;
    }

    @keyframes fadeInRight {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* Enhanced Form Controls */
    .stSelectbox > div > div {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        transition: all 0.3s ease;
    }

    .stSelectbox > div > div:hover {
        border-color: var(--primary-color);
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.2);
    }

    .stSlider > div > div > div {
        background: var(--primary-color);
    }

    .stNumberInput > div > div {
        background: var(--background-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        transition: all 0.3s ease;
    }

    .stNumberInput > div > div:hover {
        border-color: var(--primary-color);
    }

    /* Loading Animation */
    .loading-container {
        text-align: center;
        padding: 60px 20px;
    }

    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 3px solid var(--border-color);
        border-top: 3px solid var(--primary-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto 20px;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .loading-text {
        color: var(--text-secondary);
        font-size: 1.1rem;
        animation: loadingDots 1.5s ease-in-out infinite;
    }

    @keyframes loadingDots {
        0%, 20% { opacity: 0.2; }
        50% { opacity: 1; }
        100% { opacity: 0.2; }
    }

    /* Success Message */
    .success-message {
        background: linear-gradient(135deg, var(--success-color), #059669);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        margin: 20px 0;
        animation: successSlide 0.8s ease-out;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
    }

    @keyframes successSlide {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Section Headers */
    .section-header {
        text-align: center;
        margin: 60px 0 40px;
        animation: headerFade 1s ease-out;
    }

    .section-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 12px;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    @keyframes headerFade {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .recommendation-card {
            padding: 1rem;
        }
    }

    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--background-dark);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    processor = MovieDataProcessor()
    processor.load_data()
    user_item_matrix = processor.create_user_item_matrix()
    return processor, user_item_matrix

@st.cache_data
def get_cf_model(user_item_matrix):
    """Initialize and cache collaborative filtering model"""
    return CollaborativeFiltering(user_item_matrix)

@st.cache_data
def get_mf_model(user_item_matrix):
    """Initialize and cache matrix factorization model"""
    return MatrixFactorization(user_item_matrix, n_components=50)

def main():
    """Enhanced Main Streamlit app with professional UI"""
    
    # Enhanced Header with animations
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🎬 Movie Recommendation System</h1>
        <p class="main-subtitle">Discover your next favorite movie with AI-powered recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data with enhanced loading
    with st.spinner("🔄 Loading dataset..."):
        processor, user_item_matrix = load_data()
    
    # Enhanced Sidebar
    st.sidebar.markdown("### 🎛️ Configuration")
    st.sidebar.markdown("---")
    
    # Algorithm selection with enhanced styling
    st.sidebar.markdown("#### 🧠 Algorithm")
    algorithm = st.sidebar.selectbox(
        "Choose Recommendation Algorithm",
        ["User-Based CF", "Item-Based CF", "SVD Matrix Factorization"],
        help="Select the recommendation algorithm to use",
        key="algorithm_select"
    )
    
    st.sidebar.markdown("#### 👤 User Settings")
    user_id = st.sidebar.number_input(
        "User ID",
        min_value=1,
        max_value=943,
        value=1,
        help="Enter a user ID (1-943) to get recommendations",
        key="user_id_input"
    )
    
    st.sidebar.markdown("#### ⚙️ Model Parameters")
    n_recommendations = st.sidebar.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10,
        help="Number of movies to recommend",
        key="n_recs_slider"
    )
    
    n_similar = st.sidebar.slider(
        "Similar Users/Items",
        min_value=5,
        max_value=50,
        value=20,
        help="Number of similar users/items to consider",
        key="n_similar_slider"
    )
    
    # Main content with enhanced layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="section-header">
            <h2 class="section-title">🎯 Your Recommendations</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced recommendation generation
        if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
            
            # Enhanced loading animation
            loading_placeholder = st.empty()
            with loading_placeholder:
                st.markdown("""
                <div class="loading-container">
                    <div class="loading-spinner"></div>
                    <div class="loading-text">Finding your perfect movies...</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Simulate processing time for better UX
            import time
            time.sleep(2)
            
            # Initialize models
            cf_model = get_cf_model(user_item_matrix)
            mf_model = get_mf_model(user_item_matrix)
            
            # Generate recommendations based on selected algorithm
            if algorithm == "User-Based CF":
                recommendations = cf_model.recommend_user_based(
                    user_id, n_recommendations, n_similar
                )
                
            elif algorithm == "Item-Based CF":
                recommendations = cf_model.recommend_item_based(
                    user_id, n_recommendations
                )
                
            elif algorithm == "SVD Matrix Factorization":
                recommendations = mf_model.get_user_predictions(
                    user_id, n_recommendations
                )
            
            loading_placeholder.empty()
            
            # Display recommendations with enhanced styling
            if recommendations:
                st.markdown(f"""
                <div class="success-message">
                    ✅ Found {len(recommendations)} personalized recommendations for User {user_id}!
                </div>
                """, unsafe_allow_html=True)
                
                for i, rec in enumerate(recommendations, 1):
                    confidence_text = f"<p><strong>Confidence:</strong> {rec['confidence']:.2f}</p>" if 'confidence' in rec else ""
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>#{i} {rec['movie']}</h4>
                        <p class="rating">⭐ Predicted Rating: {rec['predicted_rating']:.2f}/5.0</p>
                        {confidence_text}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No recommendations found. Try a different user or algorithm.")
    
    with col2:
        st.markdown("### 📊 Dataset Statistics")
        
        # Display dataset metrics with enhanced cards
        summary = processor.get_data_summary()
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>👥 Users</h4>
            <p>{summary['n_users']:,}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎬 Movies</h4>
            <p>{summary['n_movies']:,}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>⭐ Ratings</h4>
            <p>{summary['n_ratings']:,}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Sparsity</h4>
            <p>{summary['sparsity']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced Evaluation section
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">🔍 Algorithm Evaluation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📊 Run Performance Evaluation", use_container_width=True):
        with st.spinner("🔄 Running evaluation..."):
            # Initialize evaluator
            evaluator = RecommendationEvaluator(processor.ratings, processor.movies)
            evaluator.split_data()
            
            # Run evaluation
            results = evaluator.compare_algorithms(n_users=20, k=5)
            
            # Display results with enhanced metrics
            st.markdown("### 📈 Performance Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "User-Based CF",
                    f"{results['user_based']:.4f}",
                    help="Precision@5 for User-Based Collaborative Filtering"
                )
            
            with col2:
                st.metric(
                    "Item-Based CF",
                    f"{results['item_based']:.4f}",
                    help="Precision@5 for Item-Based Collaborative Filtering"
                )
            
            with col3:
                best_algorithm = "User-Based CF" if results['user_based'] > results['item_based'] else "Item-Based CF"
                st.metric(
                    "Best Algorithm",
                    best_algorithm,
                    help="Algorithm with highest precision"
                )
    
    # Enhanced Data visualization section
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">📈 Data Visualizations</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Rating Distribution", "🎭 Genre Analysis", "📈 Dataset Overview"])
    
    with tab1:
        # Rating distribution with enhanced styling
        fig, ax = plt.subplots(figsize=(12, 6))
        processor.ratings['rating'].hist(bins=5, ax=ax, edgecolor='black', color='#6366f1', alpha=0.7)
        ax.set_title('Rating Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Rating', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with tab2:
        # Genre distribution with enhanced styling
        genre_cols = [col for col in processor.movies.columns if col not in ['movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL']]
        genre_counts = processor.movies[genre_cols].sum().sort_values(ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        genre_counts.plot(kind='barh', ax=ax, color='#8b5cf6', alpha=0.8)
        ax.set_title('Movies per Genre', fontsize=16, fontweight='bold')
        ax.set_xlabel('Number of Movies', fontsize=12)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with tab3:
        # Dataset overview with metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Rating", f"{processor.ratings['rating'].mean():.2f}")
        with col2:
            st.metric("Rating Std", f"{processor.ratings['rating'].std():.2f}")
        with col3:
            st.metric("Most Common Rating", f"{processor.ratings['rating'].mode().iloc[0]}")
        with col4:
            st.metric("Total Genres", f"{len(genre_cols)}")
    
    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #94a3b8;'>
        <h3 style='color: #6366f1; margin-bottom: 1rem;'>🎬 Movie Recommendation System</h3>
        <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>Built with Streamlit & Scikit-learn</p>
        <p style='font-size: 0.9rem;'>Dataset: MovieLens 100K | Algorithms: User-Based CF, Item-Based CF, SVD</p>
        <p style='font-size: 0.8rem; margin-top: 1rem; opacity: 0.7;'>Elevvo Pathways Machine Learning Internship</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
