"""
Complete Demo of Movie Recommendation System
Demonstrates all features and algorithms
"""
import pandas as pd
import numpy as np
from data_preprocessing import MovieDataProcessor
from collaborative_filtering import CollaborativeFiltering
from matrix_factorization import MatrixFactorization
from evaluation import RecommendationEvaluator
import time

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🎬 {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section header"""
    print(f"\n📋 {title}")
    print("-" * 40)

def main():
    """Complete demonstration of the movie recommendation system"""
    
    print_header("MOVIE RECOMMENDATION SYSTEM DEMO")
    print("Built from scratch with collaborative filtering and matrix factorization")
    
    # Step 1: Data Loading and Preprocessing
    print_section("Step 1: Data Loading and Preprocessing")
    processor = MovieDataProcessor()
    processor.load_data()
    processor.inspect_data()
    user_item_matrix = processor.create_user_item_matrix()
    
    # Step 2: Collaborative Filtering
    print_section("Step 2: Collaborative Filtering")
    cf = CollaborativeFiltering(user_item_matrix)
    
    # Test user-based recommendations
    test_user = 1
    print(f"\n🎯 User-Based Recommendations for User {test_user}:")
    user_recs = cf.recommend_user_based(test_user, n_recommendations=5)
    for i, rec in enumerate(user_recs, 1):
        print(f"  {i}. {rec['movie']} (Rating: {rec['predicted_rating']:.2f})")
    
    # Test item-based recommendations
    print(f"\n🎯 Item-Based Recommendations for User {test_user}:")
    item_recs = cf.recommend_item_based(test_user, n_recommendations=5)
    for i, rec in enumerate(item_recs, 1):
        print(f"  {i}. {rec['movie']} (Rating: {rec['predicted_rating']:.2f})")
    
    # Step 3: Matrix Factorization
    print_section("Step 3: Matrix Factorization (SVD)")
    mf = MatrixFactorization(user_item_matrix, n_components=50)
    mf.fit_svd()
    
    # Test SVD recommendations
    print(f"\n🎯 SVD Recommendations for User {test_user}:")
    svd_recs = mf.get_user_predictions(test_user, n_recommendations=5)
    for i, rec in enumerate(svd_recs, 1):
        print(f"  {i}. {rec['movie']} (Rating: {rec['predicted_rating']:.2f})")
    
    # Test similar movies
    test_movie = "Toy Story (1995)"
    print(f"\n🎯 Movies Similar to '{test_movie}':")
    similar_movies = mf.get_similar_movies(test_movie, n_similar=5)
    for i, sim in enumerate(similar_movies, 1):
        print(f"  {i}. {sim['movie']} (Similarity: {sim['similarity']:.3f})")
    
    # Step 4: Evaluation
    print_section("Step 4: Algorithm Evaluation")
    evaluator = RecommendationEvaluator(processor.ratings, processor.movies)
    evaluator.split_data()
    
    # Run evaluation
    results = evaluator.compare_algorithms(n_users=10, k=5)
    
    print(f"\n📊 Evaluation Results:")
    print(f"  User-Based CF Precision@5: {results['user_based']:.4f}")
    print(f"  Item-Based CF Precision@5: {results['item_based']:.4f}")
    
    # Step 5: Dataset Statistics
    print_section("Step 5: Dataset Statistics")
    summary = processor.get_data_summary()
    
    print(f"📈 Dataset Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Step 6: Algorithm Comparison
    print_section("Step 6: Algorithm Comparison")
    
    print(f"\n🔍 Algorithm Performance:")
    print(f"  User-Based CF: {'✅' if results['user_based'] > 0 else '❌'} {results['user_based']:.4f}")
    print(f"  Item-Based CF: {'✅' if results['item_based'] > 0 else '❌'} {results['item_based']:.4f}")
    print(f"  SVD Matrix Factorization: ✅ Implemented")
    
    # Step 7: Recommendations Summary
    print_section("Step 7: Recommendations Summary")
    
    print(f"\n🎬 All Recommendations for User {test_user}:")
    print(f"  User-Based CF: {len(user_recs)} movies")
    print(f"  Item-Based CF: {len(item_recs)} movies")
    print(f"  SVD Matrix Factorization: {len(svd_recs)} movies")
    
    # Step 8: Next Steps
    print_section("Step 8: Next Steps")
    
    print(f"\n🚀 To run the interactive demo:")
    print(f"  streamlit run app.py")
    print(f"\n📚 To explore individual components:")
    print(f"  python data_preprocessing.py")
    print(f"  python collaborative_filtering.py")
    print(f"  python matrix_factorization.py")
    print(f"  python evaluation.py")
    
    print_header("DEMO COMPLETED SUCCESSFULLY! 🎉")
    print("All recommendation algorithms implemented and tested!")
    
    return {
        'processor': processor,
        'cf': cf,
        'mf': mf,
        'evaluator': evaluator,
        'results': results
    }

if __name__ == "__main__":
    demo_results = main()
