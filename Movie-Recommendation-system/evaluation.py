"""
Evaluation System for Movie Recommendation
Step 5: Precision@K evaluation with train/test split
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collaborative_filtering import CollaborativeFiltering
import warnings
warnings.filterwarnings('ignore')

class RecommendationEvaluator:
    """Evaluate recommendation systems using Precision@K"""
    
    def __init__(self, ratings_df, movies_df):
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.train_ratings = None
        self.test_ratings = None
        self.train_matrix = None
        self.test_matrix = None
        
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("Splitting data into train/test sets...")
        
        # Split ratings into train/test
        self.train_ratings, self.test_ratings = train_test_split(
            self.ratings_df, 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"✓ Train set: {len(self.train_ratings)} ratings")
        print(f"✓ Test set: {len(self.test_ratings)} ratings")
        
        # Create user-item matrices
        self.train_matrix = self.train_ratings.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        )
        
        self.test_matrix = self.test_ratings.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        )
        
        return self.train_ratings, self.test_ratings
    
    def precision_at_k(self, recommendations, actual_ratings, k=10, threshold=4.0):
        """
        Calculate Precision@K
        
        Args:
            recommendations: List of recommended movie IDs
            actual_ratings: Series of actual ratings for the user
            k: Number of top recommendations to consider
            threshold: Minimum rating to consider as "relevant"
        
        Returns:
            Precision@K score
        """
        if len(recommendations) == 0:
            return 0.0
        
        # Get top-K recommendations
        top_k_recs = recommendations[:k]
        
        # Find relevant items (rated above threshold)
        relevant_items = actual_ratings[actual_ratings >= threshold].index
        
        # Count hits (recommended items that are relevant)
        hits = len(set(top_k_recs) & set(relevant_items))
        
        # Precision@K = hits / k
        precision = hits / k if k > 0 else 0.0
        
        return precision
    
    def evaluate_user_based_cf(self, n_users=50, k=10, threshold=4.0):
        """Evaluate user-based collaborative filtering"""
        print(f"\n🔍 Evaluating User-Based CF (n_users={n_users}, k={k})...")
        
        # Initialize CF with training data
        cf = CollaborativeFiltering(self.train_matrix)
        cf.compute_user_similarity()
        
        precisions = []
        users_evaluated = 0
        
        # Sample users for evaluation
        available_users = self.test_matrix.index.intersection(self.train_matrix.index)
        sample_users = np.random.choice(available_users, 
                                       size=min(n_users, len(available_users)), 
                                       replace=False)
        
        for user_id in sample_users:
            try:
                # Get recommendations
                recommendations = cf.recommend_user_based(user_id, n_recommendations=k)
                
                if recommendations:
                    # Extract movie IDs from recommendations
                    rec_movie_ids = []
                    for rec in recommendations:
                        # Find movie ID from title
                        movie_title = rec['movie']
                        movie_match = self.movies_df[self.movies_df['title'] == movie_title]
                        if not movie_match.empty:
                            rec_movie_ids.append(movie_match.iloc[0]['movieId'])
                    
                    # Get actual ratings for this user
                    if user_id in self.test_matrix.index:
                        actual_ratings = self.test_matrix.loc[user_id].dropna()
                        
                        if len(actual_ratings) > 0:
                            precision = self.precision_at_k(rec_movie_ids, actual_ratings, k, threshold)
                            precisions.append(precision)
                            users_evaluated += 1
                            
            except Exception as e:
                print(f"Error evaluating user {user_id}: {e}")
                continue
        
        avg_precision = np.mean(precisions) if precisions else 0.0
        print(f"✓ Evaluated {users_evaluated} users")
        print(f"✓ Average Precision@{k}: {avg_precision:.4f}")
        
        return avg_precision, precisions
    
    def evaluate_item_based_cf(self, n_users=50, k=10, threshold=4.0):
        """Evaluate item-based collaborative filtering"""
        print(f"\n🔍 Evaluating Item-Based CF (n_users={n_users}, k={k})...")
        
        # Initialize CF with training data
        cf = CollaborativeFiltering(self.train_matrix)
        cf.compute_item_similarity()
        
        precisions = []
        users_evaluated = 0
        
        # Sample users for evaluation
        available_users = self.test_matrix.index.intersection(self.train_matrix.index)
        sample_users = np.random.choice(available_users, 
                                       size=min(n_users, len(available_users)), 
                                       replace=False)
        
        for user_id in sample_users:
            try:
                # Get recommendations
                recommendations = cf.recommend_item_based(user_id, n_recommendations=k)
                
                if recommendations:
                    # Extract movie IDs from recommendations
                    rec_movie_ids = []
                    for rec in recommendations:
                        # Find movie ID from title
                        movie_title = rec['movie']
                        movie_match = self.movies_df[self.movies_df['title'] == movie_title]
                        if not movie_match.empty:
                            rec_movie_ids.append(movie_match.iloc[0]['movieId'])
                    
                    # Get actual ratings for this user
                    if user_id in self.test_matrix.index:
                        actual_ratings = self.test_matrix.loc[user_id].dropna()
                        
                        if len(actual_ratings) > 0:
                            precision = self.precision_at_k(rec_movie_ids, actual_ratings, k, threshold)
                            precisions.append(precision)
                            users_evaluated += 1
                            
            except Exception as e:
                print(f"Error evaluating user {user_id}: {e}")
                continue
        
        avg_precision = np.mean(precisions) if precisions else 0.0
        print(f"✓ Evaluated {users_evaluated} users")
        print(f"✓ Average Precision@{k}: {avg_precision:.4f}")
        
        return avg_precision, precisions
    
    def compare_algorithms(self, n_users=50, k=10, threshold=4.0):
        """Compare different recommendation algorithms"""
        print("\n" + "="*60)
        print("ALGORITHM COMPARISON")
        print("="*60)
        
        # Evaluate User-Based CF
        user_precision, user_precisions = self.evaluate_user_based_cf(n_users, k, threshold)
        
        # Evaluate Item-Based CF
        item_precision, item_precisions = self.evaluate_item_based_cf(n_users, k, threshold)
        
        # Results summary
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"User-Based CF Precision@{k}: {user_precision:.4f}")
        print(f"Item-Based CF Precision@{k}: {item_precision:.4f}")
        
        if user_precision > item_precision:
            print(f"🏆 Winner: User-Based CF (+{user_precision - item_precision:.4f})")
        elif item_precision > user_precision:
            print(f"🏆 Winner: Item-Based CF (+{item_precision - user_precision:.4f})")
        else:
            print("🤝 Tie!")
        
        return {
            'user_based': user_precision,
            'item_based': item_precision,
            'user_precisions': user_precisions,
            'item_precisions': item_precisions
        }

def main():
    """Demonstrate evaluation system"""
    from data_preprocessing import MovieDataProcessor
    
    # Load data
    processor = MovieDataProcessor()
    ratings, movies, users = processor.load_data()
    
    # Initialize evaluator
    evaluator = RecommendationEvaluator(ratings, movies)
    
    # Split data
    evaluator.split_data()
    
    # Compare algorithms
    results = evaluator.compare_algorithms(n_users=20, k=5)
    
    return evaluator, results

if __name__ == "__main__":
    evaluator, results = main()
