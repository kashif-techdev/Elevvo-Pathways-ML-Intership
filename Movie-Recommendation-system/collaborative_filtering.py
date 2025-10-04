"""
Collaborative Filtering Implementation
Step 3 & 4: User-based and Item-based Collaborative Filtering
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class CollaborativeFiltering:
    """User-based and Item-based Collaborative Filtering"""
    
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        
    def compute_user_similarity(self, method='cosine'):
        """Compute user-user similarity matrix"""
        print(f"Computing user similarity using {method}...")
        
        # Fill NaN values with 0 for similarity computation
        matrix_filled = self.user_item_matrix.fillna(0)
        
        if method == 'cosine':
            self.user_similarity_matrix = cosine_similarity(matrix_filled)
        else:
            # Pearson correlation
            self.user_similarity_matrix = matrix_filled.T.corr().values
        
        print(f"✓ User similarity matrix shape: {self.user_similarity_matrix.shape}")
        return self.user_similarity_matrix
    
    def compute_item_similarity(self, method='cosine'):
        """Compute item-item similarity matrix"""
        print(f"Computing item similarity using {method}...")
        
        # Fill NaN values with 0 for similarity computation
        matrix_filled = self.user_item_matrix.fillna(0)
        
        if method == 'cosine':
            self.item_similarity_matrix = cosine_similarity(matrix_filled.T)
        else:
            # Pearson correlation
            self.item_similarity_matrix = matrix_filled.corr().values
        
        print(f"✓ Item similarity matrix shape: {self.item_similarity_matrix.shape}")
        return self.item_similarity_matrix
    
    def get_similar_users(self, user_id, n_similar=10):
        """Get top-N similar users for a given user"""
        if self.user_similarity_matrix is None:
            self.compute_user_similarity()
        
        # Get user index
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        
        # Get similarity scores for this user
        user_similarities = self.user_similarity_matrix[user_idx]
        
        # Create DataFrame with user IDs and similarities
        similar_users_df = pd.DataFrame({
            'user_id': self.user_item_matrix.index,
            'similarity': user_similarities
        })
        
        # Sort by similarity (descending) and exclude self
        similar_users_df = similar_users_df[similar_users_df['user_id'] != user_id]
        similar_users_df = similar_users_df.sort_values('similarity', ascending=False)
        
        return similar_users_df.head(n_similar)
    
    def get_similar_items(self, movie_title, n_similar=10):
        """Get top-N similar movies for a given movie"""
        if self.item_similarity_matrix is None:
            self.compute_item_similarity()
        
        # Get movie index
        movie_idx = self.user_item_matrix.columns.get_loc(movie_title)
        
        # Get similarity scores for this movie
        movie_similarities = self.item_similarity_matrix[movie_idx]
        
        # Create DataFrame with movie titles and similarities
        similar_movies_df = pd.DataFrame({
            'movie_title': self.user_item_matrix.columns,
            'similarity': movie_similarities
        })
        
        # Sort by similarity (descending) and exclude self
        similar_movies_df = similar_movies_df[similar_movies_df['movie_title'] != movie_title]
        similar_movies_df = similar_movies_df.sort_values('similarity', ascending=False)
        
        return similar_movies_df.head(n_similar)
    
    def recommend_user_based(self, user_id, n_recommendations=10, n_similar_users=20):
        """Generate recommendations using user-based collaborative filtering"""
        print(f"Generating user-based recommendations for user {user_id}...")
        
        # Get similar users
        similar_users = self.get_similar_users(user_id, n_similar_users)
        
        # Get movies rated by similar users but not by target user
        target_user_ratings = self.user_item_matrix.loc[user_id]
        watched_movies = target_user_ratings.dropna().index.tolist()
        
        # Calculate weighted average ratings
        movie_scores = {}
        
        for _, similar_user in similar_users.iterrows():
            sim_user_id = similar_user['user_id']
            similarity = similar_user['similarity']
            
            # Get ratings from similar user
            sim_user_ratings = self.user_item_matrix.loc[sim_user_id]
            
            for movie in sim_user_ratings.index:
                if pd.notna(sim_user_ratings[movie]) and movie not in watched_movies:
                    if movie not in movie_scores:
                        movie_scores[movie] = {'weighted_sum': 0, 'weight_sum': 0}
                    
                    movie_scores[movie]['weighted_sum'] += similarity * sim_user_ratings[movie]
                    movie_scores[movie]['weight_sum'] += abs(similarity)
        
        # Calculate predicted ratings
        recommendations = []
        for movie, scores in movie_scores.items():
            if scores['weight_sum'] > 0:
                predicted_rating = scores['weighted_sum'] / scores['weight_sum']
                recommendations.append({
                    'movie': movie,
                    'predicted_rating': predicted_rating,
                    'confidence': scores['weight_sum']
                })
        
        # Sort by predicted rating
        recommendations = sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def recommend_item_based(self, user_id, n_recommendations=10):
        """Generate recommendations using item-based collaborative filtering"""
        print(f"Generating item-based recommendations for user {user_id}...")
        
        # Get movies rated by the user
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings.dropna()
        
        if len(rated_movies) == 0:
            return []
        
        # Calculate predicted ratings for unrated movies
        movie_scores = {}
        
        for movie in self.user_item_matrix.columns:
            if pd.isna(user_ratings[movie]):  # Unrated movie
                score = 0
                weight_sum = 0
                
                for rated_movie, rating in rated_movies.items():
                    # Get similarity between movies
                    movie_idx = self.user_item_matrix.columns.get_loc(movie)
                    rated_movie_idx = self.user_item_matrix.columns.get_loc(rated_movie)
                    
                    if self.item_similarity_matrix is None:
                        self.compute_item_similarity()
                    
                    similarity = self.item_similarity_matrix[movie_idx, rated_movie_idx]
                    
                    score += similarity * rating
                    weight_sum += abs(similarity)
                
                if weight_sum > 0:
                    predicted_rating = score / weight_sum
                    movie_scores[movie] = {
                        'predicted_rating': predicted_rating,
                        'confidence': weight_sum
                    }
        
        # Sort by predicted rating
        recommendations = sorted(movie_scores.items(), 
                                key=lambda x: x[1]['predicted_rating'], 
                                reverse=True)
        
        return [{'movie': movie, **scores} for movie, scores in recommendations[:n_recommendations]]

def main():
    """Demonstrate collaborative filtering"""
    from data_preprocessing import MovieDataProcessor
    
    # Load data
    processor = MovieDataProcessor()
    processor.load_data()
    user_item_matrix = processor.create_user_item_matrix()
    
    # Initialize collaborative filtering
    cf = CollaborativeFiltering(user_item_matrix)
    
    # Test user-based recommendations
    test_user = 1
    print(f"\n🎯 USER-BASED RECOMMENDATIONS for User {test_user}:")
    user_recs = cf.recommend_user_based(test_user, n_recommendations=5)
    for i, rec in enumerate(user_recs, 1):
        print(f"{i}. {rec['movie']} (Rating: {rec['predicted_rating']:.2f})")
    
    # Test item-based recommendations
    print(f"\n🎯 ITEM-BASED RECOMMENDATIONS for User {test_user}:")
    item_recs = cf.recommend_item_based(test_user, n_recommendations=5)
    for i, rec in enumerate(item_recs, 1):
        print(f"{i}. {rec['movie']} (Rating: {rec['predicted_rating']:.2f})")
    
    return cf

if __name__ == "__main__":
    cf = main()
