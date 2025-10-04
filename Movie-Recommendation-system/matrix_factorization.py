"""
Matrix Factorization Implementation
Step 7: SVD Matrix Factorization for recommendations
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MatrixFactorization:
    """Matrix Factorization using SVD for recommendations"""
    
    def __init__(self, user_item_matrix, n_components=50):
        self.user_item_matrix = user_item_matrix
        self.n_components = n_components
        self.svd_model = None
        self.user_factors = None
        self.item_factors = None
        self.reconstructed_matrix = None
        
    def fit_svd(self):
        """Fit SVD model to the user-item matrix"""
        print(f"Fitting SVD with {self.n_components} components...")
        
        # Fill NaN values with 0 for SVD
        matrix_filled = self.user_item_matrix.fillna(0)
        
        # Initialize SVD
        self.svd_model = TruncatedSVD(
            n_components=self.n_components,
            random_state=42
        )
        
        # Fit SVD
        self.user_factors = self.svd_model.fit_transform(matrix_filled)
        self.item_factors = self.svd_model.components_
        
        # Reconstruct the matrix
        self.reconstructed_matrix = np.dot(self.user_factors, self.item_factors)
        
        print(f"✓ SVD fitted successfully")
        print(f"✓ Explained variance ratio: {self.svd_model.explained_variance_ratio_.sum():.4f}")
        
        return self.reconstructed_matrix
    
    def get_user_predictions(self, user_id, n_recommendations=10):
        """Get predictions for a specific user"""
        if self.reconstructed_matrix is None:
            self.fit_svd()
        
        # Get user index
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        
        # Get predictions for this user
        user_predictions = self.reconstructed_matrix[user_idx]
        
        # Get movies already rated by user
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings.dropna().index.tolist()
        
        # Create recommendations
        recommendations = []
        for i, movie in enumerate(self.user_item_matrix.columns):
            if movie not in rated_movies:
                recommendations.append({
                    'movie': movie,
                    'predicted_rating': user_predictions[i],
                    'confidence': abs(user_predictions[i])
                })
        
        # Sort by predicted rating
        recommendations = sorted(recommendations, 
                                key=lambda x: x['predicted_rating'], 
                                reverse=True)
        
        return recommendations[:n_recommendations]
    
    def get_similar_movies(self, movie_title, n_similar=10):
        """Find movies similar to a given movie using SVD factors"""
        if self.item_factors is None:
            self.fit_svd()
        
        # Get movie index
        movie_idx = self.user_item_matrix.columns.get_loc(movie_title)
        
        # Get item factors for this movie
        movie_factors = self.item_factors[:, movie_idx]
        
        # Calculate similarity with all other movies
        similarities = []
        for i, other_movie in enumerate(self.user_item_matrix.columns):
            if other_movie != movie_title:
                other_factors = self.item_factors[:, i]
                # Cosine similarity
                similarity = np.dot(movie_factors, other_factors) / (
                    np.linalg.norm(movie_factors) * np.linalg.norm(other_factors)
                )
                similarities.append({
                    'movie': other_movie,
                    'similarity': similarity
                })
        
        # Sort by similarity
        similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:n_similar]
    
    def evaluate_svd(self, test_matrix, n_users=50, k=10, threshold=4.0):
        """Evaluate SVD model using Precision@K"""
        print(f"\n🔍 Evaluating SVD (n_users={n_users}, k={k})...")
        
        if self.reconstructed_matrix is None:
            self.fit_svd()
        
        precisions = []
        users_evaluated = 0
        
        # Sample users for evaluation
        available_users = test_matrix.index.intersection(self.user_item_matrix.index)
        sample_users = np.random.choice(available_users, 
                                       size=min(n_users, len(available_users)), 
                                       replace=False)
        
        for user_id in sample_users:
            try:
                # Get recommendations
                recommendations = self.get_user_predictions(user_id, n_recommendations=k)
                
                if recommendations:
                    # Extract movie IDs from recommendations
                    rec_movie_ids = []
                    for rec in recommendations:
                        # Find movie ID from title
                        movie_title = rec['movie']
                        # This is a simplified approach - in practice, you'd need movie ID mapping
                        rec_movie_ids.append(movie_title)  # Using title as identifier
                    
                    # Get actual ratings for this user
                    if user_id in test_matrix.index:
                        actual_ratings = test_matrix.loc[user_id].dropna()
                        
                        if len(actual_ratings) > 0:
                            # Calculate precision (simplified)
                            hits = 0
                            for rec_movie in rec_movie_ids:
                                if rec_movie in actual_ratings.index and actual_ratings[rec_movie] >= threshold:
                                    hits += 1
                            
                            precision = hits / k if k > 0 else 0.0
                            precisions.append(precision)
                            users_evaluated += 1
                            
            except Exception as e:
                print(f"Error evaluating user {user_id}: {e}")
                continue
        
        avg_precision = np.mean(precisions) if precisions else 0.0
        print(f"✓ Evaluated {users_evaluated} users")
        print(f"✓ Average Precision@{k}: {avg_precision:.4f}")
        
        return avg_precision, precisions

class AdvancedMatrixFactorization:
    """Advanced Matrix Factorization with Surprise library"""
    
    def __init__(self):
        self.model = None
        self.trainset = None
        
    def fit_surprise_svd(self, ratings_df):
        """Fit SVD using Surprise library"""
        try:
            from surprise import SVD, Dataset, Reader
            from surprise.model_selection import train_test_split as surprise_train_test_split
            
            print("Fitting SVD using Surprise library...")
            
            # Define rating scale
            reader = Reader(rating_scale=(1, 5))
            
            # Load data
            data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
            
            # Split data
            trainset, testset = surprise_train_test_split(data, test_size=0.2)
            self.trainset = trainset
            
            # Initialize and fit SVD
            self.model = SVD(n_factors=50, random_state=42)
            self.model.fit(trainset)
            
            print("✓ SVD fitted with Surprise")
            return True
            
        except ImportError:
            print("Surprise library not available. Install with: pip install surprise")
            return False
        except Exception as e:
            print(f"Error fitting SVD: {e}")
            return False
    
    def predict_rating(self, user_id, movie_id):
        """Predict rating for user-movie pair"""
        if self.model is None:
            return None
        
        try:
            prediction = self.model.predict(user_id, movie_id)
            return prediction.est
        except:
            return None
    
    def get_recommendations(self, user_id, movies_df, n_recommendations=10):
        """Get recommendations for a user"""
        if self.model is None:
            return []
        
        # Get all movies
        all_movies = movies_df['movieId'].unique()
        
        # Predict ratings for all movies
        predictions = []
        for movie_id in all_movies:
            rating = self.predict_rating(user_id, movie_id)
            if rating is not None:
                movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0]
                predictions.append({
                    'movie_id': movie_id,
                    'movie_title': movie_title,
                    'predicted_rating': rating
                })
        
        # Sort by predicted rating
        predictions = sorted(predictions, key=lambda x: x['predicted_rating'], reverse=True)
        
        return predictions[:n_recommendations]

def main():
    """Demonstrate matrix factorization"""
    from data_preprocessing import MovieDataProcessor
    
    # Load data
    processor = MovieDataProcessor()
    processor.load_data()
    user_item_matrix = processor.create_user_item_matrix()
    
    # Initialize SVD
    mf = MatrixFactorization(user_item_matrix, n_components=50)
    
    # Fit SVD
    mf.fit_svd()
    
    # Test recommendations
    test_user = 1
    print(f"\n🎯 SVD RECOMMENDATIONS for User {test_user}:")
    svd_recs = mf.get_user_predictions(test_user, n_recommendations=5)
    for i, rec in enumerate(svd_recs, 1):
        print(f"{i}. {rec['movie']} (Rating: {rec['predicted_rating']:.2f})")
    
    # Test similar movies
    test_movie = "Toy Story (1995)"
    print(f"\n🎯 SIMILAR MOVIES to '{test_movie}':")
    similar_movies = mf.get_similar_movies(test_movie, n_similar=5)
    for i, sim in enumerate(similar_movies, 1):
        print(f"{i}. {sim['movie']} (Similarity: {sim['similarity']:.3f})")
    
    return mf

if __name__ == "__main__":
    mf = main()
