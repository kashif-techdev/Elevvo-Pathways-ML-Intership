"""
Data Preprocessing for Movie Recommendation System
Step 2: Load data, merge datasets, create user-item matrix
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MovieDataProcessor:
    """Handle data loading and preprocessing for MovieLens dataset"""
    
    def __init__(self, data_path="data/ml-100k"):
        self.data_path = Path(data_path)
        self.ratings = None
        self.movies = None
        self.users = None
        self.user_item_matrix = None
        
    def load_data(self):
        """Load MovieLens 100K dataset"""
        print("Loading MovieLens 100K dataset...")
        
        # Load ratings data
        self.ratings = pd.read_csv(
            self.data_path / "u.data",
            sep='\t',
            names=['userId', 'movieId', 'rating', 'timestamp']
        )
        
        # Load movies data
        self.movies = pd.read_csv(
            self.data_path / "u.item",
            sep='|',
            encoding='latin-1',
            names=['movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                   'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        )
        
        # Load users data
        self.users = pd.read_csv(
            self.data_path / "u.user",
            sep='|',
            names=['userId', 'age', 'gender', 'occupation', 'zipcode']
        )
        
        print(f"✓ Loaded {len(self.ratings)} ratings")
        print(f"✓ Loaded {len(self.movies)} movies")
        print(f"✓ Loaded {len(self.users)} users")
        
        return self.ratings, self.movies, self.users
    
    def inspect_data(self):
        """Inspect the loaded datasets"""
        print("\n" + "="*50)
        print("DATASET INSPECTION")
        print("="*50)
        
        # Ratings info
        print("\n📊 RATINGS DATASET:")
        print(f"Shape: {self.ratings.shape}")
        print(f"Columns: {list(self.ratings.columns)}")
        print(f"Rating range: {self.ratings['rating'].min()} - {self.ratings['rating'].max()}")
        print(f"Unique users: {self.ratings['userId'].nunique()}")
        print(f"Unique movies: {self.ratings['movieId'].nunique()}")
        print(f"Average rating: {self.ratings['rating'].mean():.2f}")
        
        # Movies info
        print("\n🎬 MOVIES DATASET:")
        print(f"Shape: {self.movies.shape}")
        print(f"Sample titles: {self.movies['title'].head(3).tolist()}")
        
        # Users info
        print("\n👥 USERS DATASET:")
        print(f"Shape: {self.users.shape}")
        print(f"Age range: {self.users['age'].min()} - {self.users['age'].max()}")
        print(f"Gender distribution: {self.users['gender'].value_counts().to_dict()}")
        
        # Rating distribution
        print("\n📈 RATING DISTRIBUTION:")
        rating_counts = self.ratings['rating'].value_counts().sort_index()
        for rating, count in rating_counts.items():
            percentage = (count / len(self.ratings)) * 100
            print(f"Rating {rating}: {count:,} ({percentage:.1f}%)")
    
    def create_user_item_matrix(self):
        """Create user-item rating matrix"""
        print("\nCreating user-item matrix...")
        
        # Merge ratings with movies to get movie titles
        ratings_with_titles = self.ratings.merge(
            self.movies[['movieId', 'title']], 
            on='movieId'
        )
        
        # Create pivot table: users as rows, movies as columns, ratings as values
        self.user_item_matrix = ratings_with_titles.pivot_table(
            index='userId',
            columns='title',
            values='rating'
        )
        
        print(f"✓ User-item matrix shape: {self.user_item_matrix.shape}")
        print(f"✓ Sparsity: {(1 - self.user_item_matrix.count().sum() / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])) * 100:.2f}%")
        
        return self.user_item_matrix
    
    def get_data_summary(self):
        """Get comprehensive data summary"""
        summary = {
            'n_users': self.user_item_matrix.shape[0],
            'n_movies': self.user_item_matrix.shape[1],
            'n_ratings': self.ratings.shape[0],
            'sparsity': (1 - self.user_item_matrix.count().sum() / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])) * 100,
            'avg_rating': self.ratings['rating'].mean(),
            'rating_std': self.ratings['rating'].std()
        }
        return summary

def main():
    """Main function to demonstrate data preprocessing"""
    # Initialize processor
    processor = MovieDataProcessor()
    
    # Load data
    ratings, movies, users = processor.load_data()
    
    # Inspect data
    processor.inspect_data()
    
    # Create user-item matrix
    user_item_matrix = processor.create_user_item_matrix()
    
    # Get summary
    summary = processor.get_data_summary()
    print(f"\n📋 SUMMARY:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return processor

if __name__ == "__main__":
    processor = main()
