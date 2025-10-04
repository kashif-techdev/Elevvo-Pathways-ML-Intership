# 🎬 Movie Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Professional AI-Powered Movie Recommendations**

*Built with modern UI, advanced algorithms, and beautiful visualizations*

[🚀 Quick Start](#-quick-start) • [📊 Features](#-features) • [🎯 Demo](#-demo) • [📚 Documentation](#-documentation)

</div>

---

A comprehensive movie recommendation system built from scratch using collaborative filtering and matrix factorization techniques. This project implements multiple recommendation algorithms and provides a **professional, interactive Streamlit demo** with beautiful UI, animations, and real-time visualizations.

## 🚀 Features

- **User-Based Collaborative Filtering**: Find similar users and recommend movies they liked
- **Item-Based Collaborative Filtering**: Find similar movies to those a user has rated highly
- **SVD Matrix Factorization**: Use matrix decomposition to capture latent user and movie features
- **Precision@K Evaluation**: Comprehensive evaluation system with train/test split
- **Interactive Demo**: Streamlit web app for easy testing and visualization
- **Data Visualization**: Rating distributions, genre analysis, and more

## 📊 Dataset

- **MovieLens 100K Dataset**: 943 users, 1,682 movies, 100,000 ratings
- **Time Period**: 1990-1998 (mostly 90s movies)
- **Rating Scale**: 1-5 stars
- **Sparsity**: 93.65% (typical for recommendation systems)

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone 
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download dataset** (automatically):
```bash
python download_dataset.py
```

## 🎯 Usage

### 1. Data Preprocessing
```bash
python data_preprocessing.py
```
- Loads and inspects the MovieLens 100K dataset
- Creates user-item rating matrix
- Provides comprehensive data statistics

### 2. Collaborative Filtering
```bash
python collaborative_filtering.py
```
- Implements user-based and item-based collaborative filtering
- Uses cosine similarity for user and item comparisons
- Generates personalized recommendations

### 3. Matrix Factorization
```bash
python matrix_factorization.py
```
- Implements SVD (Singular Value Decomposition)
- Captures latent features of users and movies
- Provides movie similarity analysis

### 4. Evaluation
```bash
python evaluation.py
```
- Evaluates algorithms using Precision@K
- Compares different recommendation approaches
- Provides performance metrics

### 5. Interactive Demo
```bash
streamlit run app.py
```
- Launch the Streamlit web application
- Interactive recommendation interface
- Data visualizations and statistics

## 🔧 Algorithm Details

### User-Based Collaborative Filtering
1. **Compute User Similarity**: Calculate cosine similarity between users
2. **Find Similar Users**: Identify top-N most similar users
3. **Generate Recommendations**: Weight ratings by user similarity
4. **Filter and Rank**: Remove watched movies, sort by predicted rating

### Item-Based Collaborative Filtering
1. **Compute Item Similarity**: Calculate cosine similarity between movies
2. **Find Similar Items**: For each rated movie, find similar movies
3. **Generate Recommendations**: Weight ratings by item similarity
4. **Aggregate and Rank**: Combine scores, sort by predicted rating

### SVD Matrix Factorization
1. **Matrix Decomposition**: Factor user-item matrix into user and item factors
2. **Latent Features**: Capture hidden patterns in user preferences
3. **Reconstruction**: Predict ratings using factorized matrices
4. **Recommendation**: Rank movies by predicted ratings

## 📈 Evaluation Metrics

- **Precision@K**: Fraction of recommended items that are relevant
- **Train/Test Split**: 80/20 split for proper evaluation
- **Threshold**: Ratings ≥ 4.0 considered relevant
- **Cross-Validation**: Multiple user samples for robust evaluation

## 🎨 Streamlit App Features

### Main Interface
- **Algorithm Selection**: Choose between User-Based CF, Item-Based CF, or SVD
- **User Input**: Select user ID and number of recommendations
- **Real-time Results**: Instant recommendation generation
- **Visual Feedback**: Progress indicators and success messages

### Data Visualization
- **Rating Distribution**: Histogram of user ratings
- **Genre Analysis**: Bar chart of movies per genre
- **Dataset Statistics**: Key metrics and sparsity information
- **Algorithm Comparison**: Performance evaluation results

### Interactive Controls
- **User ID Slider**: Select any user (1-943)
- **Recommendation Count**: Adjust number of recommendations (5-20)
- **Similarity Threshold**: Control algorithm parameters
- **Evaluation Button**: Run performance analysis

## 📁 Project Structure

```
Movie-Recommendation-system/
├── data/
│   └── ml-100k/           # MovieLens 100K dataset
├── data_preprocessing.py  # Data loading and preprocessing
├── collaborative_filtering.py  # User-based and item-based CF
├── matrix_factorization.py    # SVD implementation
├── evaluation.py          # Precision@K evaluation
├── app.py                 # Streamlit demo app
├── download_dataset.py    # Dataset downloader
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔬 Technical Implementation

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms

### Recommendation Algorithms
- **Cosine Similarity**: For user and item comparisons
- **Matrix Factorization**: SVD for latent feature extraction
- **Collaborative Filtering**: User-based and item-based approaches

### Evaluation Framework
- **Train/Test Split**: Proper model evaluation
- **Precision@K**: Standard recommendation metrics
- **Cross-Validation**: Robust performance assessment

## 🎯 Key Results

- **Dataset**: 943 users, 1,682 movies, 100,000 ratings
- **Sparsity**: 93.65% (typical for recommendation systems)
- **Algorithms**: 3 different approaches implemented
- **Evaluation**: Comprehensive Precision@K analysis
- **Demo**: Interactive Streamlit application

## 🚀 Getting Started

1. **Run the complete pipeline**:
```bash
# Download dataset
python download_dataset.py

# Explore data
python data_preprocessing.py

# Test algorithms
python collaborative_filtering.py
python matrix_factorization.py

# Evaluate performance
python evaluation.py

# Launch demo app
streamlit run app.py
```

2. **Interactive Demo**: Open http://localhost:8501 in your browser
3. **Screen shots**:
   <br>
    ===>(Home Screen)<===
   <img width="1920" height="902" alt="image" src="https://github.com/user-attachments/assets/a19a167f-2f74-420b-b5d9-03f70ade7a9c" />
   <br>
   ===>(sidebar to apply the filters)<===
   <img width="930" height="858" alt="image" src="https://github.com/user-attachments/assets/4f43801c-e4b2-4fca-963b-f4321627b073" />
   <br>
    ===>(Let's see recommendation)<===
    <img width="1919" height="894" alt="image" src="https://github.com/user-attachments/assets/0d356cdf-8245-47b4-8162-b573e341e124" />
    <br>
    ===>(Algorithm Evualvation)<===
    <img width="1920" height="891" alt="image" src="https://github.com/user-attachments/assets/f34b1e0d-661f-458a-85cf-154bf1cd5252" />
    <br>
    ===>(Performance Evalvuation)<===
    <img width="1918" height="914" alt="image" src="https://github.com/user-attachments/assets/ddfc220a-2e2b-4225-b39e-6eba1e0e6f54" />
    <br>
    ===>(Data Visualization)<===
    <img width="1920" height="921" alt="image" src="https://github.com/user-attachments/assets/25211d33-1d04-4869-a1ab-7e4c5cf73929" />
    <br>
    ===>(rating Distribuition)<===
    <img width="1318" height="903" alt="image" src="https://github.com/user-attachments/assets/3e3cf31b-ba91-46f7-bfb5-619b47fab6ac" />
    <br>
    ===>(genre Analysis)<===
    <img width="1917" height="837" alt="image" src="https://github.com/user-attachments/assets/4bad1577-13a3-4326-9175-57c468dcd578" />
    <br>
    ===>(Dataset overview)<===   
    <img width="1917" height="882" alt="image" src="https://github.com/user-attachments/assets/a31fbf49-0b5d-4bde-bd4c-064a4adbe4ff" />

5. **Experiment**: Try different users, algorithms, and parameters

## 📚 Learning Outcomes

This project demonstrates:
- **Data Preprocessing**: Handling sparse matrices and missing values
- **Collaborative Filtering**: User-based and item-based approaches
- **Matrix Factorization**: SVD for recommendation systems
- **Evaluation**: Proper train/test splits and metrics
- **Visualization**: Interactive data exploration
- **Deployment**: Streamlit web application

## 🔮 Future Enhancements

- **Content-Based Filtering**: Use movie genres and features
- **Hybrid Approaches**: Combine multiple algorithms
- **Deep Learning**: Neural collaborative filtering
- **Real-time Updates**: Online learning capabilities
- **Scalability**: Handle larger datasets efficiently

## 📄 License

This project is part of the Elevvo Pathways Machine Learning Internship.

---

**Built with ❤️ using Python, Streamlit, and Scikit-learn**
