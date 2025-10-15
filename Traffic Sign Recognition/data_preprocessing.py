"""
Traffic Sign Recognition - Data Preprocessing Module
Handles data loading, preprocessing, and augmentation for GTSRB dataset
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns

class TrafficSignDataProcessor:
    """Handles all data preprocessing for traffic sign recognition"""
    
    def __init__(self, img_size=(64, 64)):
        self.img_size = img_size
        self.label_encoder = LabelEncoder()
        self.num_classes = 43
        
    def load_data_from_csv(self, csv_path, images_path):
        """
        Load data from GTSRB CSV file and corresponding images
        
        Args:
            csv_path: Path to the CSV file with labels
            images_path: Path to the images directory
            
        Returns:
            X: Array of preprocessed images
            y: Array of labels
        """
        print("Loading data from CSV and images...")
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        X = []
        y = []
        
        for idx, row in df.iterrows():
            # Load image
            img_path = os.path.join(images_path, row['Filename'])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # Preprocess image
                    img = self.preprocess_image(img)
                    X.append(img)
                    y.append(row['ClassId'])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Loaded {len(X)} images with shape {X.shape}")
        return X, y
    
    def preprocess_image(self, img):
        """
        Preprocess a single image
        
        Args:
            img: Input image (BGR format from OpenCV)
            
        Returns:
            Preprocessed image (RGB, resized, normalized)
        """
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, self.img_size)
        
        # Normalize pixel values to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Features
            y: Labels
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def encode_labels(self, y_train, y_val, y_test):
        """
        Encode labels to categorical format
        
        Args:
            y_train, y_val, y_test: Label arrays
            
        Returns:
            Encoded label arrays
        """
        # Fit label encoder on training data
        self.label_encoder.fit(y_train)
        
        # Transform all label sets
        y_train_encoded = to_categorical(y_train, num_classes=self.num_classes)
        y_val_encoded = to_categorical(y_val, num_classes=self.num_classes)
        y_test_encoded = to_categorical(y_test, num_classes=self.num_classes)
        
        return y_train_encoded, y_val_encoded, y_test_encoded
    
    def create_augmentation_generator(self, X_train, y_train, batch_size=32):
        """
        Create data augmentation generator
        
        Args:
            X_train: Training images
            y_train: Training labels
            batch_size: Batch size for generator
            
        Returns:
            Data generator with augmentation
        """
        datagen = ImageDataGenerator(
            rotation_range=15,          # Random rotation
            width_shift_range=0.1,      # Random horizontal shift
            height_shift_range=0.1,     # Random vertical shift
            zoom_range=0.1,             # Random zoom
            shear_range=0.1,            # Random shear
            horizontal_flip=False,      # Don't flip traffic signs
            fill_mode='nearest'         # Fill empty pixels
        )
        
        return datagen.flow(X_train, y_train, batch_size=batch_size)
    
    def visualize_data_distribution(self, y_train, y_val, y_test):
        """
        Visualize the distribution of classes in the dataset
        
        Args:
            y_train, y_val, y_test: Label arrays
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Training set distribution
        unique, counts = np.unique(y_train, return_counts=True)
        axes[0].bar(unique, counts)
        axes[0].set_title('Training Set Distribution')
        axes[0].set_xlabel('Class ID')
        axes[0].set_ylabel('Count')
        
        # Validation set distribution
        unique, counts = np.unique(y_val, return_counts=True)
        axes[1].bar(unique, counts)
        axes[1].set_title('Validation Set Distribution')
        axes[1].set_xlabel('Class ID')
        axes[1].set_ylabel('Count')
        
        # Test set distribution
        unique, counts = np.unique(y_test, return_counts=True)
        axes[2].bar(unique, counts)
        axes[2].set_title('Test Set Distribution')
        axes[2].set_xlabel('Class ID')
        axes[2].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_sample_images(self, X, y, class_names=None, num_samples=16):
        """
        Visualize sample images from the dataset
        
        Args:
            X: Image array
            y: Label array
            class_names: Optional class names
            num_samples: Number of samples to display
        """
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(num_samples):
            idx = np.random.randint(0, len(X))
            axes[i].imshow(X[idx])
            axes[i].set_title(f'Class: {y[idx]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_class_names(self):
        """
        Get the class names for GTSRB dataset
        Returns a dictionary mapping class IDs to names
        """
        class_names = {
            0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
            3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
            6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
            9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
            11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield',
            14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
            17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
            20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
            23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
            26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
            29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
            32: 'End of all speed and passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
            35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left',
            38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory',
            41: 'End of no passing', 42: 'End of no passing by vehicles over 3.5 metric tons'
        }
        return class_names
