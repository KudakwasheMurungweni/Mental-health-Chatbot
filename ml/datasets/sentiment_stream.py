import pandas as pd
import kaggle
import os
from typing import Iterator, Dict, Any
import logging
from ..utils.config import Config

logger = logging.getLogger(__name__)

class SentimentStream:
    """Stream sentiment analysis datasets from Kaggle"""
    
    def __init__(self):
        self.config = Config()
        # Set Kaggle credentials
        os.environ['KAGGLE_USERNAME'] = self.config.KAGGLE_USERNAME
        os.environ['KAGGLE_KEY'] = self.config.KAGGLE_KEY
        
        # Emotion label mapping
        self.emotion_labels = {
            0: "sadness",
            1: "joy", 
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise"
        }
    
    def download_emotion_datasets(self) -> list:
        """Download emotion datasets from Kaggle"""
        datasets = [
            "bhavikjikadara/emotions-dataset",  # Main emotions dataset
        ]
        
        downloaded = []
        for dataset in datasets:
            try:
                kaggle.api.dataset_download_files(dataset, path='./data/raw/', unzip=True)
                downloaded.append(dataset)
                logger.info(f"Downloaded: {dataset}")
            except Exception as e:
                logger.error(f"Failed to download {dataset}: {e}")
        
        return downloaded
    
    def stream_emotion_data(self) -> Iterator[Dict[str, Any]]:
        """Stream emotion data in batches"""
        batch_size = self.config.STREAMING_BATCH_SIZE
        
        try:
            # Load emotions dataset - assuming it has 'Text' and 'label' columns
            df = pd.read_csv('./data/raw/Emotion_final.csv')
            
            # Convert numeric labels to emotion names
            if 'label' in df.columns:
                df['emotion_name'] = df['label'].map(self.emotion_labels)
            
            logger.info(f"Loaded emotions dataset with {len(df)} records")
            
            # Stream in batches
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                batch_data = []
                for _, row in batch.iterrows():
                    batch_data.append({
                        'text': row['Text'],
                        'label_numeric': row['label'],
                        'label_name': row.get('emotion_name', 'unknown'),
                        'source': 'emotions_dataset'
                    })
                
                yield {
                    'data': batch_data,
                    'batch_id': i // batch_size,
                    'total_batches': len(df) // batch_size + 1,
                    'source': 'emotions_dataset',
                    'total_records': len(df)
                }
                
        except Exception as e:
            logger.error(f"Error streaming emotion data: {e}")
    
    def get_emotion_stats(self) -> Dict[str, Any]:
        """Get statistics about emotion dataset"""
        try:
            df = pd.read_csv('./data/raw/Emotion_final.csv')
            df['emotion_name'] = df['label'].map(self.emotion_labels)
            
            stats = {
                'total_records': len(df),
                'emotion_distribution': df['emotion_name'].value_counts().to_dict(),
                'avg_text_length': df['Text'].str.len().mean(),
                'emotion_labels': self.emotion_labels
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting emotion stats: {e}")
            return {}