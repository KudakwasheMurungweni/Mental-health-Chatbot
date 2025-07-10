import pandas as pd
import kaggle
import json
import os
from typing import Iterator, Dict, Any
import logging
from ..utils.config import Config

logger = logging.getLogger(__name__)

class ConversationStream:
    """Stream conversation datasets for mental health chatbot training"""
    
    def __init__(self):
        self.config = Config()
        os.environ['KAGGLE_USERNAME'] = self.config.KAGGLE_USERNAME
        os.environ['KAGGLE_KEY'] = self.config.KAGGLE_KEY
    
    def download_conversation_datasets(self) -> list:
        """Download mental health conversation datasets from Kaggle"""
        datasets = [
            "nguyenletruongthien/mental-health",  # Main mental health dataset
        ]
        
        downloaded = []
        for dataset in datasets:
            try:
                kaggle.api.dataset_download_files(dataset, path='./data/raw/', unzip=True)
                downloaded.append(dataset)
                logger.info(f"Downloaded from Kaggle: {dataset}")
            except Exception as e:
                logger.error(f"Failed to download {dataset}: {e}")
        
        return downloaded
    
    def stream_combined_intents(self) -> Iterator[Dict[str, Any]]:
        """Stream combined_intents.json data"""
        batch_size = self.config.STREAMING_BATCH_SIZE
        
        try:
            with open('./data/raw/combined_intents.json', 'r') as f:
                data = json.load(f)
            
            intents = data.get('intents', [])
            logger.info(f"Loaded {len(intents)} intents from combined_intents.json")
            
            # Convert intents to conversation pairs
            conversations = []
            for intent in intents:
                tag = intent.get('tag', '')
                patterns = intent.get('patterns', [])
                responses = intent.get('responses', [])
                source = intent.get('source', 'combined_intents')
                
                # Create conversation pairs from patterns and responses
                for pattern in patterns:
                    for response in responses:
                        conversations.append({
                            'input': pattern,
                            'output': response,
                            'tag': tag,
                            'source': source,
                            'type': 'intent'
                        })
            
            # Stream in batches
            for i in range(0, len(conversations), batch_size):
                batch = conversations[i:i+batch_size]
                
                yield {
                    'data': batch,
                    'batch_id': i // batch_size,
                    'total_batches': len(conversations) // batch_size + 1,
                    'source': 'combined_intents',
                    'total_records': len(conversations)
                }
                
        except Exception as e:
            logger.error(f"Error streaming combined intents data: {e}")
    
    def stream_conversations_training_csv(self) -> Iterator[Dict[str, Any]]:
        """Stream conversations_training.csv data"""
        batch_size = self.config.STREAMING_BATCH_SIZE
        
        try:
            df = pd.read_csv('./data/raw/conversations_training.csv')
            logger.info(f"Loaded {len(df)} conversations from CSV")
            
            # Stream in batches
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                batch_data = []
                for _, row in batch.iterrows():
                    batch_data.append({
                        'input': row['input'],
                        'output': row['output'],
                        'source': 'conversations_training_csv',
                        'type': 'conversation'
                    })
                
                yield {
                    'data': batch_data,
                    'batch_id': i // batch_size,
                    'total_batches': len(df) // batch_size + 1,
                    'source': 'conversations_training_csv',
                    'total_records': len(df)
                }
                
        except Exception as e:
            logger.error(f"Error streaming conversations CSV data: {e}")
    
    def stream_conversations_training_json(self) -> Iterator[Dict[str, Any]]:
        """Stream conversations_training.json data"""
        batch_size = self.config.STREAMING_BATCH_SIZE
        
        try:
            with open('./data/raw/conversations_training.json', 'r') as f:
                conversations = json.load(f)
            
            logger.info(f"Loaded {len(conversations)} conversations from JSON")
            
            # Stream in batches
            for i in range(0, len(conversations), batch_size):
                batch = conversations[i:i+batch_size]
                
                batch_data = []
                for conv in batch:
                    batch_data.append({
                        'input': conv.get('input', ''),
                        'output': conv.get('output', ''),
                        'source': 'conversations_training_json',
                        'type': 'conversation'
                    })
                
                yield {
                    'data': batch_data,
                    'batch_id': i // batch_size,
                    'total_batches': len(conversations) // batch_size + 1,
                    'source': 'conversations_training_json',
                    'total_records': len(conversations)
                }
                
        except Exception as e:
            logger.error(f"Error streaming conversations JSON data: {e}")
    
    def stream_mental_health_conversations(self) -> Iterator[Dict[str, Any]]:
        """Stream mental_health_conversations.csv data"""
        batch_size = self.config.STREAMING_BATCH_SIZE
        
        try:
            df = pd.read_csv('./data/raw/mental_health_conversations.csv')
            logger.info(f"Loaded {len(df)} mental health conversations")
            
            # Stream in batches
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                batch_data = []
                for _, row in batch.iterrows():
                    # Assuming the CSV has columns that can be mapped to input/output
                    batch_data.append({
                        'input': str(row.iloc[0]) if len(row) > 0 else '',
                        'output': str(row.iloc[1]) if len(row) > 1 else '',
                        'source': 'mental_health_conversations',
                        'type': 'mental_health'
                    })
                
                yield {
                    'data': batch_data,
                    'batch_id': i // batch_size,
                    'total_batches': len(df) // batch_size + 1,
                    'source': 'mental_health_conversations',
                    'total_records': len(df)
                }
                
        except Exception as e:
            logger.error(f"Error streaming mental health conversations: {e}")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about conversation datasets"""
        stats = {
            'sources': [],
            'total_conversations': 0
        }
        
        try:
            # Stats for combined_intents.json
            with open('./data/raw/combined_intents.json', 'r') as f:
                data = json.load(f)
                intents_count = len(data.get('intents', []))
                stats['sources'].append({
                    'name': 'combined_intents',
                    'intents': intents_count,
                    'type': 'intent_based'
                })
        except:
            pass
        
        try:
            # Stats for conversations_training.csv
            df = pd.read_csv('./data/raw/conversations_training.csv')
            stats['sources'].append({
                'name': 'conversations_training_csv',
                'conversations': len(df),
                'type': 'direct_conversations'
            })
            stats['total_conversations'] += len(df)
        except:
            pass
        
        try:
            # Stats for conversations_training.json
            with open('./data/raw/conversations_training.json', 'r') as f:
                conversations = json.load(f)
                stats['sources'].append({
                    'name': 'conversations_training_json',
                    'conversations': len(conversations),
                    'type': 'direct_conversations'
                })
                stats['total_conversations'] += len(conversations)
        except:
            pass
        
        return stats