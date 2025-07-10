from typing import Iterator, Dict, Any
import logging
from .sentiment_stream import SentimentStream
from .conversation_stream import ConversationStream
from ..utils.config import Config

logger = logging.getLogger(__name__)

class UnifiedStream:
    """Unified streaming interface for all mental health data sources"""
    
    def __init__(self):
        self.config = Config()
        self.sentiment_stream = SentimentStream()
        self.conversation_stream = ConversationStream()
    
    def setup_data_sources(self) -> Dict[str, bool]:
        """Download and setup all data sources"""
        setup_status = {}
        
        logger.info("Setting up emotion data sources...")
        emotion_datasets = self.sentiment_stream.download_emotion_datasets()
        setup_status['emotions'] = len(emotion_datasets) > 0
        
        logger.info("Setting up conversation data sources...")
        conversation_datasets = self.conversation_stream.download_conversation_datasets()
        setup_status['conversations'] = len(conversation_datasets) > 0
        
        return setup_status
    
    def stream_all_data(self, data_type: str = 'all') -> Iterator[Dict[str, Any]]:
        """Stream all data with unified format"""
        
        if data_type in ['all', 'emotions', 'sentiment']:
            logger.info("Streaming emotion data...")
            
            # Stream emotion dataset
            for batch in self.sentiment_stream.stream_emotion_data():
                # Unify format for emotions
                unified_batch = {
                    'data': [
                        {
                            'text': item['text'],
                            'label_numeric': item['label_numeric'],
                            'label_name': item['label_name'],
                            'type': 'emotion',
                            'source': item['source']
                        }
                        for item in batch['data']
                    ],
                    'batch_info': {
                        'batch_id': batch['batch_id'],
                        'total_batches': batch['total_batches'],
                        'source': batch['source'],
                        'data_type': 'emotion',
                        'total_records': batch['total_records']
                    }
                }
                yield unified_batch
        
        if data_type in ['all', 'conversations']:
            logger.info("Streaming conversation data...")
            
            # Stream combined intents
            for batch in self.conversation_stream.stream_combined_intents():
                unified_batch = {
                    'data': [
                        {
                            'input': item['input'],
                            'output': item['output'],
                            'tag': item.get('tag', ''),
                            'type': item.get('type', 'conversation'),
                            'source': item['source']
                        }
                        for item in batch['data']
                    ],
                    'batch_info': {
                        'batch_id': batch['batch_id'],
                        'total_batches': batch['total_batches'],
                        'source': batch['source'],
                        'data_type': 'conversation',
                        'total_records': batch['total_records']
                    }
                }
                yield unified_batch
            
            # Stream training conversations CSV
            for batch in self.conversation_stream.stream_conversations_training_csv():
                unified_batch = {
                    'data': [
                        {
                            'input': item['input'],
                            'output': item['output'],
                            'type': item.get('type', 'conversation'),
                            'source': item['source']
                        }
                        for item in batch['data']
                    ],
                    'batch_info': {
                        'batch_id': batch['batch_id'],
                        'total_batches': batch['total_batches'],
                        'source': batch['source'],
                        'data_type': 'conversation',
                        'total_records': batch['total_records']
                    }
                }
                yield unified_batch
            
            # Stream training conversations JSON
            for batch in self.conversation_stream.stream_conversations_training_json():
                unified_batch = {
                    'data': [
                        {
                            'input': item['input'],
                            'output': item['output'],
                            'type': item.get('type', 'conversation'),
                            'source': item['source']
                        }
                        for item in batch['data']
                    ],
                    'batch_info': {
                        'batch_id': batch['batch_id'],
                        'total_batches': batch['total_batches'],
                        'source': batch['source'],
                        'data_type': 'conversation',
                        'total_records': batch['total_records']
                    }
                }
                yield unified_batch
            
            # Stream mental health conversations
            for batch in self.conversation_stream.stream_mental_health_conversations():
                unified_batch = {
                    'data': [
                        {
                            'input': item['input'],
                            'output': item['output'],
                            'type': item.get('type', 'mental_health'),
                            'source': item['source']
                        }
                        for item in batch['data']
                    ],
                    'batch_info': {
                        'batch_id': batch['batch_id'],
                        'total_batches': batch['total_batches'],
                        'source': batch['source'],
                        'data_type': 'conversation',
                        'total_records': batch['total_records']
                    }
                }
                yield unified_batch
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get statistics about all available streams"""
        emotion_stats = self.sentiment_stream.get_emotion_stats()
        conversation_stats = self.conversation_stream.get_conversation_stats()
        
        return {
            'emotion_data': emotion_stats,
            'conversation_data': conversation_stats,
            'available_sources': [
                'emotions_dataset',
                'combined_intents',
                'conversations_training_csv',
                'conversations_training_json',
                'mental_health_conversations'
            ],
            'data_types': ['emotion', 'conversation', 'intent', 'mental_health'],
            'batch_size': self.config.STREAMING_BATCH_SIZE,
            'total_sources': 5
        }