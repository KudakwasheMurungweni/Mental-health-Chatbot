import re
import pandas as pd
from typing import Dict, Any, List
import logging
from concurrent.futures import ThreadPoolExecutor
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from ..utils.config import Config

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

class StreamPreprocessor:
    """Preprocess streaming data for mental health chatbot"""
    
    def __init__(self):
        self.config = Config()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        # Mental health specific cleaning patterns
        self.mental_health_patterns = {
            'remove_personal_info': r'\b(?:my name is|i am|i\'m)\s+\w+\b',
            'normalize_contractions': {
                "don't": "do not",
                "won't": "will not",
                "can't": "cannot",
                "n't": " not",
                "'m": " am",
                "'re": " are",
                "'ve": " have",
                "'ll": " will",
                "'d": " would"
            }
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for mental health context"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove personal information patterns
        text = re.sub(self.mental_health_patterns['remove_personal_info'], '', text)
        
        # Normalize contractions
        for contraction, expansion in self.mental_health_patterns['normalize_contractions'].items():
            text = text.replace(contraction, expansion)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep some punctuation for emotion context
        text = re.sub(r'[^\w\s\.\!\?\,\:\;\-]', '', text)
        
        # Remove very short words (less than 2 characters) except important ones
        important_short = {'i', 'me', 'my', 'no', 'ok', 'hi', 'go', 'do', 'am', 'is', 'to', 'or', 'if', 'so'}
        words = text.split()
        words = [word for word in words if len(word) >= 2 or word in important_short]
        text = ' '.join(words)
        
        return text
    
    def preprocess_emotion_data(self, batch_data: List[Dict]) -> List[Dict]:
        """Preprocess emotion/sentiment data"""
        processed = []
        
        for item in batch_data:
            text = item.get('text', '')
            cleaned_text = self.clean_text(text)
            
            # Filter out very short or very long texts
            word_count = len(cleaned_text.split())
            if 3 <= word_count <= 100:  # Reasonable length for emotion classification
                processed.append({
                    'text': cleaned_text,
                    'label_numeric': item.get('label_numeric', -1),
                    'label_name': item.get('label_name', 'unknown'),
                    'type': item.get('type', 'emotion'),
                    'source': item.get('source', ''),
                    'word_count': word_count,
                    'char_count': len(cleaned_text),
                    'original_length': len(item.get('text', ''))
                })
        
        return processed
    
    def preprocess_conversation_data(self, batch_data: List[Dict]) -> List[Dict]:
        """Preprocess conversation data"""
        processed = []
        
        for item in batch_data:
            input_text = item.get('input', '')
            output_text = item.get('output', '')
            
            cleaned_input = self.clean_text(input_text)
            cleaned_output = self.clean_text(output_text)
            
            # Filter conversations based on quality criteria
            input_words = len(cleaned_input.split())
            output_words = len(cleaned_output.split())
            
            # Quality filters for mental health conversations
            if (3 <= input_words <= 150 and 
                5 <= output_words <= 200 and
                len(cleaned_input.strip()) > 0 and 
                len(cleaned_output.strip()) > 0):
                
                processed.append({
                    'input': cleaned_input,
                    'output': cleaned_output,
                    'tag': item.get('tag', ''),
                    'type': item.get('type', 'conversation'),
                    'source': item.get('source', ''),
                    'input_word_count': input_words,
                    'output_word_count': output_words,
                    'conversation_quality_score': min(input_words, output_words) / max(input_words, output_words, 1)
                })
        
        return processed
    
    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single batch of data"""
        data_type = batch['batch_info']['data_type']
        batch_data = batch['data']
        
        if data_type == 'emotion':
            processed_data = self.preprocess_emotion_data(batch_data)
        elif data_type == 'conversation':
            processed_data = self.preprocess_conversation_data(batch_data)
        else:
            # Fallback for unknown types
            processed_data = batch_data
        
        return {
            'data': processed_data,
            'batch_info': {
                **batch['batch_info'],
                'original_size': len(batch_data),
                'processed_size': len(processed_data),
                'filter_rate': 1 - (len(processed_data) / len(batch_data)) if batch_data else 0,
                'processing_status': 'success'
            }
        }
    
    def process_stream_parallel(self, stream_batches, max_workers: int = None) -> List[Dict]:
        """Process multiple batches in parallel"""
        if max_workers is None:
            max_workers = self.config.PREPROCESSING_WORKERS
        
        processed_batches = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_batch, batch) for batch in stream_batches]
            
            for future in futures:
                try:
                    result = future.result()
                    processed_batches.append(result)
                    logger.info(f"Processed batch from {result['batch_info']['source']}: "
                              f"{result['batch_info']['processed_size']} items "
                              f"(filter rate: {result['batch_info']['filter_rate']:.2%})")
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
        
        return processed_batches
    
    def create_training_format(self, processed_data: List[Dict]) -> List[Dict]:
        """Convert processed data to training format for Phi model"""
        training_data = []
        
        for item in processed_data:
            if item.get('type') in ['conversation', 'intent', 'mental_health']:
                # Format for conversation/QA
                training_item = {
                    'messages': [
                        {
                            'role': 'user', 
                            'content': item['input']
                        },
                        {
                            'role': 'assistant', 
                            'content': item['output']
                        }
                    ],
                    'source': item['source'],
                    'type': item['type'],
                    'tag': item.get('tag', ''),
                    'quality_score': item.get('conversation_quality_score', 1.0)
                }
                training_data.append(training_item)
            
            elif item.get('type') == 'emotion':
                # Format for emotion classification as conversation
                emotion_prompt = f"How would you classify the emotion in this text: '{item['text']}'"
                emotion_response = f"The emotion expressed in this text is {item['label_name']}."
                
                training_item = {
                    'messages': [
                        {
                            'role': 'user', 
                            'content': emotion_prompt
                        },
                        {
                            'role': 'assistant', 
                            'content': emotion_response
                        }
                    ],
                    'source': item['source'],
                    'type': 'emotion_classification',
                    'label_numeric': item['label_numeric'],
                    'label_name': item['label_name'],
                    'original_text': item['text']
                }
                training_data.append(training_item)
        
        return training_data
    
    def get_preprocessing_stats(self, processed_batches: List[Dict]) -> Dict[str, Any]:
        """Generate preprocessing statistics"""
        total_original = sum(batch['batch_info']['original_size'] for batch in processed_batches)
        total_processed = sum(batch['batch_info']['processed_size'] for batch in processed_batches)
        
        source_stats = {}
        type_stats = {}
        
        for batch in processed_batches:
            source = batch['batch_info']['source']
            data_type = batch['batch_info']['data_type']
            
            if source not in source_stats:
                source_stats[source] = {'original': 0, 'processed': 0}
            source_stats[source]['original'] += batch['batch_info']['original_size']
            source_stats[source]['processed'] += batch['batch_info']['processed_size']
            
            if data_type not in type_stats:
                type_stats[data_type] = {'original': 0, 'processed': 0}
            type_stats[data_type]['original'] += batch['batch_info']['original_size']
            type_stats[data_type]['processed'] += batch['batch_info']['processed_size']
        
        return {
            'total_original_items': total_original,
            'total_processed_items': total_processed,
            'overall_filter_rate': 1 - (total_processed / total_original) if total_original > 0 else 0,
            'source_breakdown': source_stats,
            'type_breakdown': type_stats,
            'processing_efficiency': total_processed / total_original if total_original > 0 else 0
        }