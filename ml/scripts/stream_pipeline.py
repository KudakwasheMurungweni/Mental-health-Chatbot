import mlflow
import mlflow.pytorch
from typing import Dict, Any, List
import logging
import json
import os
from datetime import datetime
from ..datasets.unified_stream import UnifiedStream
from .preprocess_stream import StreamPreprocessor
from ..utils.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamPipeline:
    """Complete streaming pipeline for mental health chatbot data"""
    
    def __init__(self):
        self.config = Config()
        self.unified_stream = UnifiedStream()
        self.preprocessor = StreamPreprocessor()
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        try:
            mlflow.set_experiment(self.config.MLFLOW_EXPERIMENT_NAME)
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment: {e}")
    
    def setup_pipeline(self) -> bool:
        """Setup the complete pipeline"""
        logger.info("Setting up streaming pipeline...")
        
        # Create directories
        directories = ['./data/raw', './data/processed', './data/training', './logs']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Setup data sources
        logger.info("Downloading datasets...")
        setup_status = self.unified_stream.setup_data_sources()
        
        logger.info(f"Pipeline setup complete. Status: {setup_status}")
        return any(setup_status.values())  # Return True if at least one source is available
    
    def run_streaming_pipeline(self, data_type: str = 'all', save_processed: bool = True) -> Dict[str, Any]:
        """Run the complete streaming pipeline"""
        
        run_name = f"streaming_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            with mlflow.start_run(run_name=run_name):
                
                # Log pipeline parameters
                mlflow.log_params({
                    'data_type': data_type,
                    'batch_size': self.config.STREAMING_BATCH_SIZE,
                    'preprocessing_workers': self.config.PREPROCESSING_WORKERS,
                    'model_name': self.config.MODEL_NAME,
                    'run_timestamp': datetime.now().isoformat()
                })
                
                pipeline_stats = {
                    'total_batches': 0,
                    'total_original_items': 0,
                    'total_processed_items': 0,
                    'sources_processed': [],
                    'processing_time': 0,
                    'training_samples': 0
                }
                
                start_time = datetime.now()
                
                logger.info(f"Starting streaming pipeline for data type: {data_type}")
                
                # Collect batches from stream
                stream_batches = []
                logger.info("Collecting batches from data streams...")
                
                for batch in self.unified_stream.stream_all_data(data_type):
                    stream_batches.append(batch)
                    pipeline_stats['total_batches'] += 1
                    pipeline_stats['total_original_items'] += len(batch['data'])
                    
                    source = batch['batch_info']['source']
                    if source not in pipeline_stats['sources_processed']:
                        pipeline_stats['sources_processed'].append(source)
                    
                    logger.info(f"Collected batch {batch['batch_info']['batch_id']} from {source}: "
                              f"{len(batch['data'])} items")
                
                logger.info(f"Collected {len(stream_batches)} batches from {len(pipeline_stats['sources_processed'])} sources")
                
                if not stream_batches:
                    logger.warning("No data batches collected. Check data sources.")
                    return {
                        'status': 'failed',
                        'error': 'No data batches collected',
                        'stats': pipeline_stats
                    }
                
                # Process batches in parallel
                logger.info(f"Processing {len(stream_batches)} batches...")
                processed_batches = self.preprocessor.process_stream_parallel(stream_batches)
                
                # Calculate processing stats
                for batch in processed_batches:
                    pipeline_stats['total_processed_items'] += batch['batch_info']['processed_size']
                
                # Create unified processed data
                all_processed_data = []
                for batch in processed_batches:
                    all_processed_data.extend(batch['data'])
                
                logger.info(f"Creating training format for {len(all_processed_data)} processed items...")
                training_data = self.preprocessor.create_training_format(all_processed_data)
                pipeline_stats['training_samples'] = len(training_data)
                
                # Get detailed preprocessing stats
                preprocessing_stats = self.preprocessor.get_preprocessing_stats(processed_batches)
                
                # Save processed data
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                if save_processed and all_processed_data:
                    # Save processed data
                    processed_file = f'./data/processed/processed_data_{timestamp}.json'
                    with open(processed_file, 'w') as f:
                        json.dump(all_processed_data, f, indent=2)
                    logger.info(f"Saved processed data to {processed_file}")
                    
                    # Save training data
                    training_file = f'./data/training/training_data_{timestamp}.json'
                    with open(training_file, 'w') as f:
                        json.dump(training_data, f, indent=2)
                    logger.info(f"Saved training data to {training_file}")
                    
                    # Save statistics
                    stats_file = f'./data/processed/pipeline_stats_{timestamp}.json'
                    all_stats = {
                        'pipeline_stats': pipeline_stats,
                        'preprocessing_stats': preprocessing_stats,
                        'run_info': {
                            'timestamp': timestamp,
                            'data_type': data_type,
                            'config': {
                                'batch_size': self.config.STREAMING_BATCH_SIZE,
                                'workers': self.config.PREPROCESSING_WORKERS
                            }
                        }
                    }
                    with open(stats_file, 'w') as f:
                        json.dump(all_stats, f, indent=2)
                    
                    # Log artifacts to MLflow
                    try:
                        mlflow.log_artifact(processed_file, "processed_data")
                        mlflow.log_artifact(training_file, "training_data")
                        mlflow.log_artifact(stats_file, "statistics")
                    except Exception as e:
                        logger.warning(f"Could not log artifacts to MLflow: {e}")
                
                # Calculate final stats
                end_time = datetime.now()
                pipeline_stats['processing_time'] = (end_time - start_time).total_seconds()
                pipeline_stats['filter_rate'] = preprocessing_stats['overall_filter_rate']
                
                # Log metrics to MLflow
                try:
                    mlflow.log_metrics({
                        'total_batches': pipeline_stats['total_batches'],
                        'total_original_items': pipeline_stats['total_original_items'],
                        'total_processed_items': pipeline_stats['total_processed_items'],
                        'processing_time_seconds': pipeline_stats['processing_time'],
                        'filter_rate': pipeline_stats['filter_rate'],
                        'training_samples': pipeline_stats['training_samples'],
                        'processing_efficiency': preprocessing_stats['processing_efficiency']
                    })
                except Exception as e:
                    logger.warning(f"Could not log metrics to MLflow: {e}")
                
                logger.info("="*50)
                logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
                logger.info(f"Processed {pipeline_stats['total_processed_items']} items from {pipeline_stats['total_original_items']} original items")
                logger.info(f"Created {pipeline_stats['training_samples']} training samples")
                logger.info(f"Processing time: {pipeline_stats['processing_time']:.2f} seconds")
                logger.info(f"Filter rate: {pipeline_stats['filter_rate']:.2%}")
                logger.info("="*50)
                
                return {
                    'status': 'success',
                    'stats': pipeline_stats,
                    'preprocessing_stats': preprocessing_stats,
                    'training_data': training_data,
                    'processed_data': all_processed_data
                }
                
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            try:
                mlflow.log_param('error', str(e))
            except:
                pass
            return {
                'status': 'failed',
                'error': str(e),
                'stats': pipeline_stats
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and statistics"""
        stream_stats = self.unified_stream.get_stream_stats()
        
        return {
            'config': {
                'batch_size': self.config.STREAMING_BATCH_SIZE,
                'workers': self.config.PREPROCESSING_WORKERS,
                'model': self.config.MODEL_NAME,
                'mlflow_uri': self.config.MLFLOW_TRACKING_URI
            },
            'stream_stats': stream_stats,
            'available_data_sources': [
                'emotions_dataset (393,822 samples)',
                'combined_intents (92 intents)',
                'conversations_training_csv',
                'conversations_training_json (204 conversations)',
                'mental_health_conversations'
            ]
        }

def run_pipeline_example():
    """Example of how to run the pipeline"""
    pipeline = StreamPipeline()
    
    # Setup
    logger.info("Setting up pipeline...")
    if pipeline.setup_pipeline():
        logger.info("Pipeline setup successful!")
        
        # Get status
        status = pipeline.get_pipeline_status()
        logger.info(f"Pipeline status: {status}")
        
        # Run pipeline for all data
        logger.info("Running pipeline for all data...")
        result = pipeline.run_streaming_pipeline(data_type='all')
        
        if result['status'] == 'success':
            logger.info("SUCCESS! Pipeline completed successfully!")
            print("\n" + "="*60)
            print("FINAL RESULTS:")
            print(f"✅ Processed items: {result['stats']['total_processed_items']}")
            print(f"✅ Training samples: {result['stats']['training_samples']}")
            print(f"✅ Sources processed: {', '.join(result['stats']['sources_processed'])}")
            print(f"✅ Processing time: {result['stats']['processing_time']:.2f} seconds")
            print("="*60)
        else:
            logger.error(f"Pipeline failed: {result.get('error', 'Unknown error')}")
    else:
        logger.error("Pipeline setup failed!")

if __name__ == "__main__":
    run_pipeline_example()