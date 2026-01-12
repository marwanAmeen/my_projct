"""Logging utilities"""
import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(name: str = "med_vqa", 
                 log_dir: str = "logs",
                 level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
    
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricLogger:
    """Logger for tracking training metrics"""
    
    def __init__(self, log_dir: Optional[str] = None):
        self.metrics = {}
        self.log_dir = log_dir
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def update(self, metrics: dict, step: int):
        """Update metrics"""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append((step, value))
    
    def get_average(self, key: str, last_n: Optional[int] = None) -> float:
        """Get average of last n values"""
        if key not in self.metrics:
            return 0.0
        
        values = [v for _, v in self.metrics[key]]
        if last_n:
            values = values[-last_n:]
        
        return sum(values) / len(values) if values else 0.0
    
    def save(self, filename: str = "metrics.txt"):
        """Save metrics to file"""
        if not self.log_dir:
            return
        
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            for key, values in self.metrics.items():
                f.write(f"{key}:\n")
                for step, value in values:
                    f.write(f"  Step {step}: {value:.4f}\n")
                f.write("\n")
