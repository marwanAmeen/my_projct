"""
Universal Environment Setup for Multimodal VQA Project
=====================================================

This module provides automatic environment detection and setup for:
- Google Colab
- VS Code / Local Jupyter
- Other Jupyter environments

Usage in notebooks:
    from setup_environment import setup_project
    project_root, device, is_colab = setup_project()
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def setup_project(colab_project_path=None, verbose=True):
    """
    Universal project setup function.
    
    Args:
        colab_project_path (str): Custom path for Google Colab. 
                                If None, uses default path.
        verbose (bool): Print setup information
    
    Returns:
        tuple: (project_root, device, is_colab)
    """
    
    if verbose:
        print("UNIVERSAL PROJECT SETUP")
        print("=" * 50)
    
    # Detect environment
    try:
        import google.colab
        IN_COLAB = True
        if verbose:
            print("Environment detected: Google Colab")
    except ImportError:
        IN_COLAB = False
        if verbose:
            print("Environment detected: Local (VS Code/Jupyter)")
    
    if IN_COLAB:
        # ========== GOOGLE COLAB SETUP ==========
        if verbose:
            print("Setting up Google Colab environment...")
        
        # Mount Google Drive
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Install packages
        if verbose:
            print("Installing required packages...")
        os.system('pip install -q torch torchvision tqdm pyyaml scikit-learn pandas matplotlib seaborn Pillow')
        
        # Set project path
        if colab_project_path is None:
            # Default Colab project paths - try in order
            possible_paths = [
                "/content/drive/MyDrive/WOA7015 Advanced Machine Learning/my_projct",
                "/content/drive/MyDrive/WOA7015 Advanced Machine Learning/data", 
                "/content/drive/MyDrive/my_projct",
                "/content/drive/MyDrive/data"
            ]
            
            project_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    project_path = path
                    break
                    
            if project_path is None:
                if verbose:
                    print("ERROR: No default project path found. Available paths:")
                    base = "/content/drive/MyDrive"
                    if os.path.exists(base):
                        for item in os.listdir(base):
                            print(f"   - {os.path.join(base, item)}")
                raise FileNotFoundError(
                    "Project path not found. Please specify colab_project_path parameter."
                )
        else:
            project_path = colab_project_path
        
        os.chdir(project_path)
        project_root = Path(project_path)
        if verbose:
            print(f"SUCCESS: Project root set to {project_path}")
            
    else:
        # ========== LOCAL ENVIRONMENT SETUP ==========
        if verbose:
            print("Setting up local environment...")
        
        current_dir = Path().absolute()
        
        # Check if we're in notebooks directory or project root
        if current_dir.name == 'notebooks':
            project_root = current_dir.parent
        elif (current_dir / 'notebooks').exists():
            project_root = current_dir
        else:
            # Try to find project root by looking for key directories
            search_dir = current_dir
            for _ in range(3):  # Search up to 3 levels up
                if any((search_dir / d).exists() for d in ['src', 'data', 'notebooks']):
                    project_root = search_dir
                    break
                search_dir = search_dir.parent
            else:
                project_root = current_dir
        
        if verbose:
            print(f"SUCCESS: Project root set to {project_root}")
    
    # ========== COMMON SETUP ==========
    
    # Add to Python path
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / 'src'))
    
    # Setup PyTorch device
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # Verify project structure
        required_dirs = ['src', 'data', 'notebooks']
        missing = [d for d in required_dirs if not (project_root / d).exists()]
        
        if missing:
            print(f"WARNING: Missing directories: {missing}")
        else:
            print("SUCCESS: Project structure verified")
        
        print("Setup complete!")
        print("=" * 50)
    
    return project_root, device, IN_COLAB

def import_project_modules(project_root=None, verbose=True):
    """
    Import common project modules with error handling.
    
    Returns:
        dict: Dictionary of imported modules
    """
    modules = {}
    
    try:
        from src.data.dataset import create_multimodal_dataloaders
        modules['create_multimodal_dataloaders'] = create_multimodal_dataloaders
        
        from src.models.improved_multimodal_model import ImprovedMultimodalVQA
        modules['ImprovedMultimodalVQA'] = ImprovedMultimodalVQA
        
        if verbose:
            print("SUCCESS: Project modules imported successfully")
            
    except ImportError as e:
        if verbose:
            print(f"ERROR: Import error: {e}")
            print("   Make sure setup_project() was called first")
        raise
    
    return modules

# Convenience function for one-line setup
def quick_setup(colab_project_path=None, verbose=True):
    """
    One-line setup function that returns everything you need.
    
    Returns:
        tuple: (project_root, device, is_colab, modules_dict)
    """
    project_root, device, is_colab = setup_project(colab_project_path, verbose)
    modules = import_project_modules(project_root, verbose)
    
    return project_root, device, is_colab, modules

if __name__ == "__main__":
    # Test the setup
    project_root, device, is_colab = setup_project()
    modules = import_project_modules()
    print(f"\nTest complete: project_root={project_root}, device={device}, colab={is_colab}")