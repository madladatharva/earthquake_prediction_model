"""
Project setup script for earthquake prediction model
Run this to create the proper directory structure
"""
import os

def create_project_structure():
    """Create organized project directory structure"""
    directories = [
        'src/',
        'src/data_collection/',
        'src/preprocessing/',
        'src/models/',
        'src/visualization/',
        'src/tectonic_simulation/',
        'data/',
        'data/raw/',
        'data/processed/',
        'notebooks/',
        'results/',
        'tests/',
        'config/'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    print("\n📁 Project structure created successfully!")

if __name__ == "__main__":
    create_project_structure()
