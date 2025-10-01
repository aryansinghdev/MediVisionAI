"""
Complete pipeline script for MediVisionAI (Python version).
Alternative to run_all.sh for Windows or Python-preferred users.
"""
import os
import sys
import subprocess
import argparse
from datetime import datetime
import zipfile


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error: {description} failed!")
        sys.exit(1)
    
    print(f"✓ {description} completed successfully")


def setup_environment():
    """Setup virtual environment and install dependencies."""
    print("\nStep 1: Setting up environment...")
    
    # Check if venv exists
    if not os.path.exists('venv'):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'])
    
    # Determine pip path
    if os.name == 'nt':  # Windows
        pip = os.path.join('venv', 'Scripts', 'pip.exe')
        python = os.path.join('venv', 'Scripts', 'python.exe')
    else:  # Unix/Linux
        pip = os.path.join('venv', 'bin', 'pip')
        python = os.path.join('venv', 'bin', 'python')
    
    # Install requirements
    print("Installing requirements...")
    subprocess.run([pip, 'install', '--upgrade', 'pip'], check=True)
    subprocess.run([pip, 'install', '-r', 'requirements.txt'], check=True)
    
    return python


def generate_data(python_exe, num_samples=1000):
    """Generate synthetic data."""
    print(f"\nStep 2: Generating synthetic data ({num_samples} samples)...")
    
    code = f"""
from src.utils.data_generator import generate_synthetic_data
generate_synthetic_data(num_samples={num_samples}, seed=42)
print('Data generation completed!')
"""
    
    subprocess.run([python_exe, '-c', code], check=True)


def train_model(python_exe, image_dir, epochs=200, batch_size=64):
    """Train the model."""
    print(f"\nStep 3: Training model ({epochs} epochs, batch size {batch_size})...")
    
    cmd = [
        python_exe, 'train.py',
        '--image_dir', image_dir,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size)
    ]
    
    subprocess.run(cmd, check=True)


def evaluate_model(python_exe, image_dir, model_path='checkpoints/best_model.pth'):
    """Evaluate the model."""
    print("\nStep 4: Evaluating model...")
    
    cmd = [
        python_exe, 'evaluate.py',
        '--image_dir', image_dir,
        '--model_path', model_path
    ]
    
    subprocess.run(cmd, check=True)


def generate_gradcam(python_exe, image_path, model_path='checkpoints/best_model.pth'):
    """Generate Grad-CAM visualization."""
    print("\nStep 5: Generating Grad-CAM visualization...")
    
    cmd = [
        python_exe, 'gradcam.py',
        '--image_path', image_path,
        '--model_path', model_path,
        '--age', '50',
        '--gender', 'M'
    ]
    
    subprocess.run(cmd, check=True)


def create_archive():
    """Create output archive."""
    print("\nStep 6: Creating output archive...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_filename = f'MediVisionAI_outputs_{timestamp}.zip'
    
    files_to_archive = [
        'checkpoints/best_model.pth',
        'checkpoints/final_model.pth',
        'outputs/test_metrics.json',
        'outputs/test_metrics_report.txt',
        'outputs/gradcam.jpg',
        'config.py',
        'README.md'
    ]
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_archive:
            if os.path.exists(file):
                zipf.write(file)
                print(f"  Added: {file}")
    
    print(f"\n✓ Archive created: {zip_filename}")
    return zip_filename


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description='Run MediVisionAI complete pipeline')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--skip_setup', action='store_true',
                        help='Skip environment setup')
    parser.add_argument('--skip_data', action='store_true',
                        help='Skip data generation (use existing data)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MediVisionAI - Complete Pipeline Execution")
    print("="*60)
    
    # Setup
    if not args.skip_setup:
        python_exe = setup_environment()
    else:
        python_exe = sys.executable
    
    # Data generation
    if not args.skip_data:
        generate_data(python_exe, args.num_samples)
    
    image_dir = 'data/synthetic'
    
    # Training
    train_model(python_exe, image_dir, args.epochs, args.batch_size)
    
    # Evaluation
    evaluate_model(python_exe, image_dir)
    
    # Grad-CAM
    # Find first image in directory
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    if images:
        first_image = os.path.join(image_dir, images[0])
        generate_gradcam(python_exe, first_image)
    
    # Archive
    zip_file = create_archive()
    
    print("\n" + "="*60)
    print("Pipeline execution completed successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - Best model: checkpoints/best_model.pth")
    print("  - Final model: checkpoints/final_model.pth")
    print("  - Test metrics: outputs/test_metrics.json")
    print("  - Classification report: outputs/test_metrics_report.txt")
    print("  - Grad-CAM visualization: outputs/gradcam.jpg")
    print(f"  - Output archive: {zip_file}")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
