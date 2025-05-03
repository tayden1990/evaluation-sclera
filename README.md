# Sclera Evaluation Application

This application provides tools for evaluating sclera recognition models.

## Creating and Setting Up Environment

1. Install Python 3.8 or higher if not already installed
2. Create a new virtual environment:
   ```bash
   python -m venv sclera-env
   ```
3. Install required packages:
   ```bash
   # On Windows
   sclera-env\Scripts\activate
   
   # On Linux/Mac
   source sclera-env/bin/activate
   
   pip install torch torchvision torchaudio
   pip install opencv-python pillow numpy scikit-learn matplotlib
   pip install -r requirements.txt
   ```

## Environment Setup

Before running the application, you need to set up the proper environment:

```bash
# On Windows
sclera-env\Scripts\activate

# On Linux/Mac
source sclera-env/bin/activate
```

## Running the Application

You can run the evaluation application using the following command:

```bash
python all_in_one_evaluation.py --model_path "models\model_epoch_30.pth" --output_dir "predictions" --test_data_dir "SSBC_DATASETS_400x300\Evaluation_Sample"
```

### Command Arguments

- `--model_path`: Path to the trained model file
- `--output_dir`: Directory where prediction results will be saved
- `--test_data_dir`: Directory containing the test dataset

## Requirements

- Python 3.x
- PyTorch and torchvision
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- CUDA (optional, for GPU acceleration)

## Preparing Data

1. Download the SSBC dataset
2. Extract the dataset to the `SSBC_DATASETS_400x300` directory
3. Ensure the evaluation samples are in the `SSBC_DATASETS_400x300\Evaluation_Sample` directory
4. Create a `models` directory and place your trained model files there