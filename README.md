# Sclera Evaluation Application

This application provides tools for evaluating sclera recognition models.

## Running the Application

You can run the evaluation application using the following command:

```bash
python all_in_one_evaluation.py --model_path "models\model_epoch_30.pth" --output_dir "predictions-1" --test_data_dir "SSBC_DATASETS_400x300\Evaluation_Sample"
```

### Command Arguments

- `--model_path`: Path to the trained model file
- `--output_dir`: Directory where prediction results will be saved
- `--test_data_dir`: Directory containing the test dataset

## Requirements

- Python 3.x
- Required packages (PyTorch, etc.)