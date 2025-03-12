# Overview

This script automates the conversion of a distributed Megatron-LM checkpoint to a PyTorch format and then to Hugging Face Transformers format. 

## Dependencies

Python 3.x

SLURM workload manager (for batch job submission)

[Megatron-LM-Open-Sci](https://github.com/LAION-AI/Megatron-LM-Open-Sci)

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

[open-sci-hf](https://github.com/LAION-AI/Open-Sci-hf)

A container image compatible with the conversion process (e.g. NVIDIA Pytorch)

## Installation
Ensure you have all dependencies installed and accessible from your environment. The paths to Megatron-LM and Open-Sci repositories should be specified in the script arguments.

## Usage
Run the script with the following arguments:

```bash
python convert_megatron_to_hf.py \
    --account <SLURM_ACCOUNT> \
    --partition <SLURM_PARTITION> \
    --convert_logs_dir <LOGS_DIR> \
    --container_image <CONTAINER_IMAGE_PATH> \
    --opensci_megatron_path <PATH_TO_MEGATRON_OPEN_SCI> \
    --megatron_path <PATH_TO_MEGATRON> \
    --open_sci_hf_path <PATH_TO_OPEN_SCI_HF> \
    --train_logs_dir_or_path <TRAIN_LOGS_DIR_OR_FILE> \
    --save_checkpoints_dir <OUTPUT_CHECKPOINTS_DIR>
```

### Arguments
| Argument | Description |
|----------|-------------|
| `--account` | SLURM account name |
| `--partition` | SLURM partition name |
| `--convert_logs_dir` | Directory to store conversion logs |
| `--container_image` | Path to container image |
| `--opensci_megatron_path` | Path to Megatron-LM-Open-Sci repository |
| `--megatron_path` | Path to Megatron-LM repository |
| `--open_sci_hf_path` | Path to open-sci-hf repository |
| `--train_logs_dir_or_path` | Path to training logs (directory or specific file) |
| `--save_checkpoints_dir` | Output directory for converted checkpoints |

## Workflow
1. **Prepare Training Logs**: The script scans for `.out` files in the specified directory or processes a single file.
2. **Generate SLURM Scripts**: A conversion batch script is generated using a template.
3. **Submit SLURM Jobs**: Each training log file gets processed in a separate SLURM job.
4. **Conversion Execution**: The script inside the batch job handles checkpoint conversion.

## Output
Converted checkpoints are stored in the specified `--save_checkpoints_dir` directory.

## Troubleshooting
- Ensure all paths are correctly specified.
- Verify that the SLURM scheduler is available and properly configured.
- Check `convert_logs_dir` for logs if a job fails.
