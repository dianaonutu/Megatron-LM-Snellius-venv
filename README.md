# Megatron-LM on Snellius using virtual environment

This codebase is for Snellius users to quickly set up a virtual environment for pretraining LLMs with Megatron-LM.

## Table of Contents

1. [Create Virtual Environment](#create-virtual-environment)  
2. [Pretraining a GPT Model](#pretraining-a-gpt-model)  
   2.1 [Debug Training with One GPU](#if-you-want-to-run-the-training-for-debugging-purposes-allocate-one-gpu)  
3. [Tokenize & Preprocess Data](#tokenize--preprocess-data)  
   3.1 [Download FineWeb Dataset](#download-fineweb-dataset)  
   3.2 [Tokenization/Preprocessing](#tokenizationpreprocessing)  
4. [Acknowledgments](#acknowledgments)

## Create virtual environment 
**Estimated time:** 10 minutes.
1. Clone this repository and `cd` into it.
```
https://github.com/dianaonutu/Megatron-LM-Snellius-venv.git
```
2. Load required modules. Note, this combination of modules together with PyTorch 2.6 and CUDA 12.6 is the best (e.g., for fast flash attention installation).
```
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0
```
3. Create a virtual environment. 
```
python -m venv megatron-venv
```
4. Allocate compute node, activate virtual environment and install packages.

**Estimated time:** 7 minutes
```
salloc -n 16 -t 30:00
source megatron-venv/bin/activate
./install.sh
```
4. Once finished, exit node allocation: `exit`.

## Tokenize & Preprocess data
**Estimated time:** 45 minutes.

Load required modules, if not done yet.
```
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load cuDNN/9.5.0.50-CUDA-12.6.0
```
Allocate a compute node, activate virtual environment and set project path.
```
salloc -p gpu_h100 --gpus-per-node 1 -t 1:00:00
source megatron-venv/bin/activate
export PROJECT_SPACE=/projects/0/prjs1502
```

### Download FineWeb dataset

The 10BT shard from the HuggingFace's [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) dataset.

**Estimated time:** 8 minutes.
```
python load_fineweb.py
```

### Tokenization/Preprocessing

**Estimated time:** 34 minutes.

Set environment variables for input/output paths and worker count.
```
export FINEWEB_INPUT=$PROJECT_SPACE/datasets/FineWeb/raw/fineweb-10BT.jsonl
export FINEWEB_OUTPUT=$PROJECT_SPACE/datasets/FineWeb/fineweb-10BT
export WORKERS=${SLURM_CPUS_PER_TASK:-16}
```
Run the tokenizer.
```
python Megatron-LM/tools/preprocess_data.py --input $FINEWEB_INPUT --output-prefix $FINEWEB_OUTPUT --tokenizer-type HuggingFaceTokenizer --tokenizer-model gpt2 --append-eod --log-interval 10000 --workers $WORKERS
```
The output is an index file (idx) and the binary (bin) of the tokenizer model.

Exit allocated node: `exit`.

## Pretraining a GPT model
1. Clone the Megatron-LM repository. This codebases uses commit [`8a9e8644`](https://github.com/NVIDIA/Megatron-LM/commit/8a9e8644) .
```
git clone https://github.com/NVIDIA/Megatron-LM.git
```
2. Set permissions. You only need to run this command once.
```
chmod +x launch.sh
```
3. Submit the job.
```
sbatch train-gpt-venv.job
```

### If you want to run the training for debugging purposes, allocate one GPU
1. Load required modules.
```
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0
module load NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0 
module load cuDNN/9.5.0.50-CUDA-12.6.0
```
2. Allocate 1 GPU:
```
salloc -p gpu_h100 --gpus-per-node 1 -t 1:00:00
export SLURM_CPUS_PER_TASK=1
export SLURM_NTASKS=1
```
3. If not done yet, set permissions:
```
chmod +x train-gpt-venv.job
```
4. Run the training within `salloc`:
```
./train-gpt-venv.job
```

## Acknowledgments
Thanks to [@spyysalo](https://github.com/spyysalo) original LUMI Megatron-LM [guide](https://github.com/spyysalo/lumi-fineweb-replication) and [@tvosch](https://github.com/tvosch) [guide](https://github.com/SURF-ML/Megatron-LM-Snellius) for creating this guide. 
