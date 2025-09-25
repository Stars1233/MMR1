#!/bin/bash

set -x
which python

# Load environment variables from .env if present
if [ -f "$(dirname "$0")/../.env" ]; then
  echo "Loading environment variables from .env file"
  export $(grep -v '^#' "$(dirname "$0")/../.env" | xargs)
else
  echo "Warning: .env file not found"
fi

# Fallback for model path
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
echo "Using model path: $MODEL_PATH"

# Sanity check
if [ -z "$JOB_NAME" ]; then
  echo "ERROR: JOB_NAME is empty"
  exit 1
fi

# Prompt string
SYSTEM_PROMPT="""A conversation between User and Assistant. 
The User provides an image and asks a question. 
The Assistant first analyzes both the image and the question, 
then carefully thinks about the reasoning process step by step, 
and finally provides the User with an accurate answer. 
The Assistant must carefully checkout the correctness and validity of each reasoning step. 
If any errors or inconsistencies are found during the reasoning process, 
the Assistant reflects and corrects them logically. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, 
i.e., <think> reasoning process here, with potential reflections and corrections </think>
<answer> final answer here, with the key result enclosed in \boxed{} </answer>."""

# Go to working directory (change if needed)
WORKDIR=$PWD

# Run the main training script directly
cd $WORKDIR
export VENV=/mnt/amlfs-01/home/jingwang/PROJECTS/mmo1/rl/.venv
python -m verl.trainer.main \
  config=$WORKDIR/examples/mmr1_b200.yaml \
  data.system_prompt="${SYSTEM_PROMPT}" \
  data.train_files=mm-o1/math_vista_val \
  data.val_files=mm-o1/math_vista_val \
  worker.actor.model.model_path=${MODEL_PATH} \
  trainer.experiment_name=${JOB_NAME} \
  $@
