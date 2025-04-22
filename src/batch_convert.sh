#!/bin/bash

# Convert all the models in the checkpoint directory to the huggingface format
# Usage: ./batch_convert.sh <config_model> <checkpoint_dirs>
# Example: ./batch_convert.sh "config_model.json" "checkpoints/"

config_model=$1
checkpoint_dirs=$2

# Predifined values
hf_models="hf_models/"
fabric_checkpoint_name="lit_model.pth"
precision="bf16-true"
strategy="auto"


# Find all the models in the checkpoint file
model

# Find all the models in the checkpoint directory
models=($(find "$checkpoint_dirs" -maxdepth 1 -type f ! -path "$checkpoint_dirs" -printf "%f "))

# Print the models found
echo "Models found:"
printf "%s\n" "${models[@]}"

output_dir=$checkpoint_dirs$hf_models
mkdir -p $output_dir
# Execute the conversion script for each model
for model in "${models[@]}"; do
    #model_path=$checkpoint_dirs$model/$fabric_checkpoint_name
    model_path=$checkpoint_dirs$model
    # # Check if the model path exists
    # if [ ! -d "$model_path" ]; then
    #     echo "Model path $model_path does not exist. Skipping..."
    #     continue
    # fi

    python -m convert_fabric_to_hf_models --config_model "$config_model" --checkpoint_path "$model_path" \
        --accelerator "$strategy" --devices 1 --strategy "$strategy" --precision "$precision" --output_dir "$output_dir$model"
done

echo "All commands executed."
