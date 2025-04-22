FROM nvcr.io/nvidia/pytorch:21.10-py3

# Set the working directory inside the container
WORKDIR /workspace

# Install required Python packages
RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir \
    transformers \ 
    datasets \
    lightning==2.2.2 \
    notebook \
    wandb \
    scikit-learn \
    deepspeed \
    jsonargparse \
    openpyxl


RUN huggingface-cli login --token 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
RUN wandb login c310ac3be2b75f874ed7195790b27dbdda0db3ce

# Set up the entry point to open a shell by default
CMD ["python"]