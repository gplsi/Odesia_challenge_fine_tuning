import argparse
import os
from transformers import AutoTokenizer
import lightning as L
from models.models_class import FabricGeneration
import json


def main(args):
    # Load the model config
    with open(os.path.join(os.getcwd(), args.config_model)) as f:
        config_model = json.load(f)

    checkpoint_path = os.path.join(os.getcwd(), args.checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(config_model['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    fabric = L.Fabric(accelerator=args.accelerator, devices=args.devices, strategy=args.strategy, precision=args.precision)
    fabric.launch()
    model = FabricGeneration(config_model)
    state = {"model": model}
    print(checkpoint_path)
    fabric.load(checkpoint_path, state)
    model = fabric.setup(model)

    outputs_dir = os.path.join(os.getcwd(), args.output_dir)
    os.mkdir(outputs_dir)
    model.model.save_pretrained(outputs_dir)
    tokenizer.save_pretrained(outputs_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_model", default="/usrvol/data/Llama/llama_model.json")
    parser.add_argument("--checkpoint_path", default="/data/meta-llama/Meta-Llama-3-8B/iter-440324-ckpt.pth")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default=1)
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--precision", default="bf16-true")
    parser.add_argument("--output_dir", default="/usrvol/NAS/LLAMA/modelos/llama3-8B-mikel/hf/epoch_2")
    args = parser.parse_args()
    main(args)
