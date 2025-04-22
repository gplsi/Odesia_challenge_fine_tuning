import os
import math
import time
import json
import tqdm
import torch
import dataclasses
import lightning as L
from pathlib import Path
from tqdm.auto import tqdm, trange
from prepare_data.flan import FLAN
from datasets import load_from_disk
from torchmetrics import RunningMean
from prepare_data.preprocess import *
from utils.utils import num_parameters
from torch.utils.data import DataLoader
from utils.logger import step_csv_logger
from utils.args import EvalArgs, TrainArgs
from typing import Dict, List, Tuple, Union
from models.models_class import FabricGeneration
from pytorch_lightning.loggers import WandbLogger
from prepare_data.preprocess import get_dataloader
from lightning.fabric.strategies import FSDPStrategy
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, BloomForCausalLM, BloomTokenizerFast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

config_model="/data/5W1H/mock_model_config.json"

# Load the model config
with open(os.getcwd() + config_model) as f:
    config_model = json.load(f)

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", config_model["model_name"], flush_logs_every_n_steps=config_model['train_args']['log_interval'])
wandb_logger = WandbLogger(entity="gplsi_continual", project="fabric_Aitana_instruction", log_model="all")

def setup(
    devices: int = 4,
    resume: Union[bool, Path] = False,
    seed: int = 1337,
) -> None:
    """Finetune a model.

    Arguments:
        devices: How many devices/GPUs to use
        resume: Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
            from the latest checkpoint in ``out_dir``.
        seed: The random seed to use for reproducibility.
        config_model: Path to the model config file.
    """

    #policy = {BloomBlock}
    policy = {LlamaDecoderLayer}
    if devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy=policy,
            activation_checkpointing_policy=policy,
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=config_model["precision"], loggers=[logger, wandb_logger])
    main(fabric, resume, devices, seed, config_model)


def main(
    fabric: L.Fabric,
    resume: Union[bool, Path],
    devices: int,
    seed: int,
    config: dict,
) -> None:
    train_args = TrainArgs(**config["train_args"])
    eval_args = EvalArgs(**config["eval_args"])

    # Get the data                                 Aitana_ft_trainig_tokenized_v1
    train_dataset = load_from_disk(os.getcwd() + config["training_tokenized"])
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    validation_dataset = load_from_disk(os.getcwd() + config["eval_tokenized"])
    validation_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    train_dataloader = DataLoader(train_dataset, batch_size=train_args.micro_batch_size, shuffle=True,
                                  collate_fn=get_sft_collate_fn(max_seq_length=train_args.max_seq_length, pad_id=0, ignore_index=-100))
    #                              num_processes=fabric.world_size, process_rank=fabric.global_rank)
    val_dataloader = DataLoader(validation_dataset, batch_size=eval_args.micro_batch_size, shuffle=False,
                                       collate_fn=get_sft_collate_fn(max_seq_length=train_args.max_seq_length, pad_id=0, ignore_index=-100))
    #                             num_processes=fabric.world_size, process_rank=fabric.global_rank)

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    print("Train args: ", train_args.gradient_accumulation_iters(devices))
    print("Devices: ", devices)
    steps_per_epoch = len(train_dataloader) // train_args.gradient_accumulation_iters(devices)
    lr_max_steps = min(train_args.epochs * steps_per_epoch, (train_args.max_steps or float("inf")))
    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    with fabric.init_module():
        model = FabricGeneration(config)
    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")

    # Add to compile models with torch
    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay, betas=(train_args.beta1, train_args.beta2), foreach=False
    )
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=train_args.lr_warmup_steps, max_steps=lr_max_steps)
    state = {"model": model, "optimizer": optimizer, "scheduler": scheduler, "iter_num": 0, "step_count": 0}
    out_dir = os.getcwd() + config["checkpoint_path"]
    out_dir = Path(out_dir)
    checkpoint_dir = Path(out_dir)
    if resume is True:
        resume = max(out_dir.rglob("step-*/*.pth"), key=(lambda p: int(p.parent.name.split("-")[1])))
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)
    # else:
    #     load_checkpoint(fabric, state["model"], checkpoint_path)
    #Initial evaluation
    # val_loss = validate(fabric, model, val_dataloader, dataclasses.replace(eval_args, max_iters=len(val_dataloader)))
    # metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
    # fabric.log_dict(metrics, step=state["iter_num"])
    # fabric.print(f"Final evaluation | val loss: {val_loss.item():.3f} | val ppl: {math.exp(val_loss):.3f}")

    train_time = time.perf_counter()
    fit(fabric, state, train_dataloader, val_dataloader, devices, resume, checkpoint_dir, out_dir, train_args, eval_args)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Final evaluation
    val_loss = validate(fabric, model, val_dataloader, dataclasses.replace(eval_args, max_iters=len(val_dataloader)))
    metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
    fabric.log_dict(metrics, step=state["iter_num"])
    fabric.print(f"Final evaluation | val loss: {val_loss.item():.3f} | val ppl: {math.exp(val_loss):.3f}")

    #Save the final checkpoint at the end of training
    save_path = out_dir / "final" / "lit_model.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fabric.save(save_path, {"model": state["model"]})


def fit(
    fabric: L.Fabric,
    state: Dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    devices: int,
    resume: Union[bool, Path],
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    initial_iter = state["iter_num"]
    max_steps = train.max_steps or float("inf")
    epoch = 0
    train_iterator = trange(
        int(train.epochs), desc="Epoch", disable=False, mininterval=0)
    running_loss = RunningMean(window=train.gradient_accumulation_iters(devices), sync_on_compute=False).to(
        fabric.device
    )
    val_loss = "n/a"
    for _ in train_iterator:
        train_iterator.set_description(
            f"Running Epoch {epoch + 1} of {train.epochs}"
        )
        batch_iterator = tqdm(train_dataloader, mininterval=0, colour="blue")
        half_batches = int((len(batch_iterator) / 2) // train.gradient_accumulation_iters(devices))
        train.save_interval = half_batches
        eval.interval = half_batches

        # Iterate over each batch, getting the specific index of that batch
        for step, batch in enumerate(batch_iterator):
            #print("Step: ", step)

            # resume data loader state by fast-forwarding through all seen batches
            if resume:
                resume_t0 = time.perf_counter()
                for resume_iter in range(initial_iter):
                    if resume_iter % 1000 == 0:
                        fabric.print(f"Resuming dataset: {resume_iter} / {initial_iter}")
                    continue

                fabric.barrier()
                fabric.print(
                    f"Resuming data loader finished. Took {time.perf_counter() - resume_t0:.1f} seconds to reach iteration"
                    f" {initial_iter}."
                )

            # determine and set the learning rate for this iteration
            # fparam_group["lr"] = lr

        # while state["step_count"] < max_steps and train_iterator.epoch < train.epochs:
        #     state["iter_num"] += 1
            iter_t0 = time.perf_counter()
            is_accumulating = state["iter_num"] % train.gradient_accumulation_iters(devices) != 0 or state["iter_num"] == 0
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                loss = model.training_step(batch, step)
                fabric.backward(loss / train.gradient_accumulation_iters(devices))

            running_loss.update(loss.detach())

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                state["step_count"] += 1
            state["iter_num"] += 1

            if state["iter_num"] % train.log_interval == 0:
                loss = running_loss.compute().item()  # expensive device-to-host synchronization
                t1 = time.perf_counter()
                metrics = {
                    "loss": loss,
                    "iter": state["iter_num"],
                    "step": state["step_count"],
                    "epoch": epoch,
                    "iter_time": t1 - iter_t0,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                if isinstance(val_loss, torch.Tensor):
                    val_loss = f"{val_loss:.3f}"
                fabric.print(
                    f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} step {metrics['step']} |"
                    f" loss train: {metrics['loss']:.3f},"
                    f" val: {val_loss} |"
                    f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                    f"{' (step)' if not is_accumulating else ''}"
                )
                fabric.log_dict(metrics, step=state["iter_num"])

            if not is_accumulating and state["step_count"] % eval.interval == 0:
                t0 = time.perf_counter()
                val_loss = validate(fabric, model, val_dataloader, eval)
                t1 = time.perf_counter() - t0
                fabric.print(f"iter {state['iter_num']}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f} ms")
                metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
                fabric.log_dict(metrics, step=state["iter_num"])

            #if not is_accumulating and state["step_count"] % eval.interval == 0:
                print("Step count: ", state["step_count"])
                print("Save interval: ", train.save_interval)
                checkpoint_file = out_dir / f"step-{state['step_count']:06d}" / "lit_model.pth"
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                fabric.print(f"Saving checkpoint to {str(checkpoint_file.parent)!r}")
                fabric.save(checkpoint_file, state)
                fabric.barrier()
        epoch += 1


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, eval: EvalArgs) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(min(len(val_dataloader), eval.max_iters))
    batch_iterator = tqdm(val_dataloader, desc=f"Running Validating", mininterval=0,
                          colour="green")
    for k, batch in enumerate(batch_iterator):
        if k >= eval.max_iters:
            break
        outputs = model.validation_step(batch, k)
        losses[k] = outputs.loss

    val_loss = losses.mean()
    model.train()
    return val_loss


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    # linear warmup followed by cosine annealing
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(max_steps - warmup_steps))


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def validate_args(train: TrainArgs, eval: EvalArgs) -> None:
    issues = []
    unsupported = [(train, ["max_tokens", "max_norm", "tie_embeddings", "lr_warmup_fraction"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    required = [(train, ["epochs"]), (eval, ["max_new_tokens"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if not train.epochs and not train.max_steps:
        issues.append(f"{__file__} requires either epochs or max_steps to be set. This is set in {train}")
    if issues:
        raise ValueError("\n".join(issues))


# @torch.no_grad()
# def generate_example(fabric: L.Fabric, model: torch.nn.Module, tokenizer: None, eval: EvalArgs, data: None):
#     instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
#     fabric.print(instruction)
#     prompt = data.prompt_style.apply(instruction)
#     encoded = tokenizer.encode(prompt, device=fabric.device)
#     model.eval()
#
#     with fabric.init_tensor():
#         # do not set `max_seq_length=max_returned_token` because memory is not a concern here
#         model.set_kv_cache(batch_size=1)
#     output = generate(
#         model, encoded, max_returned_tokens=len(encoded) + eval.max_new_tokens, temperature=0.8, eos_id=tokenizer.eos_id
#     )
#     model.clear_kv_cache()
#     model.train()
#     output = tokenizer.decode(output)
#     fabric.print(output)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    from jsonargparse import CLI
    CLI(setup)
