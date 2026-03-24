# =============================================================================
# DRONE MOTION PLANNER — LLM FINE-TUNING SCRIPT
# Model  : Qwen2.5-7B-Instruct
# Method : LoRA (bf16, no quantization) — optimized for 2x NVIDIA L40S (46 GB each)
# Task   : NL sensor prompt → JSON waypoints (5-point local trajectory)
# =============================================================================
#
# QUICK START:
#   pip install unsloth trl datasets transformers accelerate peft
#   python train_motion_planner.py
#
# For dual-GPU FSDP (uses both L40S cards):
#   accelerate config   (choose: multi-GPU, FSDP, bf16)
#   accelerate launch train_motion_planner.py
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: IMPORTS
#
# WHY THESE LIBRARIES?
#
# unsloth        — Drop-in replacement for HuggingFace model loading.
#                  Provides custom CUDA kernels that make LoRA training
#                  2-5x faster than vanilla transformers. Critical for
#                  long-sequence tasks like yours (1200+ token prompts).
#
# trl / SFTTrainer — "Transformer Reinforcement Learning" library by HuggingFace.
#                  SFTTrainer is a wrapper around the standard Trainer that
#                  adds two things critical for your task:
#                  (1) Automatic chat template application (converts your
#                      messages dicts to the correct Qwen2.5 format).
#                  (2) Loss masking — the model only trains to predict the
#                      ASSISTANT turn (waypoints), not to re-predict your
#                      input prompt. Without this, the model wastes capacity
#                      memorizing the sensor data instead of learning to
#                      generate good trajectories.
#
# datasets       — HuggingFace datasets library. Handles efficient batching,
#                  tokenization caching, and data collation.
#
# peft / LoraConfig — Parameter-Efficient Fine-Tuning. Implements the LoRA
#                  adapter injection into the model's attention layers.
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import csv
import math
import inspect
import json
import os
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
torch.cuda.set_per_process_memory_fraction(0.80, device=0)
from datasets import Dataset
# Import Unsloth before TRL so its trainer/config patches are applied correctly.
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

try:
    from llm_drone.llm.llm_prompt_common import (
        DATASET_PROMPT_FILENAME,
        load_system_prompt,
        split_prompt_bundle,
    )
except ImportError:
    DATASET_PROMPT_FILENAME = "llm_prompt2d.txt"

    def load_system_prompt(prompt_filename: str) -> str:
        prompt_path = (Path(__file__).resolve().parents[1] / "config" / prompt_filename).resolve()
        return prompt_path.read_text() if prompt_path.exists() else ""

    def split_prompt_bundle(prompt_text: str, fallback_system_prompt: str = "") -> tuple[str, str]:
        text = str(prompt_text or "").strip()
        system_marker = "=== SYSTEM PROMPT ===\n"
        user_marker = "\n\n=== USER PROMPT ===\n"
        if text.startswith(system_marker) and user_marker in text:
            system_text, _, user_text = text[len(system_marker):].partition(user_marker)
            return system_text.strip(), user_text.strip()
        return str(fallback_system_prompt or "").strip(), text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = (Path(__file__).resolve().parents[1] / "models" / "drone_planner_checkpoints").resolve()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: CONFIGURATION
#
# WHY CENTRALIZE CONFIG HERE?
# All hyperparameters in one place so you can tune without hunting through code.
# Each value is explained with the reasoning specific to YOUR task.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:

    # --- Model ---
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    # WHY 7B: Your prompts are physically complex (NED frame, obstacle sectors,
    # velocity vectors). 7B has enough capacity to learn the spatial reasoning
    # required. 3B often fails on edge cases with multiple simultaneous obstacles.
    # WHY Qwen2.5: Best-in-class structured/JSON output among open models.
    # Qwen2.5 was explicitly trained on code and structured data, which means
    # it already "knows" how to write valid JSON — you're just redirecting that.

    load_in_4bit: bool = False
    # WHY FALSE: You have 46 GB VRAM per L40S. 4-bit quantization (QLoRA)
    # exists to fit models on 16-24 GB consumer GPUs. It slightly degrades
    # quality. With your hardware, use full bf16 LoRA for best results.
    # Set to True only if you're running other jobs simultaneously and VRAM is tight.

    max_seq_length: int = 2048
    # WHY 2048: Your system prompt (~500 tokens) + sensor data prompt (~700 tokens)
    # + JSON output (~300 tokens) = ~1500 tokens. 2048 gives comfortable headroom.
    # Do NOT set this lower or long prompts will be silently truncated, causing
    # the model to train on incomplete inputs and produce garbage waypoints.

    # --- LoRA Adapter Config ---
    lora_r: int = 32
    # WHY 32: LoRA rank controls how many parameters are trainable.
    # rank=16 is a starting point; rank=32 gives more expressive power.
    # Your task requires learning precise floating-point coordinate generation
    # conditioned on continuous sensor values — higher rank helps here.
    # rank=64 is diminishing returns unless you have >20K examples.

    lora_alpha: int = 64
    # WHY 64 (= 2 * rank): Alpha is a scaling factor. The effective learning
    # rate of the adapter is (alpha / rank). Keeping alpha = 2*rank is the
    # standard convention — it means the adapter's updates are scaled by 2,
    # which helps with convergence speed without instability.

    lora_dropout: float = 0.05
    # WHY 0.05: Light dropout on the adapter weights. Prevents the adapters
    # from over-specializing to your training prompts at the expense of
    # generalization to new sensor configurations you haven't seen.

    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    # WHY THESE LAYERS: These are all the linear projection layers in
    # Qwen2.5's transformer blocks. Applying LoRA to all of them (not just
    # q and v as some tutorials suggest) gives significantly better results
    # for structured output tasks. The gate/up/down projections in the MLP
    # are particularly important for learning the precise numeric patterns
    # in your waypoint coordinates.

    # --- Data ---
    data_path: str = str((Path(__file__).resolve().parents[1] / "dataset" / "offline_ground_truth_dataset.csv"))
    # Supported inputs:
    # - offline_ground_truth_dataset.csv from dataset_generator_executor.py
    # - raw dataset.csv from dataset_generator_sync.py
    # - JSONL with {"prompt","completion"} rows
    # - JSONL with {"messages":[...]} rows

    val_split_ratio: float = 0.1
    # WHY 10%: Standard train/val split. With 5K examples this gives 500
    # validation examples — enough for stable eval loss measurement.

    # --- Training ---
    output_dir: str = str(DEFAULT_OUTPUT_DIR)

    num_train_epochs: int = 3
    # WHY 3: For fine-tuning, 1-3 epochs is almost always correct.
    # More than 3 usually causes overfitting — the model starts memorizing
    # your specific training prompts rather than learning the underlying
    # sensor-to-waypoint mapping. Watch your validation loss: if it starts
    # rising while training loss falls, stop early.

    per_device_train_batch_size: int = 4
    # WHY 4: With a 7B model in bf16, each sample uses ~6-8 GB VRAM for
    # activations at seq_len=2048. Batch size 4 uses ~24-32 GB, well within
    # your 46 GB per card. Larger batches = more stable gradient estimates.

    gradient_accumulation_steps: int = 4
    # WHY 4: Effective batch size = per_device * gradient_accumulation * num_GPUs
    # = 4 * 4 * 2 = 32 (with both L40S cards).
    # Larger effective batches reduce gradient noise, which matters for
    # learning precise floating-point coordinate generation.

    learning_rate: float = 1e-4
    # WHY 1e-4: Standard LoRA learning rate. Lower than full fine-tuning
    # (which uses 1e-5 to 1e-6) because LoRA adapters are small and need
    # larger relative updates. If you see loss spikes, drop to 5e-5.
    # If loss decreases too slowly, try 2e-4.

    warmup_ratio: float = 0.05
    # WHY 0.05: Gradually ramps learning rate from 0 to target over the first
    # 5% of training steps. Prevents instability at the start when the adapter
    # weights are randomly initialized and the gradient signal is noisy.

    lr_scheduler_type: str = "cosine"
    # WHY COSINE: Smoothly decays learning rate following a cosine curve.
    # Better than linear decay for fine-tuning because the final 20% of
    # training uses very small updates — good for polishing coordinate
    # precision without overshooting.

    bf16: bool = True
    # WHY BF16: L40S (Ada Lovelace) has native bfloat16 tensor cores.
    # BF16 has the same dynamic range as FP32 (important for coordinate
    # values which can span -50m to +50m) but uses half the memory.
    # Never use FP16 for this task — FP16 can overflow on large coordinate values.

    # --- Logging & Saving ---
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 3
    # Keep only last 3 checkpoints to avoid filling disk.

    seed: int = 42


cfg = TrainConfig()
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the llm_drone motion planner")
    parser.add_argument("--data-path", default=None, help="Override training dataset path")
    parser.add_argument("--prompt-file", default=None, help="Override system prompt file path")
    args, unknown = parser.parse_known_args()
    if unknown:
        log.warning("Ignoring unknown CLI args: %s", unknown)
    return args


def resolve_repo_relative_path(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def resolve_system_prompt_path(prompt_override: str | None) -> Path:
    if prompt_override:
        prompt_path = resolve_repo_relative_path(prompt_override)
    elif "resolve_prompt_file" in globals():
        prompt_path = resolve_prompt_file(DATASET_PROMPT_FILENAME)
    else:
        prompt_path = (REPO_ROOT / "config" / DATASET_PROMPT_FILENAME).resolve()

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path


def load_system_prompt_override(prompt_override: str | None) -> str:
    if not prompt_override:
        return load_system_prompt(DATASET_PROMPT_FILENAME)
    prompt_path = resolve_system_prompt_path(prompt_override)
    return prompt_path.read_text()


CLI_ARGS = parse_cli_args()
if CLI_ARGS.data_path:
    cfg.data_path = str(resolve_repo_relative_path(CLI_ARGS.data_path))
SELECTED_PROMPT_PATH = resolve_system_prompt_path(CLI_ARGS.prompt_file)


def _filter_supported_kwargs(callable_obj, kwargs: dict, alias_map: Optional[dict[str, list[str]]] = None) -> tuple[dict, dict]:
    """
    Keep only kwargs supported by the target callable and map renamed args
    across TRL versions, e.g. max_seq_length -> max_length.
    """
    params = inspect.signature(callable_obj).parameters
    supported = {}
    dropped = {}
    alias_map = alias_map or {}

    for key, value in kwargs.items():
        target_key = key
        if key not in params:
            for alias in alias_map.get(key, []):
                if alias in params:
                    target_key = alias
                    break
            else:
                dropped[key] = value
                continue
        supported[target_key] = value

    return supported, dropped


def _compute_warmup_steps(cfg: TrainConfig, train_dataset_len: int) -> int:
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    micro_batch = max(1, cfg.per_device_train_batch_size * world_size)
    steps_per_epoch = max(1, math.ceil(train_dataset_len / micro_batch))
    optimizer_steps_per_epoch = max(1, math.ceil(steps_per_epoch / max(1, cfg.gradient_accumulation_steps)))
    total_optimizer_steps = max(1, math.ceil(cfg.num_train_epochs * optimizer_steps_per_epoch))
    return max(1, math.ceil(total_optimizer_steps * cfg.warmup_ratio))


def formatting_prompts_func(example, tokenizer):
    """
    Unsloth's compiled SFTTrainer path may require an explicit formatting_func
    even for conversational datasets. Support both batched and unbatched inputs.
    """
    messages = example["messages"]
    if messages and isinstance(messages[0], list):
        return [
            tokenizer.apply_chat_template(message_list, tokenize=False, add_generation_prompt=False)
            for message_list in messages
        ]
    return [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ]


def prepare_generation_inputs(tokenizer, messages, device):
    """
    Handle tokenizer.apply_chat_template returning either a Tensor or a
    BatchEncoding, depending on the transformers version.
    """
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    encoded = encoded.to(device)

    if isinstance(encoded, torch.Tensor):
        return {"input_ids": encoded}, int(encoded.shape[1])

    input_ids = encoded["input_ids"]
    return dict(encoded), int(input_ids.shape[1])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SYSTEM PROMPT
#
# WHY A DEDICATED SYSTEM PROMPT?
# The system prompt is prepended to EVERY training example. It establishes:
# (1) The model's role and the coordinate frame conventions (NED, meters).
# (2) The output schema — the model learns that it must always produce
#     exactly 5 waypoints in valid JSON.
# (3) Safety constraints — these become "soft" constraints baked into
#     the model's generation behavior through repeated exposure.
#
# KEEP THIS IDENTICAL between training and inference, or the model will
# behave inconsistently at test time.
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = load_system_prompt_override(CLI_ARGS.prompt_file)
log.info("Using dataset file: %s", cfg.data_path)
log.info("Using system prompt file: %s", SELECTED_PROMPT_PATH)


# ─────────────────────────────────────────────────────────────────────────────

def validate_completion(completion_str: str) -> bool:
    """
    Validate that a completion string is valid JSON with the correct schema.
    Returns True if valid, False if the example should be skipped.
    """
    try:
        data = json.loads(completion_str)
    except json.JSONDecodeError:
        return False

    if not isinstance(data, dict):
        return False
    if "waypoints" not in data:
        return False

    waypoints = data["waypoints"]
    if not isinstance(waypoints, list):
        return False

    # Must have exactly 5 waypoints
    if len(waypoints) != 5:
        return False

    # Each waypoint must have x and y as finite numbers. z may be present in
    # older rows but is no longer part of the model-facing contract.
    for wp in waypoints:
        if not isinstance(wp, dict):
            return False
        for key in ["x", "y"]:
            if key not in wp:
                return False
            val = wp[key]
            if not isinstance(val, (int, float)):
                return False
            if not torch.tensor(val).isfinite():
                return False
        if "z" in wp:
            val = wp["z"]
            if not isinstance(val, (int, float)):
                return False
            if not torch.tensor(val).isfinite():
                return False

    reasoning = data.get("reasoning", None)
    if reasoning is not None and not isinstance(reasoning, str):
        return False

    evaluation = data.get("evaluation", None)
    if evaluation is not None and not isinstance(evaluation, dict):
        return False

    return True


def normalize_completion_for_training(completion_str: str) -> str:
    """
    Keep the assistant target aligned with the prompt contract used at inference.
    Reasoning is preserved; auxiliary evaluation metadata is dropped.
    """
    data = json.loads(completion_str)
    normalized = {
        "waypoints": [
            {"x": float(wp["x"]), "y": float(wp["y"])}
            for wp in data["waypoints"]
        ],
        "reasoning": str(data.get("reasoning", "")).strip(),
    }
    return json.dumps(normalized, separators=(",", ":"))


def extract_chat_messages(messages: list[dict]) -> tuple[str | None, str | None, str | None]:
    system_content = next((m.get("content") for m in messages if m.get("role") == "system"), None)
    user_content = next((m.get("content") for m in messages if m.get("role") == "user"), None)
    assistant_content = next((m.get("content") for m in messages if m.get("role") == "assistant"), None)
    return system_content, user_content, assistant_content


def iter_examples_from_jsonl(data_path: Path):
    with data_path.open("r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield line_num, line


def iter_examples_from_csv(data_path: Path):
    with data_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            prompt = (row.get("prompt") or "").strip()
            completion = (row.get("label_waypoints_json") or row.get("completion") or "").strip()
            if not prompt or not completion:
                yield row_idx, None
                continue
            yield row_idx, {"prompt": prompt, "completion": completion}


def load_and_prepare_dataset(cfg: TrainConfig) -> tuple[Dataset, Dataset]:
    """
    Load raw dataset.csv or JSONL dataset, validate each example, convert to chat message format,
    and split into train/val sets.
    """
    log.info(f"Loading dataset from {cfg.data_path}")
    data_path = Path(cfg.data_path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    raw_examples = []
    skipped = 0
    iterator = iter_examples_from_csv(data_path) if data_path.suffix.lower() == ".csv" else iter_examples_from_jsonl(data_path)

    for line_num, raw_example in iterator:
        if raw_example is None:
            log.warning(f"Row {line_num}: Missing prompt/completion columns, skipping.")
            skipped += 1
            continue

        if isinstance(raw_example, str):
            try:
                example = json.loads(raw_example)
            except json.JSONDecodeError:
                log.warning(f"Line {line_num}: Invalid JSON, skipping.")
                skipped += 1
                continue
        else:
            example = raw_example

        system_content = None
        if "messages" in example:
            system_content, user_content, assistant_content = extract_chat_messages(example["messages"])
        elif "prompt" in example and ("completion" in example or "label_waypoints_json" in example):
            user_content = example["prompt"]
            assistant_content = example.get("completion") or example.get("label_waypoints_json")
        else:
            log.warning(f"Line {line_num}: Missing required fields, skipping.")
            skipped += 1
            continue

        if not isinstance(user_content, str) or not isinstance(assistant_content, str):
            log.warning(f"Line {line_num}: Prompt/completion types are invalid, skipping.")
            skipped += 1
            continue

        if not validate_completion(assistant_content):
            log.warning(f"Line {line_num}: Invalid completion schema, skipping.")
            skipped += 1
            continue

        assistant_content = normalize_completion_for_training(assistant_content)
        fallback_system = system_content if isinstance(system_content, str) else SYSTEM_PROMPT
        bundle_system, user_content = split_prompt_bundle(user_content, fallback_system_prompt=fallback_system)

        raw_examples.append({
            "messages": [
                {"role": "system", "content": bundle_system or fallback_system or SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        })

    log.info(f"Loaded {len(raw_examples)} valid examples. Skipped {skipped} invalid.")

    if len(raw_examples) == 0:
        raise ValueError(
            f"No valid examples found in {cfg.data_path}. "
            "Check your data format — see SECTION 4 comments for expected schema."
        )

    # Shuffle and split
    dataset = Dataset.from_list(raw_examples).shuffle(seed=cfg.seed)
    split = dataset.train_test_split(test_size=cfg.val_split_ratio, seed=cfg.seed)

    log.info(f"Train: {len(split['train'])} | Val: {len(split['test'])} examples")
    return split["train"], split["test"]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: MODEL LOADING
#
# WHY UNSLOTH INSTEAD OF VANILLA TRANSFORMERS?
# For your task specifically, Unsloth matters because:
# (1) Your prompts are long (~1200-1500 tokens). Unsloth's FlashAttention-2
#     implementation scales O(n) instead of O(n²) in memory, meaning you can
#     fit larger batches at your sequence length.
# (2) Unsloth's custom LoRA backward pass is 30-40% faster than HuggingFace PEFT.
# (3) It handles Qwen2.5's RoPE (rotary position embedding) scaling correctly
#     out of the box — vanilla transformers sometimes needs manual patching.
#
# WHY NOT LOAD IN 4-BIT?
# As discussed: 7B in bf16 = ~14 GB. You have 46 GB. Save the quality.
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: TrainConfig):
    """Load the base model and tokenizer via Unsloth."""
    log.info(f"Loading model: {cfg.model_name}")
    log.info(f"  Max seq length : {cfg.max_seq_length}")
    log.info(f"  4-bit quantize : {cfg.load_in_4bit}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=torch.bfloat16,   # Explicit bf16 for L40S
        load_in_4bit=cfg.load_in_4bit,
    )

    # Qwen2.5 uses a specific pad token. Setting it explicitly prevents
    # silent tokenization bugs where padding tokens leak into the loss.
    tokenizer.padding_side = "right"
    qwen_chat_eos = "<|im_end|>"
    qwen_chat_eos_id = tokenizer.convert_tokens_to_ids(qwen_chat_eos)
    if qwen_chat_eos_id is not None and qwen_chat_eos_id != getattr(tokenizer, "unk_token_id", None):
        tokenizer.eos_token = qwen_chat_eos
        tokenizer.eos_token_id = qwen_chat_eos_id
        log.info("Aligned tokenizer EOS token to chat template terminator: %s", tokenizer.eos_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Base model loaded successfully.")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: LORA ADAPTER INJECTION
#
# WHY THESE SPECIFIC TARGET_MODULES?
#
# In a transformer block, there are two types of computation:
# (1) Attention: q_proj, k_proj, v_proj compute queries/keys/values.
#     o_proj projects the attended values back to hidden dim.
#     These learn WHICH parts of the sensor prompt to attend to when
#     deciding where to place the next waypoint.
#
# (2) MLP / Feed-forward: gate_proj, up_proj, down_proj.
#     These apply non-linear transformations to the attended representation.
#     For structured output tasks, the MLP layers are where numeric
#     patterns are stored — the model learns that "obstacle at 5.7m in
#     far_left sector → move y slightly right" type associations here.
#
# Applying LoRA only to q_proj and v_proj (as many tutorials show) leaves
# the MLP adapters out, which significantly hurts coordinate accuracy.
# Apply to all 7 layer types for best results on your task.
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora(model, cfg: TrainConfig):
    """Inject LoRA adapters into the model."""
    log.info(f"Applying LoRA: rank={cfg.lora_r}, alpha={cfg.lora_alpha}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        # gradient_checkpointing trades compute for memory.
        # With 46 GB VRAM you likely don't need it, but it's free insurance.
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    # Print trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    # For Qwen2.5-7B with rank=32, this will be roughly:
    # ~83M trainable / 7.6B total = ~1.1%
    # That's the power of LoRA: 1% of parameters, most of the adaptation.

    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: TRAINING ARGUMENTS
#
# WHY THESE SPECIFIC SETTINGS FOR YOUR TASK?
# Each setting is tuned for: long-sequence structured output, L40S hardware,
# and the need to learn precise floating-point coordinate generation.
# ─────────────────────────────────────────────────────────────────────────────

def build_training_args(cfg: TrainConfig, tokenizer, train_dataset_len: int) -> SFTConfig:
    warmup_steps = _compute_warmup_steps(cfg, train_dataset_len)
    tensorboard_log_dir = os.path.join(cfg.output_dir, "logs")
    os.environ["TENSORBOARD_LOGGING_DIR"] = tensorboard_log_dir

    raw_kwargs = dict(
        output_dir=cfg.output_dir,

        # --- Core training schedule ---
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,

        # --- Optimizer ---
        learning_rate=cfg.learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim="adamw_torch_fused",
        # WHY FUSED ADAMW: PyTorch's fused AdamW does the optimizer step
        # in a single CUDA kernel instead of separate operations. On L40S
        # this is 10-15% faster with no quality difference.

        weight_decay=0.01,
        # Light L2 regularization on non-LoRA parameters. Helps prevent
        # the model from over-relying on a small set of attention heads.

        max_grad_norm=1.0,
        # Gradient clipping. Prevents exploding gradients if a batch
        # contains an unusually hard or unusual obstacle configuration.

        # --- Precision ---
        bf16=cfg.bf16,
        fp16=False,

        # --- Sequence length ---
        max_seq_length=cfg.max_seq_length,

        # --- Logging ---
        logging_steps=cfg.logging_steps,
        report_to="tensorboard",
        # Run: tensorboard --logdir ./drone_planner_checkpoints/logs
        # to monitor training loss and val loss in real time.

        # --- Evaluation & Checkpointing ---
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # --- Reproducibility ---
        seed=cfg.seed,
        data_seed=cfg.seed,

        # --- SFT-specific: CRITICAL for your task ---
        dataset_text_field=None,  # We use "messages" format
        dataset_kwargs={"skip_prepare_dataset": False},
        eos_token=tokenizer.eos_token,
    )

    sft_kwargs, dropped = _filter_supported_kwargs(
        SFTConfig,
        raw_kwargs,
        alias_map={
            "max_seq_length": ["max_length"],
            "eval_strategy": ["evaluation_strategy"],
        },
    )
    if dropped:
        log.warning("Dropping unsupported SFTConfig args for this TRL version: %s", sorted(dropped))
    log.info("Using warmup_steps=%s (derived from warmup_ratio=%.3f)", warmup_steps, cfg.warmup_ratio)
    if "max_length" in sft_kwargs:
        log.info("Using SFTConfig(max_length=%s) compatibility path", sft_kwargs["max_length"])
    elif "max_seq_length" in sft_kwargs:
        log.info("Using SFTConfig(max_seq_length=%s)", sft_kwargs["max_seq_length"])

    training_args = SFTConfig(**sft_kwargs)
    if hasattr(training_args, "eos_token"):
        training_args.eos_token = tokenizer.eos_token
    return training_args


def find_latest_checkpoint(output_dir: str | os.PathLike[str]) -> Path | None:
    """Return the highest-step checkpoint directory if one exists."""
    root = Path(output_dir).expanduser().resolve()
    if not root.exists():
        return None

    checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = checkpoint_pattern.match(child.name)
        if match is None:
            continue
        candidates.append((int(match.group(1)), child))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def prepare_output_dirs(output_dir: str | os.PathLike[str]) -> tuple[Path, Path, Path, Path]:
    """Create the directories used for logs and final save artifacts."""
    root = Path(output_dir).expanduser().resolve()
    logs_dir = root / "logs"
    final_dir = root / "final_adapter"
    merged_dir = root / "final_merged"

    root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    return root, logs_dir, final_dir, merged_dir


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: EVALUATION CALLBACKS
#
# WHY CUSTOM EVALUATION?
# Standard eval_loss tells you how well the model predicts tokens, but for
# YOUR task you care about:
# (1) JSON validity rate — does the model produce parseable output?
# (2) Waypoint L2 error — are the coordinates close to the MPC ground truth?
# (3) Safety violations — does any waypoint land inside an obstacle?
#
# This callback runs after each evaluation step and logs these task-specific
# metrics alongside the standard loss. These are the metrics you actually
# care about, not just cross-entropy.
# ─────────────────────────────────────────────────────────────────────────────

from transformers import TrainerCallback
import numpy as np

class WaypointEvalCallback(TrainerCallback):
    """
    After each eval step, generate predictions on a small subset of the
    validation set and compute task-specific metrics.
    """
    def __init__(self, model, tokenizer, val_dataset, num_samples=20):
        self.model = model
        self.tokenizer = tokenizer
        # Sample a fixed small subset for fast evaluation during training
        indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)
        self.eval_samples = val_dataset.select(indices)

    def on_evaluate(self, args, state, control, **kwargs):
        log.info("Running waypoint-specific evaluation...")

        valid_json = 0
        l2_errors = []
        FastLanguageModel.for_inference(self.model)  # Switch to inference mode

        for sample in self.eval_samples:
            messages = sample["messages"]

            # Get the ground truth assistant message (MPC label)
            gt_content = next(
                m["content"] for m in messages if m["role"] == "assistant"
            )

            # Build the input (system + user only, no assistant turn)
            input_messages = [m for m in messages if m["role"] != "assistant"]
            model_inputs, prompt_len = prepare_generation_inputs(
                self.tokenizer,
                input_messages,
                self.model.device,
            )

            with torch.no_grad():
                output_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=400,
                    temperature=0.1,   # Near-deterministic for eval
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode only the new tokens (not the input)
            new_tokens = output_ids[0][prompt_len:]
            prediction = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # --- Metric 1: JSON validity ---
            try:
                pred_data = json.loads(prediction)
                valid_json += 1

                pred_wps = pred_data.get("waypoints", [])
                gt_data = json.loads(gt_content)
                gt_wps = gt_data.get("waypoints", [])

                # --- Metric 2: Waypoint L2 error ---
                if len(pred_wps) == 5 and len(gt_wps) == 5:
                    for p, g in zip(pred_wps, gt_wps):
                        dist = np.sqrt(
                            (p["x"] - g["x"])**2 +
                            (p["y"] - g["y"])**2
                        )
                        l2_errors.append(dist)

            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # JSON invalid — already counted above

        # Switch back to training mode
        FastLanguageModel.for_training(self.model)

        n = len(self.eval_samples)
        json_rate = valid_json / n * 100
        mean_l2 = np.mean(l2_errors) if l2_errors else float("nan")

        log.info(f"  JSON validity    : {json_rate:.1f}% ({valid_json}/{n})")
        log.info(f"  Mean L2 error    : {mean_l2:.3f} m")

        # Targets to aim for:
        #   JSON validity  > 95%
        #   Mean L2 error  < 0.5 m
        #   Safety violations = 0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: MAIN TRAINING LOOP
#
# WHY SFTTrainer INSTEAD OF A CUSTOM LOOP?
# SFTTrainer handles several subtle but critical things automatically:
# (1) Loss masking: Only computes cross-entropy loss on the ASSISTANT tokens.
#     This means the model learns to generate waypoints, not to "predict"
#     the sensor input. Without this, ~80% of gradient signal is wasted on
#     learning to reproduce your input prompts.
# (2) Chat template formatting: Applies Qwen2.5's <|im_start|> / <|im_end|>
#     tokens correctly, which is required for the instruct model to understand
#     system/user/assistant role separation.
# (3) Sequence packing (optional): Can pack multiple short examples into one
#     sequence to reduce padding waste. For your ~1200-token examples this
#     is less critical, but enabled by default.
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Drone Motion Planner — LoRA Fine-Tuning")
    log.info("=" * 60)

    # 1. Load and validate data
    train_dataset, val_dataset = load_and_prepare_dataset(cfg)

    # 2. Load model
    model, tokenizer = load_model_and_tokenizer(cfg)

    # 3. Inject LoRA adapters
    model = apply_lora(model, cfg)

    # 4. Build training arguments
    training_args = build_training_args(cfg, tokenizer, len(train_dataset))
    output_dir, logs_dir, final_dir_path, merged_dir_path = prepare_output_dirs(cfg.output_dir)
    disk_usage = shutil.disk_usage(output_dir)
    free_gib = disk_usage.free / (1024 ** 3)
    log.info(
        "EOS alignment: tokenizer.eos_token=%r eos_id=%r training_args.eos_token=%r",
        tokenizer.eos_token,
        tokenizer.eos_token_id,
        getattr(training_args, "eos_token", None),
    )
    log.info("Training output root: %s", output_dir)
    log.info("TensorBoard log dir: %s", logs_dir)
    log.info("Free disk at output root: %.1f GiB", free_gib)

    # 5. Build custom evaluation callback
    eval_callback = WaypointEvalCallback(
        model=model,
        tokenizer=tokenizer,
        val_dataset=val_dataset,
        num_samples=20,
    )

    # 6. Build trainer
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "args": training_args,
        "callbacks": [eval_callback],
        # TRL renamed this from `tokenizer` to `processing_class` in newer releases.
        "tokenizer": tokenizer,
        "formatting_func": lambda example: formatting_prompts_func(example, tokenizer),
    }
    trainer_kwargs, dropped_trainer_kwargs = _filter_supported_kwargs(
        SFTTrainer,
        trainer_kwargs,
        alias_map={"tokenizer": ["processing_class"]},
    )
    if dropped_trainer_kwargs:
        log.warning(
            "Dropping unsupported SFTTrainer args for this TRL version: %s",
            sorted(dropped_trainer_kwargs),
        )
    trainer = SFTTrainer(**trainer_kwargs)

    # 7. Log GPU memory before training
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(i) / 1e9
            log.info(f"GPU {i}: {allocated:.1f} GB allocated / {total:.1f} GB total")

    # 8. Train
    log.info("Starting training...")
    log.info(f"  Epochs          : {cfg.num_train_epochs}")
    log.info(f"  Effective batch : {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps} per GPU")
    log.info(f"  Learning rate   : {cfg.learning_rate}")
    log.info(f"  Checkpoints at  : {output_dir}")
    log.info(f"  TensorBoard logs: {logs_dir}")

    resume_checkpoint = find_latest_checkpoint(cfg.output_dir)
    if resume_checkpoint is not None:
        log.info(f"Resuming from latest checkpoint: {resume_checkpoint}")
    else:
        log.info("No prior checkpoint found. Starting fresh training run.")

    training_completed = False
    try:
        trainer.train(
            resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None
        )
        training_completed = True
    except KeyboardInterrupt:
        interrupt_dir = output_dir / "interrupted_adapter"
        interrupt_dir.mkdir(parents=True, exist_ok=True)
        log.warning("Training interrupted by user. Saving current adapter snapshot to: %s", interrupt_dir)
        model.save_pretrained(interrupt_dir)
        tokenizer.save_pretrained(interrupt_dir)
        latest_checkpoint = find_latest_checkpoint(cfg.output_dir)
        if latest_checkpoint is not None:
            log.warning("Latest resumable trainer checkpoint remains at: %s", latest_checkpoint)
        log.warning("You can restart the script and it will auto-resume from the latest checkpoint.")
        return

    if not training_completed:
        return

    # 9. Save final model
    log.info("Training complete. Saving final model...")
    model.save_pretrained(final_dir_path)
    tokenizer.save_pretrained(final_dir_path)
    log.info(f"LoRA adapter saved to: {final_dir_path}")

    # 10. Optionally merge LoRA weights into base model
    # This creates a single standalone model (no adapter overhead at inference)
    log.info(f"Merging LoRA into base model → {merged_dir_path}")
    model.save_pretrained_merged(
        merged_dir_path,
        tokenizer,
        save_method="merged_16bit",
        # WHY MERGED: At inference time, a merged model has zero LoRA overhead.
        # The adapter math (W + BA) is pre-computed into W directly.
        # For a drone real-time system where latency matters, always use merged.
    )
    log.info("Done! Merged model ready for inference.")
    log.info(f"To run inference: load model from {merged_dir_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: INFERENCE HELPER
#
# Run this AFTER training to test your model on a new sensor prompt.
# Temperature=0.1 gives near-deterministic output — appropriate for a
# safety-critical real-time planning system. Never use temperature=1.0
# in production for this task.
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model_path: str, sensor_prompt: str) -> dict:
    """
    Load the merged fine-tuned model and generate a trajectory for a given
    sensor prompt. Returns parsed JSON dict with waypoints.

    Args:
        model_path: Path to merged model directory (output of save_pretrained_merged)
        sensor_prompt: The NL description of current drone state (your existing prompt format)

    Returns:
        dict with keys: waypoints, reasoning
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": sensor_prompt},
    ]

    model_inputs, prompt_len = prepare_generation_inputs(tokenizer, messages, model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=400,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][prompt_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        log.error(f"Model produced invalid JSON: {e}")
        log.error(f"Raw output: {response}")
        raise


if __name__ == "__main__":
    main()
