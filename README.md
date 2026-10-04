# Reproducing CapImagine-7B on V* Bench

This repo reproduces the **V\*** result for **CapImagine-7B** from *Imagination Helps Visual Reasoning, But Not Yet in Latent Space* (Li et al., ICML 2026, [arXiv:2602.22766](https://arxiv.org/abs/2602.22766)). It uses the released weights ([Michael4933/CapImagine-7B](https://huggingface.co/Michael4933/CapImagine-7B)) and follows the paper and the official repo ([AI9Stars/CapImagine](https://github.com/AI9Stars/CapImagine)).

## Target (paper, Table 1)

| Model | V* Overall | Attribute | Spatial |
|---|---|---|---|
| **CapImagine-7B** | **85.9** (164/191) | **87.8** (101/115) | **82.9** (63/76) |
| Qwen2.5-VL-7B (base model, sanity check) | 76.4 | 77.4 | 75.0 |

One question is worth 0.52 points overall, 0.87 on Attribute and 1.32 on Spatial.

## Evaluation protocol and where it comes from

The official CapImagine repo has no evaluation code. Its README says inference runs through the official Qwen2.5-VL code. The paper (§5.1) says the benchmarks follow "the experimental protocol of Monet" and that final answers are extracted with an "LLM-as-a-judge protocol". So the settings below come from [Monet's README](https://github.com/NOVAglow646/Monet#evaluation) and from [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), the harness Monet used.

| Setting | Value | Source |
|---|---|---|
| Harness | VLMEvalKit, dataset `VStarBench` (191 multiple-choice questions: 115 attribute, 76 spatial) | Monet README: "We evaluate Monet-7B on VLMEvalKit" |
| VLMEvalKit version | commit [`0bac5c0`](https://github.com/open-compass/VLMEvalKit/commit/0bac5c064e8216037141d2cbd525e66b4b1dce99) (2026-02-25) | The commit current when the CapImagine weights were released (2026-02-25). Later commits changed MCQ answer matching (2026-06-17) and judge defaults (2026-08-28). |
| Model wrapper | `Qwen2VLChat`: plain Qwen2.5-VL, no latent-token decoding | CapImagine README: inference "through the official code from Qwen2.5-VL" |
| System prompt | `You are a helpful multimodal assistant. You are required to answer the question based on the image provided. Put your final answer in \boxed{}.` | Monet README, "To accurately reproduce the result" |
| User prompt | image, then `Question: {q}\nOptions:\nA. …\nB. …\nPlease select the correct answer from the options above. \n` | VLMEvalKit `ImageMCQDataset.build_prompt` (`use_custom_prompt=False`) |
| Image resolution | `min_pixels = 1280·28²`, `max_pixels = 16384·28²` | VLMEvalKit preset for `Qwen2.5-VL-7B-Instruct` |
| Decoding | greedy (transformers `top_k=1`; vLLM `temperature=0`), `max_new_tokens=2048` | VLMEvalKit `Qwen2VLChat` defaults |
| Answer extraction | VLMEvalKit's rule-based matcher first, then an LLM judge when the rules can't find an option | Paper §5.1; Monet README: "we replace the original exact matching judgement with API judge" |
| Judge model | `chatgpt-0125` (= `gpt-3.5-turbo-0125`), temperature 0 | VLMEvalKit's default judge for MCQ datasets at `0bac5c0` |
| Libraries | Python 3.10, `transformers==4.54.0`, `vllm==0.10.0` (pulls `torch==2.7.1`), `qwen-vl-utils==0.0.11`, `flash-attn==2.8.2` | Monet `requirements.txt` (CapImagine was trained with the Monet codebase, §5.1) |

### Choices the authors did not specify

These are documented best guesses. Each one could move the score by a question or two.

- **Judge model.** The paper does not name it, so this uses VLMEvalKit's default. If `gpt-3.5-turbo-0125` is no longer available on your account, use `JUDGE=gpt-4o-mini`.
- **Inference backend.** HF transformers is the default, following the CapImagine README. Monet used vLLM (`USE_VLLM=1`). Both decode greedily, but bf16 kernels differ, so a few long generations can diverge.
- **System prompt.** Monet's SFT code trains with `You are a helpful assistant.`. Monet's eval prompt (used here) adds the `\boxed{}` instruction. To try the training prompt, set `"system_prompt": null` in the config; the Qwen chat template then inserts `You are a helpful assistant.`.
- **Checkpoint.** The paper picks the best checkpoint during training (§5.1). This assumes the released weights are that checkpoint.
- **Baseline protocol.** The Qwen2.5-VL-7B number is the authors' own run. Its exact settings aren't stated; here it gets the same protocol as CapImagine.

## How to run

**You need:** Linux; one NVIDIA GPU with ≥40 GB (A100/H100/A800), because V* images are high-resolution and take up to ~16k visual tokens; conda; ~25 GB of disk; access to huggingface.co; and an API key for the judge.

```bash
git clone https://github.com/payamsiabd/capimage.git && cd capimage

bash scripts/setup_env.sh          # conda env "capimagine", pinned libs, VLMEvalKit@0bac5c0 in third_party/
conda activate capimagine

# VLMEvalKit reads judge credentials from its .env file
echo "OPENAI_API_KEY=sk-..." >> third_party/VLMEvalKit/.env

bash scripts/download.sh           # weights -> models/CapImagine-7B, VStarBench.tsv -> ~/LMUData
bash scripts/run_vstar.sh          # inference + judge + comparison with the paper
```

`run_vstar.sh` checks the judge before it starts. VLMEvalKit would otherwise fall back to exact matching without telling you, which lowers the score on long reasoning outputs.

| Env var | Default | Meaning |
|---|---|---|
| `MODEL` | `capimagine` | `qwen25vl` evaluates the Qwen2.5-VL-7B base model to sanity-check the pipeline (download it with `WITH_BASELINE=1 bash scripts/download.sh`). `custom` evaluates your own merged fine-tune at `MODEL_PATH` (see [LoRA fine-tuning](#lora-fine-tuning-on-capimagine-data)). |
| `MODEL_NAME` | basename of `MODEL_PATH` | Output name for `MODEL=custom` |
| `REFERENCE` | `CapImagine-7B` for `custom` | Which paper row the summary compares against |
| `JUDGE` | `chatgpt-0125` | Any VLMEvalKit judge name, e.g. `gpt-4o-mini`. `exact_matching` turns the LLM judge off (not the paper protocol). |
| `USE_VLLM` | `0` | `1` uses VLMEvalKit's vLLM backend (faster; doesn't need flash-attn) |
| `NGPU` | `1` | Number of data-parallel processes for the transformers backend (uses `torchrun`) |
| `MODE` | `all` | `infer` generates predictions only. `eval` re-scores existing predictions, e.g. with a different `JUDGE`. |
| `REUSE` | `1` for `MODE=eval`, else `0` | `1` reuses the latest earlier predictions (VLMEvalKit `--reuse`). Without it, VLMEvalKit moves earlier outputs from the same day into `bak_*` and starts over. |
| `MODEL_PATH` | `models/<name>`, or the HF repo id | Where to load the weights from |

**No OpenAI key?** Serve any instruct LLM with vLLM's OpenAI-compatible server and add this to `third_party/VLMEvalKit/.env`. It departs from the paper's judge, so report it as such.

```
OPENAI_API_KEY=EMPTY
OPENAI_API_BASE=http://localhost:8000/v1/chat/completions
LOCAL_LLM=<served model name>
```

## Output

The run ends with a table like this:

```
| Split | n | correct | ours | paper | diff | boxed-only (diagnostic) |
|---|---|---|---|---|---|---|
| Attribute | 115 | … | … | 87.8 | … | … |
| Spatial | 76 | … | … | 82.9 | … | … |
| Overall | 191 | … | … | 85.9 | … | … |
```

The files are in `outputs/CapImagine-7B/T<date>_G0bac5c06/`:

| File | Contents |
|---|---|
| `CapImagine-7B_VStarBench.xlsx` | Raw model outputs, one row per question |
| `CapImagine-7B_VStarBench_openai_result.xlsx` | Per-question verdict (`hit`) and the matcher/judge log |
| `CapImagine-7B_VStarBench_acc.csv` | VLMEvalKit's own accuracy table |
| `CapImagine-7B_VStarBench_openai_result_summary.json` | The comparison table above, as JSON |

The **boxed-only** column is a judge-free check that reads the answer only from the last `\boxed{}` in each output. It is not the paper's metric; it shows how much the judge changes the score. The summary also warns if the judge fell back to exact matching or if VLMEvalKit substituted a random option after judge failures.

## LoRA fine-tuning on CapImagine-Data

`train/` fine-tunes Qwen2.5-VL-7B-Instruct with LoRA adapters (PEFT) on [CapImagine-Data](https://huggingface.co/datasets/Michael4933/CapImagine-Data), the 17k-sample text-imagination dataset from the paper (§4.2). The paper itself fine-tunes all of the language model's weights with the Monet codebase. This is the LoRA counterpart, so expect it to land near CapImagine-7B, not necessarily on it.

| Setting | Value | Source |
|---|---|---|
| Base model | Qwen2.5-VL-7B-Instruct, vision tower frozen | Paper §5.1; Monet `src/main.py` freezes `model.visual` |
| Global batch | 128 (8 GPUs × batch 1 × grad accum 16; `GLOBAL_BATCH` keeps it constant on fewer GPUs) | Paper §5.1 and Monet's SFT launch script |
| Epochs / warmup / weight decay / precision | 4 / 10 steps / 0.01 / bf16, linear decay | Monet `script_examples/sft_stage1.sh` and `src/main.py` |
| System prompt | `You are a helpful assistant.` (replaces the data's own) | Monet `src/task.py` |
| Image budget | ≤2000 visual tokens per sample, ≤1280 per image once over budget | Monet `resize_by_token_budget` defaults |
| Loss | Assistant tokens only, including the closing `<|im_end|>`, averaged over all tokens in an optimizer step | Standard SFT masking (see the note on loss scaling below) |
| LoRA | rank 64, alpha 128, dropout 0.05, lr 1e-4, on every q/k/v/o and gate/up/down projection of the language model | Our choice; common LoRA settings, not from the paper |

**Loss scaling.** In transformers 4.54, Qwen2.5-VL's `forward` accepts `num_items_in_batch` but never uses it. Trainer then assumes the model already normalizes the loss and skips dividing it by the accumulation steps, so with the built-in loss every step's loss and gradients are 16× too large. `train_lora.py` passes its own `compute_loss_func` to fix this, and a test checks that the loss of a random model starts near ln(vocab).

**The data format could not be inspected.** Hugging Face was unreachable from the sandbox where this code was written. The loader expects Monet's layout (`{"data": [{"role", "content": [{"type": "image" | "text", ...}]}]}`), since CapImagine-Data was rewritten from Monet-SFT-125K. It also accepts OpenAI-style `messages` and LLaVA-style `conversations` records. Anything else fails at load time with the offending record printed. CapImagine verbalizes intermediate images as text (`<think_image>…</think_image>`), so an image inside an assistant turn is treated as a format error. Pass `--assistant_images drop` to strip such images instead. **Run `inspect_data` before training** and check that its output looks like the paper's Figure 3.

```bash
bash scripts/setup_env.sh && conda activate capimagine   # now also installs peft==0.17.1
bash scripts/download_data.sh                             # CapImagine-Data (+ unzip images) and the base model
python -m train.inspect_data --data_path data/CapImagine-Data/<annotation>.json
bash scripts/train_lora.sh                                # all visible GPUs; adapters -> checkpoints/capimagine-lora
python -m train.merge_lora --adapter checkpoints/capimagine-lora --output checkpoints/capimagine-lora-merged
MODEL=custom MODEL_PATH=checkpoints/capimagine-lora-merged bash scripts/run_vstar.sh
```

`train_lora.sh` takes `DATA`, `IMAGE_ROOT`, `BASE_MODEL`, `OUTPUT_DIR`, `NGPU` and `GLOBAL_BATCH` as env vars. Any extra arguments go to `train.train_lora`, which also accepts every `transformers.TrainingArguments` flag, e.g. `--num_train_epochs 2 --report_to wandb`. An adapter is saved after each epoch in `checkpoint-*`. The paper picks the best checkpoint (§5.1), so merge and evaluate each one. GPU memory has not been measured. If you run out, lower `--max_length` (default 8192) or `--global_max_image_tokens`.

### Goldfish-loss variant

`scripts/train_lora_goldfish.sh` runs the same LoRA training with one change: the standard next-token loss is replaced by the **goldfish loss** from *Be like a Goldfish, Don't Memorize!* (Hans et al., NeurIPS 2024; [ahans30/goldfish-loss](https://github.com/ahans30/goldfish-loss)). The loss skips a pseudo-random ~1 in k of the supervised tokens, and the choice depends only on the surrounding text, so the model can't learn to reproduce training replies verbatim. Every other setting matches the standard run.

```bash
bash scripts/train_lora_goldfish.sh          # adapters -> checkpoints/capimagine-lora-goldfish
python -m train.merge_lora --adapter checkpoints/capimagine-lora-goldfish --output checkpoints/capimagine-lora-goldfish-merged
MODEL=custom MODEL_PATH=checkpoints/capimagine-lora-goldfish-merged bash scripts/run_vstar.sh
```

| Setting | Value | Source |
|---|---|---|
| Mask | `hash-table`: multiply the token ids of the `h` targets ending at each target (that target included), look the product up in a fixed random table (1,000,003 entries, seed 2971215073), and drop the target if the value is < 1/k. The first `h−1` targets of a sequence are never dropped. | `apply_goldfish` in the official `lit_gpt/utils.py`, the strategy behind the paper's main results |
| k (drop 1 in k) | 4 (`K_GOLDFISH`) | Paper default; released config |
| h (context width) | 13 (`GOLDFISH_H`) | Paper §3.1; released config `tinyllama-1b-equal-supervised-tokens.yaml` |
| Normalization | Mean over the kept tokens (paper Eq. 2), across the whole optimizer step and all GPUs | Paper Eq. 2 |
| Scope | Only assistant tokens are supervised to begin with. The hash still reads the true token ids, prompt included, so the start of a reply is masked by its actual context. | Adaptation to chat SFT; the reference hashes raw next tokens |

The flags are also available directly: `--goldfish_strategy hash-table --k_goldfish 4 --goldfish_context_width 13`. The mask is applied to the labels in the collator, so Trainer's token count already excludes dropped tokens; this is what makes the normalization exactly Eq. 2. The training log reports how many tokens of the first example were dropped.

**Fidelity notes.**
- On CPU the mask is bit-identical to the reference code: `tests/test_goldfish.py` runs a verbatim copy of it.
- The reference builds its random table on the training device. Here it's built on CPU, so every machine masks identically, but the values differ from the reference's CUDA table. The drop rule and the 1-in-k rate are the same.
- Two quirks of the reference's product hash are kept on purpose. The key ignores token order, and any window containing token id 0 (`!` in Qwen's vocabulary) gets key 0. With this table, `table[0] = 0.234 < 1/4`, so at k=4 the 13 targets starting at each `!` are always dropped.

**Fewer supervised tokens.** k=4 drops about 25% of the supervised tokens. The paper (§5.2) recovers the standard loss's quality by training on proportionally more tokens: a k/(k−1) larger batch, or more steps. That isn't applied here, so the comparison stays a single change. To try it, set `GLOBAL_BATCH=168` on 8 GPUs (≈ 128 × 4/3, rounded to a multiple of 8).

## Repository layout

```
configs/vstar_capimagine.json   VLMEvalKit model + dataset config for CapImagine-7B
configs/vstar_qwen25vl.json     same protocol for the Qwen2.5-VL-7B base model
scripts/setup_env.sh            pinned environment + VLMEvalKit@0bac5c0
scripts/download.sh             model weights and V* data
scripts/run_vstar.sh            judge check -> VLMEvalKit inference + evaluation -> summary
scripts/check_judge.py          fails fast if the judge API does not answer
scripts/summarize_vstar.py      Attribute / Spatial / Overall vs. the paper, plus sanity warnings
scripts/download_data.sh        CapImagine-Data and the Qwen2.5-VL-7B-Instruct base model
scripts/train_lora.sh           torchrun launcher with the paper's global batch size
scripts/train_lora_goldfish.sh  the same run with the goldfish loss (k=4, h=13)
train/data.py                   CapImagine-Data records -> Qwen2.5-VL chat messages
train/collator.py               tokenization, Monet's image budget, assistant-only labels
train/train_lora.py             PEFT LoRA + HF Trainer, with a gradient-accumulation-safe loss
train/goldfish.py               goldfish-loss token mask, ported from the official implementation
train/merge_lora.py             merges an adapter into a standalone model for evaluation
train/inspect_data.py           schema and image-path check of the dataset, no GPU needed
tests/                          offline tests (pytest): summary logic; VLMEvalKit's own V* prompt and
                                scoring code on synthetic outputs; LoRA training, label masking and
                                merging on a tiny random Qwen2.5-VL (CPU); goldfish mask vs. a verbatim
                                copy of the official code
```
