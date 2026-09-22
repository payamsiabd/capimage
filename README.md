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
| `MODEL` | `capimagine` | `qwen25vl` evaluates the Qwen2.5-VL-7B base model to sanity-check the pipeline (download it with `WITH_BASELINE=1 bash scripts/download.sh`) |
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

## Repository layout

```
configs/vstar_capimagine.json   VLMEvalKit model + dataset config for CapImagine-7B
configs/vstar_qwen25vl.json     same protocol for the Qwen2.5-VL-7B base model
scripts/setup_env.sh            pinned environment + VLMEvalKit@0bac5c0
scripts/download.sh             model weights and V* data
scripts/run_vstar.sh            judge check -> VLMEvalKit inference + evaluation -> summary
scripts/check_judge.py          fails fast if the judge API does not answer
scripts/summarize_vstar.py      Attribute / Spatial / Overall vs. the paper, plus sanity warnings
tests/                          offline tests (pytest): summary logic, plus VLMEvalKit's own V* prompt
                                and scoring code on synthetic outputs (runs when vlmeval is installed)
```
