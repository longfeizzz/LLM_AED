# EVADE: LLM-Based Explanation Generation and Validation for Error Detection in NLI

## Notes Before Running

All commands below are written to be run from the repository root:

```bash
cd /Users/phoebeeeee/ongoing/LLM_AED
```

Several scripts in this repo use relative paths such as `../processing/...` and `../validation/...`. Those defaults only make sense if the script is launched from its own subdirectory, so this README uses explicit repo-root-relative paths to make the commands runnable from the root directory.

## Repository Structure

- `generation/`: explanation generation scripts and generated explanation folders.
- `processing/`: preprocessing script plus merged JSONL files.
- `validation/`: validation scripts, batch runner scripts, and validation outputs.
- `evaluation/`: thresholding and evaluation scripts plus saved evaluation results.
- `dataset/`: datasets used in the project.
- `fine-tuning/`: downstream fine-tuning shell scripts.
- `notebooks/`: notebooks for data preparation and analysis.
- `src/`: plotting and analysis scripts used during the project.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Explanation Generation

### Qwen Models

```bash
CUDA_VISIBLE_DEVICES=0 python generation/generate_explanation_qwen.py \
  --model_name Qwen/Qwen2.5-72B-Instruct \
  --jsonl_path dataset/varierr/varierr.json \
  --output_dir generation/qwen_72b_generation_raw
```

### LLaMA Models

```bash
CUDA_VISIBLE_DEVICES=0 python generation/generate_explanation_llama.py \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --jsonl_path dataset/varierr/varierr.json \
  --output_dir generation/llama_70b_generation_raw
```

- `--model_name`: model name or path.
- `--jsonl_path`: input VariErr JSONL file.
- `--output_dir`: output folder for one model's generations.

Current script status:

- `generation/generate_explanation_qwen.py` eagerly loads a default model at import time before parsing CLI arguments.
- `generation/generate_explanation_llama.py` currently references `args.model_id` internally although the CLI argument is `--model_name`.

So the output paths above are correct, but both generation scripts need code fixes before they can be reproduced reliably from this checkout.

### Generation Outputs

For each sample, the script writes files under:

- `generation/<model>_generation_raw/<sample_id>/`

The generation scripts currently write files named:

- `E` or `E_0.txt`: explanations for entailment / true
- `N` or `N_0.txt`: explanations for neutral / undetermined
- `C` or `C_0.txt`: explanations for contradiction / false

The preprocessing script reads the cleaned files named exactly `E`, `N`, and `C`. If you manually clean generations, keep the cleaned files in the same sample directory with those exact filenames.

## Preprocessing

### Manual Cleaning

Generated outputs are manually inspected and cleaned. For each sample, keep the cleaned files as:

- `generation/<model>_generation_raw/<sample_id>/E`
- `generation/<model>_generation_raw/<sample_id>/N`
- `generation/<model>_generation_raw/<sample_id>/C`

### Merge Cleaned Explanations

```bash
python processing/processing.py \
  --generation_dir generation \
  --input_jsonl dataset/varierr/varierr.json \
  --processing_dir processing \
  --all_dir generation_all.jsonl
```

- `--generation_dir`: directory containing all `*_generation_raw` folders.
- `--input_jsonl`: original VariErr JSONL file.
- `--processing_dir`: directory for per-model merged JSONL files.
- `--all_dir`: merged filename written inside `processing_dir`.

### Preprocessing Outputs

After this step, files are saved to:

- `processing/<model>_generation_raw.jsonl`
- `processing/generation_all.jsonl`

## Explanation Validation

### `one_expl`

Validate one explanation per prompt:

```bash
CUDA_VISIBLE_DEVICES=0 python validation/one_expl.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --model_type llama \
  --input_path processing/llama_8b_generation_raw.jsonl \
  --output_dir validation/validation_results/one_expl/llama_8b
```

Output:

- `validation/validation_results/one_expl/<model>/scores.json`

### `one_llm`

Validate all explanations from one source LLM in one prompt:

```bash
CUDA_VISIBLE_DEVICES=0 python validation/one_llm.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --model_type llama \
  --input_path processing/llama_8b_generation_raw.jsonl \
  --output_dir validation/validation_results/one_llm/llama_8b
```

Output:

- `validation/validation_results/one_llm/<model>/scores.json`

### `all_llm`

Validate explanations from multiple source LLMs together:

```bash
CUDA_VISIBLE_DEVICES=0 python validation/all_llm.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --model_type llama \
  --input_path processing/generation_all.jsonl \
  --output_dir validation/validation_results/all_llm/llama_8b
```

Output:

- `validation/validation_results/all_llm/<model>/scores.json`

### Run All Validation Modes

The batch scripts are inside `validation/`, but they do not currently run successfully as-is in this checkout:

- they expect to be launched from inside `validation/`
- they call `one-expl.py` and `one-llm.py`, while the actual files are `one_expl.py` and `one_llm.py`

So for now, run the three validation commands manually from the repository root instead of relying on:

```bash
bash validation/run_llama_all.sh 0
bash validation/run_qwen_all.sh 0
```

### Validation Output Format

All three validation scripts save:

- `<output_dir>/scores.json`

For `one_expl` and `one_llm`, keys look like:

```json
{
  "<sample_id>_<label_code>-<index>": 0.87
}
```

For `all_llm`, keys look like:

```json
{
  "<source_model>_<sample_id>_<label_code>-<index>": 0.87
}
```

## Evaluation

### Thresholding

Apply validation thresholds and save thresholded JSONL files:

```bash
cd evaluation
bash run_val_threshold.sh
```

Outputs are written to:

- `evaluation/<mode>/<model>/threshold/with_validation_<threshold>.jsonl`

### Distribution Comparison

Run KL / JSD comparison for one mode and one model at a time:

```bash
cd evaluation
bash run_kld_jsd.sh all-llm Llama-3.1-8B
```

Outputs are written to:

- `evaluation/<mode>/<model>/kld_jsd/*`

Typical files include:

- `<prefix>_summary.csv`
- `<prefix>_chaos_jsd_kl.csv`
- `<prefix>_varierr_jsd_kl.csv`
- `<prefix>_merged_errors.csv`

### Overlap of Validated Labels

The overlap summary CSVs are saved under:

- `evaluation/<mode>/<model>/validated_overlap/results_summary.csv`

The current repo contains [evaluation/precision_recall.py](/Users/phoebeeeee/ongoing/LLM_AED/evaluation/precision_recall.py:1), but the shell wrapper [evaluation/run_pre_re.sh](/Users/phoebeeeee/ongoing/LLM_AED/evaluation/run_pre_re.sh:1) uses hard-coded absolute paths and needs adjustment before it can be run successfully in this checkout.

### AED

The repo contains [evaluation/run_aed.sh](/Users/phoebeeeee/ongoing/LLM_AED/evaluation/run_aed.sh:1), but it currently points to `../LLM_AED/validation/validation_results` and to `evaluate.py`, which is not present in this repository. This step therefore needs script fixes before it can be reproduced from this checkout.

## Downstream Fine-Tuning

Use these notebooks for dataset preparation:

- `notebooks/chaosnli_dist.ipynb`
- `notebooks/removing_llm_errors.ipynb`
- `notebooks/auxiliary.ipynb`

Fine-tuning scripts are located in `fine-tuning/`, not `scripts/`:

```bash
cd fine-tuning
bash varierr_tune_bert.sh
bash varierr_tune_roberta.sh
```

These scripts currently save outputs to:

- `output/bert_repeated_errorless/run.log`
- `output/roberta_repeated_errorless/run.log`

They also reference `run.py`, `../train_chaosnli_dist`, `../bert_finetuned`, `../roberta_finetuned`, `../dataset/dev_cleaned.json`, and `../dataset/test_cleaned.json`, so those resources must exist before the scripts can run successfully.
