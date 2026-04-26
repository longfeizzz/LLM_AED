# EVADE: LLM-Based Explanation Generation and Validation for Error Detection in NLI

## Running Convention

All commands below are intended to be run from the repository root.


## Repository Structure

- `generation/`: explanation generation scripts and generated explanation folders.
- `processing/`: preprocessing script and merged JSONL outputs.
- `validation/`: validation scripts, helper shell scripts, and validation results.
- `evaluation/`: thresholding and evaluation scripts plus evaluation outputs.
- `dataset/`: datasets used in this project.
- `fine-tuning/`: downstream fine-tuning shell scripts.
- `notebooks/`: notebooks for data preparation and analysis.
- `src/`: plotting and miscellaneous analysis scripts.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Explanation Generation

### Qwen

```bash
CUDA_VISIBLE_DEVICES=0 python generation/generate_explanation_qwen.py \
  --model_name \
  --jsonl_path \
  [--output_dir]
```

### LLaMA

```bash
CUDA_VISIBLE_DEVICES=0 python generation/generate_explanation_llama.py \
  --model_name \
  --jsonl_path \
  [--output_dir]
```

- `--model_name`: Model name` \
- `--jsonl_path`: Path to input JSONL file (default: .../dataset/varierr.json) `\
- `--output_dir`: Output directory. Auto-generated from model name if not specified (default: ../generation/<model_name>_generation_raw)`

Saved output: 

- `generation/<model>_generation_raw/<sample_id>/`

The generation scripts write one file per target label inside each sample folder:

- `E_0.txt`: entailment / true explanations
- `N_0.txt`: neutral / undetermined explanations
- `C_0.txt`: contradiction / false explanations

## Preprocessing
### Manual Cleaning

After manual inspection, keep the cleaned files in the same sample directory and name them exactly:

- `generation/<model>_generation_raw/<sample_id>/E`
- `generation/<model>_generation_raw/<sample_id>/N`
- `generation/<model>_generation_raw/<sample_id>/C`

These cleaned files are what the preprocessing script actually reads.

### Preprocessing and Merging

```bash
python processing/processing.py \
  --generation_dir \
  --input_jsonl \
  --processing_dir \
  --all_dir 
```
- `--generation_dir`: Directory containing `<model>_generation_raw` folders
- `--input_jsonl`: Original dataset JSONL file
- `--processing_dir`: Directory to save per-model JSONL files
- `--all_dir`: Final merged output filename

Saved output:


- one merged file per model:
  `processing/<model>_generation_raw.jsonl`
- one merged file across all models:
  `processing/generation_all.jsonl`

## Explanation Validation

### *one_expl*

Validate one explanation per prompt:

```bash
CUDA_VISIBLE_DEVICES=0 python validation/one_expl.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --model_type llama \
  --input_path processing/llama_8b_generation_raw.jsonl \
  --output_dir validation/validation_results/one_expl/llama_8b
```
- `--model_name_or_path`: Model name
- `--model_type`: `llama` or `qwen`
- `--input_path`: Path to input JSONL file. Auto-generated from model name if not specified (default: `../processing/<model_name>_generation_raw.jsonl`)
- `--output_dir`: Output directory. Auto-generated from model name if not specified


Saved output:

- `validation/validation_results/one_expl/<model>/scores.json`

### *one_llm*

Validate all explanations from one source LLM in one prompt:

```bash
CUDA_VISIBLE_DEVICES=0 python validation/one_llm.py \
  --model_name_or_path \
  --model_type \
  --input_path \
  --output_dir
```
- `--model_name_or_path`: Model name
- `--model_type`: `llama` or `qwen`
- `--input_path`: Path to input JSONL file. Auto-generated from model name if not specified (default: `../processing/<model_name>_generation_raw.jsonl`)
- `--output_dir`: Output directory. Auto-generated from model name if not specified


Saved output:

- `validation/validation_results/one_llm/<model>/scores.json`

### *all_llm*

Validate explanations from multiple source LLMs together:

```bash
CUDA_VISIBLE_DEVICES=0 python validation/all_llm.py \
  --model_name_or_path \
  --model_type \
  --input_path \
  --output_dir
```

- `--model_name_or_path`: Model name
- `--model_type`: `llama` or `qwen`
- `--input_path`: Path to input JSONL file. Auto-generated from model name if not specified (default: `../processing/<model_name>_generation_raw.jsonl`)
- `--output_dir`: Output directory. Auto-generated from model name if not specified


Saved output:

- `validation/validation_results/all_llm/<model>/scores.json`

### Validation Output Format

All validation scripts save a JSON file mapping explanation IDs to probabilities.

For `one_expl` and `one_llm`, the key format is:

```json
{
  "<sample_id>_<label_code>-<index>": 0.87
}
```

For `all_llm`, the source model is also included:

```json
{
  "<source_model>_<sample_id>_<label_code>-<index>": 0.87
}
```



## Evaluation
### Thresholding

Apply validation tags to generated explanations across a range of thresholds (0.0–1.0) to support further analysis.

```bash
cd evaluation
bash run_val_threshold.sh
```

Saved output:

- `validation/validation_results/<mode>/<model>/threshold/with_validation_<threshold>.jsonl`
- `evaluation/<mode>/<model>/threshold/with_validation_<threshold>.jsonl`

### Distribution Comparison

Compare the validated label distribution's alignment with ChaosNLI and VariErr Distribution before and after validation, with different thresholds, ChaosNLI contains 100 annotations per instance.

```bash
bash run_kld_jsd.sh
```

Saved output:


- `evaluation/<mode>/<model>/kld_jsd/`


### Overlap of Validated Labels

Evaluate the model predictions using Precision, Recall by comparing LLM-validated labels against VariErr-validated labels across different thresholds.

```bash
bash evaluation/run_pre_re.sh
```

Saved output:

- `evaluation/<mode>/<model>/validated_overlap/results_summary.csv`

### Explanation Similarity

We analyze the linguistic similarity between human and LLM-generated explanations before and after validation from three perspectives: lexical, syntactic and semantic.

#### within-human
This script measures the diversity of human-written explanations in the VariErr dataset.
For each instance and each label, we compute pairwise similarity between explanations along three dimensions: lexical, syntactic and semantic.

```bash
python -m spacy download en_core_web_md
python similarity_within_human.py
```

#### within-LLM

This script measures the diversity of explanations generated by LLMs within the same instance and label. Default thresholds are set the same as the ones reported in the paper.

```bash
bash run_similarity_within_llm.sh
```

#### LLM-vs-human
This script compares LLM-generated explanations with human-written explanations from the VariErr dataset. Default thresholds are set the same as the ones reported in the paper.

```bash
bash run_similarity_llm_human.sh
```



### AED

Report average precision (AP), as well as precision and recall at the top 100 predictions (P@100 and R@100).

```bash
bash evaluation/run_aed.sh
```

Output will be displayed in terminal.



## Downstream Fine-Tuning
### Preprocessing

We clean the input data for fine-tuning by converting label sets into soft label distributions.
Each file is in JSONL format. Each line (instance) has the following structure:
```
{
  "uid": "123",
  "premise": "A man is running",
  "hypothesis": "Someone is moving",
  "label": [0.5, 0.5, 0.0]
}
```

#### VariErr baseline R1 and R2

```bash
cd evaluation
python baseline_r1_r2.py
```

After running the script, two files are generated in the `dataset/` directory.


#### Fine-tuning with EVADE labels (setup (a))



```bash
bash run_llm_fine_tuning.sh
```

After running the script, two files are saved under `LLM_AED/evaluation/<mode>/<model>/LLM-cleaned`. Default thresholds are set the same as the ones reported in the paper.


#### Remove EVADE errors from VariErr R1 (setup (b))

```bash
bash run_remove_llm_error.sh
```




Useful notebooks in the current repo:

- `notebooks/chaosnli_dist.ipynb`
- `notebooks/removing_llm_errors.ipynb`
- `notebooks/auxiliary.ipynb`

The fine-tuning shell scripts are located in `fine-tuning/`, not `scripts/`:

```bash
cd fine-tuning
bash varierr_tune_bert.sh
bash varierr_tune_roberta.sh
```

Saved Outputs

