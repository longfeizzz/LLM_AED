# EVADE: LLM-Based Explanation Generation and Validation for Error Detection in NLI


Repository of the ACL 2026 Findings paper ["Evade: LLM-Based Explanation Generation and Validation for Error Detection in NLI"](https://arxiv.org/pdf/2511.08949).


## Install Dependencies

```bash
pip install -r requirements.txt
```

## Explanation Generation

### Qwen

```bash
CUDA_VISIBLE_DEVICES=0 python generation/generate_explanation_qwen.py \
  --model_name \
  [--json_path] \
  [--output_dir]
```

### LlaMA

```bash
CUDA_VISIBLE_DEVICES=0 python generation/generate_explanation_llama.py \
  --model_name \
  [--json_path] \
  [--output_dir]
``` 

- `--model_name`: Huggingface model name.
- `--json_path`:  Original VariErr dataset JSON file (optional).
- `--output_dir`: Output directory. Auto-generated from model name if not specified (optional).

Saved output: 

- `generation/<model>_generation_raw/<sample_id>/`

The generation scripts write one file per target label inside each sample folder:

- `E_0.txt`: entailment / true explanations
- `N_0.txt`: neutral / undetermined explanations
- `C_0.txt`: contradiction / false explanations

## Preprocessing
### Manual Cleaning

After manual inspection, keep the cleaned files in the same sample directory and name them as:

- `generation/<model>_generation_raw/<sample_id>/E`
- `generation/<model>_generation_raw/<sample_id>/N`
- `generation/<model>_generation_raw/<sample_id>/C`

make sure every _0.txt file has corresponding ENC file
These cleaned files are what the preprocessing script actually reads.

### Preprocessing and Merging

```bash
python processing/processing.py \
  [--generation_dir] \
  [--input_json] \
  [--processing_dir] \
  [--all_dir] 
```
- `--generation_dir`: Directory containing `<model>_generation_raw` folders (optional).
- `--input_jsonl`: Original VariErr dataset JSON file (optional).
- `--processing_dir`: Directory to save per-model JSONL files (optional).
- `--all_dir`: Final merged output filename (optional).

Saved output:


- One merged file per model:
  `processing/<model>_generation_raw.jsonl`
- One merged file across all models:
  `processing/generation_all.jsonl`

## Explanation Validation

### *one_expl*

Validate one explanation per prompt:

```bash
CUDA_VISIBLE_DEVICES=0 python validation/one_expl.py \
  --model_name_or_path \
  --model_type \
  [--input_path] \
  [--output_dir]
```
- `--model_name_or_path`: Huggingface model name.
- `--model_type`: `llama` or `qwen`
- `--input_path`: Path to input JSONL file. (optional, default: `../processing/<model_name>_generation_raw.jsonl`)
- `--output_dir`: Output directory (optional).


Saved output:

- `validation/validation_results/one_expl/<model>/scores.json`

### *one_llm*

Validate all explanations from one source LLM in one prompt:

```bash
CUDA_VISIBLE_DEVICES=0 python validation/one_llm.py \
  --model_name_or_path \
  --model_type \
  [--input_path] \
  [--output_dir]
```
- `--model_name_or_path`: Huggingface model name.
- `--model_type`: `llama` or `qwen`
- `--input_path`: Path to input JSONL file. Auto-generated from model name if not specified (optional, default: `../processing/<model_name>_generation_raw.jsonl`)
- `--output_dir`: Output directory (optional).


Saved output:

- `validation/validation_results/one_llm/<model>/scores.json`

### *all_llm*

Validate explanations from multiple source LLMs together:

```bash
CUDA_VISIBLE_DEVICES=0 python validation/all_llm.py \
  --model_name_or_path \
  --model_type \
  [--input_path] \
  [--output_dir]
```

- `--model_name_or_path`: Huggingface model name.
- `--model_type`: `llama` or `qwen`
- `--input_path`: Path to input JSONL file. Auto-generated from model name if not specified (optional, default: `../processing/<model_name>_generation_raw.jsonl`)
- `--output_dir`: Output directory (optional).


Saved output:

- `validation/validation_results/all_llm/<model>/scores.json`


***


```bash
cd validation
bash run_llama_all.sh
bash run_qwen_all.sh
```
You can also use these scripts to run the full validation workflow for the LLaMA and Qwen models used in the paper.

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

- `evaluation/<mode>/<model>/threshold/with_validation_<threshold>.jsonl`

### Distribution Comparison

Compare the validated label distribution's alignment with ChaosNLI and VariErr Distribution before and after validation, with different thresholds.

```bash
bash run_kld_jsd.sh <mode> <model>
```

- `<model>`: model directory name used in previous experiments (e.g., Llama-3.1-8B),
- `<mode>`: `one_expl`, `one_llm` or `all_llm`


Saved output:


- `evaluation/<mode>/<model>/kld_jsd/`


### Overlap of Validated Labels

Evaluate the model predictions using Precision, Recall by comparing LLM-validated labels against VariErr-validated labels across different thresholds.

```bash
cd evaluation
bash run_pre_re.sh
```

Saved output:

- `evaluation/<mode>/<model>/validated_overlap.csv`

### Explanation Similarity

We analyze the linguistic similarity between human and LLM-generated explanations before and after validation from three perspectives: lexical, syntactic and semantic.

#### within-human
This script measures the diversity of human-written explanations in the VariErr dataset.
For each instance and each label, we compute pairwise similarity between explanations along the three dimensions.

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

Report average precision (AP), as well as precision and recall at the top 100 predictions (P@100 and R@100). Metrics are computed using the average score for each label of each instance.

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
cd fine-tuning
python baseline_r1_r2.py
```

After running the script, two files are generated in the `LLM_AED/fine-tuning/processed_data/baselines/`directory.


#### Fine-tuning with EVADE labels (setup (a))



```bash
bash run_llm_fine_tuning.sh
```

After running the script, two files are saved under `LLM_AED/fine-tuning/processed_data/llm_cleaned/<mode>/<model>/` for all the modes and models. Default thresholds are set the same as the ones reported in the paper.


#### Remove EVADE errors from VariErr R1 (setup (b))

```bash
bash run_remove_llm_error.sh
```

Processed data will be saved under `LLM_AED/fine-tuning/processed_data/without_llm_error/<mode>/<model>/`


### Model Fine-tuning
Place the processed training data into the designated folder, then update the corresponding directory paths in the scripts before running them.

First, run the following commands so that the model obtains a basic understanding of the NLI task:

```bash
cd fine-tuning
bash mnli_train.sh
bash mnli_train_roberta.sh
```

After that, start fine-tuning with:


```bash
bash varierr_tune_bert.sh
bash varierr_tune_roberta.sh
```
The development and test sets are located at `LLM_AED/dataset/dev_test`.


