# EVADE: LLM-Based Explanation Generation and Validation for Error Detection in NLI


## Repository Structure


- `src/`: Complete source code used to reproduce all experiments and results presented in the thesis.

- `scripts/`: Python and shell scripts for data preprocessing, running experiments, and evaluating results.

- `dataset/`: All datasets used in this study, including processed versions of VariErr and ChaosNLI.

- `results/`: Output files, including model scores and evaluation metrics, for reference and verification.

- `generation/`: LLM-generated explanations, along with intermediate preprocessing outputs.



## Install dependencies:

```bash
pip install -r requirements.txt
```


## Explanation Generation

### Using Qwen Models:

```bash
cd generation

CUDA_VISIBLE_DEVICES=0 python generate_explanation_qwen.py \
    --model_name  \
    --jsonl_path  \
    [--output_dir ]
 ```

### Using Llama Models

```bash
cd generation

CUDA_VISIBLE_DEVICES=0 python generate_explanation_llama.py \
    --model_name  \
    --jsonl_path  \
    [--output_dir ]
```
- `--model_name`: Model name
- `--jsonl_path`: Path to input JSONL file (default: `.../dataset/varierr.json`)
- `--output_dir`: Output directory. Auto-generated from model name if not specified (default: `../generation/<model_name>_generation_raw`)

### Output Format

For each instance in VariErr, three files are generated under `<output_dir>/<sample_id>/`:
- `E_0.txt` — explanations for why the statement is **true**
- `N_0.txt` — explanations for why the statement is **neutral**
- `C_0.txt` — explanations for why the statement is **false**

## Preprocessing

**Manual Cleaning**: We manually inspecte all generated outputs and filtered out low-quality generations, including incomplete outputs, fallback responses, and non-English explanations. \
For each sample, we create cleaned files named `E`, `N`, and `C` under the same directory (if no changes were made, the original file is copied as-is).

**Merging**: We then merge all cleaned generations per model into individual JSONL files, and further combine all models into a single file for the *all-llm* prompting scenario.

```bash
cd processing

python processing.py \
    --generation_dir ../generation \
    --input_jsonl ../dataset/varierr/varierr.json \
    --processing_dir ../new_processing \
    --all_dir generation_all.jsonl
```

- `--generation_dir`: Directory containing `<model>_generation_raw` folders (default: `../generation`)
- `--input_jsonl`: Original dataset JSONL file (default: `../dataset/varierr/varierr.json`)
- `--processing_dir`: Directory to save per-model JSONL files (default: `../processing`)
- `--all_dir`: Final merged output filename (default: `generation_all.jsonl`)



## Explanation Validation

### GPT-4.1 (via API)

```bash
export OPENAI_API_KEY=your-api-key 
python src/llm_validation_gpt.py gpt-4.1
 ```

### Llama model

```bash
python src/llm_validation_llama.py
 ```
*Requires input file "model_explanation_raw.jsonl" with the following format:*
  ```json
  {
    "id": "sample_id",
    "premise": "...",
    "hypothesis": "...",
    "generated_explanations": [["reason_1", "label_1"], ["reason_2", "label_2"], ...]
  }
  ```


## Evaluation

### Distributional Alignment
- `notebooks/distribution_comparison.ipynb`: KLD and JSD between model and human distributions from VariErr or ChaosNLI.

### Ranking Alignment
- `notebooks/generate_ranking.ipynb`: Ranking based comparison between model and human distributions from VariErr or ChaosNLI.


### Evaluation Protocol from [VariErr](https://aclanthology.org/2024.acl-long.123.pdf).

```bash
python src/evaluation_protocol_varierr.py
 ```
- Report average precision (AP), as well as precision and recall at the top 100 predictions (P@100 and R@100).





## Downstream Fine-Tuning
####  Evaluate error removal impact on downstream NLI task performance.

Use these notebooks to generate different training sets — with or without injected noise or filtered errors:
- `notebooks/injecting_noise_self_validated.ipynb` 
- `notebooks/injecting_noise_peer_validated.ipynb`
- `notebooks/chaosnli_dist.ipynb`
- `notebooks/removing_llm-errors.ipynb` 


Run the following scripts to fine-tune and evaluate:

```bash
bash scripts/varierr_tune_bert.sh
bash scripts/varierr_tune_roberta.sh
 ```
