# Beyond Noise: Detecting Annotation Error from Human Label Variation using LLMs

This repository contains the code and resources for the master’s thesis “Beyond Noise: Detecting Annotation Error from Human Label Variation using LLMs” by Longfei Zuo, carried out at the MaiNLP group, CIS, LMU Munich.


## Abstract


High-quality labeled datasets are essential for Natural Language Processing (NLP) research, but ensuring data quality remains a major challenge. Human Label Variation (HLV) is prevalent in tasks such as Natural Language Inference (NLI), where multiple labels may be valid for the same instance. This inherent ambiguity makes it even more difficult to distinguish annotation errors from plausible variations . In this thesis, we propose a framework that leverages large language models (LLMs) to detect annotation errors through explanation-based validation. Specifically, the LLM first generates diverse, label-specific explanations for each instance and then validate them by assigning a validity score to each explanation. If none of the explanations under a given label are validated,
the label is considered erroneous.

We perform a comprehensive analysis comparing human and LLM explanations across
distribution, validation results, and impact on model learning. Our experiments show that LLM-generated explanations align well with human annotations in terms of label distributiong and removing LLM-detected errors from the training data leads to improved performance on downstream tasks. These demonstrate that LLMs can detect annotation errors and offer complementary insights to human annotators, highlighting the potential of explanation-based pipelines to scale validation with minimal human effort, offering a practical approach to improving dataset quality in the presence of label variation.

## Repository Structure


- `src/`: Complete source code used to reproduce all experiments and results presented in the thesis.

- `scripts/`: Python and shell scripts for data preprocessing, running experiments, and evaluating results.

- `dataset/`: All datasets used in this study, including processed versions of VariErr and ChaosNLI.

- `results/`: Output files, including model scores and evaluation metrics, for reference and verification.

- `generation/`: LLM-generated explanations, along with intermediate preprocessing outputs.


## Run Experiments

### 1. Install dependencies:

```bash
pip install -r requirements.txt
```


### 2. Explanation Generation

#### Using GPT-4.1 (via OpenAI API):

```bash
export OPENAI_API_KEY=your-api-key 
python src/generate_explanation_gpt.py
 ```

#### Using Llama models

```bash
python src/generate_explanation_llama.py
 ```

### 3. Deduplication
- `notebooks/deduplication.ipynb`: Notebook for the three step deduplication process.

### 4. Explanation Validation

#### GPT-4.1 (via API)

```bash
export OPENAI_API_KEY=your-api-key 
python src/llm_validation_gpt.py gpt-4.1
 ```

#### Llama model

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
