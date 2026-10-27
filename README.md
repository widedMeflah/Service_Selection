# LLM-Based Negotiation for QoS-Aware Cloud Service Selection and Provider Resource Completion

Implementation, evaluation dataset, prompts, per-case results, and user study of the paper *LLM-Based Negotiation for QoS-Aware Cloud Service Selection and Provider Resource Completion* (submitted to IEEE Transactions on Services Computing).

## Contents

```
src/           implementation of the approach (selection, negotiation, completion, verification) and of the baselines
run.py         runs the approach on one provider-independent TOSCA template
run_eval.py    runs the evaluation
prompts/       prompts of the paper
data/eval/     evaluation dataset
results/       per-case results of the evaluation
user_study/    scenarios and response file of the user study
```

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env          # add your own API keys to .env (never commit this file)
```

## Usage

```bash
python run.py --tosca data/eval/dataset/cases/B-aws-vm/template.tosca.yaml \
              --discovery data/eval/dataset/cases/B-aws-vm/discovery.json
python run_eval.py --frozen data/eval/snapshots/ --run run1
```

The LLMs of the evaluation are set in `eval_config.json`.

## Prompts

| Prompt | Paper |
|---|---|
| `prompts/1_phrasing_of_the_options.txt` | Fig. 2a |
| `prompts/2_interpretation_of_the_answer.txt` | Fig. 2b |
| `prompts/3_extraction_of_dependent_resources.txt` | Fig. 3 |
| `prompts/4_proposal_of_documentation_pages.txt` | Section 6.1 |
| `prompts/5_correction_of_a_failing_resource.txt` | Section 6.2 |
| `prompts/6_simulated_user.txt` | Section 7.1 |

## Evaluation dataset

| Folder | Content |
|---|---|
| `data/eval/dataset/cases/` | 167 cases, each a provider-independent TOSCA template with the candidate offers of its cloud services and its ground truth |
| `data/eval/annotation/` | Dependent resources annotated manually on the official pages |
| `data/eval/dataset/verification/` | 92 verification cases |
| `data/eval/dataset/staleness/` | 10 staleness cases |
| `data/eval/snapshots/` | Frozen copies of the official documentation pages |
| `data/catalogs/` | Catalogs of the candidate offers |

## Per-case results

| Folder | Paper |
|---|---|
| `results/selection/` | Table 3 |
| `results/negotiation/` | Section 7.3 |
| `results/completion/` | Table 5 |
| `results/staleness/` | Table 6 (top) |
| `results/verification/` | Table 6 (bottom) |
| `results/ablation/` | Table 7 |
| `results/cost/` | Fig. 4 |
| `results/scalability/` | Fig. 5 |
| `results/memory/` | Fig. 6 |

## User study

`user_study/scenarios.md` gives the messages and questions shown to the participants, and `user_study/responses_template.csv` the format of their anonymized answers (Table 4).

## License

MIT License (see `LICENSE`).
