# sempa-look
Absolutely. For GitHub, I’d make it **research-grade but visually polished**: a strong header, badges, architecture diagram, mathematical intuition, algorithm table, installation, usage, results, roadmap, and reproducibility notes.

Here is the consolidated `README.md` I recommend using:

````markdown
# SEMPA-LAB

### Semantic Packet Aggregation for Token Communication

<p align="center">

**Reproducing, analyzing, and extending SemPA-Look for semantic communication**

<br><br>

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MPS-black?logo=apple)](https://developer.apple.com/metal/pytorch/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#testing)
[![Status](https://img.shields.io/badge/status-active%20research-orange)](#project-status)

</p>

---

## Overview

**SEMPA-LAB** is a research implementation and experimental framework for studying **semantic packet aggregation (SemPA)** in token communication.

The project reproduces and analyzes the 2026 IEEE Transactions on Communications paper:

> **Low-Complexity Semantic Packet Aggregation for Token Communication via Lookahead Search**  
> S. Lee, J. Park, J. Choi, and H. Park, IEEE Transactions on Communications, 2026.

The central problem is simple to state:

> **How should the tokens of a message be grouped into packets so that the message retains as much semantic meaning as possible when packets are lost?**

Unlike conventional bit communication, tokens are not independent. The semantic meaning of a token depends on its surrounding context. Consequently, packetizing tokens becomes a **combinatorial optimization problem**.

SEMPA-LAB aims to go beyond simply reproducing the paper:

```text
                    SEMPA-LAB
                        │
          ┌─────────────┼─────────────┐
          │             │             │
      REPRODUCE      ANALYZE        EXTEND
       the paper     its limits     the method
          │             │             │
          └─────────────┼─────────────┘
                        │
                 Research System
````

The research workflow is deliberately ordered:

```text
WHAT THE PAPER CLAIMS
          ↓
WHAT THE CODE IMPLEMENTS
          ↓
WHAT THE EXPERIMENTS SHOW
          ↓
WHAT CAN BE IMPROVED
```

Extensions are only introduced after the baseline has been verified.

---

# Research Question

> **Can we reproduce SemPA-Look locally, explain why it works, identify where it breaks, and build a more adaptive version that improves the quality–robustness–computation trade-off?**

The project therefore focuses on three dimensions:

* **Semantic quality**
* **Robustness to packet loss**
* **Computational complexity**

The goal is not merely to maximize semantic similarity.

The goal is to understand the **quality–complexity trade-off**.

---

# Why Semantic Packet Aggregation?

Consider a message consisting of `K` tokens:

```text
W = {w₁, w₂, ..., wₖ}
```

These tokens are divided into `N` subpackets, each containing `M` tokens:

```text
N = K / M
```

For example:

```text
Message:

[a] [man] [riding] [a] [bicycle] [near] [the] [beach]

                 ↓ packetization

C₁ = {a, riding, bicycle, beach}
C₂ = {man, a, near, the}
```

If a packet is lost, all of its tokens disappear from the reconstructed message.

The important observation is that **which tokens are grouped together matters**.

A good packetization should make the remaining tokens preserve as much of the original semantic meaning as possible.

---

# The Combinatorial Problem

The number of valid ways to partition `K` tokens into `N = K/M` unordered subpackets of size `M` is

$$
\frac{K!}{(M!)^N N!}.
$$

This grows extremely quickly.

For example:

| `K` | `M` | Number of partitions |
| --: | --: | -------------------: |
|   8 |   4 |                   35 |
|   9 |   3 |                  280 |
|  12 |   4 |                5,775 |
|  12 |   3 |               15,400 |
|  16 |   4 |            2,627,625 |

And evaluating a single partition requires considering the possible packet-loss states.

With `N` subpackets, there are:

$$
2^N
$$

possible received/lost-subpacket combinations.

Therefore:

```text
More tokens
    ↓
More possible partitions
    ↓
More semantic encoder evaluations
    ↓
Rapidly increasing computational cost
```

This is the central problem SemPA-Look attempts to address.

---

# Core Idea: SemPA-Look

The paper introduces two key ideas:

1. **Residual Semantic Score (RSS)** as a token-level surrogate for the message-level ATS objective.
2. **Lookahead search** to avoid the exponential cost of exhaustive partition search.

Instead of evaluating every possible complete partition, SemPA-Look makes the packetization decision sequentially.

At each depth:

```text
Remaining tokens
       │
       ▼
Sample P candidate groups
       │
       ├──────────────┐
       ▼              ▼
 Candidate 1      Candidate 2      ...
       │              │
       ▼              ▼
 Lookahead         Lookahead
 evaluations       evaluations
       │              │
       └───────┬──────┘
               ▼
        Score candidates
               │
               ▼
        Commit best group
               │
               ▼
       Remove its tokens
               │
               ▼
             Repeat
```

For each candidate `C`, the algorithm evaluates its residual semantic score together with `k` lookahead candidates:

$$
\Psi(C)
=
\frac{1}{k+1}
\left[
\psi(C,W)
+
\sum_{i=1}^{k}\psi(\hat C_i,W)
\right].
$$

The candidate with the highest score is committed.

The process continues until the complete partition is constructed.

---

# Algorithms

SEMPA-LAB implements the packetization strategies as independent modules so they can be compared under the same semantic encoder, channel model, objective, and instrumentation.

| Algorithm             | Role                                  |
| --------------------- | ------------------------------------- |
| **Full Search**       | Exact optimum for small `K`           |
| **Random PA**         | Random packetization baseline         |
| **No Packetization**  | Zero-aggregation baseline             |
| **Greedy TSS**        | High-packet-success surrogate         |
| **Greedy RSS**        | Residual semantic score surrogate     |
| **SemPA-Look**        | Main proposed lookahead method        |
| **Genetic Algorithm** | Evolutionary optimization baseline    |
| **SemPA-GBeam**       | Optional predecessor baseline         |
| **MCTS-UCT**          | Optional generic tree-search baseline |

The distinction between `greedy_rss` and `sempa_look` is intentional:

```text
greedy_rss
    ≈
SemPA-Look with k = 0
```

Keeping them as separate implementations allows the contribution of lookahead to be measured experimentally.

---

# Evaluation Metrics

## Average Token Similarity — ATS

ATS is the primary semantic objective.

It measures the expected similarity between the original message and the reconstructed message under packet loss.

The optimization objective can therefore be written conceptually as:

```text
Choose packetization
        ↓
maximize expected semantic similarity
        ↓
under packet losses
```

---

## Weighted Token Similarity — WTS

SEMPA-LAB also implements WTS.

This is important because cosine-based ATS can compress differences between different packetization strategies.

The paper reports that WTS can expose considerably larger differences between optimized and random packet aggregation than ATS.

Therefore, final experiments should generally report:

```text
ATS
+
WTS
+
Computational Cost
```

rather than ATS alone.

---

# Computational Instrumentation

Every algorithm returns a structured result rather than only a partition.

The result records:

```text
best_partition
best_score
objective

partitions_evaluated

logical_calls
miss_calls
cache_hits
encoder_batches

runtime_seconds
```

This allows us to measure both:

### Quality

```text
ATS
WTS
relative performance
```

### Complexity

```text
logical evaluations
encoder evaluations
cache utilization
encoder batches
wall-clock runtime
```

This is particularly important because semantic encoder computation is one of the dominant costs of the system.

---

# Encoder Architecture

The semantic evaluation pipeline is:

```text
                    Message
                       │
                       ▼
                     Tokens
                       │
                       ▼
                 Packetization
                       │
                       ▼
                Packet Loss Model
                       │
                       ▼
                Reconstruction
                       │
                       ▼
              Semantic Encoder
                       │
                       ▼
                Embeddings
                       │
                       ▼
              Cosine Similarity
                       │
                       ▼
                    ATS/WTS
```

The encoder wrapper provides:

* Original-message embeddings
* Reconstructed-message embeddings
* Batch encoding
* Embedding caching
* Encoder instrumentation

Caching is particularly important because different partition evaluations repeatedly encounter the same token subsets.

---

# Current Encoder

The development implementation uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The reproduction configuration can use the encoder specified by the corresponding experiment.

The system is designed so that the encoder can be replaced without rewriting the packetization algorithms.

---

# Hardware Support

SEMPA-LAB is designed to run on consumer hardware.

### Supported

* CPU
* Apple Silicon / MPS
* CUDA-compatible GPUs

The implementation does **not** assume CUDA.

On Apple Silicon, MPS is automatically preferred when available.

For unsupported MPS operations, enable:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Important

MPS is not guaranteed to be faster for every workload.

For small batches of short candidate strings, CPU execution can sometimes be faster because of dispatch and device-transfer overhead.

SEMPA-LAB therefore measures encoder runtime rather than assuming that GPU/MPS is always faster.

---

# Repository Structure

```text
iot/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── src/
│   └── sempalab/
│       │
│       ├── __init__.py
│       ├── types.py
│       ├── partition.py
│       ├── packetize.py
│       ├── encoder.py
│       ├── metrics.py
│       ├── objectives.py
│       │
│       ├── channel/
│       │   └── __init__.py
│       │
│       └── algorithms/
│           ├── __init__.py
│           ├── full_search.py
│           ├── random_pa.py
│           ├── no_packetization.py
│           ├── greedy_tss.py
│           ├── greedy_rss.py
│           ├── sempa_look.py
│           └── genetic.py
│
├── tests/
│
├── scripts/
│
├── configs/
│
├── data/
│
├── results/
│
└── notebooks/
```

### Directory Guide

| Directory       | Purpose                       |
| --------------- | ----------------------------- |
| `src/sempalab/` | Core implementation           |
| `algorithms/`   | Packetization algorithms      |
| `channel/`      | Channel models                |
| `tests/`        | Unit and regression tests     |
| `scripts/`      | Reproducible experiments      |
| `configs/`      | Experiment configurations     |
| `data/`         | Datasets and evaluation sets  |
| `results/`      | CSV/JSON experimental results |
| `notebooks/`    | Exploratory analysis          |

---

# Requirements

## Software

Recommended:

```text
Python >= 3.11
```

Core dependencies:

```text
numpy
scipy
torch
sentence-transformers
scikit-learn
PyYAML
tqdm
```

Install everything with:

```bash
pip install -r requirements.txt
```

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd iot
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For development:

```bash
pip install -e .
```

---

# Quick Start

Once the environment is installed:

```bash
pytest
```

To run an individual algorithm test:

```bash
python tests/test_genetic.py
```

Example output:

```text
GA partition:
((0, 2, 3, 6), (1, 4, 5, 7))

GA ATS:
0.8050889603793621

Logical calls:
240

Miss calls:
35

Cache hits:
445

Encoder batches:
35

Runtime:
0.5545s

Genetic Algorithm test passed
```

The exact values may vary with encoder/model/runtime configuration.

---

# Example API

A typical message is represented as:

```python
from sempalab.types import Message

message = Message(
    message_id=1,
    text="a man riding a bicycle near the beach",
    tokens=(
        "a",
        "man",
        "riding",
        "a",
        "bicycle",
        "near",
        "the",
        "beach",
    ),
)
```

A packetization algorithm can then be evaluated using the shared encoder and objective infrastructure.

The design intentionally separates:

```text
Message
   ↓
Partition
   ↓
Objective
   ↓
Encoder
   ↓
Search Algorithm
   ↓
SearchResult
```

This makes algorithms independently testable and reusable.

---

# Genetic Algorithm

The Genetic Algorithm serves as an evolutionary baseline.

Each chromosome represents a valid partition:

```text
G = [C₁, C₂, ..., Cₙ]
```

where every subpacket contains exactly `M` tokens and every token appears exactly once.

The implementation uses:

* Random population initialization
* Tournament selection
* Crossover
* Mutation
* Elitism
* Exact ATS fitness

The fitness function deliberately uses **exact ATS**.

This is computationally expensive because every chromosome requires evaluation over all `2^N` packet-loss states.

Therefore, GA experiments should generally be restricted to:

```text
K <= 12
```

unless additional compute is available.

---

# Reproducibility Modes

The project distinguishes between two modes:

```text
paper_faithful
research
```

### `paper_faithful`

Used for reproducing the original method.

Once the reproduction baseline passes, this mode should be treated as frozen.

### `research`

Used for modifications, experiments, and proposed improvements.

This prevents an experimental improvement from accidentally changing the baseline and making the reproduction impossible to audit.

---

# Reproducibility Principles

SEMPA-LAB follows six engineering principles.

### 1. Vertical slice before breadth

The first goal is a complete end-to-end pipeline:

```text
sentence
   ↓
tokens
   ↓
partition
   ↓
packet loss
   ↓
reconstruction
   ↓
semantic encoder
   ↓
similarity
```

### 2. Paper-faithful mode is frozen

Baseline reproduction and research modifications must remain separate.

### 3. Assumptions are documented

If the paper leaves an implementation detail unspecified, the choice is explicitly documented.

### 4. Every algorithm reports cost

No algorithm returns only a partition.

### 5. Batch and cache from the beginning

Encoder calls are expensive and must be treated as a first-class computational resource.

### 6. Results are files

Experimental results belong in:

```text
results/
```

as machine-readable CSV/JSON files.

Figures should be generated from these results rather than manually assembled.

---

# Stage 2 — Algorithmic Validation

The current algorithmic stage contains:

```text
[x] Full Search
[x] Random PA
[x] No Packetization
[x] Greedy TSS
[x] Greedy RSS
[x] SemPA-Look
[x] Genetic Algorithm

[ ] SemPA-GBeam      optional
[ ] MCTS-UCT         optional
[ ] 100-message acceptance experiment
```

## Stage 2 Acceptance Criteria

At:

```text
K = 8
M = 4
```

using the designated 100-message evaluation set:

### Full Search

Full Search must achieve a score greater than or equal to every other method on every message.

### SemPA-Look

SemPA-Look should remain within:

```text
3%
```

of Full Search.

### Lookahead Ablation

The experiments should demonstrate the benefit of lookahead against the relevant greedy baselines.

With:

```text
k_max = N - 1
```

the target regime is:

```text
k >= 0.2 * k_max
```

These acceptance criteria are the gate before moving to the full communication-system reproduction.

---

# Stage 3 — Communication System

After algorithmic validation, the project introduces the actual communication system.

Planned components:

```text
[ ] BPSK modulation
[ ] AWGN channel
[ ] Hard demodulation
[ ] Empirical packet-error validation
[ ] Batch packet aggregation
[ ] Packet deaggregation
[ ] Dummy padding
[ ] Packet-bit accounting
[ ] YAML configuration system
```

The analytical packet-error model will be validated against an independent Monte-Carlo BPSK/AWGN simulation.

This provides two independent correctness checks:

```text
Analytical model
       ↕
Physical simulation
```

---

# Stage 4 — Paper Reproduction

The main reproduction dataset is **MS-COCO 2017 caption annotations**.

The initial experiments use captions only; the images are not required for the text packetization experiments.

Planned reproduction targets include:

| Experiment   | Configuration         | Target                      |
| ------------ | --------------------- | --------------------------- |
| Fig. 3       | `K=8, M=4`            | TSS/RSS behavior across SNR |
| Fig. 6       | `K=8, M=4, P=10, k=4` | SemPA-Look ≈ Full Search    |
| Table III    | `K=24, M=4, N=6`      | Complexity comparison       |
| Fig. 8       | `K=12`                | ATS vs. `M`                 |
| Fig. 9       | `K=12`                | `M` behavior at 4/8 dB      |
| Fig. 10      | Sweep `k`             | Lookahead saturation        |
| Fig. 11      | Sweep `P`             | Diminishing returns         |
| Fig. 12      | Sweep `K`             | Scaling behavior            |
| Tokenization | Word vs. subword      | Robustness comparison       |

---

# Reproduction Philosophy

Exact numerical equality with the paper is **not** the primary criterion.

The paper leaves some implementation details unspecified, including aspects of:

* reconstruction convention
* caption subset selection
* random-number generation

Therefore, reproduction should primarily compare:

```text
✓ Curve shape
✓ Relative ordering
✓ Gap direction
✓ Rough gap magnitude
✓ Complexity scaling
```

rather than expecting identical absolute ATS values.

A discrepancy is not automatically a failure.

An unexplained discrepancy is.

---

# ATS Saturation

One important characteristic of the evaluation is that cosine-based semantic similarity can compress the difference between packetization strategies.

As a result:

```text
Different packetizations
        ↓
Similar ATS values
```

does not necessarily mean that the algorithms behave similarly.

This is why WTS and computational cost are included in the evaluation framework.

The final reproduction should therefore report:

```text
ATS
WTS
Runtime
Logical Calls
Encoder Calls
Cache Hits
```

---

# Configuration

The long-term experiment interface uses YAML configuration.

Example:

```yaml
experiment: repro_fig6

dataset: ms_coco
tokenization: word

K: 12
M: 4
B: 100

P: 10
k: 4

snr_db:
  - 2
  - 4
  - 6
  - 8
  - 10

packet_bits: 300

encoder: all-mpnet-base-v2

mode: paper_faithful

seed: 42
```

Experiments will eventually be launched through:

```bash
python -m sempalab.run --config configs/repro_fig6.yaml
```

This allows the same experiment to be reproduced without modifying source code.

---

# Results

All experiment outputs should be stored under:

```text
results/
```

Example:

```text
results/
├── stage2/
│   ├── acceptance.csv
│   ├── acceptance.json
│   └── summary.json
│
├── reproduction/
│   ├── fig3.csv
│   ├── fig6.csv
│   ├── fig8.csv
│   └── fig9.csv
│
└── figures/
    ├── fig3.png
    ├── fig6.png
    └── ...
```

The principle is:

```text
Experiment
    ↓
CSV / JSON
    ↓
Analysis script
    ↓
Figure / Table
```

not:

```text
Experiment
    ↓
Screenshot
```

---

# Research Extensions

Only after reproduction and analysis will SEMPA-LAB introduce an original research contribution.

Potential directions include:

### Adaptive Surrogate Weighting

Adapt the relative importance of surrogate terms according to packet-loss probability.

### Adaptive Lookahead

Choose `k` dynamically based on the current search state or marginal gain.

### Risk-Aware Packetization

Optimize not only expected semantic quality but also the risk of severe semantic degradation.

### Bursty Channels

Replace independent packet erasures with a correlated channel model such as a Gilbert–Elliott process.

### Saliency-Aware Sampling

Bias candidate generation toward semantically important tokens.

### Learned Surrogate

Train a lightweight model to approximate expensive semantic evaluations.

The final extension will be selected based on weaknesses discovered during the reproduction and analysis stages.

---

# Research Workflow

The intended research loop is:

```text
                ┌──────────────────┐
                │     PAPER        │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │    REPRODUCE     │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │     ANALYZE      │
                │  Where does it   │
                │     break?       │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │     EXTEND       │
                │  Research mode   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │     ABLATION     │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │    EVIDENCE      │
                └──────────────────┘
```

This prevents the common research failure mode of modifying a baseline before establishing that the baseline itself is correct.

---

# Project Status

### Current milestone

**Core algorithmic implementation**

Completed:

* [x] Project architecture
* [x] Core data structures
* [x] Partition enumeration
* [x] Packet reconstruction
* [x] Semantic encoder wrapper
* [x] Encoder batching
* [x] Encoder caching
* [x] Semantic similarity
* [x] ATS
* [x] WTS
* [x] Channel model foundation
* [x] Full Search
* [x] Random PA
* [x] No Packetization
* [x] Greedy TSS
* [x] Greedy RSS
* [x] SemPA-Look
* [x] Genetic Algorithm
* [x] Algorithm-level tests

Current next milestone:

* [ ] 100-message Stage 2 acceptance experiment

Then:

* [ ] Physical communication system
* [ ] MS-COCO reproduction
* [ ] Paper reproduction report
* [ ] Limitation analysis
* [ ] Research extension
* [ ] Final ablation and evaluation

---

# Citation

If you use this repository or build upon the implementation, please cite the original work:

```bibtex
@article{lee2026sempalook,
  author  = {Lee, Seunghun and Park, Jihong and Choi, Jinho and Park, Hyuncheol},
  title   = {Low-Complexity Semantic Packet Aggregation for Token Communication via Lookahead Search},
  journal = {IEEE Transactions on Communications},
  year    = {2026}
}
```

---

# Acknowledgements

This project is an independent research implementation based on the SemPA-Look framework.

The original work introduces semantic packet aggregation with residual semantic scoring and lookahead search to achieve near-optimal semantic performance while substantially reducing computational complexity.

---

# License

This repository is intended for research and educational purposes.

The original paper, datasets, pretrained models, and third-party dependencies remain subject to their respective licenses.

---

<p align="center">

### SEMPA-LAB

**Reproduce → Analyze → Extend**

*Semantic communication research through reproducible engineering.*

</p>
```

### `requirements.txt`

I'd pair that README with this clean requirements file:

```txt
numpy
scipy
torch
sentence-transformers
scikit-learn
PyYAML
tqdm
pytest
```

One thing I would **not** put in the public README yet is a claim like *“we reproduced the paper”*. We're currently at the **algorithmic implementation milestone**; the 100-message Stage 2 acceptance experiment and the actual MS-COCO reproduction still need to be run. That distinction is important for the credibility of the GitHub repo. 

The paper itself reports that SemPA-Look is intended to approach exhaustive-search semantic quality while reducing computational complexity, so the README frames the project around exactly that quality–complexity trade-off rather than presenting it as merely an implementation exercise. 
