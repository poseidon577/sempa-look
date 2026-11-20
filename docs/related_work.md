# Related work

Where this repository sits relative to prior work, and how each of our
contributions differs from the closest papers.

**Status.** This page collects the work already cited in
[limitations.md](limitations.md), [assumptions.md](assumptions.md) and the
project plan. It is not yet a systematic search. Before CASA is claimed as new
in a paper, we still need to search for papers that cite the SemPA-Look and
SemPA-GBeam preprints (arXiv 2506.19451, 2504.19591), search arXiv cs.IT and
eess.SP for "semantic packetization", "token communication" and "packet loss
semantic communication", and check for follow-ups from the same group. That
search is tracked in the README's [Project Status](../README.md#project-status).
The descriptions below summarise what our earlier reviews recorded about each
paper. They have not been re-checked against each paper for this page.

## 1. The paper and its predecessor

| Work | What it does | Role here |
|---|---|---|
| Lee, Park, Choi, Park, *Low-Complexity Semantic Packet Aggregation for Token Communication via Lookahead Search*, IEEE TCOM vol. 74, pp. 9639–9654, 2026 ([arXiv 2506.19451](https://arxiv.org/abs/2506.19451)) | SemPA-Look: residual semantic score plus sampled lookahead to group tokens into packets under i.i.d. erasures | The paper we reproduce ([reproduction.md](reproduction.md)). arXiv v1 used a CLIP text encoder and no WTS, so we follow the published TCOM version |
| SemPA-GBeam, IEEE SPAWC 2025 ([arXiv 2504.19591](https://arxiv.org/abs/2504.19591)) | Genetic beam search for the same aggregation problem | Direct predecessor. Implemented as a baseline (assumption A13) |
| Same group, *Semantic packet aggregation and repeated transmission* (TCOM ref. [19]) | Aggregation with repetition | Background. The TCOM paper drops the repetition axis |

## 2. Burst losses and interleaving (closest to CASA)

| Work | What it does | How CASA differs |
|---|---|---|
| [arXiv 2511.16983](https://arxiv.org/abs/2511.16983) | Random grouping / interleaving of latent feature units, which turns burst losses into roughly independent errors | It uses blind, random grouping of feature units. CASA groups text tokens by their semantics *and* picks which packet slot each group rides in, both optimised for the channel's joint loss law. We also compare against the interleaving idea directly: a packet interleaver and an interleaver-seeded swap search are baselines in [channel_aware.md](channel_aware.md) |
| SITP ([arXiv 2512.09291](https://arxiv.org/html/2512.09291v1)) | Semantic transport over burst fades | Different layer (transport). Cite as evidence that bursts matter for semantic links |
| TONIC ([arXiv 2605.21553](https://arxiv.org/html/2605.21553v1)) | Evaluates semantic communication under Rayleigh/Rician block fading | Evaluation setting only; no aggregation |
| BERT concealment with interleaving ([arXiv 2408.04535](https://arxiv.org/abs/2408.04535)), TokCode ([arXiv 2604.12931](https://arxiv.org/html/2604.12931v1)) | Interleave tokens, then recover lost ones at the receiver | Receiver-side recovery with fixed interleaving. CASA keeps the paper's receiver and optimises the grouping. Recovery-aware aggregation is our open direction 1 |

## 3. Receiver-side token recovery

| Work | What it does | Relevance |
|---|---|---|
| TokCom ([arXiv 2502.12096](https://arxiv.org/pdf/2502.12096)) | Token communication with language-model recovery at the receiver | Motivates recovery-aware aggregation (direction 1 in [limitations.md](limitations.md)) |
| MLM-prior iterative detection ([arXiv 2601.17770](https://arxiv.org/abs/2601.17770v1), [arXiv 2605.02123](https://arxiv.org/pdf/2605.02123)) | Masked-LM priors in detection; shares co-authors with the TCOM paper | Risk of overlap for direction 1, which is why we led with channel-aware aggregation |
| Short codes plus LM correction ([arXiv 2505.08536](https://arxiv.org/abs/2505.08536)) | Channel coding with language-model post-correction | Related to the paper's Fig. 18, which we did not reproduce |

## 4. Coding, unequal protection and retransmission

| Work | What it does | Relevance |
|---|---|---|
| Token UEP with expanding-window fountain codes ([arXiv 2511.22859](https://arxiv.org/html/2511.22859v1)) | Unequal error protection for image tokens | No equivalent for aggregated text packets that we know of (direction 6) |
| Semantic rate matching ([arXiv 2608.11551](https://arxiv.org/pdf/2608.11551)), IK-HARQ ([arXiv 2203.06634](https://ar5iv.labs.arxiv.org/html/2203.06634)) | Rate adaptation and HARQ for semantic links | Background for a coded or retransmitting extension |

## 5. Search, surrogates and robust optimisation

| Work | What it does | Relevance |
|---|---|---|
| Deletion-robust submodular maximisation ([Bogunovic et al. 2017, arXiv 1706.04918](https://arxiv.org/pdf/1706.04918); [arXiv 2201.13128](https://arxiv.org/pdf/2201.13128)); robust submodular maximisation ([Orlin et al., arXiv 1507.06616](https://arxiv.org/abs/1507.06616)) | Constant-factor guarantees when items can be deleted | Possible bounds for aggregation, if φ is close to submodular (untested) |
| FastSHAP ([arXiv 2107.07436](https://arxiv.org/pdf/2107.07436)); [Yang et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.19998) | Amortised Shapley-value prediction | Cheaper fitting of CASA's pairwise surrogate |
| Swap local search / annealing for assignment problems ([arXiv 1312.0841](https://arxiv.org/pdf/1312.0841), [arXiv 1409.5223](https://arxiv.org/pdf/1409.5223)) | Local search for quadratic assignment | CASA searches its quadratic assignment surrogate with swaps; swap search on the RSS sum is a baseline |
| NeuroCUT ([arXiv 2310.11787](https://arxiv.org/html/2310.11787v3)) | Learned graph partitioning | Learned packetiser (direction 7, not pursued) |
| AdaDec ([arXiv 2506.08980](https://arxiv.org/abs/2506.08980)) | Uncertainty-guided adaptive lookahead for LLM decoding | Prior art for any adaptive-k variant of SemPA-Look: frame such a variant as a transfer, not an invention |

## 6. Metrics and encoders

| Work | Relevance |
|---|---|
| Anisotropy of contextual embeddings ([Ethayarajh 2019](https://ar5iv.arxiv.org/html/1909.00512)); rogue dimensions in cosine similarity ([Timkey & van Schijndel 2021](https://arxiv.org/pdf/2109.04404)) | Why cosine ATS compresses differences between methods |
| Negation blindness ([SNCSE, arXiv 2201.05979](https://arxiv.org/pdf/2201.05979); [arXiv 2504.00584](https://arxiv.org/pdf/2504.00584)) | A failure mode cosine ATS cannot see |
| [MTEB](https://arxiv.org/abs/2210.07316v3) | No single embedder dominates, so we re-score with MiniLM and BGE-small |
| BERTScore ([arXiv 1904.09675](https://arxiv.org/pdf/1904.09675v2)), BLEURT ([arXiv 2004.04696](https://ar5iv.arxiv.org/html/2004.04696)), COMET ([arXiv 2407.12832](https://arxiv.org/pdf/2407.12832)), G-Eval ([arXiv 2303.16634](https://arxiv.org/abs/2303.16634v3)) | Non-cosine metrics for the planned metric-robustness check of CASA |

## Positioning in one paragraph

SemPA-Look and SemPA-GBeam optimise how tokens are grouped under i.i.d.
erasures. Interleaving work such as arXiv 2511.16983 handles burst losses by
blind, random grouping. CASA sits between them: it keeps semantic grouping but
fits a channel-aware pairwise surrogate and optimises both the grouping and the
slot order for a correlated loss law. It is compared against SemPA-Look,
interleaving, interleaver-seeded swap search and the exhaustive oracle in
[channel_aware.md](channel_aware.md). Whether anything closer exists is the
open question for the systematic search above.
