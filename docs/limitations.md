# SemPA-Look's gains shrink under scrutiny

The core of SemPA-Look holds up. Our MPNet reproduction matches its absolute SemPA-Look and Full Search numbers within about 1–2%. What does not hold up is the size of the advantage the paper claims and the conditions under which it holds. With MPNet, SemPA-Look's ATS gain over Random PA is **+2–6% at 6 dB, +1–3% at 7 dB and under 1% at 8 dB and above**. The paper's headline **+14.32% (Fig. 6a) and +18.5% (Table II)** both rest on a 6 dB Random PA point that is internally inconsistent or about 12 standard errors below our value. Beyond the numbers, the paper's design depends on three idealisations it never tests. Packets are erased independently over uncoded BPSK/AWGN with no retransmission. The receiver simply drops lost tokens. One sentence encoder both defines and scores the objective. The complexity claims are also kept under three different counting rules. The useful lesson for our next stage: the remaining headroom in better *search* on the paper's objective is small, at most a few hundredths of ATS. The open, defensible novelty lies in changing the *problem*. The strongest candidates are (1) aggregation that knows the receiver will try to recover lost tokens, (2) channel-aware aggregation under correlated or fading erasures, and (3) a closed-form pairwise surrogate that takes the encoder out of the search loop and allows certified bounds.

*Sources and conventions.* The authoritative paper is Lee, Park, Choi, Park, "Low-Complexity Semantic Packet Aggregation for Token Communication via Lookahead Search," IEEE TCOM vol. 74, pp. 9639–9654, 2026 (cited below as "TCOM, p. N"). It uses the all-MPNet-base-v2 encoder. Facts that come only from arXiv v1 ([arXiv 2506.19451v1](https://arxiv.org/html/2506.19451v1)) are labelled "arXiv v1". That version used a CLIP text encoder and did not define WTS, and neither of those describes the published paper. Every reproduction number comes from this repo with `sentence-transformers/all-mpnet-base-v2`, 100 messages per point unless noted, paired per-message statistics and bootstrap 95% CIs (5,000 resamples). Text marked **Inference** is our reasoning, not a statement in the paper or a measured result.

## Five classes of limitation, from idealised channel to shifting complexity counts

### System model: a memoryless, single-shot erasure channel with a passive receiver

The paper sends uncoded BPSK over AWGN with fixed 300-bit packets. Packet error probability is p = 1 − (1 − Q(√(2γ)))^d, and "for simplicity, we do not assume the channel coding method" (TCOM, pp. 9641–9642, eq. 1). The ATS objective (eq. 6) hard-codes **i.i.d. Bernoulli(p) erasures with the same p for every packet**. Dummy messages are padded so that each subpacket "experiences independent packet erasures" (TCOM, p. 9641). Each packet group "is transmitted once" (TCOM, p. 9642). This drops the repetition axis of the authors' own earlier SemPA work, ref. [19], "Semantic packet aggregation and repeated transmission". The receiver takes the union of surviving tokens. Its reconstruction rule is not specified, so we had to assume survivors joined in original order and φ(∅, W) = 0 ([assumptions.md](assumptions.md), A1, P2). Channel coding is argued away as something that "only changes the effective packet error probability p". The evidence is one rate-5/6 convolutional code, compared at coded 4 dB (15.9% gain) against uncoded 6 dB (14.3% gain) (TCOM, pp. 9652–9653, Fig. 18a).

**Inference:** independence is the load-bearing assumption. Spreading important tokens across packets helps only because packets fail independently. Under block fading or bursts spanning a batch's packets, a message's subpackets fail together and the benefit shrinks. The uncoded channel is also a steep waterfall: p = 0.977 at 4 dB, 0.512 at 6 dB, 0.207 at 7 dB and 0.0556 at 8 dB (TCOM, pp. 9648, 9652). So the informative SNR window is only about 6–8 dB. Batching B = 100 messages implies buffering latency, which is never analysed. The ordering bits are never weighed against spending the same bits on FEC (about 60 of 300 bits at JM = 12, K = 24 by our arithmetic).

### Objective and metric: optimised and graded by the same encoder

φ is the cosine similarity of all-MPNet-base-v2 embeddings (TCOM, pp. 9642, 9646). The same encoder drives SemPA-Look's RSS surrogate (eqs. 18, 22) and scores ATS, so the headline results are **partly circular**. The authors concede that cosine "tends to assign relatively high similarity scores to many token configurations, thereby compressing the margin" (TCOM, p. 9648). They also chose it because it "exhibits the largest and most stable ATS gap" (0.8649 → 0.8777 vs 0.8600 for Random PA) (TCOM, p. 9651). External validation is thin:
- BLEURT and BERTScore are reported at one setting (K = 24, M = 4, 7 dB), with gaps of about 0.716 → 0.73 and 0.947 → 0.953–0.956 and no error bars (Fig. 16).
- LPIPS is computed on Stable Diffusion images conditioned on a Canny map of the original image.
- CLIP score (Table VI) measures agreement between the caption and the generated image, not fidelity to the original.
- There is no human study and no downstream NLP task.

The literature explains why this matters. Contextual embeddings are anisotropic, so cosines crowd near 1 ([Ethayarajh 2019](https://ar5iv.arxiv.org/html/1909.00512)). A few rogue dimensions dominate cosine similarity, and post-processing changes its correlation with human judgments ([Timkey & van Schijndel 2021](https://arxiv.org/pdf/2109.04404)). Sentence embeddings score negated pairs as too similar ([SNCSE](https://arxiv.org/pdf/2201.05979); [arXiv 2504.00584](https://arxiv.org/pdf/2504.00584)). No single embedder dominates across tasks ([MTEB](https://arxiv.org/abs/2210.07316v3)).

The encoder itself changed between arXiv v1, which used CLIP text, and TCOM, which uses MPNet ([arXiv v1](https://arxiv.org/html/2506.19451v1)). **Inference:** this shows the method was never tied to one encoder, yet no cross-encoder evaluation exists. A second structural gap is in the objective. Optimal packetization depends on which tokens are *predictable* only if the receiver tries to recover them, and this receiver never does.

### Algorithm and complexity claims: three counting rules, no bound, SNR-blind

The paper gives no optimality bound and leaves one to future work (TCOM, p. 9646). NP-hardness is asserted without a reduction (TCOM, pp. 9639, 9642). The RSS surrogate is justified as a first-order expansion "around p → 0" (eq. 15). Yet most headline numbers are taken at 6 dB, where p = 0.512. SemPA-Look's score Ψ (eq. 22) contains no p, so it returns the same partition at every SNR.

The encoder-call bookkeeping is inconsistent. Table III gives SemPA-Look (N−1)P(k+1) = 250 calls at K = 24, M = 4. The algorithm as specified caps lookahead at the number of unassigned subpackets ([sempa_look.py](../src/sempalab/algorithms/sempa_look.py), `effective_k = min(k, future_groups)`), so it actually makes **190 calls**. The paper's own Fig. 11b (260 calls at P = 130, N = 2) uses this capped count. Back-calculating from the paper's stated ratios, Fig. 10b uses the uncapped (N−1)P(k+1), Fig. 5a implies N·P·(k+1) = 350, and Fig. 5b implies about 200, which fits neither closed form (our arithmetic). The paper never says which rule each figure uses. Full Search is charged 2^K encoder calls in Table III. Sec. III-A, however, counts K!/((M!)^N N!) partitions × 2^N evaluations, which is about 2.9×10^14 at K = 24, M = 4 by our arithmetic. The speed-up claims also disagree with each other:
- The abstract says "up to 40×" versus exhaustive search.
- Sec. V-A says "five orders of magnitude".
- Table III's own ratio is about 6.7×10^4.
- Fig. 5a gives 46.8×.

"Linear complexity" holds only in encoder calls at fixed P and k. Each call costs C_enc(K), which grows quadratically in K (eq. 24). GA, the main complexity baseline, is charged exact ATS (2^N calls per chromosome), while SemPA-Look uses the cheap surrogate. A GA, beam or local search on the same surrogate would be the like-for-like comparison, and the paper does not include one.

### Evaluation and reported numbers: short texts, no statistics, internal contradictions

The evaluation covers COCO captions (K = 8–12) and WikiHow sentences (K ≤ 60), with Full Search feasible only up to K = 12. The paper's motivating use case is paragraph-length LLM output, which goes untested. Variance appears only in Fig. 3's bands and in Tables V–VI. Table V's standard deviations (e.g. 0.8716 ± 0.1594 vs 0.8138 ± 0.2002 at 7 dB) exceed many of the differences the paper interprets, such as a 0.0002 subword-versus-word change (TCOM, pp. 9648, 9652). Missing baselines include deterministic interleaving, importance-aware UEP, surrogate-based local or beam search, and the authors' own original SemPA.

The paper also contradicts itself in several places:
- **Table V text vs table.** The "respectively" pairing in the text swaps the 6 dB and 7 dB values.
- **Table IV vs Table V.** Both use K = 24, M = 6, 8 dB, yet report 0.9802 and 0.9696 for the same setting.
- **Fig. 6 caption vs axis.** K = 8, M = 4 gives N = 2, so only L = 1 is possible, but the WTS axis runs L = 1–5.
- **Fig. 7 admissibility.** B ≥ K/M is algebraically M ≥ K/B, not M ≤ K/B.
- **Fig. 13 vs Figs. 11–12.** Fig. 13 disagrees with Figs. 11 and 12 at identical configurations by 0.013–0.022 ATS (TCOM, pp. 9648–9652).

### Reproducibility: ten assumptions and eight components we could not rebuild

We had to assume ten unspecified knobs: the reconstruction rule, normalisation, the COCO and WikiHow subsets, the Fig. 6 and Fig. 3 settings, which L the WTS headline refers to, message counts, Random PA sampling, and seeds ([assumptions.md](assumptions.md), A1–A10). Some components are underspecified:
- **GA:** mutation rate, elitism and stopping rule are not given.
- **SemPA-GBeam:** κ and λ are not given.
- **MCTS:** budget, UCT constant and rollout are not given.
- **AIGC pipeline:** SD version, steps, seeds and image count are not given.
- **LLM post-correction:** prompt and decoding settings are not given.

The paper links no code. **Inference:** the least reproducible numbers are exactly the large relative gains (+14.32% ATS, +46.2% WTS, Table II's 0.5417) that carry the "semantic aggregation helps" message.

## What the MPNet reproduction reveals: real but small gains, concentrated in a narrow regime

### The 6 dB Random PA baseline is the source of the headline gap

SemPA-Look itself reproduces: 0.6341 vs the paper's 0.6421 in Table II (−1.2%), and −2.03% vs Full Search against the paper's −1.79% in Fig. 6a. Random PA does not. In Table II the paper's Random PA is **0.5417 and ours is 0.6191**. Our per-message SD is 0.0627, which puts the paper's value about 12 standard errors away ([comparison.csv](../results/stage4/comparison.csv), rows `table2_random`, `table2_sempa`; [wikihow_k24_m4/results.csv](../results/stage4/wikihow_k24_m4/results.csv)). Away from 6 dB the two agree: 0.8612 vs 0.860 at 7 dB and 0.9669 vs 0.970 at 8 dB.

For N = 2, exact ATS = (1−p)² + p(1−p)·S, where S = φ(A) + φ(B) does not depend on SNR. Solving for S from the paper's own Fig. 6a gives the following:
- **Paper Random PA:** S = 1.164 ± 0.012 at 6 dB but 1.298 ± 0.057 at 8 dB.
- **Paper SemPA-Look:** 1.468 vs 1.488, which is consistent.
- **Our Random PA:** S̄ = 1.303.

So the paper's 8 dB Random PA point agrees with ours, and its 6 dB point is the outlier. Under our S̄, the paper's 6 dB value (0.529) corresponds to p ≈ 0.546, i.e. about 5.90 dB ([coco_k8/results.csv](../results/stage4/coco_k8/results.csv); [paper_curves.yaml](../results/stage4/paper_curves.yaml), `fig6a`). This algebraic test is decisive only for N = 2. For Table II (N = 6), any monotone similarity profile that fits the 7 dB point allows 6 dB values anywhere from 0.295 to 0.781. The paper's Fig. 3 Random PA (0.49 at 6 dB, 0.95 at 8 dB) is almost exactly 1 − p (0.488, 0.944). We measure 0.6202 at K = 12, M = 3 ([reproduction.md](reproduction.md), gaps 1–2).

### The paired gains, with CIs

| Setting (MPNet, n = 100 unless noted) | SemPA-Look / Random PA | Gain [95% CI] |
|---|---|---|
| COCO K = 8, M = 4 (N = 2), 6 dB (Fig. 6a) | 0.5965 / 0.5637 | **+5.81% [4.73, 6.95]**, wins 82% |
| COCO K = 8, M = 4, 7 dB | 0.8643 / 0.8427 | +2.55% [2.09, 3.04] |
| COCO K = 8, M = 4, 8 dB | 0.9671 / 0.9602 | +0.72% [0.58, 0.85] |
| COCO K = 12, M = 2 (N = 6), 6 dB | 0.6474 / 0.6359 | +1.82% [1.28, 2.38] |
| COCO K = 12, M = 6 (N = 2), 6 dB | 0.6073 / 0.5699 | +6.57% [5.53, 7.69] |
| WikiHow K = 24, M = 4, 6 dB (Table II) | 0.6341 / 0.6191 | **+2.42% [1.71, 3.14]**, wins 79% |
| WikiHow K = 24, M = 4, 7 dB | 0.8758 / 0.8612 | +1.70% [1.42, 1.99] |
| WikiHow K = 24, M = 6, 6 dB (Table V) | — | +4.03% [3.05, 5.00] |
| WikiHow K = 60, M = 5, 7 dB (n = 20) | — | +0.69% [0.33, 1.07] |

Sources: [coco_k8](../results/stage4/coco_k8/results.csv), [coco_k12](../results/stage4/coco_k12/results.csv), [wikihow_k24_m4](../results/stage4/wikihow_k24_m4/results.csv), [wikihow_k24_m6](../results/stage4/wikihow_k24_m6/results.csv), [wikihow_len_long](../results/stage4/wikihow_len_long/results.csv).

The gain is largest for short messages split into two packets around p ≈ 0.2–0.5. Absolute gain peaks at 0.0328 at 6 dB for K = 8. The gain falls steadily with sentence length (+3.43% at K = 10 → +1.32% at K = 40 → +0.69% at K = 60), even as SemPA-Look's encoder calls rise from 20 to 490. It is exactly zero at M = 1 and M = K, where SemPA-Look still spends 490 calls at M = 1. Per message, SemPA-Look loses to Random PA on **21/100 messages at 6 dB** for K = 24, M = 4 (worst case −0.039).

SemPA-Look captures **73% of the Full Search headroom at N = 2 but only 28% at N = 6** (K = 12, 6 dB). At K = 12, M = 2 it sits 4.3% below Full Search (0.6474 vs 0.6767), so "close to full search" holds only at N = 2. It is also **SNR-blind**: its partition is identical across SNR for 100/100 messages, while Full Search changes partition for 99–100/100 and GA for 95–100/100 ([coco_k12/results.csv](../results/stage4/coco_k12/results.csv)).

### The worst-case (WTS) headline and the design motivation do not survive

At the Fig. 6c setting (K = 12, M = 2), SemPA-Look's WTS gain over Random PA is +3.6% [2.2, 4.9] at L = 1 and +10.9% [3.6, 19.2] at L = 3. At L = 4 it is **+11.9% [−0.7, 27.2]** and at L = 5 **+17.6% [−0.3, 39.1]**. Both CIs include zero, and both win rates are 52%. The paper's unattributed "+46.2%" has no counterpart in our data. The greedy TSS and RSS variants that motivate the RSS design (Fig. 3) separate by ≤0.004 ATS. They are often worse than Random PA (e.g. K = 8, M = 4, 6 dB: TSS 0.5501, RSS 0.5502, Random 0.5637). Their L = 1 WTS is far lower (0.3368/0.3372 vs Random 0.5187), and they spend more calls than SemPA-Look (566 vs 50 at K = 12, M = 4). **Inference:** SemPA-Look's gain comes from lookahead plus sampling, not from the choice between TSS and RSS.

The fixed-N sweep (Fig. 13) shows the same pattern. In our data N = 4 beats or ties N = 2 at *every* K, for Random PA as well as SemPA-Look; at K = 24 the SemPA-Look difference is +0.0114 [0.0088, 0.0140]. This is a property of φ's concavity in the fraction of tokens kept, not of the search. Our post-cache-fix (commit `741a1a5`) runs reproduce identical configurations exactly ([wikihow_fixed_n2](../results/stage4/wikihow_fixed_n2/results.csv), [wikihow_fixed_n4](../results/stage4/wikihow_fixed_n4/results.csv)).

### Cost buys little beyond the first lookahead

The k sweep at K = 24, M = 4, 7 dB:

| k | ATS | Encoder calls |
|---|---|---|
| 0 | 0.8658 | 50 |
| 1 | 0.8725 | 100 |
| 2 | 0.8748 | 140 |
| 4 (paper default) | 0.8758 | 190 |
| 5 | 0.8766 | 200 |
| Random PA (reference) | 0.8612 | 0 |

Every k ≥ 2 adds at most 0.002 ATS ([wikihow_k24_m4/summary.csv](../results/stage4/wikihow_k24_m4/summary.csv)). With more budget GA wins: 0.8871 at 2,600 calls against SemPA-Look's 0.8828 at 260 ([wikihow_p/summary.csv](../results/stage4/wikihow_p/summary.csv)). At G = 30, GA reaches 0.8891 at 19,200 calls ([wikihow_ga_g/summary.csv](../results/stage4/wikihow_ga_g/summary.csv)).

At equal calls in the N = 2 regime, SemPA-Look is the more efficient of the two: 0.8821 at 220 calls against GA's 0.8728 at 200. In the Table III setting (10 messages, CPU), the picture is:

| Method | ATS | Encoder calls | Time per message |
|---|---|---|---|
| GA | 0.7183 | 3,200 | 12.67 s |
| SemPA-Look | 0.6992 | 190 | 2.09 s |
| Random PA | 0.6890 | 0 | 0.00017 s |

The timing is indicative only ([complexity_k24/results.csv](../results/stage4/complexity_k24/results.csv)). At 190 calls against GA's 3,200, the measured cost ratio is **16.8×**, against the 12.8× implied by the paper's closed form.

## Mapping each limitation to an existing fix

| # | Limitation | What our data says | Existing approach that addresses it |
|---|---|---|---|
| L1 | Lost tokens simply dropped; no receiver recovery | Not testable in the paper's model | MLM/LLM recovery at the receiver: TokCom ([arXiv 2502.12096](https://arxiv.org/pdf/2502.12096)); MLM-prior iterative detection by an overlapping Park/Choi group ([arXiv 2601.17770](https://arxiv.org/abs/2601.17770v1), [2605.02123](https://arxiv.org/pdf/2605.02123)); BERT concealment with interleaving ([arXiv 2408.04535](https://arxiv.org/abs/2408.04535)); TokCode erasure-aware re-encoding ([arXiv 2604.12931](https://arxiv.org/html/2604.12931v1)) |
| L2 | i.i.d. erasures, AWGN only | SemPA-Look's partition is SNR-blind (100/100 identical) | Rayleigh/Rician block-fading evaluation (TONIC, [arXiv 2605.21553](https://arxiv.org/html/2605.21553v1)); burst-fade semantic transport ([SITP, arXiv 2512.09291](https://arxiv.org/html/2512.09291v1)); random interleaving against bursts |
| L3 | Surrogate Ψ ignores p | Full Search changes partition with SNR for 99–100/100 messages | Exact ATS is the multilinear extension of a set function over packets (**Inference**); second-order expansion in p; deletion-robust submodular maximisation ([Bogunovic et al. 2017](https://arxiv.org/pdf/1706.04918)) |
| L4 | No channel coding, UEP or HARQ; single shot | Gains < 1% at ≥ 8 dB, where a real link operates | Token UEP with expanding-window fountain codes ([arXiv 2511.22859](https://arxiv.org/html/2511.22859v1)); semantic rate matching ([arXiv 2608.11551](https://arxiv.org/pdf/2608.11551)); IK-HARQ ([arXiv 2203.06634](https://ar5iv.labs.arxiv.org/html/2203.06634)); short codes plus LM correction ([arXiv 2505.08536](https://arxiv.org/abs/2505.08536)) |
| L5 | Same encoder optimises and scores; cosine compresses margins | All gains are MPNet-only; arXiv v1 used a different encoder | BERTScore ([arXiv 1904.09675](https://arxiv.org/pdf/1904.09675v2)), BLEURT ([arXiv 2004.04696](https://ar5iv.arxiv.org/html/2004.04696)), COMET ([arXiv 2407.12832](https://arxiv.org/pdf/2407.12832)), G-Eval LLM judge ([arXiv 2303.16634](https://arxiv.org/abs/2303.16634v3)); cross-encoder robustness per [MTEB](https://arxiv.org/abs/2210.07316v3) |
| L6 | Negation/entity blindness of cosine | Not tested | Negation-aware embedders ([arXiv 2504.00584](https://arxiv.org/pdf/2504.00584)); weighted token metrics and NE-F1 ([survey, arXiv 2609.10714](https://arxiv.org/html/2609.10714v1)) |
| L7 | No optimality bound; NP-hardness asserted | 28–73% of Full Search headroom captured | Deletion-robust submodular maximisation over matroids with constant factors ([arXiv 2201.13128](https://arxiv.org/pdf/2201.13128)); robust submodular maximisation ([Orlin et al.](https://www.alphaxiv.org/abs/1507.06616)), conditional on (approximate) submodularity of φ |
| L8 | Encoder calls inside the search loop; inconsistent counting | 190 vs 250 calls; k ≥ 2 adds ≤ 0.002 ATS | Amortised Shapley prediction in one forward pass ([FastSHAP](https://arxiv.org/pdf/2107.07436); [Yang et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.19998)); pairwise surrogate → quadratic assignment with local search (**Inference**) |
| L9 | GA baseline charged exact ATS; missing interleaving, local-search and surrogate baselines | GA beats SemPA-Look at larger budgets | Surrogate-based GA/beam, swap local search, simulated annealing ([arXiv 1312.0841](https://arxiv.org/pdf/1312.0841), [arXiv 1409.5223](https://arxiv.org/pdf/1409.5223)); modular interleaving as in TokCode |
| L10 | Per-message search cost, no amortisation | 2.09 s/message on CPU for +1.5% ATS | Learned partitioners: GAP-style GNN, NeuroCUT RL ([arXiv 2310.11787](https://arxiv.org/html/2310.11787v3)) |
| L11 | Short texts; WTS headline; Random PA at 6 dB; no CIs | Gains fall to +0.69% at K = 60; WTS L4–L5 CIs include 0 | Paired bootstrap CIs, multiple baseline draws, longer texts. These are standard practice, not a literature fix. |
| L12 | Worst case not optimised | WTS gains real at L ≤ 3 only | Robust (min-over-deletions) submodular objectives ([arXiv 1706.04918](https://arxiv.org/pdf/1706.04918)); the paper itself names worst-case optimisation as future work (TCOM, p. 9653) |

## Ranked novelty directions for the next stage

The ranking weighs three things: how much headroom the direction opens beyond the few hundredths of ATS left on the paper's objective, whether our literature search found prior work on it, and how cheaply this repo can test it. One step comes first regardless of direction. Any new method must beat **deterministic interleaving and a swap local search on the RSS surrogate**. Neither baseline is in the paper, and without them a reviewer can dismiss any search-side gain. Both are a day's work in `src/sempalab/algorithms/`.

**1. Recovery-aware packet aggregation (addresses L1, and partly L5 and L6).** Our search found no paper that jointly optimises token-to-packet grouping with receiver-side MLM/LLM recovery for text. Recovery changes what the objective rewards. Instead of keeping important tokens alive, it should keep alive the tokens that are hard to predict from context, and preserve enough context for the rest to be recovered. Interleaving-style recovery work points the same way ([arXiv 2502.12096](https://arxiv.org/pdf/2502.12096); [arXiv 2604.12931](https://arxiv.org/html/2604.12931v1)). This is the largest change to the problem and the highest-value direction. **Risk:** the Shin et al. MLM-detection line shares the Park and Choi co-authors ([arXiv 2601.17770](https://arxiv.org/abs/2601.17770v1)), so the original authors may already be moving here. TokCode also found that receiver-only infilling barely beat dropping tokens at 40–60% loss, so the gain depends on the regime. *Effort:* medium. *First experiment:* at COCO K = 12, M ∈ {2, 3, 4}, add a fill-mask receiver (e.g. RoBERTa) that infills erased positions. Recompute exact ATS over all 2^N patterns for Random PA, interleaving, SemPA-Look and Full Search. Check whether the Full Search optimum changes and how far SemPA-Look's partition falls from it.

**2. Channel-aware aggregation under correlated erasures and fading (addresses L2, L3 and L4).** *Done: see [channel_aware.md](channel_aware.md).* SemPA-Look is SNR-blind while the true optimum is not. Our search found no semantic aggregation study under Gilbert–Elliott or block fading. **Inference:** exact ATS is the multilinear extension of φ over packets at (1−p)·1. That makes a p-dependent second-order surrogate (single- and pair-loss terms) derivable, with an explicit error term, and correlated-loss versions follow by replacing the product measure. *Effort:* low to medium, because the exact ATS machinery in [metrics.py](../src/sempalab/metrics.py) already enumerates 2^N patterns. *First experiment:* replace the Bernoulli weights in `exact_ats` with a two-state Gilbert–Elliott chain over packet order at matched average p. Measure how much Random PA, SemPA-Look and Full Search lose, and whether a burst-aware Full Search picks different partitions.

**3. Closed-form pairwise surrogate with certified search (addresses L7, L8 and L9).** Approximate φ under deletion with an additive-plus-pairwise model over tokens (Shapley-interaction or Möbius terms). Packetization then becomes a quadratic assignment problem in which ATS for any partition is closed-form. The encoder leaves the search loop, and swap or Kernighan–Lin search plus an SDP/ILP bound becomes possible. **Inference:** this gives the optimality certificate the paper lacks. Amortised Shapley prediction ([FastSHAP](https://arxiv.org/pdf/2107.07436)) could reduce even the fitting cost to one forward pass. *Effort:* medium. *First experiment:* for 100 COCO K = 12 captions, encode all 2^12 = 4,096 token subsets, fit the pairwise model, and report R². Then solve the resulting QAP exactly at K = 12 and compare its true ATS against Full Search and SemPA-Look.

**4. Metric validity and encoder robustness (addresses L5 and L6).** Optimise with MPNet and evaluate with MiniLM, a modern LLM embedder, BERTScore, BLEURT and an LLM judge. Add a small human or LLM-judge study of whether cosine ranks loss patterns correctly. Our search found no study measuring how often embedding cosine under word deletion misranks loss patterns against human judgment. This is lower in method novelty but cheap, and it hardens every other direction against the obvious reviewer objection. *Effort:* low. *First experiment:* take the stored SemPA-Look, Random PA and Full Search partitions for COCO K = 12 and re-score them under three other encoders. Report the cross-encoder ATS gains and the Kendall τ of partition rankings.

**5. Deletion-robust worst-case aggregation (addresses L12 and L7).** The paper names worst-case optimisation as future work. Robust submodular maximisation offers constant-factor guarantees *if* φ is approximately submodular ([arXiv 1706.04918](https://arxiv.org/pdf/1706.04918); [arXiv 2201.13128](https://arxiv.org/pdf/2201.13128)). Whether sentence-embedding cosine has this property is, as far as our search found, untested. *Effort:* low to medium. *First experiment:* on COCO K = 12, estimate the submodularity ratio γ and the share of diminishing-returns violations of MPNet cosine under token deletion. The answer decides whether this direction is viable.

**6. Joint aggregation with per-packet UEP or retransmission (addresses L4).** Assign code rate or MCS per packet by residual semantic score, or retransmit only the most critical packet. Token UEP exists for images ([arXiv 2511.22859](https://arxiv.org/html/2511.22859v1)), but we found none on aggregated text packets. **Inference:** the payoff is uncertain, since TokCom-UEP's gain vanished at reception overhead 1.3, and this direction needs a coded-PHY simulator that the repo lacks. *Effort:* high. *First experiment:* use two code rates and give the lower rate to the packet whose loss costs most RSS. Compare exact ATS against equal protection at equal total bits.

**7. Learned amortised packetiser (addresses L10).** A GNN or pointer-network policy trained on Full Search labels would replace per-message search with one forward pass ([NeuroCUT](https://arxiv.org/html/2310.11787v3)). It is ranked last because the gain it would amortise is only about 0.015 ATS at 7 dB for K = 24. *Effort:* high. *First experiment:* not recommended until directions 1–3 show that more headroom exists.

## Decisions (2026-10-22)

The open questions from the first version of this review were settled by the repo owner as follows.
- **Contacting the authors:** no. The 6 dB Random PA gap stays reported as unexplained, as in [reproduction.md](reproduction.md).
- **Encoder:** MPNet stays the encoder that defines the objective and the headline numbers, for continuity with the paper. Every result for a new method is also re-scored under at least two other encoders (direction 4), so the circularity objection is answered from the start rather than as a separate study.
- **Overlap with the authors' group:** lead with direction 2 (channel-aware aggregation), folding in direction 3's p-aware surrogate. Direction 1 is deferred because the authors' own group is active on MLM receivers; a recovery receiver can later be added as one more channel/receiver model in the same framework.
- **Target regime:** new results are reported against the packet error rate p (about 0.05 to 0.6) rather than uncoded SNR. That range covers uncoded BPSK at 6–8 dB and coded links at lower SNR, and it is where aggregation matters at all. SNR axes are kept only for figures compared with the paper.
- **Compute:** experiments run in the cloud container (4 CPUs, about 15 GB RAM), not on a local machine. The largest planned run, exhaustive subset encoding at K = 12 (4,096 subsets × 100 messages, about 410k short sentences, about 1.3 GB of float32 embeddings), fits without extra RAM.
- **Subword tokenization:** reproduced; see [reproduction.md](reproduction.md#subword-tokenization). Absolute numbers match within about 1 %, but the 45.5 % subword WTS gain comes out at +14.5 % (L = 3), the same Random PA gap as with words. The paper's own subword Fig. 13b has N = 4 above N = 2, which supports our Fig. 13 ordering over its word-based panel.

## Conclusion

The reproduction changes what kind of contribution is worth making. SemPA-Look is a sound, cheap heuristic that does better than random grouping, but the margin is small, regime-specific and partly inflated by one baseline point. A paper that only searches the same objective faster or better would compete for at most a few hundredths of ATS, and GA already wins that contest at larger budgets. The weaknesses that matter are in the model: the receiver recovers nothing, the channel has no memory, and the objective ignores p and is scored by the encoder that defines it. Each of these has a mature fix in adjacent literature that has never been combined with packet aggregation. That gap, not the search, is where the next stage should aim. Directions 2 and 4 are cheap enough to run first; see the decisions above for the chosen order.
