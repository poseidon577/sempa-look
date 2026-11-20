# Related work

Where this repository sits relative to prior work, and how each of our
contributions differs from the closest papers.

**Status.** Sections 1 and 3 to 6 collect the work cited in
[limitations.md](limitations.md), [assumptions.md](assumptions.md) and the
project plan. Section 2 and the verdict below come from a dedicated search for
CASA's novelty, run on 2026-11-20 before any new compute for the workshop
write-up: arXiv listings for token communication, semantic packetization and
packet loss in semantic communication, follow-ups of SemPA-Look and
SemPA-GBeam, and classical work on interleaving for bursty channels. Every
paper in Section 2 marked "read" was read in its arXiv HTML or PDF body, not
only the abstract.

## Verdict for CASA

**The core claim survives, but it must be stated narrowly.** We found no paper
that chooses the grouping of tokens into packets, or the order of those packets
in time, by optimising a semantic objective against the *joint loss law of a
channel with memory*. Every token or feature packetiser we found either assumes
i.i.d. erasures or handles bursts with a fixed or random interleaver that does
not depend on the channel's statistics.

Three things narrow what we may claim:

1. **"Burst-aware token packetisation" is not new.** Random scrambling
   (TokCom), round-robin interleaving (TokCode), strided channel grouping
   (Every Packet Counts) and random semantic interleaving (arXiv 2511.16983,
   SITP) all exist, and several say they spread losses to break bursts. CASA's
   claim is the *optimised, channel-law-dependent* assignment and slot order,
   and the finding that ordering carries most of the gain.
2. **Optimising interleaving against a Gilbert channel is classical for
   video.** Interleaved source coding (ISC) picks a frame-interleaving pattern
   for an erasure channel with a Markov decision process, and Liang,
   Apostolopoulos and Girod showed that video distortion depends on burst
   length, not only on the average loss rate. CASA transfers the idea to
   semantic token grouping, where the objective is a non-additive similarity
   over token subsets. Frame it as that transfer.
3. **The SemPA authors' group already works on correlated losses for tokens.**
   arXiv 2608.08698 (video tokens over block fading) shares an author with
   SemPA-Look. It spreads important tokens across packets with a heuristic
   concentration penalty and schedules them from current-block CSI. Its blocks
   fade independently, so it does not exploit memory, and it optimises no
   semantic objective over a loss law. It is the closest work and must be
   contrasted in any paper. It also makes a direct SemPA extension to bursty
   channels by that group plausible, so timing matters.

Suggested claim: *"We give the first semantic packet aggregation method for
text tokens whose grouping and transmission order are optimised against the
joint loss law of a channel with memory, and show that slot ordering, a degree
of freedom absent under i.i.d. erasures, carries most of the attainable gain."*

## 1. The paper and its predecessor

| Work | What it does | Role here |
|---|---|---|
| Lee, Park, Choi, Park, *Low-Complexity Semantic Packet Aggregation for Token Communication via Lookahead Search*, IEEE TCOM vol. 74, pp. 9639–9654, 2026 ([arXiv 2506.19451](https://arxiv.org/abs/2506.19451)) | SemPA-Look: residual semantic score plus sampled lookahead to group tokens into packets under i.i.d. erasures | The paper we reproduce ([reproduction.md](reproduction.md)). arXiv v1 used a CLIP text encoder and no WTS, so we follow the published TCOM version |
| SemPA-GBeam, IEEE SPAWC 2025 ([arXiv 2504.19591](https://arxiv.org/abs/2504.19591)) | Genetic beam search for the same aggregation problem | Direct predecessor. Implemented as a baseline (assumption A13) |
| Same group, *Semantic packet aggregation and repeated transmission* (TCOM ref. [19]) | Aggregation with repetition | Background. The TCOM paper drops the repetition axis |

## 2. Burst losses, interleaving and correlated loss (closest to CASA)

| Work | Unit | Loss model | How units go into packets | Depends on channel memory? | How CASA differs |
|---|---|---|---|---|---|
| Video token communication over block fading, Xie, Oh, Chen, Park, Wu, Zhang ([arXiv 2608.08698](https://arxiv.org/abs/2608.08698), Aug 2026, read) | video tokens | independent block Rayleigh fading | fixed cyclic/pseudorandom spreading; online greedy scheduler from current-block CSI with a heuristic penalty on putting important packets in one block | partly: penalises concentration within a block, but blocks are independent and no loss law is fitted | no semantic objective over a joint loss law, no memory across packets, no ordering or interleaving-depth analysis |
| Feature partitioning and semantic equalization, Yang, Ma, Liang, Shi ([arXiv 2511.16983](https://arxiv.org/abs/2511.16983), Nov 2025, read) | image CNN channels / Transformer tokens | i.i.d. erasure in the experiments | channel-wise partition plus random unit grouping, to turn bursts into near-independent losses | no | blind, random grouping. CASA groups text tokens by semantics and picks each group's slot for the channel's joint loss law; a packet interleaver and an interleaver-seeded swap search are baselines in [channel_aware.md](channel_aware.md) |
| TokCom, Qiao et al. ([arXiv 2502.12096](https://arxiv.org/abs/2502.12096), IEEE Wireless Commun. Mag., read) | image tokens | coded PHY with a PER sweep | token positions in each packet randomised "to mitigate the impact of burst errors" | no | random scrambling plus generative recovery at the receiver |
| TokCode, Hu, Wang, Li ([arXiv 2604.12931](https://arxiv.org/abs/2604.12931), GLOBECOM 2026 preliminary, read) | text tokens (prompt) | i.i.d. erasure | fixed round-robin interleave; optimises the token *content* with an erasure-rate-aware adapter | no | complementary: changes what is sent, not how it is grouped |
| BERT concealment with interleaving ([arXiv 2408.04535](https://arxiv.org/abs/2408.04535)) | text tokens | erasures | fixed interleaving, receiver-side recovery | no | CASA keeps the paper's receiver and optimises the grouping; recovery-aware aggregation is open direction 1 |
| Every Packet Counts, Wei, Zhou, Shi, Wang, Lu ([arXiv 2608.11096](https://arxiv.org/abs/2608.11096), ACM MM 2026, read) | learned-codec latent channels | trained on uniform loss, tested on Gilbert–Elliott | strided interleaved channel grouping | no | fixed grouping; bursts only as a test case |
| LGSemCom, Bo, Zhang, Tao, He, Huang ([arXiv 2609.06989](https://arxiv.org/abs/2609.06989), Sep 2026, read) | latent bits (images) | trained i.i.d., evaluated on a 3-state Markov burst model | fixed interleaver | no | same pattern: evaluated under bursts, not designed for them |
| SITP, Wang et al. ([arXiv 2512.09291](https://arxiv.org/abs/2512.09291), Dec 2025, read) | quantised image features | AWGN with a piecewise SNR fade | fixed random permutation across a batch of images (depth = batch size) | no | random interleaver at the transport layer; delay not quantified |
| TONIC ([arXiv 2605.21553](https://arxiv.org/html/2605.21553v1)) | semantic features | Rayleigh/Rician block fading | none | no | evaluation setting only; no aggregation |

Token-communication papers with no packet-loss model at all: semantics-aware
hierarchical token communication ([arXiv 2604.27641](https://arxiv.org/abs/2604.27641),
AWGN bit mapping), context-aware token masking and detection
([arXiv 2605.02123](https://arxiv.org/abs/2605.02123), per-token block fading),
and the survey [arXiv 2609.10714](https://arxiv.org/abs/2609.10714), which lists
no work on token packetisation under correlated loss.

Classical background:
- Y. J. Liang, J. G. Apostolopoulos, B. Girod, "Analysis of packet loss for
  compressed video: does burst-length matter?", ICASSP 2003
  ([doi:10.1109/ICASSP.2003.1200063](https://doi.org/10.1109/ICASSP.2003.1200063));
  journal version IEEE TCSVT 18(7), 2008.
- J. Y. Lee, H. Radha, "Interleaved source coding (ISC) for predictive video
  coded frames over the Internet", ICC 2005
  ([doi:10.1109/ICC.2005.1494542](https://doi.org/10.1109/ICC.2005.1494542)),
  and its Gilbert-channel evaluation at ICME 2005.
- The Gilbert (1960) and Elliott (1963) two-state channel.

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
erasures. Burst-aware token and feature packetisers (TokCom, TokCode,
arXiv 2511.16983, SITP) spread units with a fixed or random shuffle, and the
closest token work (arXiv 2608.08698) uses a heuristic within-block penalty
under independent block fading. CASA keeps semantic grouping but fits a
channel-aware pairwise surrogate and optimises both the grouping and the slot
order for a correlated loss law, transferring the classical idea of
channel-optimised interleaving (ISC) to a non-additive semantic objective. It
is compared against SemPA-Look, interleaving, interleaver-seeded swap search
and the exhaustive oracle in [channel_aware.md](channel_aware.md).
