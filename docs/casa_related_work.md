# Related work and novelty check for CASA

This note positions CASA (channel-aware semantic aggregation, [channel_aware.md](channel_aware.md)) against prior work. It was written before any new compute for the workshop write-up, as the project plan asks. Search date: 2026-11-20. Sources are arXiv listings and web search; every paper marked "read" was read in its arXiv HTML or PDF body, not only the abstract.

## Verdict

**The core claim survives, but it must be stated narrowly.** We found no paper that chooses the grouping of tokens into packets, or the order of those packets in time, by optimizing a semantic objective against the *joint loss law of a channel with memory*. Every token or feature packetizer we found either assumes i.i.d. erasures or handles bursts with a fixed or random interleaver that does not depend on the channel's statistics.

Three things narrow what we may claim:

1. **"Burst-aware token packetization" is not new.** Random scrambling (TokCom), round-robin interleaving (TokCode), strided channel grouping (Every Packet Counts) and random semantic interleaving (2511.16983, SITP) all exist, and several say explicitly that they spread losses to break bursts. Our claim is the *optimized, channel-law-dependent* assignment and ordering, and the finding that ordering carries most of the gain.
2. **Optimizing interleaving against a Gilbert channel is classical for video.** Interleaved source coding (ISC) chooses a frame-interleaving pattern for a two-state Gilbert channel with a Markov decision process, and Liang, Apostolopoulos and Girod showed that video distortion depends on burst length, not only on the average loss rate. CASA transfers this idea to semantic token grouping, where the objective is a non-additive semantic similarity over token subsets. We should frame it as that transfer.
3. **The SemPA authors' group already works on correlated losses for tokens.** 2608.08698 (video tokens over block fading) shares an author with SemPA-Look (J. Park). It spreads important and correlated tokens across packets with a heuristic concentration penalty and schedules packets online from current-block CSI. Its blocks fade independently, so it does not exploit memory, and it optimizes no semantic objective over the loss law. It is the closest work and must be cited and contrasted in the introduction. It also signals that a direct extension of SemPA to bursty channels by that group is plausible, so timing matters.

Suggested one-sentence claim for the paper: *"We give the first semantic packet aggregation method for text tokens whose grouping and transmission order are optimized against the joint loss law of a channel with memory, and show that slot ordering, a degree of freedom absent under i.i.d. erasures, carries most of the attainable gain."*

## Closest work

| Work | Unit | Loss model | How tokens/features go into packets | Depends on channel memory? | Contrast with CASA |
|---|---|---|---|---|---|
| **SemPA-GBeam**, arXiv:2504.19591 (read earlier, Stage 2) | text tokens | i.i.d. erasure | semantic grouping, genetic beam search on ATS | no | our baseline family |
| **SemPA-Look**, arXiv:2506.19451, IEEE TCOM 2026 (the reproduced paper) | text tokens | i.i.d. erasure ("each packet erasure event is independent") | semantic grouping, lookahead on RSS | no | our baseline; collapses under memory (channel_aware.md §3) |
| **Video token communication over block fading**, Xie, Oh, Chen, Park, Wu, Zhang, arXiv:2608.08698, Aug 2026 (read) | video tokens | independent block Rayleigh fading (i.i.d. across blocks) | fixed cyclic/pseudorandom spreading of I/P tokens; online greedy scheduler from current-block CSI with a heuristic penalty on putting important packets in one block | partly: it penalizes within-block concentration, but blocks are independent and no loss law is fitted | no semantic objective over the joint loss law; no memory across packets; no ordering or interleaving-depth analysis |
| **Feature partitioning and semantic equalization**, Yang, Ma, Liang, Shi, arXiv:2511.16983, Nov 2025 (read) | CNN channels / Transformer tokens (images) | i.i.d. erasure ("each packet is erased with probability p") | channel-wise partition plus random unit grouping to turn bursts into near-independent losses | no | random, not optimized; no burst model in the experiments |
| **TokCom**, Qiao et al., arXiv:2502.12096, IEEE Wireless Commun. Mag. (read) | image tokens | coded PHY, PER sweep; no erasure model named | token positions in each packet randomized "to mitigate the impact of burst errors" | no | random scrambling; receiver-side recovery with a generative model |
| **TokCode**, Hu, Wang, Li, arXiv:2604.12931, GLOBECOM 2026 preliminary (read) | text tokens (prompt) | i.i.d. erasure | fixed round-robin interleave over P packets; optimizes the token *content* with an erasure-rate-aware adapter | no | complementary: changes what is sent, not how it is grouped |
| **Every Packet Counts**, Wei, Zhou, Shi, Wang, Lu, ACM MM 2026, arXiv:2608.11096 (read) | learned-image-codec latent channels | trained on uniform loss; tested on Gilbert–Elliott | strided "interleaved channel grouping" | no (generalizes to GE without a GE design) | fixed grouping; shows bursts are a recognized test case |
| **LGSemCom**, Bo, Zhang, Tao, He, Huang, arXiv:2609.06989, Sep 2026 (read) | latent bits (images) | trained i.i.d.; evaluated on a 3-state Markov burst model | fixed interleaver | no | same pattern: evaluated under bursts, not designed for them |
| **SITP**, Wang et al., arXiv:2512.09291, Dec 2025 (read) | quantized image features | AWGN with a piecewise SNR fade | fixed random permutation across a batch of images (depth = batch size) | no | random interleaver; delay not quantified |

Other token-communication papers we checked have no packet-loss model at all: semantics-aware hierarchical token communication (arXiv:2604.27641, AWGN bit mapping), context-aware token masking and detection (arXiv:2605.02123, block fading per token), and the survey arXiv:2609.10714, which lists no work on token packetization under correlated loss.

## Classical background to cite

- **Burst length matters.** Y. J. Liang, J. G. Apostolopoulos, B. Girod, "Analysis of packet loss for compressed video: does burst-length matter?", ICASSP 2003, doi:10.1109/ICASSP.2003.1200063; journal version IEEE TCSVT 18(7):861–874, 2008. The motivation for evaluating semantic aggregation at matched average loss but different memory.
- **Optimized interleaving over a Gilbert channel.** Interleaved source coding (ISC), ICME 2005, which selects a frame-interleaving pattern for a two-state Gilbert erasure channel with a Markov decision process (we could not reach the PDF from this environment; authors and pages to be confirmed before citing).
- **Gilbert–Elliott model.** E. N. Gilbert (1960) and E. O. Elliott (1963), and its use for Internet packet loss.
- **Packet interleaving and its delay cost**, the honest baseline CASA is compared against (channel_aware.md, interleaving section).

## What this means for the remaining steps

- The mismatched-channel test (step 2) matters more after this check. The classical video work assumes the Markov law is known; a reviewer will ask what happens when it is estimated. We report CASA designed with a wrong memory parameter, a wrong family, a wrong p, and a Gilbert model fitted from a finite loss trace.
- The introduction must contrast with 2608.08698 and 2511.16983 explicitly, and present random/round-robin interleaving (TokCom, TokCode) as the state of practice that CASA's interleaver baselines stand in for.
