# Assumptions and paper-faithful choices

Every place where the paper (Lee et al., IEEE TCOM vol. 74, pp. 9639–9654, 2026)
does not pin down a detail, the choice we made, and why. "Paper" below means the
published TCOM version.

## Settled by the paper

| # | Topic | Choice | Evidence in the paper |
|---|---|---|---|
| P1 | Encoder | `sentence-transformers/all-mpnet-base-v2`, L2-normalised, cosine similarity | Sec. V: "We embed texts using a pre-trained all-MPNet-base-v2 model" |
| P2 | Empty reconstruction | φ(∅, W) = 0 | Eq. (5) drops the both-lost term; Fig. 7/8 give ATS = 1 − p exactly at M = K (0.793 at 7 dB) |
| P3 | WTS | (1/B) Σ_b min over losing exactly L subpackets of φ(W_b \ F(H), W_b), per L, unweighted, independent of SNR | Sec. V definition; Fig. 6c/d plot it against L; Fig. 12 uses L = 1 |
| P4 | Lookahead sampling | k lookaheads per candidate are mutually disjoint (RandomDisjoint, eq. 21), so k is capped by the subpackets left; candidates are sampled with replacement | Sec. IV-B, Algorithm 1; Fig. 11 reports 260 calls at P=130, K=24, M=12, k=4, i.e. k collapses to 1 when N = 2 |
| P5 | RSS of a lookahead | ψ(Ĉ, W) = φ(W \ Ĉ, W), always relative to the full message W | Eq. (18), (22) |
| P6 | GA generations | "G generations" = G evaluated populations, G·P·2^N encoder calls. Our `genetic()` evaluates an initial population plus G′ evolved ones, so the runner calls it with G′ = G − 1 | Table III closed form; Fig. 10b: 640 calls at G = 1, P = 10, N = 6 |
| P7 | Channel | p = 1 − (1 − Q(√(2γ)))^300, independent erasure per subpacket (B ≥ N, cyclic multiplexing) | Eq. (1), Sec. V packet length 300 bits; anchors 0.977 (4 dB), 0.512 (6 dB), 0.207 (7 dB), 0.0557 (8 dB) tested |
| P8 | Batch size B | B only decides which M are admissible (M ≥ K/B); with B ≥ N the subpackets of a message are erased independently, so exact ATS does not depend on B | Sec. II-A, Fig. 7 heatmap rows are constant across admissible B |

## Our assumptions (the paper is silent or inconsistent)

| # | Topic | Choice | Why |
|---|---|---|---|
| A1 | Reconstruction | Surviving tokens joined by single spaces in original order | Eq. (3) says "if we assume that the order of the tokens are given"; packets carry ordering bits (Sec. V) |
| A2 | Tokenization | Word-based: lowercase, punctuation removed (letters, digits, `'` and `-` kept), whitespace split | Paper uses word tokens "split by space" for most figures; subword runs are not reproduced |
| A3 | COCO messages | Unique normalised captions from `captions_train2017.json` with exactly K words; 100 drawn with seed 42; frozen in `results/stage4/eval_sets/` | Paper does not say which split, how captions were filtered, or which 100 |
| A4 | WikiHow messages | Sentences of article bodies in `wikihowAll.csv` (first 20,000 articles), split at `. ! ? ;` or line breaks, normalised as in A2, exactly K words | Paper only says WikiHow "comprises longer sentences than MS-COCO" |
| A5 | Fig. 6 setting | ATS panel (6a) at the captioned K = 8, M = 4. WTS panel (6c) at K = 12, M = 2 (N = 6), 6 dB | The caption says K = 8, M = 4, but N = 2 allows only L = 1 while the plot runs L = 1..5. N = 6 with full search feasible means K = 12, M = 2 (the setting of Figs. 5b and 7) |
| A6 | Fig. 6c headline +46.2 % | Compared at L = 3, with all L shown | The paper does not say which L (or average) the 46.2 % refers to |
| A7 | Fig. 3 setting | Greedy TSS (eq. 17) and greedy RSS (eq. 18) at K = 12, M = 3 | K and M are not stated; at N = 2 the two greedy rules pick the same partition, so N > 2 is needed; M = 3 is the paper's preferred subpacket size |
| A8 | Message counts | 100 messages per point, except GA-vs-G (Fig. 10b) and the P sweep (Fig. 11) with 30, and K = 50, 60 in Fig. 12 with 20 | CPU-only budget: these points need 10^4–10^6 MPNet forwards per message |
| A9 | Random PA | One uniformly random partition per message, fixed across SNRs | Paper: "G is generated as a random partition" |
| A10 | Seeds | SemPA-Look / GA / Random PA seed for message i: 42·1,000,003 + i | Reproducible per message, independent of algorithm order |

## Not reproduced

- Subword tokenization (Figs. 6b/6d, 7b, 8b, subword curves of 10–13).
- SemPA-GBeam, MCTS (Table II column), R-DeepSC / DeepSC / UTF-8 / Huffman (Table IV).
- Fig. 4 sampling ablations (with/without replacement, RSS-only, lookahead-only variants).
- AIGC / LPIPS (Figs. 14, 15), BLEURT / BERTScore (Fig. 16 lower panels), LLM post-correction (Tables V–VI with correction, Fig. 17), channel coding and CRC (Fig. 18).
- Wall-clock comparisons: the paper's complexity metric is encoder calls, which we report; our cache-prefetching runner makes per-algorithm runtimes meaningless.
