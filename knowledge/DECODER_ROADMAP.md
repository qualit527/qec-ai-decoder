# AI QEC Decoder Roadmap (April 2026)

**Audience**: (a) the AutoQEC Ideator LLM, which reads this verbatim each round, and (b) the 3-person project team using it for top-level scientific planning.
**Source base**: `knowledge/INDEX.md` (81 papers), `knowledge/bibliography.bib`, `knowledge/papers_md/*`.
**Convention**: inline citations use `[@bibkey]` where `bibkey` matches `bibliography.bib`.

---

## 1. Executive summary

Nine years of neural decoding (2017-2026) have converged on a short list of working recipes and a much longer list of dead ends. The Ideator should internalize four take-homes before proposing anything:

- **Surface code is solved at small distance.** Recurrent transformer with per-stabilizer tokens [@bausch2024alphaqubit; @deepmind2025alphaqubit2] is at or near optimal for d≤11 on circuit-level and real Sycamore data. Fully-convolutional 3D [@fullyconv3d2025] and Mamba/SSM [@mamba2025scalable] match accuracy at much lower compute. The wedge on surface code is **latency, scaling, and generality**, not raw LER.
- **qLDPC / BB codes are the live frontier.** GNN-on-Tanner-graph (Astra [@maan2025astra], HyperNQ [@hypernq2025], Efficient-Universal [@universal2025efficient]) and transformer [@ml2025bivariate] decoders now beat BP+OSD [@roffe2020decoding] on [[72,12,6]] and extrapolate to d≈25-34. This is the most defensible AutoQEC target.
- **Strong classical baselines must be beaten explicitly.** Relay-BP [@improvedbp2025], Collision-Clustering [@riverlane2024realtime], and Local-Clustering-Decoder [@microsoft2024lcd] are *already* real-time and accurate; a neural decoder that only matches PyMatching is not publishable.
- **Architecture search for QEC decoders is an open niche.** The only published NAS-like work is a small FFN hardware sweep [@overwater2022neural]. No NAS paper for transformer / GNN / SSM decoders exists as of Apr 2026 — this is AutoQEC's novelty wedge.

---

## 2. Architecture families — taxonomy and current status

For each family: (a) signature, (b) canonical keys, (c) noise × code × distance × latency sweet-spot, (d) building-block vocabulary, (e) AutoQEC priority for the 2-env MVP (`surface_d5_depol`, `bb72_depol`).

### 2.1 FFN / MLP
(a) Flatten syndrome → dense layers → logical-parity output.
(b) [@torlai2017neural; @krastanov2017deep; @varsamopoulos2017decoding; @varsamopoulos2018comparing].
(c) Excels: d≤5 code-capacity; tiny FLOPs. Fails: d≥7 circuit-level (parameter blow-up from flattened input), temporal syndrome history.
(d) Dense, GELU/ReLU, dropout, skip connection, 1-hot syndrome embedding.
(e) **Primary seed for MVP surface_d5** — cheap baseline; the Ideator must propose an FFN within the first three rounds so the later round's transformers have a comparison point.

### 2.2 CNN
(a) 2D/3D convolutions over spatially arranged syndromes.
(b) [@zhang2023convolutional; @ni2020neural; @fullyconv3d2025; @meinerz2022scalable].
(c) Excels: translation-invariant surface / toric, large distances, FPGA-friendly kernels. Fails: non-Euclidean codes (BB, color), long-range correlations.
(d) 2D conv, 3D conv (space×time), dilated conv, residual block, group norm, strided pool.
(e) **Primary seed for surface_d5** — [@fullyconv3d2025] is SOTA-competitive at much lower compute than AlphaQubit and is the obvious low-FLOPs rival.

### 2.3 RNN / LSTM
(a) One hidden state per QEC cycle; process syndrome-rounds as a temporal sequence.
(b) [@baireuther2018machine; @baireuther2019neural; @chamberland2018deep; @varsamopoulos2018comparing].
(c) Excels: streaming inference, mid-distance circuit-level. Fails: training difficulty at large d (vanishing gradient), long contexts.
(d) LSTM cell, GRU cell, teacher forcing, per-cycle loss, bidirectional wrap.
(e) **Secondary** — RNN alone is dominated by transformers; include only as a component (recurrent wrapper around transformer, as in AlphaQubit [@bausch2024alphaqubit]).

### 2.4 Transformer (AlphaQubit family)
(a) Per-stabilizer tokens + multi-head self-attention + recurrent state across QEC cycles.
(b) [@bausch2024alphaqubit; @deepmind2025alphaqubit2; @parallel2025selfcoord; @selfcoord2026parallel; @moevit2026; @ml2025bivariate; @neuraluqa2025; @scalpractical2026].
(c) Excels: surface d=3-11, color d≤9, Sycamore real data, BB [[72,12,6]] circuit-level. Fails: O(d⁴) compute scaling; latency without sliding window; memory on large d.
(d) Per-stabilizer token embedding, attention bias (2D geometry), dilated convolution interleave [@bausch2024alphaqubit], gated dense block, pool+readout head, sliding-window attention [@parallel2025selfcoord], MoE router [@moevit2026].
(e) **Primary seed for both MVP envs** — AlphaQubit recipe is the known-good template the Ideator should emit on round 1.

### 2.5 Mamba / SSM
(a) Selective state-space model replaces attention; O(d²) vs O(d⁴).
(b) [@mamba2025scalable].
(c) Excels: real-time surface code under decoder-induced noise (threshold 1.04% vs transformer's 0.97% in this regime). Fails: less established; single paper.
(d) Selective scan, SSM block, residual SSM, gating.
(e) **Primary seed for surface_d5 latency-Pareto** — if AutoQEC's goal is the low-FLOPs corner, this is the template.

### 2.6 GNN on Tanner graph
(a) Learned message-passing over the check-variable graph.
(b) [@lange2025data; @ninkovic2024decoding; @universal2025efficient; @maan2025astra; @hypernq2025; @learnbpqldpc2026; @lowlatgnn2026].
(c) Excels: qLDPC (BB, Gross), arbitrary-code generality, transfer across distances (Astra: trained at d=11/18, extrapolates to d=25/34). Fails: surface code (CNN/transformer cheaper at equal accuracy), higher-order stabilizer couplings (HyperNQ mitigates).
(d) Node embedding, check-node update, variable-node update, message MLP, gated aggregation, hypergraph layer [@hypernq2025], linear-attention fusion [@universal2025efficient], edge features (Pauli type, syndrome bit).
(e) **Primary seed for bb72_depol** — highest-leverage known architecture for qLDPC and the one published baselines struggle most against.

### 2.7 Neural Belief Propagation (deep-unfolded BP)
(a) Unroll K iterations of BP; learn per-edge / per-iteration scaling weights.
(b) [@nachmani2016learning; @liu2019neural; @nachmani2019learned; @nbp2022overcomplete; @quaternary2023nbp; @learnbpqldpc2026; @improvedbp2025].
(c) Excels: qLDPC with degeneracy-aware loss, hardware-friendly inference, strong inductive bias. Fails: trapping sets without overcomplete checks; limited modeling power vs GNN when BP is already near-optimal.
(d) Message-passing iteration, learned damping, per-edge weight, overcomplete check matrix, GF(4) Pauli-aware update, min-sum approximation, relay chaining [@improvedbp2025].
(e) **Primary seed for bb72_depol** — interpretable, hardware-friendly, and directly comparable to the BP+OSD baseline. A 5-iter unrolled Neural-BP is probably the cheapest hypothesis the Ideator can beat BP+OSD with.

### 2.8 Reinforcement Learning
(a) Agent outputs correction moves; reward = logical fidelity.
(b) [@andreasson2019quantum; @sweke2020reinforcement; @fitzek2020deep; @decodingrl2022; @rleadvqec2026; @rlgreedy2025].
(c) Excels: exploratory code discovery, small d. Fails: d≥9 ([@andreasson2019quantum] sub-MWPM at d=9); sample complexity.
(d) Q-network, policy net, reward shaping, action masking.
(e) **Skip for MVP as a primary hypothesis.** RL fine-tuning of a supervised decoder [@rleadvqec2026] is a viable post-MVP add-on. RL for *code discovery* [@nautrup2019optimizing; @discoveryrl2023; @lowweightrl2025] is orthogonal to decoding and out of scope.

### 2.9 Ensemble / hybrid / pre-decoder
(a) NN supplies priors or local patches; classical decoder resolves globally.
(b) [@meinerz2022scalable; @varbanov2024optimization; @microsoft2024lcd; @riverlane2024realtime; @pinball2025; @acc2025bposd].
(c) Excels: real-time deployment, leakage adaptation (LCD), easy drop-in. Fails: upper-bounded by the classical stage.
(d) Sliding-window CNN, per-edge prior head, union-find stage, cryogenic pre-decoder, local-syndrome preprocess, ensemble averaging.
(e) **Secondary** — `varbanov2024optimization`-style prior-learning is a trivially cheap Ideator hypothesis (learn priors for PyMatching / BP-OSD and call it a Pareto win at near-zero FLOPs).

### 2.10 Latency-tailored / FPGA-aware
(a) Architecture chosen for synthesis area × delay × accuracy Pareto.
(b) [@overwater2022neural; @lowlatgnn2026; @fpgaqldpc2025; @microsoft2024lcd; @pinball2025; @effsynd2021].
(c) Excels: sub-µs decode at d≤5. Fails: not a standalone decoder family — overlays the others.
(d) Quantized weights (3-7 bit), fixed-point ops, sparse conv, pipelined MAC, sliding-window stream.
(e) **Secondary** — AutoQEC's MVP uses FLOPs as latency proxy (spec §4.8). Add quantization as a post-MVP Pareto axis.

---

## 3. Target-code × noise-model × architecture fit

Rows: code family. Cols: noise model. Cells: recommended primary / secondary architecture family.

| Code ↓ / Noise → | Code-capacity | Phenomenological | Circuit depolarizing | Biased / Pauli-asymmetric | Leakage / non-Pauli | Coherent |
|---|---|---|---|---|---|---|
| Rotated surface | FFN / CNN [@torlai2017neural; @zhang2023convolutional] | CNN / RNN [@meinerz2022scalable] | **Transformer+recurrent** [@bausch2024alphaqubit]; Mamba [@mamba2025scalable]; FullyConv3D [@fullyconv3d2025] | Transformer w/ biased prior; symmetry-equivariant [@ohnishi2025symmetry] | Transformer w/ soft readouts & leakage flags [@deepmind2025alphaqubit2; @lru2025]; LCD [@microsoft2024lcd] | Under-explored; RNN + calibrated prior |
| Toric | CNN [@ni2020neural]; Torlai-BM [@torlai2017neural] | RL [@fitzek2020deep; @andreasson2019quantum] (d≤9) | CNN / Transformer | same as surface | hybrid pre-decoder | open |
| Color | MLP + color-matching [@minsurfcolor2023] | RNN [@baireuther2019neural] | Transformer [@deepmind2025alphaqubit2] (d≤9); Vibe-decoder-inspired [@colorvibe2025] | open | open | open |
| BB / Gross qLDPC | GNN [@maan2025astra] (thr ≈17%) | GNN / Neural-BP | **Transformer** [@ml2025bivariate]; **GNN** [@universal2025efficient; @hypernq2025]; Relay-BP baseline [@improvedbp2025] | GNN w/ edge features | open ([@heavyhex2026] for heavy-hex flavor) | open |
| Random qLDPC / lifted-product | BP [@panteleev2021asymptotically]; Neural-BP [@liu2019neural] | Neural-BP + overcomplete [@nbp2022overcomplete]; quaternary NBP [@quaternary2023nbp] | GNN [@ninkovic2024decoding]; Learning-BP [@learnbpqldpc2026] | Quaternary NBP | uncertainty-aware NBP [@uncertain2025neural] | open |
| Bicycle / heavy-hex tailored | Tailored matching + NN prior | — | Transformer / GNN | [@heavyhex2026] | [@sidechannel2026] | open |

**Reading rules for the Ideator:**
1. If the env's code is surface/toric and noise is circuit-level depolarizing → transformer-recurrent is the primary seed; FullyConv3D and Mamba are the latency alternatives.
2. If the code is qLDPC (BB, lifted-product, random) → GNN or Neural-BP is primary; BP+OSD is the must-beat baseline.
3. If the noise includes leakage, soft readouts, or non-Pauli terms → AlphaQubit-2's soft-flag inputs are the only validated recipe; otherwise this is open research.

---

## 4. Evaluation & benchmark conventions

**Metrics.**
- **Logical error per round (LER)**: primary accuracy axis. Estimate from `log F(n) = log F₀ + n log(1-2ε)` regression over QEC cycles [@mamba2025scalable].
- **Threshold `p_th`**: physical error rate at which LER becomes distance-independent. Required for comparing across distances.
- **Suppression factor Λ**: `LER(d)/LER(d+2)`. Willow [@google2024below] reports Λ = 2.14 at d=7; this is the hardware ground-truth to calibrate against.
- **FLOPs**: hardware-agnostic latency proxy. Measured via `fvcore` / `ptflops`. This is AutoQEC's Pareto x-axis (spec §4.8).
- **Wall-clock**: reference only; host-dependent.
- **Parameters**: sum over `torch.nn.Module.parameters`.

**Benchmark datasets.**
- **Synthetic via Stim + sinter**: circuit-level noise sampling. The near-universal training substrate.
- **Sycamore / Willow experimental data**: [@google2024below; @bausch2024alphaqubit; @deepmind2025alphaqubit2]. Not needed for MVP.
- **IBM heavy-hex + BB shots**: [@heavyhex2026; @bravyi2024high]. Post-MVP if stretch env activated.
- **Fair-baseline suites**: [@bbfair2026] for BB codes — AutoQEC should adopt this test protocol to publish comparable numbers.

**Shot / CI conventions.** Most papers use 10⁶-10⁸ shots for training. For LER estimation: Wilson / Clopper-Pearson 95% CIs at ≥10⁴ shots per p-point; ≥10⁵ when reporting sub-10⁻⁴ rates. AutoQEC's `/verify-decoder` uses ≥5000 holdout shots (spec §4.6) — tighten to ≥10⁴ before publishing any Pareto claim.

**Adopt for AutoQEC:** (a) report LER and FLOPs together, never one alone; (b) quote BP+OSD and PyMatching (or Relay-BP [@improvedbp2025] / Collision-Clustering [@riverlane2024realtime] / LCD [@microsoft2024lcd]) baselines on every env; (c) hold out ≥5000 shots on unseen seeds for verification; (d) adopt the [@bbfair2026] baseline protocol for the BB env.

---

## 5. Architectural building-blocks library

Grouped by role. Each block: 1-line description → origin bib-key. **This is the DSL vocabulary the Ideator pulls from** (spec §6.1).

### 5.1 Encoder / frontend (syndrome → feature)
1. **One-hot syndrome flatten** — naïve input embedding. [@torlai2017neural].
2. **Per-stabilizer token embedding** — one learnable token per stabilizer, concatenated with {X,Z} type & 2D position. [@bausch2024alphaqubit].
3. **2D spatial scatter** — reshape to (d+1)×(d+1) grid with learned padding. [@bausch2024alphaqubit; @zhang2023convolutional].
4. **3D spatio-temporal tensor** — syndromes stacked along time axis. [@fullyconv3d2025].
5. **Tanner-graph node/edge encoding** — syndrome on checks, zeros on variables. [@maan2025astra; @ninkovic2024decoding].
6. **Hyperedge encoding** — captures ≥3-way stabilizer couplings. [@hypernq2025].
7. **Detector-graph node encoding** — nodes = detection events, edges = DEM adjacency. [@lange2025data].
8. **Soft-readout / leakage flag input** — analog measurement signal + leakage bit. [@deepmind2025alphaqubit2; @lru2025].
9. **Full-correlation input** — full stabilizer correlation tensor for side-channel modeling. [@sidechannel2026].

### 5.2 Core computation
10. **Dense / MLP block** — baseline. [@torlai2017neural; @krastanov2017deep].
11. **LSTM / GRU per cycle** — temporal recurrence. [@baireuther2018machine; @baireuther2019neural].
12. **2D conv block** — translation invariance. [@zhang2023convolutional].
13. **3D conv (space×time) block** — no recurrence needed. [@fullyconv3d2025].
14. **Dilated conv** — enlarged receptive field at fixed FLOPs. [@bausch2024alphaqubit].
15. **Multi-head self-attention** — per-stabilizer interactions. [@bausch2024alphaqubit].
16. **Attention bias from geometry** — inject 2D lattice priors into attention scores. [@bausch2024alphaqubit].
17. **Linear attention** — O(n) attention variant. [@universal2025efficient].
18. **Gated dense block (GLU)** — post-attention FFN. [@bausch2024alphaqubit].
19. **SSM / Mamba selective scan** — O(n) sequence mixing. [@mamba2025scalable].
20. **GNN message-passing (check↔variable)** — BP-like learned updates. [@maan2025astra; @ninkovic2024decoding].
21. **Hypergraph convolution** — higher-order messages. [@hypernq2025].
22. **Unrolled BP iteration** — learn per-edge scaling per unroll step. [@liu2019neural; @nachmani2016learning].
23. **Relay-BP block** — chained DMem-BP for trapping-set escape. [@improvedbp2025].
24. **Quaternary (GF(4)) update** — Pauli-aware NBP. [@quaternary2023nbp].
25. **Symmetry-equivariant layer** — respect surface-code group. [@ohnishi2025symmetry].
26. **Recurrent wrapper** — carry a hidden state across QEC cycles. [@bausch2024alphaqubit].
27. **Sliding-window attention / state** — real-time infinite-stream decode. [@parallel2025selfcoord; @selfcoord2026parallel].
28. **Mixture-of-Experts router** — conditional compute. [@moevit2026].
29. **RL fine-tune head** — post-supervised policy gradient. [@rleadvqec2026].
30. **Distributed / RG-style sub-decoder** — decompose + stitch. [@varsamopoulos2019distributed; @meinerz2022scalable].
31. **Prior-only MLP** — output per-edge probability for classical post-decoder. [@varbanov2024optimization].

### 5.3 Decoder head / readout
32. **Per-logical-observable sigmoid** — default FT-memory readout. [@bausch2024alphaqubit].
33. **Pooling + dense** — stabilizer-token aggregation. [@bausch2024alphaqubit].
34. **Per-qubit correction softmax** — full Pauli recovery. [@krastanov2017deep].
35. **Priors to classical decoder** — hybrid output. [@varbanov2024optimization].
36. **Beam-search post-proc** — re-rank top-k recovery candidates. [@beamsearch2025qldpc].
37. **OSD fallback** — classical fallback when NN confidence low. [@roffe2020decoding].
38. **Correction-set clustering** — union-find over local predictions. [@microsoft2024lcd].

### 5.4 Regularization & training
39. **Degeneracy-aware loss** — marginalize over equivalent recoveries. [@liu2019neural; @maan2025astra].
40. **Symbol-wise BCE per logical observable** — standard. [@bausch2024alphaqubit].
41. **Multi-p curriculum** — train over range of physical p. [@bausch2024alphaqubit; @mamba2025scalable].
42. **Uncertainty calibration head** — OOD-noise generalization. [@uncertain2025neural].
43. **Teacher forcing across cycles** — for recurrent training. [@baireuther2019neural].
44. **Transfer learning from smaller d** — initialize d=11 from d=5 weights. [@maan2025astra].
45. **Real-data fine-tune** — Sycamore data after synthetic pretrain. [@bausch2024alphaqubit].
46. **Pseudo-threshold sampling** — concentrate training shots near p_th.
47. **Label noise for robustness** — spec-style for stretch envs.

### 5.5 Post-processing
48. **Sliding-window stitching** — glue local predictions. [@meinerz2022scalable; @parallel2025selfcoord].
49. **Cryogenic pre-decoder cascade** — cheap NN pre-filter. [@pinball2025].
50. **Local-syndrome preprocess** — reduce BP input size. [@acc2025bposd].
51. **Color-code-assisted surface decode** — cross-code decoding. [@minsurfcolor2023].
52. **Correlated cross-gate decode** — algorithm-level decoding. [@corrdec2024].

Total: 52 blocks. Expand via `/add-decoder-template` contributions.

---

## 6. Known dead ends

Each entry is an approach explicitly shown inferior by a published paper. The Ideator must not re-propose these unsuffixed with a clear mitigation.

- **Pure RL decoding at d ≥ 9** — sub-MWPM at large p [@andreasson2019quantum]; sample complexity blocks scaling. RL is usable for *code discovery* [@nautrup2019optimizing; @discoveryrl2023] or as a *fine-tune* stage on a supervised decoder [@rleadvqec2026], not as the primary inference path.
- **Flattened-syndrome FFN at d ≥ 7** — parameters scale super-linearly in d and exceed RNN/CNN at equal accuracy [@varsamopoulos2018comparing]; MVP should cap FFN at d=5.
- **Boltzmann-machine / energy-based decoders** — [@torlai2017neural] was historical; all follow-ups switched to discriminative nets for good reason.
- **Pure-attention (no inductive bias) on qLDPC at circuit level** — published transformer attempts on BB without Tanner-graph structure required large models; GNN / linear-attention fusions [@universal2025efficient] dominate on the same task.
- **BP without degeneracy-aware loss for qLDPC** — [@liu2019neural] showed the loss is essential; NBP otherwise gets stuck on the "wrong" coset.
- **Single-iteration BP for qLDPC** — trapping sets kill performance; overcomplete checks [@nbp2022overcomplete] or relay chaining [@improvedbp2025] are required.
- **Feed-forward decoders for real-time streaming on large d** — latency blows up because the full history must be encoded per round; use RNN wrapper or Mamba [@mamba2025scalable] or sliding-window [@parallel2025selfcoord].
- **OSD post-processing when GPU is the target** — OSD is a CPU-heavy serial step; Astra [@maan2025astra] explicitly drops it.
- **Decoders trained at a single p ignoring the curriculum** — generalization to nearby p is poor; multi-p training [@bausch2024alphaqubit] is cheap and mandatory.

---

## 7. Open research frontiers (Apr 2026)

Under-explored places where AutoQEC can plausibly land a novel result in one or two rounds.

- **Leakage-aware neural decoders beyond AlphaQubit-2.** Only [@deepmind2025alphaqubit2] exploits soft-readouts + leakage flags at production quality; [@sidechannel2026; @lru2025] hint at richer side-channel modeling. General leakage-aware NN decoders for BB / color are wide open.
- **Erasure / qubit-loss channels.** [@ailoss2026] is a first pass; very little published on NN decoders for mixed Pauli+erasure noise.
- **Generalization across p (OOD noise).** [@uncertain2025neural] is the only calibration paper. Most decoders refuse OOD — this is a concrete Ideator target (training on p=1e-2, eval on 5e-3 and 2e-2).
- **Symmetry-equivariant architectures.** [@ohnishi2025symmetry] is a lone surface-code paper. Equivariance for BB / color is unexplored.
- **Algorithm-level decoders for non-memory circuits.** [@corrdec2024; @neuraluqa2025] — decoding during logical gates (not just idle memory) is almost untouched.
- **Hardware-Pareto NAS.** [@overwater2022neural] is the only serious hardware-aware architecture sweep, and is FFN-only. AutoQEC's wedge.
- **MoE / conditional compute.** [@moevit2026] is the first QEC MoE; wide open for qLDPC.
- **Transfer learning across code families.** Astra [@maan2025astra] transfers across distances within a family; no published work transfers *across* families (surface → BB).
- **Neural decoder self-coordination across sliding windows.** [@parallel2025selfcoord; @selfcoord2026parallel] have introduced it; open questions about consistency under logical gates.

---

## 8. Concrete MVP seed hypotheses for AutoQEC

Given `surface_d5_depol` and `bb72_depol` reference envs, these are 10 round-1 to round-3 hypotheses the Ideator can emit verbatim. Each line is ≈1 sentence in the form "family / building blocks / env / expected Pareto position / must-beat baseline." Keys in brackets cite the origin template.

1. **surface_d5 / FFN / baseline.** 3-layer MLP over flattened syndrome, hidden 256, dropout 0.1. Expected: match PyMatching at low FLOPs, d=5 only. [@varsamopoulos2018comparing]
2. **surface_d5 / CNN / low-FLOPs Pareto.** 2D conv (3×3)×3 + residual on spatially-scattered syndromes, single per-cycle head. Expected: ≤1e6 FLOPs, match PyMatching at p=5e-3. [@zhang2023convolutional]
3. **surface_d5 / FullyConv3D / accuracy Pareto.** 3D (space×time) conv block ×4 with dilation, no recurrence. Expected: Pareto-win PyMatching at all p. [@fullyconv3d2025]
4. **surface_d5 / AlphaQubit-lite / accuracy target.** Per-stabilizer token embed + 2-layer transformer w/ 2D attention bias + recurrent state across rounds. Expected: match or beat PyMatching, 10× FLOPs of (2). [@bausch2024alphaqubit]
5. **surface_d5 / Mamba / latency Pareto.** Per-stabilizer SSM block ×3, O(d²) compute. Expected: Pareto-win (4) at equal accuracy. [@mamba2025scalable]
6. **surface_d5 / PriorsOnly / free-win.** Tiny MLP learns per-edge priors for PyMatching. Expected: strict Pareto win vs unweighted PyMatching at near-zero added FLOPs. [@varbanov2024optimization]
7. **bb72_depol / BP+OSD baseline wrap.** Non-neural reference. Expected: anchor the Pareto. [@roffe2020decoding]
8. **bb72_depol / Neural-BP 5-iter / cheap neural win.** Unrolled BP with per-edge learned scalars over the [[72,12,6]] Tanner graph, degeneracy-aware loss. Expected: Pareto-win BP+OSD on (FLOPs, LER). [@liu2019neural; @nbp2022overcomplete]
9. **bb72_depol / Tanner-GNN (Astra-lite) / accuracy win.** 4-layer GNN with check-variable message MLPs, transfer-init from smaller BB code if available, degeneracy-aware loss, *no OSD*. Expected: strict LER improvement over BP+OSD, Astra-template. [@maan2025astra]
10. **bb72_depol / Transformer-BB / aggressive baseline.** Per-check-node tokens, 3-layer transformer w/ learned positional encoding from Tanner adjacency, recurrent state across rounds. Expected: strong LER but higher FLOPs than (9). [@ml2025bivariate]

The Ideator should issue 1-6 on surface within rounds 1-6, and 7-10 on BB within rounds 1-6. Rounds 7+ become *refinements*: Neural-BP variants (overcomplete [@nbp2022overcomplete], quaternary [@quaternary2023nbp], relay [@improvedbp2025]), hypergraph extensions [@hypernq2025], MoE routing [@moevit2026], symmetry equivariance [@ohnishi2025symmetry], uncertainty calibration [@uncertain2025neural], and hardware-aware quantization sweeps à la [@overwater2022neural].

---

## 9. Top-level strategic implications (for the human team)

**Does the 2-env MVP choice cover enough design space?** Broadly yes, but with one caveat. `surface_d5_depol` covers the spatial/temporal transformer-CNN-SSM family comprehensively, since surface code is the canonical proving ground and distance-5 is small enough to make the full Pareto (FFN → transformer) tractable on one GPU. `bb72_depol` is the right qLDPC choice — it hits the current research frontier, BP+OSD is a well-defined baseline, and GNN/Neural-BP have published head-to-head wins on exactly this code. **Missing dimension: the two MVP envs are both depolarizing.** The whole "leakage / non-Pauli" wedge from the original PDF pitch is deferred to the stretch env. Recommendation: keep MVP as specified for demo simplicity, but plan the `chip_leakage` stretch env to activate within ~4 weeks of MVP green-light. Without it the novelty claim rests only on "first NAS for QEC decoders," not on "decoders classical methods can't produce."

**Literature findings that should reshape the spec.** Two. (a) **Relay-BP [@improvedbp2025] is the true must-beat qLDPC baseline, not BP+OSD**: it is real-time on CPU and matches ML accuracy in several regimes. The BB env baseline list in spec §3.1 should add `relay_bp`. (b) **AlphaQubit-2 [@deepmind2025alphaqubit2] makes `surface_d5_depol` too easy a win** — the Ideator will converge to an AlphaQubit-lite and then have nowhere to go. Raising the accuracy target or swapping in d=7 + sliding-window evaluation à la [@parallel2025selfcoord] is worth considering before the demo.

**Novelty wedge.** The most defensible contribution is the combination of **(i) NAS for QEC neural decoders** (a literal open niche per [@overwater2022neural] being the only prior work, and [@nasqae2025sep; @nasqae2025nov] covering only quantum *autoencoders*), **(ii) the AutoQEC agent-driven loop** (no prior AI-Scientist-class work targets QEC decoders), and **(iii) multi-env coverage with verified Pareto fronts**. Attack (i) hardest: it is the one sentence a reviewer will latch onto. The GNN-for-BB result (hypothesis 9 above) is the most likely single publishable finding if the loop actually works.
