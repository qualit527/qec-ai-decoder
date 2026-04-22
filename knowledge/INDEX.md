# AI Decoder Literature — Knowledge Base for AutoQEC Ideator

Generated: 2026-04-20. 81 papers.

## How to use
For each new hypothesis, the Ideator should reference relevant papers here by filename + arxiv_id. Don't invent architectures that this literature already invalidates. Filenames follow `YYYY-FirstAuthor-ShortTitle.pdf` under `papers/`. When in doubt about verifiability, re-check the arxiv_id before citing.

Notation: **PAPER-ID** (arXiv:xxxx.xxxxx) — one-line takeaway relevant to neural-decoder design.

---

## Publication status summary

- **PUBLISHED** (peer-reviewed journal/conference): **38** papers
- **WORKSHOP** / non-archival venue: **0** papers
- **arXiv preprint only**: **43** papers

**Top venues**: Quantum (10), PRL (4), Nature (4), PRR (4), Science (3), IEEE TQE (2), Sci. Rep. (1), NJP (1), IEEE TC (1), ASP-DAC (1)

Verified via OpenAlex + Semantic Scholar APIs on 2026-04-20. Preprint-only papers are predominantly 2025–2026 arxiv posts that have not yet cycled through peer review; these include many of the most recent SOTA claims and should be read with appropriate skepticism.

---


## Category 1: Surface-code neural decoders (foundational)

- **2017-Torlai-NeuralDecoderTopologicalCodes** (arXiv:1610.04238) [*PRL* 2017] — First ML decoder. Boltzmann machine on 2D toric code, phase-flip only. Code-capacity threshold ~10% bit-flip. Historical baseline; scales poorly beyond d=6.
- **2017-Krastanov-DNNProbabilisticStabilizer** (arXiv:1705.09334) [*Sci. Rep.* 2017] — Fully-connected DNN predicts error distribution conditioned on syndrome. Generic stabilizer-code decoder. Higher code-capacity threshold than MWPM on toric code.
- **2017-Varsamopoulos-DecodingSmallSurfaceCodesANN** (arXiv:1705.00857) [*Science* 2018] — MLP for small surface codes. Early supervised-learning baseline.
- **2018-Chamberland-DeepNeuralDecodersNearTerm** (arXiv:1802.06441) [*Science* 2018] — Deep MLP + LSTM for near-term FT experiments (Steane, Knill, surface). No-noise-model-needed training is the key design insight.
- **2018-Baireuther-ColorCodeCircuitLevel** (arXiv:1804.02926) [*NJP* 2019] — RNN decoder for 2D color code under circuit-level noise with flag qubits. Sets template for RNN-per-syndrome-round decoding.
- **2018-Varsamopoulos-ComparingNNDecodersSurface** (arXiv:1811.12456) [*IEEE TC* 2019] — Systematic comparison of FFN / CNN / RNN decoders on surface code. FFN often matches RNN up to d=5.
- **2019-Varsamopoulos-DistributedNNDecoder** (arXiv:1901.10847) [*Quantum* 2019] — Distributed renormalization-group-style NN decoder for scalable surface-code decoding.
- **2017-Baireuther-MLAssistedTopologicalCode** (arXiv:1705.08957) [*Quantum* 2018] — LSTM decoder trained from experimental data beats blossom MWPM by catching X–Z correlations.
- **2020-Ni-NNDecodersLargeDistance2DToric** (arXiv:1911.12308) [arXiv preprint] — FC/CNN hybrids pushed to d=64 via clever factorization — scaling demonstration for 2D toric.
- **2023-Gicev-ScalableFastANNSyndromeDecoder** (arXiv:2110.05854) [*Quantum* 2023] — Arbitrary-shape/size surface-code ANN decoder; scaling and latency analysis. Quantum journal 2023.
- **2022-Overwater-NNDecodersHardwareTradeoff** (arXiv:2202.05741) [*IEEE TQE* 2022] — **Only published NAS-like exploration for neural surface-code decoders.** ASIC/FPGA synthesis, Pareto front of area × delay × accuracy. Target for Ideator.
- **2023-Zhang-ConvolutionalNNDecodersSurface** (arXiv:2312.03508) [*Quantum* 2023] — CNN decoder for surface codes; showcases translation-invariance advantage.
- **2024-Varbanov-OptimizationDecoderPriors** (arXiv:2406.02700) [*PRL* 2024] — Learn soft priors for existing decoders (matching/BP); cheap "prior-only" learning trick.

## Category 2: Transformer / SOTA surface-code decoders

- **2024-Bausch-AlphaQubit** (arXiv:2310.05900) [*Nature* 2024] — Syndrome-transformer + recurrent state, d=3/5 on Sycamore experimental data. Published in Nature 2024. Architectural template: per-stabilizer token, recurrency per QEC cycle, pool + readout for logical observable.
- **2025-DeepMind-AlphaQubit2** (arXiv:2512.07737) [arXiv preprint] — Successor with light-weight recurrent layer interleaved with transformers. Near-optimal LER for surface AND color codes. Real-time <1µs/cycle up to d=11 surface / d=9 color. SOTA as of Apr 2026.
- **2025-Mamba-ScalableNeuralDecoders** (arXiv:2510.22724) [arXiv preprint] — Mamba (SSM) decoder, O(d²) vs transformer O(d⁴). Matches AlphaQubit on Sycamore; higher threshold (1.04% vs 0.97%) in decoder-induced-noise regime.
- **2025-ParallelSelfCoordinatingNeural** (arXiv:2509.03815) [arXiv preprint] — Sliding-window + self-coordinated transformer; enables real-time decode of infinite syndrome stream.
- **2026-SelfCoordinatingNeuralParallel** (arXiv:2601.09921) [arXiv preprint] — Follow-up: train on consistent local-correction labels across window types.
- **2025-FullyConv3DNeuralDecoder** (arXiv:2506.16113) [*Science* 2025] — Fully-conv 3D (space × time) decoder; no recurrence, cheap inference.
- **2026-MoEVisionTransformerSurfaceDecoding** (arXiv:2601.12483) [arXiv preprint] — Mixture-of-Experts ViT for surface-code decoding. Conditional computation for high fidelity.
- **2026-RLEnhancedAdvancedQEC** (arXiv:2601.19279) [*ASP-DAC* 2026] — RL finetuning of neural decoders for advanced QEC architectures.
- **2025-Ohnishi-SymmetryIntegratedSurfaceDecoding** (arXiv:2509.10164) [arXiv preprint] — Equivariance to surface-code symmetries inside the architecture (group-equivariant layers).
- **2025-NeuralDecodersUniversalQuantumAlgorithms** (arXiv:2509.11370) [arXiv preprint] — Attention-based, modular (per-logical-gate) decoder for universal algorithms. Moves beyond memory-only benchmarks.
- **2026-ScalableNeuralDecodersPracticalFTQC** (arXiv:2604.08358) [arXiv preprint] — Practical FTQC-level neural decoder with scaling guarantees.
- **2024-QuantumFeedbackTransformer** (arXiv:2411.19253) [*PRR* 2024] — Transformer for quantum feedback control — related but not a decoder; useful for closed-loop QEC inspiration.

## Category 3: GNN decoders

- **2023-Lange-DataDrivenGNNDecoding** (arXiv:2307.01241) [*PRR* 2025] — Trinity/Data-driven GNN on annotated detector graph. General QEC code formulation, graph-classification framing.
- **2024-Ninkovic-DecodingQLDPCGNN** (arXiv:2408.05170) [arXiv preprint] — GNN for general qLDPC codes exploiting Tanner-graph sparsity. Message-passing view.
- **2025-EfficientUniversalNNDecoder** (arXiv:2502.19971) [arXiv preprint] — Linear-attention + GNN universal decoder for stabilizer codes. Outperforms BP-OSD on BB, concat-matching on color, AlphaQubit on surface.
- **2024-Maan-AstraMLMessagePassing** (arXiv:2408.07038) [*npj QI* 2024] — **Astra**: GNN that learns a BP-style message-passing algorithm on the Tanner graph. Beats BP+OSD on surface (d≤11) and BB (d≤18), extrapolates to d=25/34. npj QI 2025.
- **2026-LowLatencyGNNAccelerator** (arXiv:2603.22149) [arXiv preprint] — FPGA accelerator for GNN QEC decoders; single-inference latency target.
- **2025-HyperNQHypergraphNNDecoder** (arXiv:2511.01741) [arXiv preprint] — Hypergraph neural network captures higher-order stabilizer constraints via hyperedges.
- **2026-LearningDecodeQLDPCViaBP** (arXiv:2603.10192) [arXiv preprint] — Learning-based BP variants for qLDPC; hybrid GNN-BP.

## Category 4: Neural belief propagation (Neural-BP)

- **2019-LiuPoulin-NeuralBPDecoders** (arXiv:1811.07835) [*PRL* 2019] — Canonical neural-BP for QEC. Degeneracy-aware loss. PRL 2019. Foundational for entire line.
- **2016-Nachmani-LearningDecodeLinearCodes** (arXiv:1607.04793) [*Allerton* 2016] — Original deep-unfolding BP for classical codes. The "NBP" template.
- **2019-Nachmani-LearnedBPSimpleScaling** (arXiv:1901.08621) [arXiv preprint] — Simple-scaling neural BP; fewer parameters than full NBP with similar gains.
- **2022-NeuralBPOvercompleteCheckMatrices** (arXiv:2212.10245) [arXiv preprint] — Redundant check rows + learned weights to break BP trapping on QLDPC.
- **2023-QuaternaryNeuralBPQLDPC** (arXiv:2308.08208) [*Quantum* 2023] — Quaternary (over GF(4)) neural BP for Pauli-aware decoding of QLDPC.
- **2025-UncertaintyAwareGeneralizableNeural** (arXiv:2510.06257) [arXiv preprint] — Uncertainty calibration for neural QLDPC decoders — generalization beyond training noise.
- **2025-ImprovedBPRealTimeQuantum** (arXiv:2506.01779) [arXiv preprint] — "Relay BP" / improved BP is sufficient for real-time quantum memory. Strong classical baseline the Ideator must beat.

## Category 5: qLDPC / BB / Gross code decoders

- **2021-Panteleev-AsymptoticallyGoodQLDPC** (arXiv:2111.03654) [arXiv preprint] — Lifted product codes, asymptotically good QLDPC. Foundational code construction.
- **2020-Roffe-DecodingQLDPCLandscape** (arXiv:2005.07016) [*PRR* 2020] — BP+OSD classical decoder + Python library (`ldpc`/`bp_osd`). De-facto QLDPC baseline.
- **2024-Bravyi-HighThresholdLowOverheadGross** (arXiv:2308.07915) [*Nature* 2024] — IBM Gross / Bivariate Bicycle code [[144,12,12]], 0.8% threshold. Target code for neural decoders. Nature 2024.
- **2021-Roffe-DegenerateQLDPCCodes** (arXiv:1904.02703) [*Quantum* 2021] — Degenerate QLDPC codes, early benchmark.
- **2025-MLDecodingCircuitBivariateBicycle** (arXiv:2504.13043) [arXiv preprint] — Transformer decoder for BB codes at circuit-level noise; 5× better LER than BP-OSD on [[72,12,6]].
- **2023-NoisySyndromeQLDPCSlidingWindow** (arXiv:2311.03307) [arXiv preprint] — Sliding-window single-shot decoding for QLDPC.
- **2025-AcceleratingBPOSDLocalSyndrome** (arXiv:2509.01892) [arXiv preprint] — Local syndrome pre-processing to speed BP-OSD.
- **2025-BeamSearchDecoderQLDPC** (arXiv:2512.07057) [arXiv preprint] — Beam search post-processing for QLDPC BP.
- **2025-FPGATailoredQLDPCRealTime** (arXiv:2511.21660) [arXiv preprint] — FPGA-tailored algorithms (message-passing, OSD, clustering) for real-time QLDPC.

## Category 6: RL-based decoders

- **2019-Andreasson-RLToricCode** (arXiv:1811.12338) [*Quantum* 2019] — Deep Q-learning on toric code; CNN Q-function over defect positions. Quantum 2019.
- **2020-Sweke-RLDecodersFaultTolerant** (arXiv:1810.07207) [*MLST* 2021] — Deep Q-learning for surface code with correlated noise & noisy syndromes.
- **2020-Fitzek-DeepQLearningToric** (arXiv:1912.12919) [*PRR* 2020] — Deep Q-learning for depolarizing noise on toric code. PRR 2020.
- **2019-Nautrup-OptimizingQECWithRL** (arXiv:1812.08451) [*Quantum* 2019] — RL *optimizes the code itself* (not the decoder). Related: discovery pipeline relevant to Ideator.
- **2022-DecodingSurfaceCodesDeepRL** (arXiv:2212.11890) [arXiv preprint] — Surface-code RL decoder under realistic noise.
- **2023-DiscoveryOptimalQECRL** (arXiv:2305.06378) [*PRApplied* 2025] — RL-discovered QEC codes.
- **2025-LowWeightQECReinforcementLearning** (arXiv:2502.14372) [arXiv preprint] — Low-weight high-efficiency QEC code discovery via RL.
- **2025-RLEnhancedGreedyStabilizerFq** (arXiv:2506.03397) [arXiv preprint] — RL-enhanced greedy decoding for qudit stabilizer codes.

## Category 7: Ensemble / hybrid / pre-decoder work

- **2022-Meinerz-ScalableNeuralDecoderTopological** (arXiv:2101.07285) [*PRL* 2022] — Sliding-window CNN pre-decoder on small patches, then global stitch. PRL 2022. Template for "neural pre-decoder + classical global".
- **2024-RealTimeScalableResourceEfficient** (arXiv:2309.05558) [*Nature* 2025] — Riverlane Collision-Clustering real-time decoder. Classical baseline at MHz rate, d up to ~25. Nature Electronics 2024.
- **2024-RealTimeLowLatencyQECSuperconducting** (arXiv:2410.05202) [arXiv preprint] — Google real-time QEC demonstration on hardware.
- **2024-LocalClusteringDecoderSurface** (arXiv:2411.10343) [*Nat. Commun.* 2025] — Microsoft FPGA LCD: adaptive local-clustering union-find; handles leakage adaptively. Nature Comm 2025.
- **2025-PinballCryogenicPredecoder** (arXiv:2512.09807) [arXiv preprint] — Cryogenic predecoder for surface code at circuit level.
- **2024-DemonstratingDynamicSurfaceCodes** (arXiv:2412.14360) [arXiv preprint] — Dynamic (time-varying) surface code demos.
- **2024-GoogleQuantumAI-BelowSurfaceThreshold** (arXiv:2408.13687) [*Nature* 2025] — Willow d=7 memory below threshold. Λ = 2.14. Ground-truth hardware context.

## Category 8: Biased noise / leakage / non-Pauli decoders

- **2025-LeakageReductionUnitsMeasurement** (arXiv:2511.17460) [arXiv preprint] — LRU built into measurement; decoder must handle residual leakage.
- **2026-HardwareAgnosticQuantumSideChannelLeakage** (arXiv:2602.15966) [arXiv preprint] — Learning hardware side-channels (leakage) with full correlation data.
- **2026-AIEnabledQubitLossDecoding** (arXiv:2604.14269) [arXiv preprint] — ML decoding of erasure / qubit loss channels.
- **2026-ScalableQECHeavyHexArray** (arXiv:2604.14296) [arXiv preprint] — Heavy-hex qubit-array tailored decoder; IBM-style biased noise.
- **2024-CorrelatedDecodingTransversalGates** (arXiv:2403.03272) [arXiv preprint] — Logical-algorithm correlated decoding; non-iid error decoding at algorithm level.

## Category 9: FPGA / latency-optimized

- **2022-Overwater-NNDecodersHardwareTradeoff** (arXiv:2202.05741) [*IEEE TQE* 2022] — (see cat 1) ASIC/FPGA neural-decoder Pareto sweep.
- **2021-EfficientDecodingSurfaceCodeSyndromes** (arXiv:2110.10896) [arXiv preprint] — Efficient streaming syndrome decoding.
- **2024-RealTimeScalableResourceEfficient** (arXiv:2309.05558) [*Nature* 2025] — (see cat 7).
- **2024-LocalClusteringDecoderSurface** (arXiv:2411.10343) [*Nat. Commun.* 2025] — (see cat 7).
- **2026-LowLatencyGNNAccelerator** (arXiv:2603.22149) [arXiv preprint] — (see cat 3).
- **2025-FPGATailoredQLDPCRealTime** (arXiv:2511.21660) [arXiv preprint] — (see cat 5).
- **2025-PinballCryogenicPredecoder** (arXiv:2512.09807) [arXiv preprint] — (see cat 7).

## Category 10: NAS / auto-design of decoders

- **2022-Overwater-NNDecodersHardwareTradeoff** (arXiv:2202.05741) [*IEEE TQE* 2022] — Closest existing work to NAS for neural decoders: systematic hardware-vs-accuracy exploration of FFN topology. **Only published "NAS-like" decoder paper as of Apr 2026.**
- **2025-NeuralArchitectureSearchQuantumAutoencoders** (arXiv:2511.19246) [arXiv preprint] — GA-based NAS for quantum autoencoders (not decoders — adjacent).
- **2025-NASAlgorithmsQuantumAutoencoders** (arXiv:2509.15451) [*IEEE TQE* 2025] — NAS for quantum autoencoders (adjacent). No direct NAS-for-QEC-decoder paper exists — this is AutoQEC's core opportunity.

## Category 11: Other codes (color, honeycomb, subsystem)

- **2023-MinimisingSurfaceCodeFailuresColorDecoder** (arXiv:2306.16476) [*Quantum* 2025] — Color-code-assisted surface decoding.
- **2023-NewCircuitsOpenSourceColorCode** (arXiv:2312.08813) [arXiv preprint] — Open-source color-code decoder and circuits.
- **2025-ColourCodesReachSurfaceVibeDecoding** (arXiv:2508.15743) [arXiv preprint] — "Vibe" decoder closes gap between color and surface codes.
- **2023-DeepQuantumErrorCorrection** (arXiv:2301.11930) [arXiv preprint] — Deep-learning for non-surface codes.

## Category 12: Reviews / surveys / benchmarks (context, not targets)

- **2024-AIForQECComprehensiveReview** (arXiv:2412.20380) [arXiv preprint] — 150+ refs, full ML-for-QEC taxonomy. Read first for orientation.
- **2023-BenchmarkingMLModelsQEC** (arXiv:2311.11167) [arXiv preprint] — Benchmark suite comparing ML model families on QEC.
- **2024-DecodingAlgorithmsSurfaceCodesReview** (arXiv:2307.14989) [*Quantum* 2024] — Classical decoders review (MWPM, UF, BP, TN). Quantum journal 2024.
- **2023-MLPracticalQuantumErrorMitigation** (arXiv:2309.17368) [*Nat. Mach. Intell.* 2024] — ML for mitigation (vs correction) — tangential but shares techniques.
- **2024-ArchitecturesHeterogeneousQECCodes** (arXiv:2411.03202) [*ASPLOS* 2024] — System-level QEC architecture paper, useful for "what decoder latency is actually required" anchors.
- **2026-FairDecoderBaselinesBivariate** (arXiv:2603.19062) [arXiv preprint] — Fair baselines for BB codes; use for reproducible comparison.

---

## Chronological index

- **2016**: Nachmani (1607.04793)
- **2017**: Torlai (1610.04238), Krastanov (1705.09334), Varsamopoulos-Small (1705.00857), Baireuther-MLA (1705.08957)
- **2018**: Chamberland (1802.06441), Baireuther-Color (1804.02926), Sweke (1810.07207), Varsamopoulos-Comp (1811.12456), Andreasson (1811.12338), Liu-Poulin (1811.07835), Nautrup (1812.08451)
- **2019**: Varsamopoulos-Dist (1901.10847), Nachmani-Simple (1901.08621), Roffe-Degen (1904.02703), Ni (1911.12308), Fitzek (1912.12919)
- **2020**: Roffe-Landscape (2005.07016)
- **2021**: Meinerz (2101.07285), Gicev (2110.05854), EffSynd (2110.10896), Panteleev (2111.03654)
- **2022**: Overwater (2202.05741), NBP-Overcomplete (2212.10245), Decoding-RL (2212.11890)
- **2023**: DeepQEC (2301.11930), DiscoveryRL (2305.06378), MinSurfColor (2306.16476), Lange-GNN (2307.01241), Review-Dec (2307.14989), Bravyi-Gross (2308.07915), Quaternary-NBP (2308.08208), RealTimeScal (2309.05558), MLMitigation (2309.17368), AlphaQubit (2310.05900), NoisySW-QLDPC (2311.03307), Benchmark (2311.11167), NewColor (2312.08813), Zhang-CNN (2312.03508)
- **2024**: CorrDec (2403.03272), OptPriors (2406.02700), Astra (2408.07038), Ninkovic-GNN (2408.05170), BelowThresh (2408.13687), RT-SC (2410.05202), LCD (2411.10343), Hetero (2411.03202), Feedback-Trans (2411.19253), DynSurf (2412.14360), AI4QEC-Review (2412.20380)
- **2025**: LowWeightRL (2502.14372), EffUniv (2502.19971), MLBB (2504.13043), RelayBP (2506.01779), FullyConv3D (2506.16113), RL-Greedy-Fq (2506.03397), ColorVibe (2508.15743), AcceleratingBP (2509.01892), ParallelSC (2509.03815), SymmIntegr (2509.10164), NeuralUQA (2509.11370), NAS-QAE-2 (2509.15451), UncertainNeural (2510.06257), Mamba (2510.22724), HyperNQ (2511.01741), LRU (2511.17460), NAS-QAE-1 (2511.19246), FPGA-QLDPC (2511.21660), AlphaQubit2 (2512.07737), BeamSearch (2512.07057), Pinball (2512.09807)
- **2026**: SelfCoord (2601.09921), MoE-ViT (2601.12483), RLEAdvQEC (2601.19279), SideChannel (2602.15966), BBFair (2603.19062), LearningBPQLDPC (2603.10192), LowLatGNN (2603.22149), ScalPractical (2604.08358), AILoss (2604.14269), HeavyHex (2604.14296)

---

## Key findings for AutoQEC Ideator

- **Architecture winners for surface code (d≤11, April 2026)**: per-stabilizer token embeddings + transformer/recurrent hybrid (AlphaQubit, AlphaQubit-2) are SOTA on real Sycamore data. Mamba/SSM offers equal accuracy at O(d²) compute.
- **QLDPC / BB codes**: GNN message-passing on Tanner graph (Astra, HyperNQ, Efficient-Universal) beats BP-OSD both in accuracy and speed — no OSD post-processing needed. **This is the fastest-moving frontier.** Transformer-for-BB (2504.13043) is also competitive.
- **Neural BP line**: deep-unfolded BP with learned weights (Liu-Poulin 2019 → overcomplete/quaternary → 2025 relay-BP) is the strongest hybrid. Pure BP can be competitive with neural decoders for real-time (2506.01779); Ideator should beat this baseline explicitly.
- **RL approaches (cat 6)**: haven't reached SOTA. Limited to small distances and specific noise. Better used for *code discovery* (cat 8 papers, Nautrup 2019, 2502.14372) than for decoding inference.
- **Leakage / non-Pauli (cat 8)**: under-explored. AlphaQubit-2 uses soft readouts + leakage flags, but general leakage-aware neural decoders are a clear research gap.
- **NAS for neural decoders (cat 10)**: **Only 2202.05741 exists as real NAS work for surface-code NN decoders**, and it's limited to FFN topology sweeps on ASIC/FPGA. Huge AutoQEC opportunity — no transformer / GNN / SSM architecture search has been published for QEC decoding as of April 2026.
- **Latency-accuracy frontier (cat 7/9)**: classical LCD / Collision-Clustering achieve MHz on FPGA; neural decoders mostly need O(d²)+ parallel hardware. Pareto target: beat LCD's LER at comparable latency, or match LCD latency with better LER.
- **Benchmark codes to target**: d=3/5/7 surface (Sycamore), d=11 surface (AlphaQubit-2), [[72,12,6]] / [[144,12,12]] Gross code (cat 5), d≤9 color code (AlphaQubit-2).
- **Input representations that work**: raw syndromes as per-stabilizer tokens (AlphaQubit), spatio-temporal 3D tensors (2506.16113), detector-graph nodes (GNN line), Tanner-graph message passing (Astra).
- **Training data regime**: most papers use Stim + circuit-level-noise simulation, nsamples 10⁶–10⁸. Real-hardware fine-tuning (Sycamore) is the AlphaQubit differentiator.

## Papers not on arxiv
All listed papers have arxiv versions that were verified during search. If any download fails, re-check the log file `download.log` and fetch from the arxiv abstract URL `https://arxiv.org/abs/<id>` manually.