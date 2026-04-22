# AutoQEC Strategic Assessment

Strategic planning companion for the 3-person AutoQEC team. Reads the literature
at `knowledge/INDEX.md` and anchor papers. Sibling doc `DECODER_ROADMAP.md`
covers technical taxonomy / building blocks; this doc is deliberately
complementary — executive framing, feasibility, risk, scope, and fallback.

Generated: 2026-04-21.

---

## 1. Publishable-novelty positioning

**Literature verification.** `INDEX.md` Category 10 is explicit: 2202.05741
[@overwater2022neural] is the only published work that can fairly be called
"NAS for a neural QEC decoder," and even that is an FFN hyperparameter sweep
(width × depth × quantisation) driven by area–delay–accuracy, not an
architecture-space search. The two adjacent 2025 NAS papers
[@nasqae2025nov; @nasqae2025sep] are NAS for *quantum autoencoders*, not
decoders. A scan of the chronological index (late 2025 → April 2026) surfaces
SOTA neural-decoder work (AlphaQubit 2, Mamba, MoE-ViT, GraphQEC, Astra,
HyperNQ, Self-Coordinating) but no auto-search / agent-driven
decoder-discovery preprint. Conclusion: **the NAS-for-neural-QEC-decoder
whitespace still exists as of April 2026**, and the specific sub-whitespace
of *LLM-agent-driven* search has zero published competitors.

**Recommended claim framing: B + partial C, not A.** Framing A ("first
LLM-driven loop producing SOTA-competitive decoders") over-claims on two
axes reviewers will attack immediately: (i) "SOTA" for surface memory is
[@deepmind2025alphaqubit2], a Google DeepMind training run on many millions
of Stim samples with model ensembles — a three-person, 50–200-round harness
will not match it on absolute LER and reviewers will know this; (ii)
"LLM-driven" as the headline invites the "LLM hype" bin. Framing C ("agent
reproduces SOTA and extends") is appealing but "reproduces" is risky if
reproductions are partial.

Framing B is defensible and novel: **"A code-and-noise-agnostic harness that
discovers Pareto-front decoders across (code, noise, hardware-constraint)
triples without per-environment manual redesign, driven by an LLM-proposer +
evaluator loop."** This lands in the gap between per-environment
hand-designed SOTAs (AlphaQubit for surface, GraphQEC [@universal2025efficient]
for multi-code, MLBB [@ml2025bivariate] for BB) — each of which still
required human architects. The Pareto framing (LER × latency × params ×
code-generality) converts "we might not be absolute SOTA" from a weakness
into the methodological contribution: Pareto *discovery* is the output, not
a single leaderboard number.

Recommended headline: *"AutoQEC: Pareto-front discovery of neural QEC
decoders across codes and noise models via an agent-driven search loop."*

**Venue fit.**

| Venue | Fit | Rationale |
|---|---|---|
| *Quantum* (journal) | **Strong** | Accepts methodology + open-source harness papers; publishes neural-decoder work (Gicev 2023 [@gicev2023scalable], Zhang 2023 [@zhang2023convolutional], Andreasson 2019 [@andreasson2019quantum]); recent BB fair-baselines paper is journal-style [@bbfair2026]. |
| *IEEE QCE* (conf.) | **Strong** | Accepts reproducibility / systems / harness contributions; good for Framing D (decoder zoo). |
| *npj QI* | Medium | Precedent from Astra [@maan2025astra]; prefers experimental angle or clear threshold improvement. |
| *PRX Quantum* | Medium-hard | Requires either a threshold or a fundamental methods advance. |
| NeurIPS / ICML | Medium-hard | Fits "AutoML × physics" niche, but reviewers will demand ML-methods novelty in the *loop*, not just in the decoder. |
| *Nature QI* / *Science* | **Skip** | AlphaQubit 1/2 already anchor the top-tier narrative; the team cannot out-compute DeepMind. |

**Primary target: *Quantum*. Secondary: IEEE QCE (for systems/harness
emphasis). Both accept open-source deliverables and explicit Pareto-front
methodology.**

**Reviewer-1 stress test — what hits first and how we preempt.**

1. *"You don't beat AlphaQubit 2."* → Preempt by never promising to. Position
   Pareto discovery + cross-environment generality + cost-controlled search
   as the contribution. Report AQ2-at-DeepMind-scale as an oracle upper bound
   the *harness* cannot be expected to hit.
2. *"Your BB numbers are unfair."* — [@bbfair2026] just hammered the BB
   community on this exact point. Mitigation: adopt their methodology
   (200 k shots, bootstrap CIs, seeded reproducibility, erasure-aware baselines
   when relevant) **verbatim** in MVP env spec, cite them, and foreground
   fair-baseline adherence in §3 of the eventual paper.
3. *"The LLM is memorising the literature, not discovering."* → Mitigate by
   held-out "decoy codes" / unfamiliar noise biases and by reporting
   proposer-hit rate on a clean train/test split of the solution space.
4. *"Search budget is too small to be SOTA."* → Answer with Pareto framing
   + compute-disclosure table + "oracle bound via manual transcription of
   AQ2 into the DSL" ablation.

---

## 2. Baseline-beating feasibility

**What the anchors say.**

*Surface d=5 circuit-level noise.*
- **Sycamore d=5 experimental data** (the most-cited benchmark):
  AlphaQubit 1 [@bausch2024alphaqubit] reports LER = 2.748% ± 0.015% per
  cycle, beating TN (2.915%), MWPM-BP (3.059%), MWPM-Corr (3.597%),
  PyMatching (4.356%). Fine-tuned 20-model ensemble.
- Mamba [@mamba2025scalable] "matches AlphaQubit at d=3/5 with 100 M
  pretrain samples" (vs AQ1's 1 B); d=5 Mamba LER ≈ 3.03%, AQ1 (their
  replicated) ≈ 2.98% — i.e. **the accuracy floor at d=5 on Sycamore is not
  an intrinsic property of the model family; it is dominated by pretraining
  data volume**, a fact the Mamba authors emphasise. This is good news for
  AutoQEC: architectural cleverness has ~0.05% LER headroom over Mamba at d=5
  on Sycamore, but training-data scale has much more.
- SI1000 p=0.15% simulated (AQ2 regime [@deepmind2025alphaqubit2]): d=5
  LER reaches ~10⁻³, d=11 ~10⁻⁵, near optimal. Real-time variant AQ2-RT
  decodes d=11 surface <1 µs/cycle on commercial accelerators. Not
  reported: FLOPs per inference.

*BB [[72,12,6]] phenomenological / circuit-level noise.*
- MLBB [@ml2025bivariate]: at p=0.1% circuit-level, logical error rate
  ~4.5× lower than BP-OSD; runtime consistent vs BP-OSD's heavy tails.
- GraphQEC [@universal2025efficient]: at p=0.005 depolarising, [[72,12,6]]
  "break-even improved" (exact LER not reported in summary); on [[144,12,12]]
  LER = 9.55×10⁻⁵, 157 µs/cycle — claimed ~18× over BP-OSD.
- BP-OSD baseline [@roffe2020decoding; @bbfair2026] remains the canonical
  "must-beat" reference. [@improvedbp2025] (Relay-BP) is the classical
  baseline the Ideator must *explicitly* beat or it's dead on arrival.

**Can a 50–200-round × ~30-min-per-round loop match these?** For absolute
LER on either benchmark, **almost certainly no**. The LER floors above come
from 10⁸–10⁹ sample pretraining runs (AQ1/2), tens of M-parameter models,
and multi-week multi-GPU training. 200 rounds × 30 min ≈ 100 GPU-hours.

However, **matching the *matching / BP baselines* is very plausible.** On
Sycamore d=5, PyMatching = 4.36% LER; MWPM-BP = 3.06%; TN = 2.92%. A harness
that reaches ~3.0% LER on Sycamore d=5 is strictly between MWPM-BP and TN
and already a publishable result given Framing B (discovered without
hand-design). Similarly on BB [[72,12,6]] circuit-level, matching vanilla
BP-OSD at p=0.3–0.7% is realistic; matching MLBB [@ml2025bivariate] is the
stretch.

**Gap analysis.**
- *Data-scale gap.* Biggest single driver. Ideator cannot generate 10⁹
  training samples per proposal. Mitigation: shared pretraining cache
  per env; fine-tune-only search; distillation from a frozen oracle model.
- *Architecture-space coverage gap.* Low-risk; DSL can enumerate the winning
  motifs (attention, SSM, Tanner-GNN, 3D-conv) directly.
- *Ensembling gap.* AQ1/2 ensemble 2–20 models. Cheap to replicate inside
  the DSL at evaluation time.

**Recommendation: keep MVP envs, but re-frame the win condition.** Target
"beat the fastest classical decoder (PyMatching / vanilla BP-OSD) at
comparable latency on both envs" as MVP success, and "Pareto-dominate
MWPM-BP / non-MLBB transformer at a stated compute budget" as stretch.
**Do not promise to beat AlphaQubit 2 or GraphQEC in headline numbers.**

**Fair-baselines compliance [@bbfair2026].** Three methodological
requirements propagate into the AutoQEC env spec immediately: (i) use both
an uninformed *and* a decoder-matched baseline in BB evals — reporting only
against the weaker one is the canonical unfairness the paper calls out;
(ii) 200 k shots per data point, seeded, bootstrap CIs; (iii) record
package versions. Add these to `/verify-decoder` output and the env
manifest.

---

## 3. Compute feasibility

On a single RTX 4090 (24 GB, ~82 TFLOPs fp16), one round = "train a
candidate + eval on held-out syndromes in ≤30 min." Anchor numbers:

| Architecture | Typical params | Dataset size | Reported compute | 4090-feasible in <30 min? |
|---|---|---|---|---|
| FFN d=5 [@overwater2022neural] | 10⁴–10⁵ | 10⁶–10⁷ samples | not reported / minutes scale | **Yes** |
| CNN 3D [@fullyconv3d2025] | 10⁶ | not reported | not reported | Likely |
| Transformer-surface (AQ1 arch) d=5 | ~10⁷ | 2×10⁹ pretrain + 2×10⁴ fine-tune | not reported; multi-GPU-week implied | **Fine-tune only**: yes; full pretrain: no |
| Mamba [@mamba2025scalable] d=5 | ~10⁶–10⁷ | 10⁸ samples | RTX 4090 used for inference benchmarks | Full pretrain: no; 10⁷ sample train: borderline |
| GNN / Astra [@maan2025astra] d≤11 | ~10⁵–10⁶ | not reported | not reported | Likely |
| BB Transformer [@ml2025bivariate] [[72,12,6]] | not reported in summary | not reported | A100 for inference | Borderline; may need smaller variant |
| AQ2 SI1000 d=11 [@deepmind2025alphaqubit2] | ~10⁸ | 10⁸+ curriculum | multi-TPU-week | **H100 / TPU only**; not feasible |
| MoE-ViT [@moevit2026] | not reported | not reported | not reported | Probably not; MoE training overhead |

**DSL schema implications (hard caps to encode).**
- Params ≤ **10 M** (4090 memory + 30 min budget).
- Training samples per round ≤ **5×10⁶** (new) or use a **shared pretrained
  backbone + fine-tune** budget of ≤ 10⁵ samples.
- Context / rolling window ≤ **~4d cycles** in time; full memory experiments
  up to ~25 cycles is fine at d=5; d=11+ needs windowed architectures
  (SSM, sliding transformer) to fit in 30 min.
- Ensembling limited to ≤ **4 models per candidate** to preserve budget.
- Pinball pre-decoder / pre-processing [@pinball2025] allowed but counted
  in the latency metric.

**Resource note.** If the team has opportunistic access to an H100 / A100
for a small number of "finalist" runs, reserve that budget for the last
~5–10 rounds rather than spreading it thin. Otherwise plan entirely within
the 4090 envelope and treat "oracle upper-bound runs" as a separate,
later-stage activity.

---

## 4. QEC-specific reward-hacking risks

Beyond the generic agent-spec §8 items, QEC pipelines have specific
attack surfaces a naive eval harness will miss:

1. **Val-syndrome memorisation.** Stim is deterministic from seed. If the
   proposer sees val seeds or fingerprints, a candidate can memorise the
   val syndrome → logical-error map. Mitigation: generate val set from a
   *seed family the Ideator cannot observe*; include a *held-out-seed check*
   in `/verify-decoder` that re-rolls 100 k samples at an unseen seed and
   compares LER within tolerance.

2. **Degenerate-to-linear-layer latency hack.** If latency is rewarded
   purely on wall-clock, the search trivially discovers a 1-layer linear
   identity-ish decoder that is fast and has "acceptable" LER only at
   near-zero physical error. Mitigation: require `/verify-decoder` to eval
   at **two physical error rates** (one well below pseudo-threshold, one
   near) and report slope; reject candidates whose LER vs p slope is flat
   (non-discriminating).

3. **Val-pickling into weights.** A candidate could encode the val labels
   into a lookup table disguised as embeddings. Mitigation: run a
   *generalisation probe* — sample 10 k fresh shots from a noise model with
   a small perturbation (e.g. p ± 20%, or depolarising → slightly biased)
   and require LER to remain close to an analytic interpolation of the
   clean-p LERs; sharp deviations → flag.

4. **Gradient leakage via Stim seed reuse.** If training and eval share
   RNG state, "augmentations" in training silently include eval shots.
   Mitigation: enforce non-overlapping seed ranges per env, logged in the
   env manifest, and assert at `/verify-decoder` entry that the candidate
   cannot name its val seeds.

5. **Label-leak through detector-error-model (DEM).** AQ1's fine-tuning
   used DEMs fit to val distributions [@bausch2024alphaqubit]. A careless
   Ideator could have the candidate *itself* fit a DEM to val and then use
   it as a prior → effectively label leakage. Mitigation: DEMs used for
   training must be derived only from a declared train split.

6. **Metric over-fitting to LER slope.** Regressing LER vs cycle count
   (standard AQ1/AQ2 protocol) produces a single scalar; candidates can
   tune to the regression specifically. Mitigation: report full curve +
   fit residuals, and include a cycle-count extrapolation point (e.g. train
   at 2d+1, eval at 8d+4 cycles, as [@mamba2025scalable] already do).

Add all six as `/verify-decoder` sanity checks before MVP launch.

---

## 5. MVP scope validation

**Proposed MVP envs.**
- `surface_d5_depol`: rotated surface code d=5, circuit-level depolarising
  (SI1000-ish, p ≈ 0.1–0.3%), 10-round memory. Baselines: PyMatching,
  MWPM-BP, MWPM-Corr.
- `bb72_depol`: [[72,12,6]] BB code, circuit-level depolarising p ≈ 0.3–0.7%,
  memory experiment of NR rounds where NR ∈ {6, 12, 18}. Baselines:
  BP-OSD (order 0 and 10), Relay-BP [@improvedbp2025].

**Explicit "env N proves X" claims.**
- *Env 1 (`surface_d5_depol`) proves:* the harness can discover competitive
  decoders on the most-studied code family — a mandatory credibility gate.
  If AutoQEC can't match MWPM-BP here, no one will read further. Success
  also validates the per-stabiliser tokenised input representation,
  recurrent-over-time pattern, and the DSL's ability to express the AQ1
  family.
- *Env 2 (`bb72_depol`) proves:* the harness is not surface-code-specific
  and can handle non-local Tanner graph topologies. BB is architecturally
  very different (non-local stabilisers, higher rate, Tanner-graph GNN is
  the natural winner), so success demonstrates cross-code generalisation —
  which is the core of Framing B.

**Orthogonality check.** These envs are well-separated along the most
important axes: (code locality: local vs non-local), (input structure:
2D grid vs Tanner graph), (baseline family: matching vs BP-OSD), (winning
ML motif: transformer/CNN vs GNN/linear-attention). They are **not**
orthogonal on noise type (both depolarising) or on memory-vs-algorithm
(both memory). That's fine for MVP.

**BB pathologies to avoid.**
- *Unfair baselines* [@bbfair2026]: see §2 / §4 mitigations.
- *OSD order matters a lot* [@ml2025bivariate]: MLBB compares against 3rd-order
  OSD; using order-0 as your baseline inflates your win. Env spec must fix
  OSD order (recommend: report both 0 and 10, as [@bbfair2026] does).
- *X-only vs X+Z decoding* [@ml2025bivariate Sec 1.3 & note 2]: BP-OSD is
  usually run X-only for tractability; ML decoders can do both. Declaring
  which is being compared is part of fairness.
- *Circuit depth / connectivity assumptions*: [[72,12,6]] is not local, so
  "latency" means something different than on surface. Report
  parallel-depth of the stabiliser-extraction circuit alongside the
  decoder latency.

**Stretch env (leakage / biased noise).** This is *not* a pure plug-in in
Stim: leakage requires non-Pauli circuit-level simulation (`stim` supports
limited leakage via custom instructions; AQ1 fit soft-readout + leakage
flags post-hoc). Biased noise *is* plug-in (dephasing-biased depolarising
is a one-line Stim change). Verdict: **biased noise = plug-in, leakage =
infra change**. Add biased-noise to stretch; push full leakage to a v2
milestone.

**Verdict on scope.** **Keep the two MVP envs; add a third biased-noise
env (`surface_d5_biased`, e.g. Z-biased at η ∈ {3, 10}) as a cheap, *plug-in*
stretch** that tests whether AutoQEC is noise-model-adaptive without BB's
baseline-fairness complications. Leakage stays as v2.

---

## 6. Fallback thesis if MVP underperforms

If AutoQEC does not produce decoders that Pareto-dominate or match MWPM-BP /
BP-OSD baselines, there is still a real paper:

**Minimum viable publishable thesis.** *"An agent-driven, code-agnostic
harness for QEC decoder discovery: capabilities, failure modes, and a
reproducible benchmark suite." (IEEE QCE systems/benchmarks track, or
Quantum journal methodology section).* Contributions in descending order
of required performance:

1. **Harness + DSL + benchmark suite** (always publishable; zero
   performance dependency). A reproducible framework covering surface +
   BB + biased-noise envs, fair-baselines-compliant [@bbfair2026], with
   seeded Stim configs, baseline decoders, and a scored DSL schema. This
   alone is a systems paper worth of contribution.

2. **Reproduction study** ("harness autonomously reproduces known SOTA
   motifs"): transformer-on-surface, GNN-on-BB, Mamba-on-surface emerge
   from agent proposals without human nudging. Lower bar than "extends
   SOTA"; demonstrates the loop learns from literature without
   regurgitating specific architectures.

3. **Failure analysis paper**: "Where LLM loops fall short on QEC decoder
   design" — a taxonomy of the Ideator's recurring mistakes (what it
   proposes that can't possibly work, what it gets stuck on, what priors
   it lacks). This is genuinely novel content because no one has
   published such analysis for QEC, and it has cross-domain interest for
   the agentic-ML community.

4. **Decoder zoo**: release ~20–50 discovered candidates across the
   Pareto front, with reproducible training/eval scripts. Citeable
   artifact value.

Ordering the thesis narrative by (1) + (4) as primary, (2) + (3) as
secondary means the paper is safe-to-submit **independently of whether
AutoQEC reaches any particular LER threshold**, which removes nearly all
schedule risk.

---

## Summary

Recommended positioning: **Framing B (Pareto-front discovery across
(code, noise, constraint) triples)**, targeting *Quantum* + IEEE QCE. MVP
envs (`surface_d5_depol`, `bb72_depol`) stay; add `surface_d5_biased` as
plug-in stretch; leakage is v2. Mandatory methodological adoption of
[@bbfair2026] fair-baseline protocol. Six QEC-specific reward-hacking
guards added to `/verify-decoder`. DSL caps at 10 M params and 5×10⁶
fresh samples per round. Fallback thesis (harness + zoo + failure
analysis) is independently publishable, which decouples publication from
absolute-LER risk.
