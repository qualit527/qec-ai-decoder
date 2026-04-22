"""Download arxiv PDFs for the AutoQEC knowledge base.

Usage: python download.py
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPERS_DIR = HERE / "papers"
PAPERS_DIR.mkdir(exist_ok=True)

# (arxiv_id, filename-without-extension)
# Filename format: YYYY-FirstAuthor-ShortTitle
PAPERS: list[tuple[str, str]] = [
    # ---------- Category 1: Surface-code neural decoders (foundational) ----------
    ("1610.04238", "2017-Torlai-NeuralDecoderTopologicalCodes"),
    ("1705.09334", "2017-Krastanov-DNNProbabilisticStabilizer"),
    ("1705.00857", "2017-Varsamopoulos-DecodingSmallSurfaceCodesANN"),
    ("1802.06441", "2018-Chamberland-DeepNeuralDecodersNearTerm"),
    ("1804.02926", "2018-Baireuther-ColorCodeCircuitLevel"),
    ("1811.12456", "2018-Varsamopoulos-ComparingNNDecodersSurface"),
    ("1901.10847", "2019-Varsamopoulos-DistributedNNDecoder"),
    ("1705.08957", "2017-Baireuther-MLAssistedTopologicalCode"),
    ("1911.12308", "2020-Ni-NNDecodersLargeDistance2DToric"),
    ("2110.05854", "2023-Gicev-ScalableFastANNSyndromeDecoder"),
    ("2202.05741", "2022-Overwater-NNDecodersHardwareTradeoff"),
    ("2312.03508", "2023-Zhang-ConvolutionalNNDecodersSurface"),
    ("2406.02700", "2024-Varbanov-OptimizationDecoderPriors"),

    # ---------- Category 2: Transformer / SOTA surface-code decoders ----------
    ("2310.05900", "2024-Bausch-AlphaQubit"),
    ("2512.07737", "2025-DeepMind-AlphaQubit2"),
    ("2510.22724", "2025-Mamba-ScalableNeuralDecoders"),
    ("2509.03815", "2025-ParallelSelfCoordinatingNeural"),
    ("2601.09921", "2026-SelfCoordinatingNeuralParallel"),
    ("2506.16113", "2025-FullyConv3DNeuralDecoder"),
    ("2601.12483", "2026-MoEVisionTransformerSurfaceDecoding"),
    ("2601.19279", "2026-RLEnhancedAdvancedQEC"),
    ("2509.10164", "2025-Ohnishi-SymmetryIntegratedSurfaceDecoding"),
    ("2509.11370", "2025-NeuralDecodersUniversalQuantumAlgorithms"),
    ("2604.08358", "2026-ScalableNeuralDecodersPracticalFTQC"),
    ("2411.19253", "2024-QuantumFeedbackTransformer"),

    # ---------- Category 3: GNN decoders ----------
    ("2307.01241", "2023-Lange-DataDrivenGNNDecoding"),
    ("2408.05170", "2024-Ninkovic-DecodingQLDPCGNN"),
    ("2502.19971", "2025-EfficientUniversalNNDecoder"),
    ("2408.07038", "2024-Maan-AstraMLMessagePassing"),
    ("2603.22149", "2026-LowLatencyGNNAccelerator"),
    ("2511.01741", "2025-HyperNQHypergraphNNDecoder"),
    ("2603.10192", "2026-LearningDecodeQLDPCViaBP"),

    # ---------- Category 4: Neural belief propagation ----------
    ("1811.07835", "2019-LiuPoulin-NeuralBPDecoders"),
    ("1607.04793", "2016-Nachmani-LearningDecodeLinearCodes"),
    ("1901.08621", "2019-Nachmani-LearnedBPSimpleScaling"),
    ("2212.10245", "2022-NeuralBPOvercompleteCheckMatrices"),
    ("2308.08208", "2023-QuaternaryNeuralBPQLDPC"),
    ("2510.06257", "2025-UncertaintyAwareGeneralizableNeural"),
    ("2506.01779", "2025-ImprovedBPRealTimeQuantum"),

    # ---------- Category 5: qLDPC / BB / Gross code decoders ----------
    ("2111.03654", "2021-Panteleev-AsymptoticallyGoodQLDPC"),
    ("2005.07016", "2020-Roffe-DecodingQLDPCLandscape"),
    ("2308.07915", "2024-Bravyi-HighThresholdLowOverheadGross"),
    ("1904.02703", "2021-Roffe-DegenerateQLDPCCodes"),
    ("2504.13043", "2025-MLDecodingCircuitBivariateBicycle"),
    ("2311.03307", "2023-NoisySyndromeQLDPCSlidingWindow"),
    ("2509.01892", "2025-AcceleratingBPOSDLocalSyndrome"),
    ("2512.07057", "2025-BeamSearchDecoderQLDPC"),
    ("2511.21660", "2025-FPGATailoredQLDPCRealTime"),

    # ---------- Category 6: RL-based decoders ----------
    ("1811.12338", "2019-Andreasson-RLToricCode"),
    ("1810.07207", "2020-Sweke-RLDecodersFaultTolerant"),
    ("1912.12919", "2020-Fitzek-DeepQLearningToric"),
    ("1812.08451", "2019-Nautrup-OptimizingQECWithRL"),
    ("2212.11890", "2022-DecodingSurfaceCodesDeepRL"),
    ("2305.06378", "2023-DiscoveryOptimalQECRL"),
    ("2502.14372", "2025-LowWeightQECReinforcementLearning"),
    ("2506.03397", "2025-RLEnhancedGreedyStabilizerFq"),

    # ---------- Category 7: Ensemble / hybrid / pre-decoder ----------
    ("2101.07285", "2022-Meinerz-ScalableNeuralDecoderTopological"),
    ("2309.05558", "2024-RealTimeScalableResourceEfficient"),
    ("2410.05202", "2024-RealTimeLowLatencyQECSuperconducting"),
    ("2411.10343", "2024-LocalClusteringDecoderSurface"),
    ("2512.09807", "2025-PinballCryogenicPredecoder"),
    ("2412.14360", "2024-DemonstratingDynamicSurfaceCodes"),
    ("2408.13687", "2024-GoogleQuantumAI-BelowSurfaceThreshold"),

    # ---------- Category 8: Biased noise / leakage / non-Pauli decoders ----------
    ("2511.17460", "2025-LeakageReductionUnitsMeasurement"),
    ("2602.15966", "2026-HardwareAgnosticQuantumSideChannelLeakage"),
    ("2604.14269", "2026-AIEnabledQubitLossDecoding"),
    ("2604.14296", "2026-ScalableQECHeavyHexArray"),
    ("2403.03272", "2024-CorrelatedDecodingTransversalGates"),

    # ---------- Category 9: FPGA / latency-optimized ----------
    # (largely covered in Cat 7; adding specific FPGA neural accelerator papers)
    ("2110.10896", "2021-EfficientDecodingSurfaceCodeSyndromes"),

    # ---------- Category 10: NAS / auto-design ----------
    # 2202.05741 already included; space exploration of NN decoder architectures
    ("2511.19246", "2025-NeuralArchitectureSearchQuantumAutoencoders"),
    ("2509.15451", "2025-NASAlgorithmsQuantumAutoencoders"),

    # ---------- Category 11: Other codes (color, honeycomb, subsystem) ----------
    ("2306.16476", "2023-MinimisingSurfaceCodeFailuresColorDecoder"),
    ("2312.08813", "2023-NewCircuitsOpenSourceColorCode"),
    ("2508.15743", "2025-ColourCodesReachSurfaceVibeDecoding"),
    ("2301.11930", "2023-DeepQuantumErrorCorrection"),

    # ---------- Category 12: Reviews / surveys / benchmarks (for Ideator context) ----------
    ("2412.20380", "2024-AIForQECComprehensiveReview"),
    ("2311.11167", "2023-BenchmarkingMLModelsQEC"),
    ("2307.14989", "2024-DecodingAlgorithmsSurfaceCodesReview"),
    ("2309.17368", "2023-MLPracticalQuantumErrorMitigation"),
    ("2411.03202", "2024-ArchitecturesHeterogeneousQECCodes"),
    ("2603.19062", "2026-FairDecoderBaselinesBivariate"),
]


def download_one(arxiv_id: str, filename: str) -> tuple[bool, str]:
    dest = PAPERS_DIR / f"{filename}.pdf"
    if dest.exists() and dest.stat().st_size > 10_000:
        return True, "skipped-exists"
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "AutoQEC-KB/1.0 (contact: jiahanchen@proton.me)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if len(data) < 10_000:
            return False, f"too-small ({len(data)} bytes)"
        dest.write_bytes(data)
        return True, f"ok ({len(data)//1024} KB)"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, f"err {type(e).__name__}: {e}"


def main() -> int:
    ok = 0
    failed: list[tuple[str, str, str]] = []
    total_bytes = 0
    print(f"Downloading {len(PAPERS)} papers to {PAPERS_DIR}")
    for i, (aid, name) in enumerate(PAPERS, 1):
        success, msg = download_one(aid, name)
        tag = "OK " if success else "FAIL"
        print(f"[{i:3d}/{len(PAPERS)}] {tag} {aid:15s} {name[:60]:60s}  {msg}", flush=True)
        if success:
            ok += 1
            p = PAPERS_DIR / f"{name}.pdf"
            if p.exists():
                total_bytes += p.stat().st_size
        else:
            failed.append((aid, name, msg))
        # arxiv recommends <=1 request / 3s
        if i < len(PAPERS):
            time.sleep(3.1)

    print()
    print(f"Done: {ok}/{len(PAPERS)} succeeded, total {total_bytes/1024/1024:.1f} MB")
    if failed:
        print("\nFailures:")
        for aid, name, msg in failed:
            print(f"  - {aid} {name}: {msg}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
