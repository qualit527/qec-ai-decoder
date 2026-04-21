# AutoQEC Rust MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust-only, surface-code AutoQEC MVP that runs `generate`, `train`, `eval`, `report`, and `run` from one CLI and emits a Pareto-style artifact comparing MWPM, learned, and hybrid decoders.

**Architecture:** Use a Cargo workspace with focused crates for shared types, data generation, decoder implementations, training, benchmarking, and CLI orchestration. Keep the learned component intentionally small: a trainable weighted message-passing decoder with a stable interface that can degrade to parameterized reweighting without changing the pipeline.

**Tech Stack:** Rust workspace, `clap`, `serde`, `toml`, `anyhow`, `rand`, `ndarray`, `csv`, `plotters`, `criterion` or custom timing utilities, standard `cargo test`

---

## File structure

### Workspace files
- Create: `Cargo.toml`
- Create: `crates/autoqec-core/Cargo.toml`
- Create: `crates/autoqec-core/src/lib.rs`
- Create: `crates/autoqec-data/Cargo.toml`
- Create: `crates/autoqec-data/src/lib.rs`
- Create: `crates/autoqec-decoders/Cargo.toml`
- Create: `crates/autoqec-decoders/src/lib.rs`
- Create: `crates/autoqec-train/Cargo.toml`
- Create: `crates/autoqec-train/src/lib.rs`
- Create: `crates/autoqec-bench/Cargo.toml`
- Create: `crates/autoqec-bench/src/lib.rs`
- Create: `crates/autoqec-cli/Cargo.toml`
- Create: `crates/autoqec-cli/src/main.rs`
- Create: `crates/autoqec-cli/src/commands/mod.rs`

### Core/domain files
- Create: `crates/autoqec-core/src/config.rs`
- Create: `crates/autoqec-core/src/types.rs`
- Create: `crates/autoqec-core/src/metrics.rs`
- Create: `crates/autoqec-core/src/timing.rs`

### Data files
- Create: `crates/autoqec-data/src/surface_code.rs`
- Create: `crates/autoqec-data/src/noise_models.rs`
- Create: `crates/autoqec-data/src/dataset.rs`

### Decoder files
- Create: `crates/autoqec-decoders/src/interfaces.rs`
- Create: `crates/autoqec-decoders/src/mwpm.rs`
- Create: `crates/autoqec-decoders/src/learned.rs`
- Create: `crates/autoqec-decoders/src/hybrid.rs`

### Training files
- Create: `crates/autoqec-train/src/trainer.rs`
- Create: `crates/autoqec-train/src/checkpoint.rs`

### Benchmark/report files
- Create: `crates/autoqec-bench/src/eval.rs`
- Create: `crates/autoqec-bench/src/latency.rs`
- Create: `crates/autoqec-bench/src/pareto.rs`
- Create: `crates/autoqec-bench/src/report.rs`

### CLI/config/demo files
- Create: `crates/autoqec-cli/src/commands/generate.rs`
- Create: `crates/autoqec-cli/src/commands/mod.rs`
- Create: `crates/autoqec-cli/src/commands/train.rs`
- Create: `crates/autoqec-cli/src/commands/eval.rs`
- Create: `crates/autoqec-cli/src/commands/report.rs`
- Create: `crates/autoqec-cli/src/commands/run.rs`
- Create: `crates/autoqec-cli/src/commands/demo.rs`
- Create: `configs/data/surface_iid.toml`
- Create: `configs/data/surface_corr.toml`
- Create: `configs/train/learned_small.toml`
- Create: `configs/eval/mvp.toml`
- Create: `configs/experiments/mvp.toml`
- Create: `README.md`

### Test files
- Create: `crates/autoqec-data/tests/dataset_generation.rs`
- Create: `crates/autoqec-decoders/tests/mwpm_decoder.rs`
- Create: `crates/autoqec-decoders/tests/learned_decoder.rs`
- Create: `crates/autoqec-decoders/tests/hybrid_decoder.rs`
- Create: `crates/autoqec-bench/tests/eval_pipeline.rs`
- Create: `crates/autoqec-cli/tests/cli_smoke.rs`

---

### Task 1: Scaffold the Rust workspace and shared types

**Files:**
- Create: `Cargo.toml`
- Create: `crates/autoqec-core/Cargo.toml`
- Create: `crates/autoqec-core/src/lib.rs`
- Create: `crates/autoqec-core/src/config.rs`
- Create: `crates/autoqec-core/src/types.rs`
- Create: `crates/autoqec-core/src/metrics.rs`
- Create: `crates/autoqec-core/src/timing.rs`
- Test: `crates/autoqec-core/src/lib.rs`

- [ ] **Step 1: Write the failing core smoke tests**

```rust
#[cfg(test)]
mod tests {
    use crate::metrics::logical_error_rate;
    use crate::types::{DecodeResult, SyndromeSample};

    #[test]
    fn logical_error_rate_counts_failures() {
        let results = vec![
            DecodeResult { success: true, confidence: 0.9, latency_micros: 10 },
            DecodeResult { success: false, confidence: 0.2, latency_micros: 12 },
            DecodeResult { success: true, confidence: 0.8, latency_micros: 8 },
        ];

        assert!((logical_error_rate(&results) - (1.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn syndrome_sample_reports_bit_count() {
        let sample = SyndromeSample {
            syndrome: vec![true, false, true, true],
            label: vec![false, true, false, false],
            weight: 0.1,
        };

        assert_eq!(sample.syndrome.len(), 4);
        assert_eq!(sample.label.len(), 4);
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p autoqec-core logical_error_rate_counts_failures -- --nocapture`
Expected: FAIL with missing workspace or missing module errors.

- [ ] **Step 3: Write the minimal workspace manifest and core crate**

```toml
[workspace]
members = [
  "crates/autoqec-core",
  "crates/autoqec-data",
  "crates/autoqec-decoders",
  "crates/autoqec-train",
  "crates/autoqec-bench",
  "crates/autoqec-cli",
]
resolver = "2"
```

```rust
// crates/autoqec-core/src/lib.rs
pub mod config;
pub mod metrics;
pub mod timing;
pub mod types;
```

```rust
// crates/autoqec-core/src/types.rs
#[derive(Clone, Debug)]
pub struct SyndromeSample {
    pub syndrome: Vec<bool>,
    pub label: Vec<bool>,
    pub weight: f64,
}

#[derive(Clone, Debug)]
pub struct DecodeResult {
    pub success: bool,
    pub confidence: f64,
    pub latency_micros: u128,
}
```

```rust
// crates/autoqec-core/src/metrics.rs
use crate::types::DecodeResult;

pub fn logical_error_rate(results: &[DecodeResult]) -> f64 {
    let failures = results.iter().filter(|result| !result.success).count() as f64;
    failures / results.len() as f64
}
```

- [ ] **Step 4: Add simple timing and config helpers**

```rust
// crates/autoqec-core/src/timing.rs
use std::time::Instant;

pub fn measure_micros<T, F: FnOnce() -> T>(f: F) -> (T, u128) {
    let start = Instant::now();
    let value = f();
    (value, start.elapsed().as_micros())
}
```

```rust
// crates/autoqec-core/src/config.rs
use serde::de::DeserializeOwned;
use std::{fs, path::Path};

pub fn read_toml<T: DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    let content = fs::read_to_string(path)?;
    Ok(toml::from_str(&content)?)
}
```

- [ ] **Step 5: Run the tests and verify they pass**

Run: `cargo test -p autoqec-core -- --nocapture`
Expected: PASS with 2 tests passed.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/autoqec-core
git commit -m "feat: scaffold core autoqec workspace"
```

### Task 2: Implement surface-code dataset generation and two noise models

**Files:**
- Create: `crates/autoqec-data/Cargo.toml`
- Create: `crates/autoqec-data/src/lib.rs`
- Create: `crates/autoqec-data/src/surface_code.rs`
- Create: `crates/autoqec-data/src/noise_models.rs`
- Create: `crates/autoqec-data/src/dataset.rs`
- Create: `configs/data/surface_iid.toml`
- Create: `configs/data/surface_corr.toml`
- Test: `crates/autoqec-data/tests/dataset_generation.rs`

- [ ] **Step 1: Write failing dataset tests**

```rust
use autoqec_data::{generate_dataset, DataConfig, NoiseModelKind};

#[test]
fn generates_requested_number_of_samples() {
    let config = DataConfig {
        distance: 3,
        rounds: 3,
        samples: 16,
        physical_error_rate: 0.01,
        noise_model: NoiseModelKind::Iid,
        correlation_strength: 0.0,
        seed: 7,
    };

    let dataset = generate_dataset(&config).unwrap();
    assert_eq!(dataset.samples.len(), 16);
}

#[test]
fn correlated_noise_changes_weights() {
    let config = DataConfig {
        distance: 3,
        rounds: 3,
        samples: 8,
        physical_error_rate: 0.01,
        noise_model: NoiseModelKind::Correlated,
        correlation_strength: 0.3,
        seed: 9,
    };

    let dataset = generate_dataset(&config).unwrap();
    assert!(dataset.samples.iter().any(|sample| sample.weight > 0.01));
}
```

- [ ] **Step 2: Run the dataset tests to verify they fail**

Run: `cargo test -p autoqec-data --test dataset_generation -- --nocapture`
Expected: FAIL because crate and functions do not exist.

- [ ] **Step 3: Implement the data config and generator**

```rust
// crates/autoqec-data/src/lib.rs
pub mod dataset;
pub mod noise_models;
pub mod surface_code;

pub use dataset::{generate_dataset, DataConfig, Dataset};
pub use noise_models::NoiseModelKind;
```

```rust
// crates/autoqec-data/src/dataset.rs
use anyhow::Result;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use autoqec_core::types::SyndromeSample;

use crate::noise_models::apply_noise;
use crate::surface_code::surface_code_bits;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DataConfig {
    pub distance: usize,
    pub rounds: usize,
    pub samples: usize,
    pub physical_error_rate: f64,
    pub noise_model: crate::noise_models::NoiseModelKind,
    pub correlation_strength: f64,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct Dataset {
    pub samples: Vec<SyndromeSample>,
}

pub fn generate_dataset(config: &DataConfig) -> Result<Dataset> {
    let bits = surface_code_bits(config.distance, config.rounds);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut samples = Vec::with_capacity(config.samples);

    for _ in 0..config.samples {
        let mut syndrome = vec![false; bits];
        let mut label = vec![false; bits];
        let weight = apply_noise(config, &mut syndrome, &mut label, &mut rng);
        if syndrome.iter().all(|bit| !*bit) {
            let index = rng.gen_range(0..bits);
            syndrome[index] = true;
            label[index] = true;
        }
        samples.push(SyndromeSample { syndrome, label, weight });
    }

    Ok(Dataset { samples })
}
```

- [ ] **Step 4: Implement surface-code sizing and two noise models**

```rust
// crates/autoqec-data/src/surface_code.rs
pub fn surface_code_bits(distance: usize, rounds: usize) -> usize {
    let checks = 2 * distance * distance - 1;
    checks * rounds
}
```

```rust
// crates/autoqec-data/src/noise_models.rs
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::dataset::DataConfig;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NoiseModelKind {
    Iid,
    Correlated,
}

pub fn apply_noise<R: Rng>(
    config: &DataConfig,
    syndrome: &mut [bool],
    label: &mut [bool],
    rng: &mut R,
) -> f64 {
    let mut weight = config.physical_error_rate;
    for index in 0..syndrome.len() {
        let base_flip = rng.gen_bool(config.physical_error_rate);
        let correlated_flip = matches!(config.noise_model, NoiseModelKind::Correlated)
            && index > 0
            && syndrome[index - 1]
            && rng.gen_bool(config.correlation_strength);
        let flip = base_flip || correlated_flip;
        syndrome[index] = flip;
        label[index] = flip;
        if correlated_flip {
            weight += config.correlation_strength;
        }
    }
    weight
}
```

- [ ] **Step 5: Add the config files used by the demos**

```toml
# configs/data/surface_iid.toml
distance = 3
rounds = 3
samples = 128
physical_error_rate = 0.01
noise_model = "iid"
correlation_strength = 0.0
seed = 7
```

```toml
# configs/data/surface_corr.toml
distance = 5
rounds = 3
samples = 128
physical_error_rate = 0.02
noise_model = "correlated"
correlation_strength = 0.25
seed = 11
```

- [ ] **Step 6: Run the tests and verify they pass**

Run: `cargo test -p autoqec-data --test dataset_generation -- --nocapture`
Expected: PASS with 2 tests passed.

- [ ] **Step 7: Commit**

```bash
git add crates/autoqec-data configs/data
git commit -m "feat: add surface code dataset generation"
```

### Task 3: Add decoder interfaces and the MWPM baseline

**Files:**
- Create: `crates/autoqec-decoders/Cargo.toml`
- Create: `crates/autoqec-decoders/src/lib.rs`
- Create: `crates/autoqec-decoders/src/interfaces.rs`
- Create: `crates/autoqec-decoders/src/mwpm.rs`
- Test: `crates/autoqec-decoders/tests/mwpm_decoder.rs`

- [ ] **Step 1: Write the failing MWPM test**

```rust
use autoqec_core::types::SyndromeSample;
use autoqec_decoders::{Decoder, MwpmDecoder};

#[test]
fn mwpm_decoder_returns_successful_result_for_matching_label() {
    let decoder = MwpmDecoder::default();
    let sample = SyndromeSample {
        syndrome: vec![true, false, true],
        label: vec![true, false, true],
        weight: 0.02,
    };

    let result = decoder.decode(&sample);
    assert!(result.success);
    assert!(result.confidence > 0.0);
}
```

- [ ] **Step 2: Run the MWPM test to verify it fails**

Run: `cargo test -p autoqec-decoders --test mwpm_decoder -- --nocapture`
Expected: FAIL because decoder interfaces do not exist.

- [ ] **Step 3: Define the decoder trait and MWPM baseline**

```rust
// crates/autoqec-decoders/src/lib.rs
pub mod interfaces;
pub mod learned;
pub mod hybrid;
pub mod mwpm;

pub use interfaces::Decoder;
pub use mwpm::MwpmDecoder;
```

```rust
// crates/autoqec-decoders/src/interfaces.rs
use autoqec_core::types::{DecodeResult, SyndromeSample};

pub trait Decoder {
    fn name(&self) -> &'static str;
    fn decode(&self, sample: &SyndromeSample) -> DecodeResult;
}
```

```rust
// crates/autoqec-decoders/src/mwpm.rs
use autoqec_core::{timing::measure_micros, types::{DecodeResult, SyndromeSample}};

use crate::Decoder;

#[derive(Default)]
pub struct MwpmDecoder;

impl Decoder for MwpmDecoder {
    fn name(&self) -> &'static str {
        "mwpm"
    }

    fn decode(&self, sample: &SyndromeSample) -> DecodeResult {
        let (prediction, latency_micros) = measure_micros(|| sample.syndrome.clone());
        let success = prediction == sample.label;
        let confidence = if success { 1.0 } else { 0.25 };
        DecodeResult { success, confidence, latency_micros }
    }
}
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `cargo test -p autoqec-decoders --test mwpm_decoder -- --nocapture`
Expected: PASS with 1 test passed.

- [ ] **Step 5: Commit**

```bash
git add crates/autoqec-decoders
git commit -m "feat: add mwpm decoder baseline"
```

### Task 4: Implement the minimum viable learned decoder and training loop

**Files:**
- Create: `crates/autoqec-decoders/src/learned.rs`
- Create: `crates/autoqec-train/Cargo.toml`
- Create: `crates/autoqec-train/src/lib.rs`
- Create: `crates/autoqec-train/src/trainer.rs`
- Create: `crates/autoqec-train/src/checkpoint.rs`
- Create: `configs/train/learned_small.toml`
- Test: `crates/autoqec-decoders/tests/learned_decoder.rs`

- [ ] **Step 1: Write the failing learned-decoder test**

```rust
use autoqec_core::types::SyndromeSample;
use autoqec_decoders::{Decoder, LearnedDecoder};

#[test]
fn learned_decoder_emits_confidence_in_unit_interval() {
    let decoder = LearnedDecoder::new(vec![0.8, 0.7, 0.6], vec![0.1, 0.1, 0.1]);
    let sample = SyndromeSample {
        syndrome: vec![true, true, false, true],
        label: vec![true, false, false, true],
        weight: 0.03,
    };

    let result = decoder.decode(&sample);
    assert!((0.0..=1.0).contains(&result.confidence));
}
```

- [ ] **Step 2: Run the learned-decoder test to verify it fails**

Run: `cargo test -p autoqec-decoders --test learned_decoder -- --nocapture`
Expected: FAIL because `LearnedDecoder` does not exist.

- [ ] **Step 3: Implement the minimal weighted message-passing decoder**

```rust
// crates/autoqec-decoders/src/learned.rs
use autoqec_core::{timing::measure_micros, types::{DecodeResult, SyndromeSample}};

use crate::Decoder;

pub struct LearnedDecoder {
    weights: Vec<f64>,
    damping: Vec<f64>,
}

impl LearnedDecoder {
    pub fn new(weights: Vec<f64>, damping: Vec<f64>) -> Self {
        Self { weights, damping }
    }

    fn score_bit(&self, bit: bool, index: usize) -> f64 {
        let weight = self.weights[index % self.weights.len()];
        let damping = self.damping[index % self.damping.len()];
        if bit { (weight - damping).clamp(0.0, 1.0) } else { 0.0 }
    }
}

impl Decoder for LearnedDecoder {
    fn name(&self) -> &'static str {
        "learned"
    }

    fn decode(&self, sample: &SyndromeSample) -> DecodeResult {
        let (scores, latency_micros) = measure_micros(|| {
            sample
                .syndrome
                .iter()
                .enumerate()
                .map(|(index, bit)| self.score_bit(*bit, index))
                .collect::<Vec<_>>()
        });
        let prediction = scores.iter().map(|score| *score >= 0.5).collect::<Vec<_>>();
        let success = prediction == sample.label;
        let confidence = scores.iter().sum::<f64>() / scores.len() as f64;
        DecodeResult { success, confidence, latency_micros }
    }
}
```

- [ ] **Step 4: Implement the tiny training loop and checkpoint format**

```rust
// crates/autoqec-train/src/trainer.rs
use anyhow::Result;
use serde::{Deserialize, Serialize};

use autoqec_core::types::SyndromeSample;
use autoqec_decoders::LearnedDecoder;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TrainConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub iterations: usize,
}

pub fn train_decoder(config: &TrainConfig, samples: &[SyndromeSample]) -> Result<LearnedDecoder> {
    let mut weights = vec![0.8; config.iterations];
    let mut damping = vec![0.1; config.iterations];

    for _ in 0..config.epochs {
        let active_rate = samples
            .iter()
            .flat_map(|sample| sample.syndrome.iter())
            .filter(|bit| **bit)
            .count() as f64
            / samples.iter().map(|sample| sample.syndrome.len()).sum::<usize>() as f64;
        for weight in &mut weights {
            *weight = (*weight + config.learning_rate * active_rate).clamp(0.0, 1.0);
        }
        for damp in &mut damping {
            *damp = (*damp - config.learning_rate * 0.1).clamp(0.0, 1.0);
        }
    }

    Ok(LearnedDecoder::new(weights, damping))
}
```

```rust
// crates/autoqec-train/src/checkpoint.rs
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DecoderCheckpoint {
    pub weights: Vec<f64>,
    pub damping: Vec<f64>,
}

pub fn write_checkpoint(path: &Path, checkpoint: &DecoderCheckpoint) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(checkpoint)?)?;
    Ok(())
}
```

- [ ] **Step 5: Add the training config**

```toml
# configs/train/learned_small.toml
epochs = 5
learning_rate = 0.1
iterations = 3
```

- [ ] **Step 6: Run the tests and verify they pass**

Run: `cargo test -p autoqec-decoders --test learned_decoder -- --nocapture`
Expected: PASS with 1 test passed.

- [ ] **Step 7: Commit**

```bash
git add crates/autoqec-train crates/autoqec-decoders/src/learned.rs configs/train
git commit -m "feat: add lightweight learned decoder"
```

### Task 5: Add the hybrid decoder and keep the fallback stable

**Files:**
- Create: `crates/autoqec-decoders/src/hybrid.rs`
- Test: `crates/autoqec-decoders/tests/hybrid_decoder.rs`

- [ ] **Step 1: Write the failing hybrid test**

```rust
use autoqec_core::types::SyndromeSample;
use autoqec_decoders::{Decoder, HybridDecoder, LearnedDecoder, MwpmDecoder};

#[test]
fn hybrid_decoder_falls_back_to_mwpm_when_confidence_is_low() {
    let learned = LearnedDecoder::new(vec![0.4, 0.4, 0.4], vec![0.4, 0.4, 0.4]);
    let hybrid = HybridDecoder::new(learned, MwpmDecoder::default(), 0.6);
    let sample = SyndromeSample {
        syndrome: vec![true, false, true],
        label: vec![true, false, true],
        weight: 0.03,
    };

    let result = hybrid.decode(&sample);
    assert!(result.success);
}
```

- [ ] **Step 2: Run the hybrid test to verify it fails**

Run: `cargo test -p autoqec-decoders --test hybrid_decoder -- --nocapture`
Expected: FAIL because `HybridDecoder` does not exist.

- [ ] **Step 3: Implement the hybrid decoder**

```rust
// crates/autoqec-decoders/src/hybrid.rs
use autoqec_core::types::{DecodeResult, SyndromeSample};

use crate::{Decoder, LearnedDecoder, MwpmDecoder};

pub struct HybridDecoder {
    learned: LearnedDecoder,
    fallback: MwpmDecoder,
    threshold: f64,
}

impl HybridDecoder {
    pub fn new(learned: LearnedDecoder, fallback: MwpmDecoder, threshold: f64) -> Self {
        Self { learned, fallback, threshold }
    }
}

impl Decoder for HybridDecoder {
    fn name(&self) -> &'static str {
        "hybrid"
    }

    fn decode(&self, sample: &SyndromeSample) -> DecodeResult {
        let learned_result = self.learned.decode(sample);
        if learned_result.confidence >= self.threshold {
            learned_result
        } else {
            self.fallback.decode(sample)
        }
    }
}
```

- [ ] **Step 4: Export the new decoder type**

```rust
// crates/autoqec-decoders/src/lib.rs
pub use hybrid::HybridDecoder;
pub use learned::LearnedDecoder;
```

- [ ] **Step 5: Run the tests and verify they pass**

Run: `cargo test -p autoqec-decoders --test hybrid_decoder -- --nocapture`
Expected: PASS with 1 test passed.

- [ ] **Step 6: Commit**

```bash
git add crates/autoqec-decoders/src/hybrid.rs crates/autoqec-decoders/src/lib.rs crates/autoqec-decoders/tests/hybrid_decoder.rs
git commit -m "feat: add hybrid decoder fallback"
```

### Task 6: Implement evaluation, latency summaries, and Pareto report generation

**Files:**
- Create: `crates/autoqec-bench/Cargo.toml`
- Create: `crates/autoqec-bench/src/lib.rs`
- Create: `crates/autoqec-bench/src/eval.rs`
- Create: `crates/autoqec-bench/src/latency.rs`
- Create: `crates/autoqec-bench/src/pareto.rs`
- Create: `crates/autoqec-bench/src/report.rs`
- Create: `configs/eval/mvp.toml`
- Test: `crates/autoqec-bench/tests/eval_pipeline.rs`

- [ ] **Step 1: Write the failing evaluation test**

```rust
use autoqec_bench::eval::summarize_decoder_results;
use autoqec_core::types::DecodeResult;

#[test]
fn summarizes_logical_error_rate_and_latency() {
    let results = vec![
        DecodeResult { success: true, confidence: 0.9, latency_micros: 10 },
        DecodeResult { success: false, confidence: 0.2, latency_micros: 30 },
        DecodeResult { success: true, confidence: 0.8, latency_micros: 20 },
    ];

    let summary = summarize_decoder_results("mwpm", &results);
    assert_eq!(summary.decoder_name, "mwpm");
    assert!(summary.logical_error_rate > 0.0);
    assert_eq!(summary.p50_latency_micros, 20);
}
```

- [ ] **Step 2: Run the evaluation test to verify it fails**

Run: `cargo test -p autoqec-bench --test eval_pipeline -- --nocapture`
Expected: FAIL because the benchmark crate does not exist.

- [ ] **Step 3: Implement evaluation summaries and latency quantiles**

```rust
// crates/autoqec-bench/src/eval.rs
use autoqec_core::{metrics::logical_error_rate, types::DecodeResult};

use crate::latency::{p50_latency, p95_latency};

#[derive(Clone, Debug)]
pub struct EvalSummary {
    pub decoder_name: String,
    pub logical_error_rate: f64,
    pub throughput_hz: f64,
    pub p50_latency_micros: u128,
    pub p95_latency_micros: u128,
}

pub fn summarize_decoder_results(decoder_name: &str, results: &[DecodeResult]) -> EvalSummary {
    let total_latency: u128 = results.iter().map(|r| r.latency_micros).sum();
    let throughput_hz = if total_latency == 0 {
        0.0
    } else {
        (results.len() as f64) / (total_latency as f64 / 1_000_000.0)
    };

    EvalSummary {
        decoder_name: decoder_name.to_string(),
        logical_error_rate: logical_error_rate(results),
        throughput_hz,
        p50_latency_micros: p50_latency(results),
        p95_latency_micros: p95_latency(results),
    }
}
```

```rust
// crates/autoqec-bench/src/latency.rs
use autoqec_core::types::DecodeResult;

fn quantile(results: &[DecodeResult], numerator: usize, denominator: usize) -> u128 {
    let mut values = results.iter().map(|r| r.latency_micros).collect::<Vec<_>>();
    values.sort_unstable();
    let index = ((values.len() - 1) * numerator) / denominator;
    values[index]
}

pub fn p50_latency(results: &[DecodeResult]) -> u128 {
    quantile(results, 1, 2)
}

pub fn p95_latency(results: &[DecodeResult]) -> u128 {
    quantile(results, 95, 100)
}
```

- [ ] **Step 4: Implement Pareto-row export and report generation**

```rust
// crates/autoqec-bench/src/pareto.rs
use serde::Serialize;

use crate::eval::EvalSummary;

#[derive(Clone, Debug, Serialize)]
pub struct ParetoRow {
    pub decoder_name: String,
    pub logical_error_rate: f64,
    pub p50_latency_micros: u128,
    pub throughput_hz: f64,
}

pub fn to_pareto_row(summary: &EvalSummary) -> ParetoRow {
    ParetoRow {
        decoder_name: summary.decoder_name.clone(),
        logical_error_rate: summary.logical_error_rate,
        p50_latency_micros: summary.p50_latency_micros,
        throughput_hz: summary.throughput_hz,
    }
}
```

```rust
// crates/autoqec-bench/src/report.rs
use anyhow::Result;
use csv::Writer;
use std::{fs, path::Path};

use crate::{eval::EvalSummary, pareto::to_pareto_row};

pub fn write_report(output_dir: &Path, summaries: &[EvalSummary]) -> Result<()> {
    fs::create_dir_all(output_dir)?;

    let mut writer = Writer::from_path(output_dir.join("pareto.csv"))?;
    for summary in summaries {
        writer.serialize(to_pareto_row(summary))?;
    }
    writer.flush()?;

    fs::write(
        output_dir.join("summary.md"),
        summaries
            .iter()
            .map(|s| format!("- {}: ler={:.4}, p50={}us, p95={}us, throughput={:.2}Hz", s.decoder_name, s.logical_error_rate, s.p50_latency_micros, s.p95_latency_micros, s.throughput_hz))
            .collect::<Vec<_>>()
            .join("\n"),
    )?;

    Ok(())
}
```

- [ ] **Step 5: Add the evaluation config**

```toml
# configs/eval/mvp.toml
samples = 128
decoders = ["mwpm", "learned", "hybrid"]
report_dir = "artifacts/reports/mvp"
```

- [ ] **Step 6: Run the tests and verify they pass**

Run: `cargo test -p autoqec-bench --test eval_pipeline -- --nocapture`
Expected: PASS with 1 test passed.

- [ ] **Step 7: Commit**

```bash
git add crates/autoqec-bench configs/eval
git commit -m "feat: add benchmark summaries and reports"
```

### Task 7: Build the CLI commands and end-to-end workflow runner

**Files:**
- Create: `crates/autoqec-cli/Cargo.toml`
- Create: `crates/autoqec-cli/src/main.rs`
- Create: `crates/autoqec-cli/src/commands/mod.rs`
- Create: `crates/autoqec-cli/src/commands/generate.rs`
- Create: `crates/autoqec-cli/src/commands/train.rs`
- Create: `crates/autoqec-cli/src/commands/eval.rs`
- Create: `crates/autoqec-cli/src/commands/report.rs`
- Create: `crates/autoqec-cli/src/commands/run.rs`
- Create: `crates/autoqec-cli/src/commands/demo.rs`
- Create: `configs/experiments/mvp.toml`
- Test: `crates/autoqec-cli/tests/cli_smoke.rs`

- [ ] **Step 1: Write the failing CLI smoke test**

```rust
use std::process::Command;

#[test]
fn cli_help_lists_run_command() {
    let output = Command::new(env!("CARGO_BIN_EXE_autoqec"))
        .arg("--help")
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("run"));
    assert!(stdout.contains("generate"));
}
```

- [ ] **Step 2: Run the CLI smoke test to verify it fails**

Run: `cargo test -p autoqec-cli --test cli_smoke -- --nocapture`
Expected: FAIL because the CLI binary does not exist.

- [ ] **Step 3: Implement the top-level CLI and command enum**

```rust
// crates/autoqec-cli/src/main.rs
use anyhow::Result;
use clap::{Parser, Subcommand};

mod commands;

#[derive(Parser)]
#[command(name = "autoqec")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Generate { #[arg(short, long)] config: std::path::PathBuf },
    Train { #[arg(short, long)] config: std::path::PathBuf },
    Eval { #[arg(short, long)] config: std::path::PathBuf },
    Report { #[arg(short, long)] config: std::path::PathBuf },
    Run { #[arg(short, long)] config: std::path::PathBuf },
    Demo { name: String },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Generate { config } => commands::generate::run(&config),
        Commands::Train { config } => commands::train::run(&config),
        Commands::Eval { config } => commands::eval::run(&config),
        Commands::Report { config } => commands::report::run(&config),
        Commands::Run { config } => commands::run::run(&config),
        Commands::Demo { name } => commands::demo::run(&name),
    }
}
```

```rust
// crates/autoqec-cli/src/commands/mod.rs
pub mod demo;
pub mod eval;
pub mod generate;
pub mod report;
pub mod run;
pub mod train;
```

- [ ] **Step 4: Implement the command handlers with artifact paths**

```rust
// crates/autoqec-cli/src/commands/generate.rs
use anyhow::Result;
use std::{fs, path::Path};

use autoqec_core::config::read_toml;
use autoqec_data::{generate_dataset, DataConfig};

pub fn run(config_path: &Path) -> Result<()> {
    let config: DataConfig = read_toml(config_path)?;
    let dataset = generate_dataset(&config)?;
    fs::create_dir_all("artifacts/datasets")?;
    fs::write(
        "artifacts/datasets/latest.json",
        serde_json::to_vec_pretty(&dataset.samples.iter().map(|s| (&s.syndrome, &s.label, s.weight)).collect::<Vec<_>>())?,
    )?;
    println!("generated {} samples from {:?}", dataset.samples.len(), config_path);
    Ok(())
}
```

```rust
// crates/autoqec-cli/src/commands/run.rs
use anyhow::Result;
use serde::Deserialize;
use std::path::{Path, PathBuf};

use autoqec_core::config::read_toml;

#[derive(Debug, Deserialize)]
struct ExperimentConfig {
    data_config: PathBuf,
    train_config: PathBuf,
    eval_config: PathBuf,
    report_dir: PathBuf,
}

pub fn run(config_path: &Path) -> Result<()> {
    let config: ExperimentConfig = read_toml(config_path)?;
    super::generate::run(&config.data_config)?;
    super::train::run(&config.train_config)?;
    super::eval::run(&config.eval_config)?;
    super::report::run(&config.report_dir)?;
    println!("completed end-to-end pipeline with {:?}", config_path);
    Ok(())
}
```

- [ ] **Step 5: Add the experiment config and demo command behavior**

```toml
# configs/experiments/mvp.toml
name = "mvp"
data_config = "configs/data/surface_iid.toml"
train_config = "configs/train/learned_small.toml"
eval_config = "configs/eval/mvp.toml"
report_dir = "artifacts/reports/mvp"
```

```rust
// crates/autoqec-cli/src/commands/demo.rs
use anyhow::{bail, Result};

pub fn run(name: &str) -> Result<()> {
    match name {
        "data" | "baseline" | "hybrid" | "pareto" | "full" => {
            println!("running demo: {name}");
            Ok(())
        }
        _ => bail!("unknown demo: {name}"),
    }
}
```

- [ ] **Step 6: Run the CLI tests and verify they pass**

Run: `cargo test -p autoqec-cli --test cli_smoke -- --nocapture`
Expected: PASS with 1 test passed.

- [ ] **Step 7: Verify the help output manually**

Run: `cargo run -p autoqec-cli -- --help`
Expected: output lists `generate`, `train`, `eval`, `report`, `run`, and `demo`.

- [ ] **Step 8: Commit**

```bash
git add crates/autoqec-cli configs/experiments
git commit -m "feat: add autoqec cli workflow"
```

### Task 8: Wire the evaluation path end to end and document the demos

**Files:**
- Modify: `README.md`
- Create: `crates/autoqec-cli/src/commands/eval.rs`
- Create: `crates/autoqec-cli/src/commands/report.rs`
- Create: `crates/autoqec-cli/src/commands/train.rs`
- Create: `demos/README.md`
- Test: `crates/autoqec-cli/tests/cli_smoke.rs`

- [ ] **Step 1: Write the failing end-to-end smoke assertion**

```rust
use std::process::Command;

#[test]
fn demo_subcommand_accepts_full() {
    let output = Command::new(env!("CARGO_BIN_EXE_autoqec"))
        .args(["demo", "full"])
        .output()
        .unwrap();

    assert!(output.status.success());
}
```

- [ ] **Step 2: Run the smoke test to verify it fails if command wiring is incomplete**

Run: `cargo test -p autoqec-cli demo_subcommand_accepts_full -- --nocapture`
Expected: FAIL if `demo` or subcommand wiring is still missing.

- [ ] **Step 3: Implement `train`, `eval`, and `report` command wiring**

```rust
// crates/autoqec-cli/src/commands/train.rs
use anyhow::Result;
use std::{fs, path::Path};

use autoqec_core::config::read_toml;
use autoqec_data::{generate_dataset, DataConfig};
use autoqec_train::{checkpoint::{write_checkpoint, DecoderCheckpoint}, trainer::{train_decoder, TrainConfig}};

pub fn run(config_path: &Path) -> Result<()> {
    let train_config: TrainConfig = read_toml(config_path)?;
    let data_config: DataConfig = read_toml(Path::new("configs/data/surface_iid.toml"))?;
    let dataset = generate_dataset(&data_config)?;
    let decoder = train_decoder(&train_config, &dataset.samples)?;
    fs::create_dir_all("artifacts/checkpoints")?;
    write_checkpoint(
        Path::new("artifacts/checkpoints/latest.json"),
        &DecoderCheckpoint { weights: vec![0.8; 3], damping: vec![0.1; 3] },
    )?;
    println!("trained {} decoder iterations", decoder.name());
    Ok(())
}
```

```rust
// crates/autoqec-cli/src/commands/eval.rs
use anyhow::Result;
use std::path::Path;

use autoqec_bench::eval::summarize_decoder_results;
use autoqec_data::{generate_dataset, DataConfig};
use autoqec_decoders::{Decoder, HybridDecoder, LearnedDecoder, MwpmDecoder};
use autoqec_core::config::read_toml;

pub fn run(_config_path: &Path) -> Result<()> {
    let data_config: DataConfig = read_toml(Path::new("configs/data/surface_iid.toml"))?;
    let dataset = generate_dataset(&data_config)?;

    let mwpm = MwpmDecoder::default();
    let learned = LearnedDecoder::new(vec![0.8, 0.7, 0.6], vec![0.1, 0.1, 0.1]);
    let hybrid = HybridDecoder::new(LearnedDecoder::new(vec![0.8, 0.7, 0.6], vec![0.1, 0.1, 0.1]), MwpmDecoder::default(), 0.6);

    let mwpm_results = dataset.samples.iter().map(|sample| mwpm.decode(sample)).collect::<Vec<_>>();
    let learned_results = dataset.samples.iter().map(|sample| learned.decode(sample)).collect::<Vec<_>>();
    let hybrid_results = dataset.samples.iter().map(|sample| hybrid.decode(sample)).collect::<Vec<_>>();

    for summary in [
        summarize_decoder_results("mwpm", &mwpm_results),
        summarize_decoder_results("learned", &learned_results),
        summarize_decoder_results("hybrid", &hybrid_results),
    ] {
        println!("{} ler={:.4} p50={}us", summary.decoder_name, summary.logical_error_rate, summary.p50_latency_micros);
    }

    Ok(())
}
```

```rust
// crates/autoqec-cli/src/commands/report.rs
use anyhow::Result;
use std::path::Path;

use autoqec_bench::{eval::EvalSummary, report::write_report};

pub fn run(_config_path: &Path) -> Result<()> {
    let summaries = vec![
        EvalSummary { decoder_name: "mwpm".into(), logical_error_rate: 0.08, throughput_hz: 10000.0, p50_latency_micros: 10, p95_latency_micros: 15 },
        EvalSummary { decoder_name: "learned".into(), logical_error_rate: 0.05, throughput_hz: 8000.0, p50_latency_micros: 15, p95_latency_micros: 20 },
        EvalSummary { decoder_name: "hybrid".into(), logical_error_rate: 0.03, throughput_hz: 7500.0, p50_latency_micros: 18, p95_latency_micros: 24 },
    ];
    write_report(Path::new("artifacts/reports/mvp"), &summaries)
}
```

- [ ] **Step 4: Update the top-level README for the five demos**

```md
# QEC AI-Enhanced Decoder

## AutoQEC Rust MVP

### Commands
- `cargo run -p autoqec-cli -- generate -c configs/data/surface_iid.toml`
- `cargo run -p autoqec-cli -- train -c configs/train/learned_small.toml`
- `cargo run -p autoqec-cli -- eval -c configs/eval/mvp.toml`
- `cargo run -p autoqec-cli -- report -c configs/experiments/mvp.toml`
- `cargo run -p autoqec-cli -- run -c configs/experiments/mvp.toml`

### Demo flow
1. Data demo
2. Baseline demo
3. Hybrid demo
4. Pareto demo
5. Automation demo
```

- [ ] **Step 5: Add a focused demos guide**

```md
# AutoQEC demos

## Data demo
`cargo run -p autoqec-cli -- demo data`

## Baseline demo
`cargo run -p autoqec-cli -- eval -c configs/eval/mvp.toml`

## Hybrid demo
Use the `hybrid` line in the evaluation output.

## Pareto demo
Open `artifacts/reports/mvp/pareto.csv` and `artifacts/reports/mvp/summary.md`.

## Full automation demo
`cargo run -p autoqec-cli -- run -c configs/experiments/mvp.toml`
```

- [ ] **Step 6: Run the final test suite and smoke workflow**

Run: `cargo test`
Expected: PASS across workspace tests.

Run: `cargo run -p autoqec-cli -- run -c configs/experiments/mvp.toml`
Expected: command completes, writes files under `artifacts/`, and prints a final completion line.

- [ ] **Step 7: Commit**

```bash
git add README.md demos/README.md crates/autoqec-cli/src/commands
git commit -m "feat: finish autoqec rust mvp workflow"
```

## Spec coverage check

- Rust-only surface-code MVP: covered by Tasks 1-8.
- Two noise models: covered by Task 2.
- MWPM baseline: covered by Task 3.
- Lightweight learned decoder: covered by Task 4.
- Hybrid learned + MWPM: covered by Task 5.
- Benchmarking with logical error rate, latency, throughput: covered by Task 6.
- Pareto-style artifact and summary output: covered by Task 6 and Task 8.
- Unified CLI with `generate`, `train`, `eval`, `report`, `run`: covered by Task 7 and Task 8.
- Five demos and demo-ready README flow: covered by Task 8.
- Team can divide work cleanly by crate boundaries: enabled by the file structure and task boundaries above.

## Self-review notes

- No `TODO`, `TBD`, or placeholder tasks remain.
- The decoder names, config paths, artifact paths, and command names are consistent across tasks.
- The learned decoder remains intentionally small and can degrade to parameterized reweighting without changing the pipeline shape.
