# Rift — Autonomous TensorRT Failure Recovery

> **Automatically diagnose, repair, and verify TensorRT migration failures** when naive PyTorch → ONNX → TensorRT pipelines break.

Rift is an evidence-driven recovery agent for NVIDIA TensorRT deployments. It runs an untouched baseline first, classifies failures from real error output, applies targeted repair strategies, and verifies every candidate engine against a PyTorch reference using cosine similarity — all with full JSON audit trails.

This repository contains the **Colab T4 benchmark suite**, benchmark artifacts, and repair trajectories from a five-model evaluation across vision, NLP, detection, and audio domains.

---

## Table of Contents

- [Why Rift Exists](#why-rift-exists)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [Failure Classification](#failure-classification)
- [Repair Toolkit](#repair-toolkit)
- [Benchmark Suite](#benchmark-suite)
- [Benchmark Results](#benchmark-results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [JSON Artifact Schemas](#json-artifact-schemas)
- [Integrity Rules](#integrity-rules)
- [Configuration Reference](#configuration-reference)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)

---

## Why Rift Exists

Migrating PyTorch models to TensorRT is brittle. Common failure modes include:

| Pain Point | Example |
|---|---|
| **PyTorch export API changes** | `dynamic_axes` → `dynamic_shapes` break in PyTorch 2.11+ |
| **TensorRT version migrations** | `--fp16`, `--int8`, and other flags removed in TensorRT 11 (strong typing) |
| **Dynamic shape mismatches** | Missing optimization profiles cause silent shape errors |
| **Unsupported operators** | Custom or rare ONNX ops fail during TRT parsing |
| **Precision drift** | FP16 engines produce outputs below accuracy thresholds |
| **Resource exhaustion** | OOM or workspace allocation failures during build |

Teams often spend **days to weeks** debugging these issues manually. Rift automates the diagnostic → repair → verify loop so engineers get a verified engine — or a clear, auditable failure reason.

---

## Key Features

| Feature | Description |
|---|---|
| **Baseline-first integrity** | Repair logic never runs until an untouched baseline is recorded |
| **Evidence-based classification** | Failure categories derived from actual stderr/stdout, not assumptions |
| **Deterministic repair tools** | Each failure category maps to a specific, auditable repair strategy |
| **Cosine similarity verification** | Every candidate engine validated against PyTorch reference (threshold: 0.99) |
| **Bounded retries** | Max 3 attempts per model, 480s timeout — no unbounded agent loops |
| **Full trajectory logging** | Every attempt, build, profile, and precision check saved as JSON |
| **Human approval gate** | Final engines copied to persistent storage only after explicit approval |
| **Subprocess sandbox** | Crash isolation tested via SIGSEGV sandbox before orchestration |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                    │
│         PyTorch Model  +  Test Case Config (shapes, export flags)           │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1 — UNTOUCHED BASELINE                             │
│         ONNX Export  ──►  trtexec Build  ──►  Baseline JSON Artifact        │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2 — CLASSIFICATION                                 │
│         classify_failure()  ──►  Failure Category Label                     │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3 — RECOVERY AGENT                                 │
│                         recover_model() Orchestrator                        │
│    ┌──────────────┬──────────────┬──────────────┬──────────────────────┐    │
│    │ Export API   │ Dynamic      │ Node         │ Precision Flag       │    │
│    │ Migration    │ Profile Inj. │ Surgery      │ Migration            │    │
│    └──────────────┴──────────────┴──────────────┴──────────────────────┘    │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 4 — VERIFICATION                                 │
│    TRT Engine Build  ──►  Engine Inference  ──►  Cosine Similarity Check    │
│                              (threshold: 0.99)                              │
└──────────────────────────┬─────────────────────┬────────────────────────────┘
                           │                     │
                    pass ≥ 0.99            fail (retry)
                           │                     │
                           ▼                     └──► back to Phase 3
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 5 — DELIVERY                                       │
│    Trajectory JSON  ──►  Human Approval Gate  ──►  Final .engine File       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Core Module | Depends On | Role |
|---|---|---|
| **Baseline Exporter** | PyTorch, ONNX, Transformers / timm / Ultralytics | Export model to ONNX and run naive trtexec build |
| **Failure Classifier** | TensorRT / trtexec stderr | Label failure category from error evidence |
| **Repair Tools** | PyTorch, ONNX, GraphSurgeon, TensorRT | Apply category-specific graph/export fixes |
| **Precision Verifier** | PyTorch, TensorRT runtime | Run engine inference and compare to reference |
| **Orchestrator** | All core modules above | Route failures to tools, enforce retry/timeout limits |

---

## End-to-End Pipeline

| Step | Actor | Action |
|:---:|---|---|
| 1 | **User** | Trigger untouched ONNX export + trtexec build |
| 2 | **Baseline** | Write baseline JSON (success or failure) |
| 3 | **Classifier** | If failed → extract error evidence → assign category label |
| 4 | **Rift Agent** | Select repair tool matching the failure category |
| 5 | **trtexec** | Build candidate TensorRT engine from repaired ONNX |
| 6 | **Verifier** | Run engine inference; compute cosine similarity vs PyTorch |
| 7 | **Rift Agent** | If cosine ≥ 0.99 → mark `repaired_and_verified`; else retry (max 3, 480s) |
| 8 | **Approval Gate** | Prompt user `[y/N]` before copying engine to persistent storage |
| 9 | **User** | Approve → final `.engine` delivered to Drive |

**If baseline already succeeded:** steps 3–7 are skipped; status is `baseline_already_succeeded`.

---

## Failure Classification

Rift classifies failures using **evidence extracted from error output**, not pre-assigned labels. The `expected_category` in each test case is a **hypothesis**; the `actual_category` comes from runtime evidence.

### Supported Categories

| Category | Trigger Signals | Repair Tool |
|---|---|---|
| `export_api_incompatibility` | `Failed to convert 'dynamic_axes' to 'dynamic_shapes'` | `export_shapes_migration()` |
| `precision_flag_removed` | `Unknown option: --fp16` (or `--int8`, `--bf16`, etc.) | `precision_flag_migration()` |
| `shape_mismatch` | `dimension mismatch`, `optimization profile`, `dynamic shape` | `dynamic_profile_injection()` |
| `unsupported_operator` | `unsupported operator`, `no importer`, `not implemented` | `node_surgery_splicing()` |
| `precision_drift` | Measured cosine < 0.99 after successful build | `precision_flag_migration()` (FP32 fallback) |
| `resource_bound` | `out of memory`, `workspace size`, `insufficient memory` | Not yet implemented (requires diagnostics) |
| `unknown` | No pattern match | No repair attempted |

### Classification Decision Tree

```
                    Parse stderr + stdout
                              │
                              ▼
              ┌───────────────────────────────┐
              │ dynamic_axes → dynamic_shapes │
              │         error present?        │
              └───────────────┬───────────────┘
                     Yes      │      No
                      ▼       │       ▼
         export_api_          │   ┌─────────────────────────┐
         incompatibility      │   │ Unknown option + removed│
                              │   │   precision flag?       │
                              │   └───────────┬─────────────┘
                              │        Yes    │    No
                              │         ▼     │     ▼
                              │  precision_   │  Score keyword
                              │  flag_removed │  patterns
                              │               │     │
                              │               │     ▼
                              │               │  best score > 0?
                              │               │   Yes │  No
                              │               │    ▼  │   ▼
                              │               │  shape_mismatch /     unknown
                              │               │  unsupported_operator /
                              │               │  precision_drift /
                              │               │  resource_bound
```

### Removed TensorRT Precision Flags (TRT 10 → 11)

These flags are detected explicitly to avoid misclassification as numerical drift:

```
--fp16  --int8  --bf16  --fp8  --int4  --best
```

---

## Repair Toolkit

### 1. Export API Migration (`export_shapes_migration`)

Repairs the PyTorch 2.11+ export break where `dynamic_axes` is no longer accepted.

| Attempt | Strategy | Description |
|---|---|---|
| 1 | `legacy_torchscript_dynamic_axes` | Re-export via legacy exporter with `dynamo=False` |
| 2 | `dynamo_dynamic_shapes` | Re-export with `torch.export.Dim` dynamic shapes |
| 3 | `static_shapes_no_dynamic` | Force static-shape export (escalation on retry) |

### 2. Dynamic Profile Injection (`dynamic_profile_injection`)

Replaces symbolic ONNX shape dimensions with concrete bounds for TensorRT optimization profiles.

### 3. Node Surgery (`node_surgery_splicing`)

Conservative ONNX graph surgery. Currently supports **Identity node bypass** — a provably semantics-preserving substitution.

### 4. Precision Flag Migration (`precision_flag_migration`)

Repairs TensorRT 11's removal of CLI precision flags by rebuilding with **strong typing** (no `--fp16` flag). Optionally bakes FP16 into the ONNX graph via `onnxconverter-common`.

### Repair Routing Matrix

| Failure Category | Attempt 1 | Attempt 2+ | Max Retries |
|---|---|---|---|
| `export_api_incompatibility` | Dynamic legacy export | Static export | 3 |
| `shape_mismatch` | Profile injection | Same tool | 3 |
| `unsupported_operator` | Identity bypass | Same tool | 3 |
| `precision_flag_removed` | Strongly-typed FP32 build | Same tool | 3 |
| `precision_drift` | FP32 fallback (if baseline was FP16) | — | 3 |

---

## Benchmark Suite

Five models spanning four ML domains, each configured to surface a specific failure mode under TensorRT 11 + PyTorch 2.11.

| # | Model | Domain | Input Shape | Intended Category | Export Config |
|---|---|---|---|---|---|
| 1 | **ResNet-50** | Vision (CNN) | `1×3×224×224` | `shape_mismatch` | Dynamic axes |
| 2 | **ViT-Base** | Vision (Transformer) | `1×3×224×224` | `shape_mismatch` | Dynamic axes |
| 3 | **BERT-Base** | NLP | `1×32` (3 inputs) | `unsupported_operator` | Dynamic axes |
| 4 | **YOLOv8n** | Object Detection | `1×3×640×640` | `precision_flag_removed` | Static + forced `--fp16` |
| 5 | **Audio Spectrogram Transformer** | Audio | `1×1024×128` | `precision_flag_removed` | Static + forced `--fp16` |

> **Note:** ResNet-50 succeeds on baseline in the current environment. ViT and BERT actually fail with `export_api_incompatibility` (PyTorch export break), not their intended categories — demonstrating Rift's evidence-based classification.

---

## Benchmark Results

**Environment:** Google Colab Tesla T4 · PyTorch 2.11.0+cu128 · TensorRT 11.2.1.2 · Python 3.13.15

### Summary

| Model | Baseline Status | Rift Status | Max Cosine | Retries | Repair Time |
|---|---|---|---|---|---|
| ResNet-50 | SUCCESS | baseline_already_succeeded | 1.000 | 0 | — |
| ViT-Base | MODEL_EXPORT_FAILURE | **repaired_and_verified** | 1.000 | 1 | 109.3s |
| BERT-Base | MODEL_EXPORT_FAILURE | exhausted_retries | 0.900 | 3 | 266.5s |
| YOLOv8n | TENSORRT_BUILD_FAILURE | **repaired_and_verified** | 1.000 | 1 | 97.6s |
| Audio Spectrogram Transformer | TENSORRT_BUILD_FAILURE | **repaired_and_verified** | 1.000 | 1 | 63.0s |

### Success Rate

```
Baseline failures:     4 / 5 models
Rift recovered:        3 / 4 failures  (75%)
Overall verified:      4 / 5 models    (80%)
```

**Baseline outcomes (5 models)**

| Outcome | Count | Share |
|---|---|---|
| Success (no repair needed) | 1 | 20% |
| Export Failure | 2 | 40% |
| TRT Build Failure | 2 | 40% |

**Rift recovery on failed baselines (4 models)**

| Outcome | Count | Share |
|---|---|---|
| Repaired & Verified | 3 | 75% |
| Exhausted Retries | 1 | 25% |

### Inference Latency (Repaired Engines)

| Model | Mean GPU Latency | Median | P99 |
|---|---|---|---|
| ViT-Base | 13.13 ms | 13.06 ms | 13.37 ms |
| YOLOv8n | 3.57 ms | 3.56 ms | 3.67 ms |
| Audio Spectrogram Transformer | 83.40 ms | 83.44 ms | 84.72 ms |

### Per-Model Repair Details

#### ViT-Base — Export API Incompatibility

| Field | Value |
|---|---|
| Actual category | `export_api_incompatibility` |
| Repair strategy | `legacy_torchscript_dynamic_axes` |
| Build duration | 58.8s |
| Cosine similarity | 0.9999999999988204 |

#### YOLOv8n — Precision Flag Removed

| Field | Value |
|---|---|
| Actual category | `precision_flag_removed` |
| Repair strategy | Drop `--fp16`; build strongly-typed FP32 |
| Build duration | 86.2s |
| Cosine similarity | 0.9999999999999799 |

#### BERT-Base — Export Recovered, Precision Failed

| Attempt | Strategy | Cosine | Passed |
|---|---|---|---|
| 1 | `legacy_torchscript_dynamic_axes` | 0.888 | No |
| 2 | `static_shapes_no_dynamic` | 0.895 | No |
| 3 | `static_shapes_no_dynamic` | 0.900 | No |

BERT demonstrates a known limitation: export can succeed while numerical parity with the PyTorch reference remains below the 0.99 threshold, likely due to multi-input attention dynamics and Int64 binding warnings.

---

## Project Structure

```
.
├── README.md                                          # This file
├── Rift_TRT_Migrator_Colab_FIXED_v3 (2).ipynb         # Main benchmark notebook (24 cells)
│
├── logs/
│   ├── environment.json                               # Runtime environment snapshot
│   ├── versions.json                                  # Exact package versions
│   ├── final_summary.json                             # Aggregated benchmark results
│   └── sandbox_crash_test.json                        # SIGSEGV isolation proof
│
└── trajectories/
    ├── {model}.json                                   # Full repair trajectories (rift-trajectory-v2)
    └── {model}_classification.json                    # Failure classification evidence (rift-classification-v1)
```

### Runtime Directories (Created by Notebook)

| Path | Purpose |
|---|---|
| `/content/rift_workspace/onnx/` | Baseline and repaired ONNX models |
| `/content/rift_workspace/engines/` | Candidate and baseline TensorRT engines |
| `/content/rift_workspace/logs/baseline/` | Untouched baseline JSON records |
| `/content/rift_workspace/logs/approval/` | Human approval gate records |
| `/content/rift_workspace/trajectories/` | Live trajectory copies during execution |
| `/content/drive/MyDrive/Projects/micro1/` | Persistent Google Drive storage |

---

## Getting Started

### Prerequisites

| Requirement | Version (Benchmark) |
|---|---|
| GPU | NVIDIA Tesla T4 (or any CUDA-capable GPU) |
| Python | 3.13+ |
| PyTorch | 2.11.0+cu128 |
| TensorRT | 11.2.1.2 |
| CUDA | 12.8 |
| Runtime | Google Colab with GPU enabled |

### Quick Start (Google Colab)

1. **Open the notebook**
   ```
   Rift_TRT_Migrator_Colab_FIXED_v3 (2).ipynb
   ```

2. **Select a GPU runtime**
   - Runtime → Change runtime type → **T4 GPU**

3. **Run all cells in order** (Cells 1–24)
   - Cell 1: Mount Google Drive and create directories
   - Cell 2: Install dependencies
   - Cell 3: Verify environment (CUDA, TensorRT, trtexec)
   - Cells 6–10: Load models and run **untouched baseline**
   - Cells 12–13: Classify baseline failures
   - Cells 14–19: Define repair tools and orchestrator
   - Cell 20: Run Rift recovery agent
   - Cell 21–22: Human approval gate
   - Cell 23: Generate results from JSON artifacts
   - Cell 24: Copy artifacts to Drive

4. **Review results**
   - Check `logs/final_summary.json` for aggregated outcomes
   - Inspect `trajectories/*.json` for full repair audit trails

### Dependencies

Installed automatically in Cell 2:

```
onnx>=1.17,<1.23
onnxscript>=0.5,<1.0
onnx-graphsurgeon>=0.5,<1.0
transformers>=4.50,<5.0
timm>=1.0,<2.0
ultralytics>=8.3,<9.0
scipy>=1.11,<2.0
onnxconverter-common>=1.14,<2.0
tensorrt
```

---

## JSON Artifact Schemas

### `rift-baseline-v2` — Baseline Record

```json
{
  "schema": "rift-baseline-v2",
  "model": "vit_base",
  "display_name": "ViT-Base",
  "intended_category": "shape_mismatch",
  "export": { "success": false, "error": "..." },
  "policy": {
    "single_attempt": true,
    "manual_hints": false,
    "repair_logic": false
  },
  "status": "MODEL_EXPORT_FAILURE",
  "success": false,
  "total_duration_sec": 0.01
}
```

**Status values:** `SUCCESS` · `MODEL_EXPORT_FAILURE` · `TENSORRT_BUILD_FAILURE` · `INVALID_ENVIRONMENT`

### `rift-classification-v1` — Failure Classification

```json
{
  "schema": "rift-classification-v1",
  "model": "vit_base",
  "classification": {
    "label": "export_api_incompatibility",
    "scores": {},
    "evidence_snippet": "RuntimeError: Failed to convert 'dynamic_axes' to 'dynamic_shapes'..."
  }
}
```

### `rift-trajectory-v2` — Repair Trajectory

```json
{
  "schema": "rift-trajectory-v2",
  "model": "vit_base",
  "intended_category": "shape_mismatch",
  "actual_category": "export_api_incompatibility",
  "attempts": [
    {
      "attempt": 1,
      "category": "export_api_incompatibility",
      "repair": {
        "success": true,
        "strategy": "legacy_torchscript_dynamic_axes",
        "output": "/path/to/repair_vit_base_1.onnx"
      },
      "build": { "returncode": 0, "engine_exists": true },
      "precision": {
        "cosine_similarity": 0.9999999999988204,
        "threshold": 0.99,
        "passed": true
      }
    }
  ],
  "final_status": "repaired_and_verified",
  "total_repair_time_sec": 109.29
}
```

**Final status values:** `repaired_and_verified` · `exhausted_retries` · `baseline_already_succeeded` · `unknown_failure` · `timeout`

---

## Integrity Rules

Rift enforces strict experimental integrity to ensure benchmark results are reproducible and honest:

| Rule | Rationale |
|---|---|
| **Baseline runs before repair** | Repair logic cannot influence baseline measurements |
| **Environment failures excluded** | Missing packages (`onnxscript`, etc.) are not counted as model failures |
| **Evidence over hypothesis** | `actual_category` comes from error output; `intended_category` is only a hypothesis |
| **Results from JSON artifacts** | Summary tables and charts are computed from saved JSON, not live variables |
| **Human approval for delivery** | Final engines copied to persistent storage only after explicit `[y/N]` confirmation |
| **Measured precision only** | `precision_drift` requires sub-threshold cosine, not keyword matching on "fp16" in stderr |
| **Bounded orchestration** | Max 3 retries, 480s timeout — no infinite agent loops |
| **Conservative surgery only** | Node surgery limited to provably safe Identity bypass |

---

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `MAX_RETRIES` | `3` | Maximum repair attempts per model |
| `MODEL_TIMEOUT` | `480` | Total repair budget in seconds |
| Cosine threshold | `0.99` | Minimum similarity for verification pass |
| ONNX opset | `17` | Export opset version |
| Warmup iterations | `100` | trtexec profiling warmup |
| Profile duration | `10s` | trtexec inference measurement window |

---

## Known Limitations

| Limitation | Impact | Status |
|---|---|---|
| **BERT precision parity** | Multi-input transformers may export and build but fail cosine verification | Open — needs attention-specific repair |
| **Resource-bound repairs** | OOM/workspace failures not yet auto-repaired | Planned — requires workspace diagnostics |
| **Custom ONNX plugins** | Plugin-dependent models need manual plugin paths | Out of scope for v1 |
| **GPU-specific engines** | Engines built on T4 won't run on A100 without rebuild | By design (TensorRT constraint) |
| **Colab-only workflow** | No CLI, Docker, or CI integration yet | Roadmap item |
| **Single-node surgery** | Only Identity bypass supported | Extensible architecture |

---

## Roadmap

```
2026 Q1          2026 Q2                    2026 Q3                 2026 Q4+
────────────────────────────────────────────────────────────────────────────────
[v0.1 CURRENT]
 Colab benchmark · 5-model suite · JSON trajectories
                  │
                  ▼
              [v0.2] CLI + Docker · local GPU support
                  │
                  ├──────────────────────────────► [v0.3] GitHub Action / CI
                  │                                batch model processing
                  ▼
              [v0.4] Transformer attention repairs · plugin registry
                  │
                  ├──────────────────────────────► Resource-bound auto-tuning
                  ▼
              [v1.0] SaaS / on-prem · approval workflows · team dashboards
```

| Phase | Target | Deliverable |
|---|---|---|
| **v0.1** | Current | Colab benchmark, 5-model suite, JSON trajectories |
| **v0.2** | Q1 2026 | CLI tool, Docker image, local GPU support |
| **v0.3** | Q2 2026 | CI/CD plugin (GitHub Actions), batch model processing |
| **v0.4** | Q2–Q3 2026 | Transformer-specific repairs, plugin registry |
| **v1.0** | Q4 2026+ | SaaS/on-prem platform, approval workflows, team dashboards |

---

## License

GPL-3.0 license

---

## Acknowledgments

Built on the NVIDIA TensorRT ecosystem:

- [TensorRT](https://developer.nvidia.com/tensorrt)
- [ONNX](https://onnx.ai/)
- [ONNX GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
