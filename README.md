# WaymoCoverageAnalyzer


> **On 200 real Waymo Motion V1.3.1 scenarios, one cluster holds 94× more data than the rarest.
> One outlier scenario has a peak agent speed of 68 m/s and acceleration 282× the corpus mean —
> the kind of annotation artifact that silently poisons a training distribution unless you surface it first.**

<p align="center">
  <img src="https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/docs/images/vehicle-3D-labeling-example.png" width="30%" alt="Waymo vehicle 3D label"/>
  <img src="https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/docs/images/pedestrian-3D-labeling-example.png" width="30%" alt="Waymo pedestrian 3D label"/>
  <img src="https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/docs/images/cyclist-3D-labeling-example.png" width="30%" alt="Waymo cyclist 3D label"/>
</p>
<p align="center"><em>Waymo Open Dataset — vehicle · pedestrian · cyclist labels ingested and clustered by this pipeline</em></p>

A pipeline that ingests Waymo Open Dataset Motion `.tfrecord` files, extracts 15 kinematic features
per scenario via a **C++20 engine** (pybind11 / Eigen3), clusters scenarios with KMeans + PCA to
surface coverage gaps, and renders a 4-panel interactive **Plotly dashboard** — offline, laptop-CPU,
no GPU required.

---

## Why This Exists

Waymo's **Release Evaluation** team needs confidence that test scenarios cover the full kinematic
distribution of real-world driving before each software release. The **Multiverse** team generates
counterfactual variants to stress-test edge cases that are underrepresented in collected data.
Both workflows share the same bottleneck: fast, correct extraction of per-agent kinematics at scale.

This project mirrors the production C++/Python split exactly:
- **C++** owns the performance-critical compute kernel — speed, longitudinal and lateral acceleration,
  jerk, curvature, path length, and point-mass TTC across all 91 timesteps per agent, zero Python
  overhead in the inner loop
- **Python** owns proto parsing, pipeline orchestration, clustering, and visualization
- The benchmark table quantifies the speedup

---

## Real Findings — Waymo Motion Dataset V1.3.1, Shard 0

*200 scenarios · `training.tfrecord-00000-of-01000` · 430 MB · **200/200 parsed, 0 errors***

### Coverage distribution (8-cluster KMeans on 15-feature vectors)

| Cluster | n | Share | Character |
|--------:|--:|------:|-----------|
| C0 | 94 | 47.0% | Dense low-speed urban — mean 2.7 m/s, ~54 agents, stop-and-go |
| C6 | 54 | 27.0% | Moderate-speed urban — mean 3.4 m/s, elevated pedestrian mix |
| C2 | 26 | 13.0% | Higher-speed scenes — mean 8.6 m/s, sparser agent fields |
| C1 | 20 | 10.0% | Crowded intersections — mean **172 agents**, packed crossings |
| C5 |  3 |  1.5% | Pedestrian-dense — mean **66 pedestrians**, slow (0.8 m/s) |
| **C3** | **1** | **0.5%** | **Gap** — 8 agents, 9.8 m/s, TTC **0.38 s** (genuine near-miss) |
| **C4** | **1** | **0.5%** | **Gap** — max speed **68 m/s**, accel **282× mean** (annotation outlier) |
| C7 |  1 |  0.5% | Curvature outlier — mean curvature **20,000× global average** |

**The largest cluster holds 94× more scenarios than the smallest.**
C0 alone contains 47× more data than the three smallest clusters combined.

### What the gaps reveal

**Cluster 3 — genuine near-miss on a sparse road.**
Only 8 agents (vs corpus mean 65.8), moving at 9.8 m/s (2.8× corpus mean). TTC = 0.38 s — the
only cluster with a non-zero mean TTC, meaning agents are genuinely converging, not merely
collocated stationary objects. This is the rarest scenario type in this shard: a small number of
fast-moving vehicles on a clear road with a real interaction.

**Cluster 4 — annotation artifact that would corrupt a training run.**
Max speed 67.9 m/s (244 km/h). Mean acceleration 5.3 m/s² — **282× the corpus mean**.
Max lateral acceleration 506 m/s² (**12× the next-highest cluster**). Almost certainly a tracking
dropout or label-propagation error. A coverage analyzer exists precisely to surface this before it
enters a model's training distribution undetected.

**Cluster 7 — curvature outlier.**
Mean curvature 20,042 m⁻¹ vs corpus mean 485 m⁻¹. A single scenario where the raw curvature
formula (`|v × a| / |v|³`) amplifies near-zero speed, flagging it as a distinct distribution edge
case worth manual inspection.

### Speed and interaction structure

No highway cluster appears in this shard — consistent with Waymo's published dataset composition
being predominantly urban/suburban. 84% of scenarios cluster in the 1.8–3.4 m/s range. The
moderate-speed cluster (C2, mean 8.6 m/s) is the only one approaching arterial speeds.
Interaction density averages **1.74 agents within 10 m**, peaking at 4.3 in the pedestrian-dense
cluster — confirming the corpus is weighted toward dense urban intersection scenarios.

> **On TTC**: The point-mass TTC approximation (`t = -r⃗·v⃗ / |v⃗|²`) returns 0 s whenever two
> collocated stationary agents share a position — a known artifact in dense Waymo annotation where
> parked vehicles may overlap in map coordinates. TTC is useful as a **relative ranking signal**
> across scenarios, not as an absolute collision threshold. Cluster 3 is the exception: its
> TTC of 0.38 s represents a verified converging-agent interaction.

---

## Architecture

```
 Waymo Motion .tfrecord
        │
        ▼
  parser.py  ─── TFRecordDataset (lazy TF import)
  (Python)   ─── scenario_pb2  (compiled stub, works on all platforms)
        │           field numbers confirmed via wire-format scan
        ▼ list[ScenarioData]   (Pydantic, frozen, ParseStats error tracking)
        │
  features.py
  (Python)
        │                pybind11 — zero-copy numpy arrays via buffer protocol
        │          ┌──────────────────────────────────────────────────┐
        │          │         waymo_kinematics.so  (C++20 / Eigen3)    │
        │          │                                                  │
        │          │  KinematicsEngine::compute(AgentTrajectory)      │
        │          │  ├─ speed  =  √(vx²+vy²)  per valid timestep    │
        │          │  ├─ long. accel  = Δspeed/dt  (signed)          │
        │          │  ├─ lateral accel  =  |v×a| / |v|  (=speed²·κ) │
        │          │  ├─ jerk  = Δaccel/dt                           │
        │          │  ├─ curvature  =  |v×a| / |v|³                 │
        │          │  ├─ circular heading variance  (wrap-safe)      │
        │          │  ├─ path_length,  displacement                  │
        │          │  └─ TTC  (point-mass, min over all pairs)       │
        │          └──────────────────────────────────────────────────┘
        │
        ▼ list[ScenarioFeatureVector]   (15 features, Pydantic)
        │
  clustering.py ── StandardScaler → KMeans(k=8) → PCA(2D)
  (Python / sklearn)
        │
        ▼ ClusteringResult   (labels, gap indices, PCA centers)
        │
  perturbation.py ── speed-scale + heading-rotate + Euler re-integration
  (Python)       ──  counterfactual variants for gap scenarios
        │
        ▼
  dashboard.py ── 4-panel Plotly HTML
                  1. PCA scatter coloured by cluster (gaps amber)
                  2. Cluster size bar chart
                  3. Feature heatmap (Z-score normalised, clustered rows)
                  4. Perturbation overlay (original + variants)
```

---

## Benchmark

MacBook Pro M2, 16 GB RAM. Each scenario: ~66 agents × 91 timesteps (real Waymo data).

| Scenarios | C++ Engine (s) | NumPy Baseline (s) | Speedup |
|----------:|---------------:|-------------------:|--------:|
|        10 |          0.003 |              0.021 |   ~7×   |
|        50 |          0.014 |              0.105 |   ~7×   |
|       100 |          0.028 |              0.211 |   ~7×   |

> M2 NEON SIMD flatters NumPy on short inner loops; expect **3–5× on x86 Linux** with AVX2.
> Reproduce: `python scripts/benchmark.py` → `outputs/benchmark.json`.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<your-handle>/waymo-coverage-analyzer.git
cd waymo-coverage-analyzer

# 2. Create a virtual environment (Python 3.11 recommended)
python3.11 -m venv venv && source venv/bin/activate

# 3. System dependencies
brew install eigen cmake          # macOS
# sudo apt install libeigen3-dev cmake  # Ubuntu/Debian

# 4. Python dependencies
pip install -r requirements.txt

# 5. Build C++ shared library + run gtests
bash build.sh

# 6. Run the demo — no real data required
python scripts/demo.py            # generates outputs/demo_dashboard.html

# 7. Run on real data  (Waymo Open Dataset registration required)
gcloud storage cp \
  "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/training/training.tfrecord-00000-of-01000" \
  data/
waymo-coverage analyze data/training.tfrecord-00000-of-01000 --max-scenarios 200
waymo-coverage cluster outputs/features.csv --n-clusters 8
open outputs/dashboard.html
```

> The `scenario/` format stores raw `Scenario` proto bytes per record — use this, not `tf_example/`.
> The WOD SDK is **not required**. `waymo_coverage/scenario_pb2.py` is a compiled proto stub whose
> field numbers were confirmed by raw wire-format inspection of the actual tfrecord.

---

## CLI Reference

```bash
# Extract features from a tfrecord → outputs/features.csv
waymo-coverage analyze data/training.tfrecord-00000-of-01000 \
    --max-scenarios 200 \
    --output-dir outputs/

# Cluster + build dashboard
waymo-coverage cluster outputs/features.csv \
    --n-clusters 8 \
    --output-dir outputs/

# Generate counterfactual variants for one scenario
waymo-coverage perturb data/training.tfrecord-00000-of-01000 \
    --scenario-id "28fe360951cf98d6" \
    --n-variants 5 \
    --output-dir outputs/

# Serve dashboard locally
waymo-coverage serve --dashboard-path outputs/dashboard.html --port 8050
```

---

## Project Structure

```
waymo-coverage-analyzer/
├── .github/workflows/ci.yml       # Ubuntu + macOS: cmake → gtest → pytest → demo
├── CMakeLists.txt                 # builds waymo_kinematics.so + gtest binary
├── build.sh                       # one-command build (cmake + gtest)
├── requirements.txt
├── pyproject.toml
├── setup.py                       # CMake integration for pip install -e .
│
├── cpp/
│   ├── include/kinematics.h       # KinematicFeatures struct, AgentTrajectory, KinematicsEngine
│   ├── src/kinematics.cpp         # compute() + compute_ttc()  —  C++20, Eigen3
│   ├── src/bindings.cpp           # pybind11 module, buffer-protocol numpy arrays
│   └── tests/test_kinematics.cpp  # 7 gtests: constant-velocity, braking, circular motion, TTC
│
├── waymo_coverage/
│   ├── scenario_pb2.py            # compiled proto stub (wire-confirmed field numbers)
│   ├── _proto_src/scenario.proto  # minimal Scenario/Track/ObjectState proto source
│   ├── parser.py                  # TFRecordDataset → list[ScenarioData], ParseStats
│   ├── features.py                # C++ engine + NumPy baseline (benchmarking)
│   ├── clustering.py              # StandardScaler → KMeans → PCA
│   ├── perturbation.py            # speed-scale + heading-rotate + Euler integration
│   ├── dashboard.py               # 4-panel Plotly HTML
│   └── cli.py                     # typer: analyze / cluster / perturb / serve
│
├── tests/                         # 39 pytest cases (no real data required)
│   ├── conftest.py                # synthetic ScenarioData fixtures
│   ├── test_parser.py
│   ├── test_features.py
│   ├── test_clustering.py
│   └── test_perturbation.py
│
├── scripts/
│   ├── demo.py                    # 200 synthetic scenarios → full pipeline end-to-end
│   └── benchmark.py              # C++ vs NumPy wall-clock table
│
└── data/                          # place .tfrecord files here (gitignored)
```

---

## Tests

```bash
# C++ unit tests (7 cases: constant velocity, braking, circular motion, TTC)
./build/kinematics_tests

# Python tests (39 cases — no real data required)
pytest tests/ -v --cov=waymo_coverage

# End-to-end smoke test
python scripts/demo.py
```

CI runs all three on every push to `main`, on **Ubuntu and macOS in parallel**.
Test cases exercise edge cases explicitly: all-invalid trajectories, single-timestep inputs,
zero-velocity agents, and circular-motion lateral acceleration verified against `v²/r`.

---

## Engineering Notes

**C++ correctness decisions worth reading:**

- **Circular heading variance** (`1 - |mean(e^{iθ})|`) instead of raw angle variance — heading values
  near ±π would give enormous raw variance for a nearly straight agent. The circular formulation is
  wrap-safe by construction.
- **Longitudinal vs lateral acceleration split** — `mean_acceleration` differentiates scalar speed
  magnitude (longitudinal); `mean_lateral_acceleration = |v×a|/|v|` captures centripetal load
  independently. A sharp turn at constant speed has zero longitudinal acceleration but non-zero
  lateral. Both feed the clustering.
- **TTC as point-mass closest-approach time** — `t = -r⃗·v⃗/|v⃗|²` returns time to minimum
  separation, not physical collision time (no bounding-box term). Documented in the source as an
  intentional approximation.
- **`std::vector<bool>` footgun** — `std::vector<bool>` is a bitfield specialisation and cannot
  construct `std::span<const bool>`. The `AgentTrajectory` doc comment explicitly states the
  contract; tests use `std::vector<uint8_t>` with `reinterpret_cast`.

**Proto parsing without the WOD SDK:**

The official `waymo-open-dataset-tf-2-12-0` wheel has no macOS ARM build. Rather than requiring a
Linux VM, `waymo_coverage/scenario_pb2.py` is compiled from a minimal `scenario.proto` whose field
numbers were confirmed by decoding raw protobuf wire bytes from the actual tfrecord — not from
documentation. The stub covers only the fields consumed by `parser.py`; unknown fields in the real
data (map geometry, object metadata) are silently ignored by the protobuf wire format.

---

## Extending the C++ Engine

To add a new kinematic feature (e.g. `longitudinal_jerk_rms`):

1. Add the field to `KinematicFeatures` in [cpp/include/kinematics.h](cpp/include/kinematics.h)
2. Compute it in `KinematicsEngine::compute()` in [cpp/src/kinematics.cpp](cpp/src/kinematics.cpp)
3. Expose it via `def_readwrite` in [cpp/src/bindings.cpp](cpp/src/bindings.cpp)
4. Add a gtest in [cpp/tests/test_kinematics.cpp](cpp/tests/test_kinematics.cpp) with an analytic
   expected value (e.g. constant-jerk trajectory where RMS is known)
5. Add the field to `ScenarioFeatureVector` in [waymo_coverage/features.py](waymo_coverage/features.py)
   and append its value to `feature_vector`
6. `bash build.sh`

## Authors

Built by **Param Madan** and **Dhwanil Panchani**.

- Param Madan 
- Dhwanil Panchani 

