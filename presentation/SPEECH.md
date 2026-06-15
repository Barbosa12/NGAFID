# Presentation Speech — CS3315 Final Project
**Aviation Engine Health Diagnosis via Task Decomposition (NGAFID)**
44 slides · target ~12–15 min · concise script per slide

---

### 1 — Title
Good morning. I'm Iván Barbosa. My project tackles aviation engine health diagnosis on the NGAFID dataset, comparing classical machine learning against deep learning across three tasks: anomaly detection, fault classification, and end-to-end diagnosis.

### 2 — Agenda
I'll cover motivation and the problem, the dataset, then results for each of the three tasks, closing with two critical questions.

### 3 — Motivation
General aviation accounts for about 65% of US aviation accidents, with mechanical failures a persistent factor. Condition-based maintenance can catch developing faults early — but only if we can detect and identify them automatically.

### 4 — Problem Statement
I decompose diagnosis into three sub-problems: detect if a flight is anomalous (AD), identify the specific fault type from 19 categories (FC), and combine both end-to-end (E2E).

### 5 — Two Critical Questions
Everything builds toward two questions. **CQ1:** can classical ML with statistical features compete with deep learning — and if not, where and why does it fail? **CQ2:** when we chain the stages, where does the error concentrate — detection or classification?

### 6 — Dataset Overview
NGAFID: 11,446 Cessna-172 flights, 23 sensors at 1 Hz, resampled to 2,048 timesteps. 19 fault classes. AD is near-balanced at 1.04 to 1; FC is severely imbalanced at 38 to 1.

### 7 — Contributions / EDA
My EDA uncovered three structural challenges that shaped every modeling decision: target component sparsity, multi-task coupling, and dual class imbalance.

---

## Exploratory Data Analysis

### 8 — EDA Overview
These three challenges are not incidental — each one directly motivates a feature representation and a modeling choice.

### 9 — Flight Duration
Flight durations range from 10 minutes to over 5 hours, median 88 minutes. Crucially, healthy and anomalous flights have identical duration profiles — so duration is not a confounder.

### 10 — Distribution of Observations
Here's the imbalance. Intake gasket has 2,098 anomalous flights, rocker cover 1,024 — together 56% of all faults. The rarest, baffle rivet, has only 55. That 38-to-1 ratio is why F1-macro, not accuracy, is our metric.

### 11 — Sensor Traces (Repr A)
The raw signal. Healthy in blue, anomalous in gold. At this scale they look nearly identical — the fault signature is a subtle perturbation, not an obvious shift. That's challenge one.

### 12 — Correlation
Inter-sensor correlation. The four cylinder-head sensors correlate above 0.92 because all cylinders share the same combustion. Under a fault, that symmetry breaks — which becomes our key engineered feature.

### 13 — Flight Phase Coupling
The mean flight shows five distinct regimes: taxi, climb, cruise, descent, landing. Each has a completely different sensor distribution — that's multi-task coupling. A single global statistic mixes all five.

### 14 — Sensor Volatility Shift
Which sensors change under a fault? The cylinder thermal sensors — CHT and EGT — show the largest volatility increase, consistent with combustion-related failures. Flight-state sensors barely move.

### 15 — PCA Analysis
PCA confirms challenge one: the dominant components encode the flight envelope — altitude, airspeed — not the fault. The fault signal lives in low-variance components. Global statistics alone can't surface it.

### 16 — Four-Representation Strategy
So I built four representations: **Repr A** raw signal for deep learning; **Repr B** 184 global statistics as baseline; **Repr B-seg** for phase coupling; and **Repr F++**, which adds 40 cross-cylinder difference features to capture exactly that broken symmetry.

### 17 — Representation Dimensions
This is the scale: Repr A is 47,000 values per flight; the engineered representations compress to 184 up to 1,840. Repr F++ is a compact 224. Each trades fidelity against the constraints of classical ML.

---

## Anomaly Detection

### 18 — AD Task
Anomaly detection: binary classification, healthy versus anomalous, within a 2-day maintenance window. Near-balanced, so F1-macro behaves well.

### 19 — AD Models
I evaluated six classical models — Logistic Regression, SVM, XGBoost, Random Forest, an Ensemble, and kNN — across all representations under 5-fold cross-validation.

### 20 — AD Heatmap per Repr
The full grid: models by representations. The Ensemble consistently tops every column. And Repr F++ edges out Repr B — the cross-cylinder features help even here.

### 21 — F1 and AUC by Model
The Ensemble leads on both F1 and AUC. kNN is consistently weakest — Euclidean distance in high-dimensional stat-space is poorly calibrated. Segmentation actually hurts AD, because the anomaly signal is global, not phase-local.

### 22 — Hyperparameter Tuning
I tuned the top-3 with RandomizedSearch. Gains were modest — at most 0.006 F1 — confirming the Ensemble was already near its ceiling.

### 23 — Winner: Ensemble × Repr F++
The AD winner: tuned Ensemble on Repr F++, F1 of 0.703, AUC 0.767. That's within 9.6 points of the deep-learning benchmark.

### 24 — Feature Importance Top-25
What did it learn? The top features are dominated by CHT and EGT sensors — and the cross-cylinder differences appear prominently, validating the Repr F++ design.

### 25 — Sensor Group Importance
Aggregated by group, the model matches the EDA prediction exactly: cylinder thermal sensors dominate, flight-state sensors contribute little. The model learned engine health, not flight trajectory.

### 26 — AD Final Summary
The performance ladder. Best classical 0.703, FNR 0.307 — meaning roughly one in three anomalies is still missed at the default threshold, which matters for safety.

---

## Fault Classification

### 27 — FC Task
Now the harder task: given an anomalous flight, identify which of 19 fault types. Scarce data, extreme imbalance.

### 28 — FC Models
Same six models, but now on 5,602 anomalous flights only, with balanced class weights throughout.

### 29 — FC Heatmap per Repr
The pattern flips from AD. XGBoost dominates, and the absolute numbers are far lower — FC is intrinsically hard.

### 30 — FC F1 by Model and Repr
Here Repr F++ gives a large gain — seven times larger than in AD. The cross-cylinder asymmetry is exactly the signal that tells fault types apart. Segmentation hurts again — too many dimensions for too few samples.

### 31 — FC Tuning
Tuning XGBoost on Repr F++ with RandomizedSearch.

### 32 — Default vs Tuned
Tuning lifts XGBoost from 0.169 to 0.207 — a meaningful 23% gain, much larger than in AD, because regularization matters under severe imbalance.

### 33 — Confusion Matrix
The 19-by-19 confusion matrix. Errors concentrate within physical groups — baffle faults confused with other baffle faults — and the two majority classes act as gravitational sinks.

### 34 — Top Misclassifications
Concretely: most errors are rare classes misclassified as intake gasket or rocker cover — the majority-class absorption effect. This motivates the hierarchical grouping idea.

### 35 — Per-Class Analysis
Per-class F1 versus class size: a strong positive correlation. Classes above 1,000 samples are detected; classes below 150 collapse to near-zero F1. Data scarcity, not the model, is the limit.

### 36 — FC Feature Importance
The importances confirm it: cross-cylinder differential features rank at the top — the explicit signature of which cylinder is failing, something global statistics simply cannot encode.

---

## End-to-End Diagnosis

### 37 — E2E Task
End-to-end: one pipeline, raw flight in, a fault type or "healthy" out — a 20-class problem. Two architectures: cascade, or direct.

### 38 — Option A: Cascade
Cascade: AD first, then FC only on flights flagged anomalous. The strong AD stage acts as a filter, propagating little error.

### 39 — Option B: Direct
Direct: a single 20-class model. No inter-stage propagation, but now healthy competes directly against 19 fault classes.

### 40 — Architecture Comparison
The cascade wins on both F1 — 0.235 versus 0.113 — and on safety: AD-recall of 0.688 versus 0.509. The direct model misses far more faults.

### 41 — Error Decomposition
And here's the answer to CQ2. The dominant error is S2 at 20% — anomaly detected, but wrong fault type. Missed detections are only 15%. The bottleneck is fault classification, not detection.

---

## Final Results

### 42 — Final Results (divider)
Let me close by answering the two critical questions directly.

### 43 — CQ1 Answer
**Can classical ML compete?** Yes — and it depends on the data regime. Classical wins FC and E2E, where data is scarce: RF with SMOTE reaches 0.315, beating our best deep model. It falls short only on AD, where deep CNN-plus-attention hits 0.740 — because abundant balanced data lets attention exploit the raw signal's temporal shape. The verdict: no universal winner — match each task to its regime. The best system is a **hybrid**.

### 44 — CQ2 Answer
**Where does error concentrate?** In fault typing, not detection. The cascade decomposition shows 20% wrong-type errors versus 15% missed detections, and the Oracle analysis confirms 63% of the ceiling is lost to grouping errors. The implication is clear: to improve diagnosis, improve classification — detection is already a strong filter.

---

## Closing line
In summary: physics-anchored feature engineering, a model-specific imbalance strategy, and task decomposition together let classical ML match or beat deep learning everywhere except detection — and the best deployable system pairs deep learning for AD with classical RF-plus-SMOTE for FC. Thank you.
