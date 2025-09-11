
# 🌌 Theory of AEON (Adaptive Evolutionary Online Neural Network)

## 1. Motivation

* Traditional training = offline, static hyperparams, epochs.
* AEON = **continuous, feedback-driven, adaptive**.
* Inspiration: control systems, DSP, adaptive filters, resonance.

---

## 2. Formal System Definition

### 2.1 State Vector

$$
x_t =
\begin{bmatrix}
\theta_t, h_t, \alpha_t, \mu_t, \sigma_t, r_t, m_t
\end{bmatrix}
$$

### 2.2 Dynamics

$$
\theta_{t+1} = \theta_t - u_t + \sigma_t D(\theta_t)\eta_t + K_r r_t
$$

* **Plant** = model parameters
* **Controller** = optimizer update $u_t$
* **Scheduler** = adaptive law updating $\alpha_t, \mu_t, \sigma_t$
* **Resonance** = filtered error memory $m_t$
* **Recurrence** = state feedback $r_t$

---

## 3. Feedback Channels

* Loss/error: $\ell_t$
* Reward (optional RL): $R_t$
* Derived signals: $\Delta \ell_t, v_t$
* Resonance $m_t$, Recurrence $r_t$

All flow back into the scheduler.

---

## 4. Adaptive Laws

1. **Learning Rate ($\alpha_t$)**: increases if loss decreases, decreases otherwise.
2. **Momentum ($\mu_t$)**: adapts to reward/loss variance.
3. **Exploration Noise ($\sigma_t$)**: increases during plateau, decays during progress.
4. **Resonance Gate ($m_t$)**: balances plasticity vs. stability.
5. **Recurrence ($r_t$)**: maintains long-range dynamics.

---

## 5. Stability Principles

* Clip hyperparams (safe bounds).
* Decay resonance ($|\rho|<1$).
* Gate updates if system is stable (resonance frozen).
* Lyapunov heuristic: expected $\ell_t$ should decrease.

---

## 6. Interpretations

* **Cybernetic:** AEON = feedback control loop.
* **DSP:** Optimizer = adaptive filter; hyperparams = filter coefficients.
* **Reinforcement:** Reward channels steer adaptation.
* **Resonance:** Long-term memory prevents catastrophic forgetting.

---

## 7. Practical Consequences

* Training becomes **streaming**, no epochs.
* Hyperparams evolve online.
* Plasticity–stability balance is intrinsic.
* Exploration vs exploitation handled by dither + resonance.
* Works in SFT and RL contexts seamlessly.

---

✨ **In one line:**
**AEON = learning reframed as an adaptive control system, where models evolve continuously via closed-loop feedback, adaptive resonance, and evolutionary dither.**

---


# ✅ AEON Evaluation Problems

## A. Core Sanity & Ablations

1. **Online LM vs Fixed LR (Baseline)**

* Setup: Stream WikiText (train\[:1%]) line-by-line. AEON vs AdamW with fixed LR/WD, no dither, no gating.
* Metrics: online loss (EMA), time-to-improvement (steps to 90% of best loss), area under learning curve (AULC).
* Pass: AEON ≥ baseline on AULC; faster time-to-improvement.

2. **Ablate Each Knob**

* Setup: AEON full vs {no σ, no LR adapt, no WD adapt, no resonance gate}.
* Metrics: AULC; stability violations (loss spikes > Xσ); % steps gated (“gate duty cycle”).
* Pass: Full AEON best or tied; each ablation degrades at least one metric.

3. **Noise Robustness**

* Setup: Add Gaussian noise to gradients at fixed SNR levels.
* Metrics: loss variance, recovery time after spikes.
* Pass: AEON maintains lower variance and faster recovery than fixed LR.

---

## B. Non-Stationarity / Concept Drift

4. **Topic Shift (In-Domain)**

* Setup: WikiText stream by sections: biographies → math → sports (hard switches every N steps).
* Metrics: adaptation half-life (steps to regain 90% of pre-shift performance), forgetting score (loss on a small replay buffer from previous topic).
* Pass: Shorter half-life and lower forgetting vs baseline.

5. **Domain Shift (Out-of-Domain)**

* Setup: WikiText → small StackExchange slice (technical Q\&A).
* Metrics: half-life; % LR bumps after shift; σ pulse amplitude; gate duty cycle change.
* Pass: Clear LR/σ response at shift and quicker convergence than baseline.

6. **Gradual Drift**

* Setup: Mix WikiText topics with slowly changing mixture weights.
* Metrics: tracking error (loss difference vs oracle that always trains on the current mixture).
* Pass: AEON tracks closer to oracle than fixed LR.

---

## C. Stability–Plasticity (Resonance Gate)

7. **Plateau then Shock**

* Setup: Hold distribution constant for 500 steps (plateau), then introduce a new topic.
* Metrics: % steps gated during plateau; gate release latency on shock; overshoot (max loss spike after gate opens).
* Pass: High gating during plateau; rapid, controlled un-gating on shock; limited overshoot.

8. **Rare Revisit (Catastrophic Forgetting Probe)**

* Setup: Mostly topic A; occasionally insert a sample from topic B last seen long ago.
* Metrics: B-sample loss over time; impact on A immediately after a B update (interference).
* Pass: With resonance + small replay (e.g., 8 samples), B loss improves without large A regression.

---

## D. Exploration / Dither

9. **Stuck Region Escape**

* Setup: Freeze LR adapt; compare σ∈{0, AEON σ} on a mini task that often plateaus (short contexts).
* Metrics: # of plateaus escaped (loss drop > δ within K steps); extra compute overhead.
* Pass: AEON σ escapes more plateaus with minimal overhead (<5%).

10. **Directional vs Isotropic Dither (Optional)**

* Setup: Replace isotropic noise with grad-aligned perturbations occasionally.
* Metrics: improvement per step; variance.
* Pass: Any structured dither improves improvement/variance tradeoff.

---

## E. Data Curriculum as Control

11. **Difficulty-Aware Sampling**

* Setup: Maintain a small hardness score per example (recent loss). AEON biases sampling to “hard-but-not-impossible.”
* Metrics: AULC; steadier LR profile; fewer oscillations.
* Pass: Better AULC and reduced oscillation vs uniform sampling.

---

## F. Efficiency / PEFT Compatibility

12. **LoRA/QLoRA Compatibility**

* Setup: Train with small LoRA adapters only; AEON adapts LR/σ/WD on adapter params.
* Metrics: AULC; memory/compute; generation quality.
* Pass: AEON retains benefits under PEFT with lower memory.

13. **Sparse Updating (Top-k Grad Gate)**

* Setup: Update only parameters with top-k gradient norms; AEON controls k via loss variance.
* Metrics: tokens/sec, AULC.
* Pass: Similar AULC with higher throughput vs dense updates.

---

## G. Robustness & Safety

14. **Outlier Injection**

* Setup: Insert adversarial junk lines at 1–5% rate.
* Metrics: spike magnitude; recovery time; automatic LR back-off.
* Pass: AEON auto-backs off LR/boosts WD and recovers faster.

15. **Latency Sensitivity**

* Setup: Enforce per-step compute budgets (e.g., limit AEON bookkeeping).
* Metrics: overhead %, retained AULC.
* Pass: <5–10% overhead with negligible loss.

---

# 📊 Standardized Logging (for every test)

* Loss (raw + EMA), LR α, exploration σ, weight decay, gradient norm, gate duty cycle, plateau detector state.
* Events: shift points, gate open/close, LR/σ spikes.

# 📈 Primary Scores to Report

* **AULC** (higher is better).
* **Adaptation Half-Life** after shift.
* **Forgetting Score** on replay buffer.
* **Stability**: loss spike rate/size.
* **Overhead**: % compute vs baseline.


