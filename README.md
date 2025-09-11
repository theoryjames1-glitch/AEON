
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


### TEST1 PSEUDOCODES


# 🧪 Demo: AEON (Delayed + Reversible Lock) vs Fixed LR vs Scheduler

```python
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from datasets import load_dataset
import random
import matplotlib.pyplot as plt

# ---------------------------
# 1. Setup
# ---------------------------
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def make_model():
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 2. Dataset
# ---------------------------
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
examples = [ex["text"] for ex in dataset if len(ex["text"].strip()) > 20]

def get_batch():
    text = random.choice(examples)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    labels = enc["input_ids"].clone()
    enc["labels"] = labels
    return enc

# ---------------------------
# 3. Resonance gate
# ---------------------------
def resonance_gate(loss, prev_loss, step, max_steps, base_tol=0.3, min_tol=0.05):
    if step < max_steps // 2:
        tol = base_tol - (base_tol - min_tol) * (step / (max_steps // 2))
    else:
        tol = min_tol
    if prev_loss is None:
        return True
    return abs(prev_loss - loss) > tol

# ---------------------------
# 4. AEON scheduler (delayed + reversible lock)
# ---------------------------
def update_hyperparams(loss, prev_loss, var_window, alpha, sigma, step, max_steps,
                       best_loss, lock_mode, unlock_tolerance=1.0, lock_tolerance=0.5):
    delta = prev_loss - loss if prev_loss is not None else 0.0

    # --- Lock mode ---
    if lock_mode:
        alpha = max(alpha * 0.95, 8e-6)   # slow LR decay
        sigma = max(sigma * 0.7, 1e-7)    # shrink exploration
        # unlock if loss drifts up too far from best
        if loss > best_loss + unlock_tolerance:
            lock_mode = False
        return alpha, sigma, lock_mode, best_loss

    # --- Normal adaptive mode ---
    if delta > 0:
        alpha *= 1.05
    else:
        alpha *= 0.7

    lr_floor = 1e-5 if step < max_steps // 2 else 2e-5
    alpha = max(lr_floor, min(alpha, 5e-5))

    if len(var_window) > 5 and torch.var(torch.tensor(var_window[-5:])) < 1e-2:
        sigma *= 1.1
    else:
        sigma *= 0.9
    sigma = max(1e-6, min(sigma, 1e-5))

    if loss - prev_loss > 2.0:
        alpha = lr_floor

    if step % 50 == 0:
        if loss > prev_loss:
            alpha = min(alpha * 1.2, 5e-5)
        elif loss < prev_loss:
            alpha = min(alpha * 1.3, 5e-5)

    # --- Trigger delayed lock ---
    if step > max_steps // 3 and loss < best_loss - lock_tolerance:
        best_loss = loss
        lock_mode = True

    return alpha, sigma, lock_mode, best_loss

# ---------------------------
# 5. Training loop
# ---------------------------
def run_experiment(mode="aeon", steps=200):
    model = make_model()
    optimizer = AdamW(model.parameters(), lr=3e-5)

    if mode == "sched":
        scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=50,
            num_training_steps=steps
        )
    else:
        scheduler = None

    alpha, sigma = 3e-5, 1e-6
    loss_ema, var_window = None, []
    losses = []

    best_loss = float("inf")
    lock_mode = False

    for step in range(1, steps+1):
        enc = get_batch()
        outputs = model(**enc)
        loss = outputs.loss

        prev_loss = loss_ema if loss_ema is not None else loss.item()
        loss_ema = 0.9 * prev_loss + 0.1 * loss.item()
        var_window.append(loss.item())

        learn_flag = True
        if mode == "aeon" and not lock_mode:
            learn_flag = resonance_gate(loss.item(), prev_loss, step, steps)

        if learn_flag:
            optimizer.zero_grad()
            loss.backward()

            if mode == "aeon":
                if not lock_mode and len(var_window) > 5 and torch.var(torch.tensor(var_window[-5:])) < 1.0:
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad += sigma * torch.randn_like(p.grad)
                for g in optimizer.param_groups:
                    g["lr"] = alpha

            optimizer.step()

        if mode == "aeon":
            alpha, sigma, lock_mode, best_loss = update_hyperparams(
                loss.item(), prev_loss, var_window, alpha, sigma, step, steps,
                best_loss, lock_mode
            )

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())

        if step % 50 == 0:
            lr_val = optimizer.param_groups[0]["lr"]
            print(f"{mode.upper()} Step {step} | Loss {loss.item():.4f} | LR={lr_val:.2e} | σ={sigma:.1e} | Learn={learn_flag} | Lock={lock_mode}")

    return losses

# ---------------------------
# 6. Run experiments
# ---------------------------
steps = 300
print("Running AEON (delayed + reversible lock)...")
aeon_losses = run_experiment("aeon", steps)

print("Running Fixed LR...")
fixed_losses = run_experiment("fixed", steps)

print("Running Standard Scheduler...")
sched_losses = run_experiment("sched", steps)

# ---------------------------
# 7. Plot
# ---------------------------
plt.figure(figsize=(8,4))
plt.plot(aeon_losses, label="AEON Adaptive (Delayed + Reversible Lock)")
plt.plot(fixed_losses, label="Fixed LR (3e-5)")
plt.plot(sched_losses, label="Cosine Scheduler")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Online LM on WikiText: AEON vs Fixed LR vs Scheduler")
plt.legend()
plt.show()
```

---

## 🔍 What’s New

* **Delayed lock:** AEON only locks after 1/3 of training (after \~100 steps).
* **Soft lock:** LR/σ decay gradually, not instant freeze.
* **Reversible lock:** if loss worsens +1.0 above best, AEON unlocks and resumes exploration.

This should give you the “explore → consolidate → re-explore” behavior.

