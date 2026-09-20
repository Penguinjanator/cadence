import CadenceFlagship.Settling
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Fixed budgets and conditional convergence certificates

A finite iteration budget does not by itself establish convergence. These results
quantify the step count needed for one supplied contraction certificate. They do
not certify arbitrary trained weights, negative target nudges, temporal anchors,
or the Python solver. `PatchNet` measures its full equilibrium residual and
rejects parameter updates when a required phase exhausts its budget above tolerance.

The certificate below bounds distance to a fixed point. It is not the same
quantity as the differential-equation residual reported by the Python runtime.

The arithmetic. `CadenceFlagship/Settling.lean` certifies a settling step as a contraction
with rate `q < 1` (`settle_contracting`) and gives the a priori certificate
`dist (f^[k] v) v* ≤ dist v (f v) * q ^ k / (1 - q)` (`apriori_bound`). Writing
`r₀ = dist v (f v) / (1 - q)`, the certificate after `k` steps is `q ^ k * r₀`, and

  `q ^ k * r₀ ≤ tol  ↔  k ≥ log (tol / r₀) / log q`

for `0 < q < 1` and `0 < tol, r₀` (`pow_mul_le_iff_log`). So a budget below that threshold
cannot certify the tolerance (`certificate_gt_of_budget_lt`). The certificate is an upper
bound, so this alone does not put the state outside tolerance; the scalar contraction
`x ↦ q x` realises it exactly (`scalar_outside_tolerance`), which is the transient-contrast
hypothesis in its sharpest form: for that map a budget below the threshold leaves the state
outside tolerance.
-/

namespace Cadence

open CadenceFlagship

/-- Assumes `0 < q < 1`, `0 < r₀` and `0 < tol`. The geometric certificate `q ^ k * r₀`
is within `tol` exactly when `k ≥ log (tol / r₀) / log q`. -/
theorem pow_mul_le_iff_log {q r₀ tol : ℝ} (hq0 : 0 < q) (hq1 : q < 1) (hr : 0 < r₀)
    (htol : 0 < tol) (k : ℕ) :
    q ^ k * r₀ ≤ tol ↔ Real.log (tol / r₀) / Real.log q ≤ k := by
  have hlq : Real.log q < 0 := Real.log_neg hq0 hq1
  rw [div_le_iff_of_neg hlq, ← Real.log_pow, Real.log_le_log_iff (pow_pos hq0 k)
    (div_pos htol hr), le_div_iff₀ hr]

/-- Assumes `0 < q < 1`, `0 < r₀` and `0 < tol`. A budget `k` below
`log (tol / r₀) / log q` leaves the certificate above `tol`. -/
theorem tol_lt_pow_mul_of_budget_lt {q r₀ tol : ℝ} (hq0 : 0 < q) (hq1 : q < 1) (hr : 0 < r₀)
    (htol : 0 < tol) {k : ℕ} (hk : (k : ℝ) < Real.log (tol / r₀) / Real.log q) :
    tol < q ^ k * r₀ := by
  by_contra h
  push Not at h
  exact absurd ((pow_mul_le_iff_log hq0 hq1 hr htol k).mp h) (not_le.mpr hk)

section Settling

variable {n : ℕ} {K : NNReal} {f : (Fin n → ℝ) → (Fin n → ℝ)}

/-- The a priori certificate after `k` settling steps, as `q ^ k * r₀` with
`r₀ = dist v (f v) / (1 - q)`. -/
theorem residual_le_pow_mul (hf : ContractingWith K f) (v : Fin n → ℝ) (k : ℕ) :
    dist (f^[k] v) (equilibrium hf) ≤ (K : ℝ) ^ k * (dist v (f v) / (1 - K)) := by
  calc dist (f^[k] v) (equilibrium hf) ≤ dist v (f v) * (K : ℝ) ^ k / (1 - K) :=
        apriori_bound hf v k
    _ = (K : ℝ) ^ k * (dist v (f v) / (1 - K)) := by ring

/-- Conditional budget certificate. Assumes `0 < K`, `0 < tol` and a first movement
`dist v (f v) > 0`. The a priori certificate of a `k`-step budget is within `tol` exactly
when `k ≥ log (tol / r₀) / log K`, `r₀ = dist v (f v) / (1 - K)`. -/
theorem certificate_le_iff (hf : ContractingWith K f) (hK : 0 < (K : ℝ)) (v : Fin n → ℝ)
    (hv : 0 < dist v (f v)) {tol : ℝ} (htol : 0 < tol) (k : ℕ) :
    (K : ℝ) ^ k * (dist v (f v) / (1 - K)) ≤ tol ↔
      Real.log (tol / (dist v (f v) / (1 - K))) / Real.log K ≤ k :=
  pow_mul_le_iff_log hK hf.1 hv' htol k
where
  hv' : 0 < dist v (f v) / (1 - K) := div_pos hv hf.one_sub_K_pos

/-- Contrapositive budget certificate. Assumes `0 < K`, `0 < tol` and
`dist v (f v) > 0`. A budget below `log (tol / r₀) / log K` leaves the a priori certificate
above `tol`: the budget cannot certify the tolerance. -/
theorem certificate_gt_of_budget_lt (hf : ContractingWith K f) (hK : 0 < (K : ℝ))
    (v : Fin n → ℝ) (hv : 0 < dist v (f v)) {tol : ℝ} (htol : 0 < tol) {k : ℕ}
    (hk : (k : ℝ) < Real.log (tol / (dist v (f v) / (1 - K))) / Real.log K) :
    tol < (K : ℝ) ^ k * (dist v (f v) / (1 - K)) :=
  tol_lt_pow_mul_of_budget_lt hK hf.1 (div_pos hv hf.one_sub_K_pos) htol hk

end Settling

/-! ### The scalar contraction realises the certificate exactly -/

theorem iterate_mul (q x : ℝ) (k : ℕ) : (fun y => q * y)^[k] x = q ^ k * x := by
  induction k with
  | zero => simp
  | succ k ih => rw [Function.iterate_succ_apply', ih]; ring

/-- Scalar sharpness example. Assumes `0 < q < 1`, `x ≠ 0` and `0 < tol`. For
the scalar contraction `x ↦ q x`, whose equilibrium is `0`, a budget `k` below
`log (tol / |x|) / log q` leaves the state outside tolerance. -/
theorem scalar_outside_tolerance {q x tol : ℝ} (hq0 : 0 < q) (hq1 : q < 1) (hx : x ≠ 0)
    (htol : 0 < tol) {k : ℕ} (hk : (k : ℝ) < Real.log (tol / |x|) / Real.log q) :
    tol < |(fun y => q * y)^[k] x - 0| := by
  rw [iterate_mul, sub_zero, abs_mul, abs_of_pos (pow_pos hq0 k)]
  exact tol_lt_pow_mul_of_budget_lt hq0 hq1 (abs_pos.mpr hx) htol hk

end Cadence
