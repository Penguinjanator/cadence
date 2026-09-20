import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Finite temporal overlap in a scalar quadratic example

The current value `z` minimizes a supplied positive quadratic energy plus
an overlap penalty with the preceding value `c`. The preceding value is a
fixed boundary during minimization; it becomes the next boundary only after
that solve. With `a > 0` and finite nonnegative strength `k`, the exact repair
is `k / (a + k) * c`. Repeated unforced repairs therefore fade to zero.

This is an exact convex scalar example, not a theorem about every nonlinear
Cadence trajectory. It supplies no learned relevance criterion, basin
protection, biological identification, floating-point error bound, or formal
refinement of a numerical solver. New external forcing, parameter learning
and multiple minima are excluded. A zero input with a nonzero intrinsic
preferred state would instead require centering the coordinate on that state.
-/

set_option autoImplicit false

namespace CadenceMission.TemporalOverlap

noncomputable section

def energy (a k c z : ℝ) : ℝ := a * z ^ 2 / 2 + k * (z - c) ^ 2 / 2

def rate (a k : ℝ) : ℝ := k / (a + k)

def repair (a k c : ℝ) : ℝ := rate a k * c

def iterate (a k : ℝ) : Nat → ℝ → ℝ
  | 0, c => c
  | n + 1, c => repair a k (iterate a k n c)

/-- Finite overlap with a positive intrinsic curvature gives a strict rate. -/
theorem rate_bounds {a k : ℝ} (ha : 0 < a) (hk : 0 ≤ k) :
    0 ≤ rate a k ∧ rate a k < 1 := by
  have hden : 0 < a + k := by linarith
  constructor
  · exact div_nonneg hk hden.le
  · exact (div_lt_one hden).mpr (by linarith)

/-- Completing the square identifies the exact minimizer without assuming it. -/
theorem energy_gap {a k : ℝ} (ha : 0 < a) (hk : 0 ≤ k) (c z : ℝ) :
    energy a k c z - energy a k c (repair a k c) =
      (a + k) / 2 * (z - repair a k c) ^ 2 := by
  have hden : a + k ≠ 0 := ne_of_gt (by linarith)
  unfold energy repair rate
  field_simp [hden]
  ring

/-- The supplied overlap rule is the global minimizer of this scalar energy. -/
theorem repair_minimizes {a k : ℝ} (ha : 0 < a) (hk : 0 ≤ k) (c z : ℝ) :
    energy a k c (repair a k c) ≤ energy a k c z := by
  have hcoef : 0 ≤ (a + k) / 2 := by positivity
  have hnonneg := mul_nonneg hcoef (sq_nonneg (z - repair a k c))
  have hgap := energy_gap ha hk c z
  linarith

/-- Positive intrinsic curvature makes the minimum unique at each fixed boundary. -/
theorem repair_unique {a k : ℝ} (ha : 0 < a) (hk : 0 ≤ k) (c z : ℝ)
    (h : energy a k c z = energy a k c (repair a k c)) : z = repair a k c := by
  have hcoef : 0 < (a + k) / 2 := by positivity
  have hzero : (a + k) / 2 * (z - repair a k c) ^ 2 = 0 := by
    rw [← energy_gap ha hk c z, h, sub_self]
  have hsquare := (mul_eq_zero.mp hzero).resolve_left (ne_of_gt hcoef)
  exact sub_eq_zero.mp (sq_eq_zero_iff.mp hsquare)

/-- Each new boundary is exactly the geometrically attenuated original value. -/
theorem iterate_closed_form (a k c : ℝ) (n : Nat) :
    iterate a k n c = rate a k ^ n * c := by
  induction n with
  | zero => simp [iterate]
  | succ n ih =>
    simp only [iterate, repair, ih, pow_succ]
    ring

/-- The separation of two histories fades by the same known factor. -/
theorem history_separation {a k : ℝ} (ha : 0 < a) (hk : 0 ≤ k)
    (c d : ℝ) (n : Nat) :
    |iterate a k n c - iterate a k n d| = rate a k ^ n * |c - d| := by
  have hq := (rate_bounds ha hk).1
  rw [iterate_closed_form, iterate_closed_form, ← mul_sub, abs_mul,
      abs_of_nonneg (pow_nonneg hq n)]

/-- Every nonzero value shrinks strictly after one unforced repair. -/
theorem nonzero_strictly_shrinks {a k c : ℝ} (ha : 0 < a) (hk : 0 ≤ k)
    (hc : c ≠ 0) : |repair a k c| < |c| := by
  obtain ⟨hq0, hq1⟩ := rate_bounds ha hk
  unfold repair
  rw [abs_mul, abs_of_nonneg hq0]
  have habs : 0 < |c| := abs_pos.mpr hc
  nlinarith

/-- This convex example has no permanent nonzero memory under repeated repair. -/
theorem iterate_tends_to_zero {a k : ℝ} (ha : 0 < a) (hk : 0 ≤ k) (c : ℝ) :
    Filter.Tendsto (fun n : Nat => iterate a k n c) Filter.atTop (nhds 0) := by
  obtain ⟨hq0, hq1⟩ := rate_bounds ha hk
  have h := (tendsto_pow_atTop_nhds_zero_of_lt_one hq0 hq1).mul_const c
  simpa only [iterate_closed_form, zero_mul] using h

end

end CadenceMission.TemporalOverlap
