import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Algebra.Field.GeomSum
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.Analysis.Calculus.Deriv.Basic
import CadenceFlagship.Activation

/-!
# The update rule: the RMS-normalised step and the rectified neuron

Two claims from the 2026-09-15 experiments, stated exactly.

## The RMS-normalised step

After `t ≥ 1` updates the learner has seen the contrasts `g 0, …, g (t-1)` of one synapse
(`g (t-1)` is the latest). With momentum rate `a` and normalising rate `b` it keeps the
bias-corrected exponential averages

  `m_t = (1 - a) ∑_{k<t} a^(t-1-k) g_k / (1 - a^t)`,
  `v_t = (1 - b) ∑_{k<t} b^(t-1-k) g_k^2 / (1 - b^t)`,

and the step is `d_t = η m_t / (√v_t + floor)`. This is `plasticity.py`: `velocity` is the
uncorrected `m`, `second_moment` the uncorrected `v`, `1 - rate^updates` the corrections, and
`floor = 1e-3`. `ema_eq` connects the recursion the code runs to the closed form used here.

The weights `(1 - a) a^(t-1-k) / (1 - a^t)` are nonnegative and sum to one, so `m_t` is a
weighted mean of the contrasts and `v_t` the same weighted mean of their squares. With equal
rates the Cauchy-Schwarz inequality gives `|m_t| ≤ √v_t`, hence `|d_t| ≤ η`. When every
contrast is at most `G` in size, `|m_t| ≤ G` and so `|d_t| ≤ η G / floor`: for contrasts far
below the floor the step scales linearly with the contrast. When the contrast is a constant
`c > 0` the step is exactly `η c / (c + floor)`, and for `c ≥ c₀` it is at least
`η c₀ / (c₀ + floor)`, a fixed fraction of `η` however large the contrast grows.

Remark (no proof). The runs use `a = 0.8` and `b = 0.98`. The bound `|m_t| ≤ √v_t` is proved
here for the equal-rate case `a = b`; the unequal case is checked numerically.

## The rectified neuron

With slope `k`, threshold `θ`, leak zero and the logistic `σ`, the neuron publishes

  `s v = max 0 (σ (k (v - θ)) - σ (-k θ)) / (1 - σ (-k θ))`.

It is zero for `v ≤ 0`, monotone nondecreasing, has derivative zero for `v < 0` (no contrast
passes through a neuron below rest), and has positive derivative for `v > 0`. This is
`CadenceFlagship.activation k θ 0` (`rectified_eq_activation`).
-/

namespace Cadence

open Finset

/-! ### Weighted means -/

section WeightedMean

variable {ι : Type*}

/-- Jensen for the square. With nonnegative weights summing to one, the square of the
weighted mean is at most the weighted mean of the squares. -/
theorem sq_weighted_mean_le (s : Finset ι) (w g : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (h1 : ∑ i ∈ s, w i = 1) :
    (∑ i ∈ s, w i * g i) ^ 2 ≤ ∑ i ∈ s, w i * g i ^ 2 := by
  have h := sum_mul_sq_le_sq_mul_sq s (fun i => Real.sqrt (w i)) (fun i => Real.sqrt (w i) * g i)
  have e1 : ∑ i ∈ s, Real.sqrt (w i) * (Real.sqrt (w i) * g i) = ∑ i ∈ s, w i * g i :=
    sum_congr rfl fun i hi => by rw [← mul_assoc, Real.mul_self_sqrt (hw i hi)]
  have e2 : ∑ i ∈ s, Real.sqrt (w i) ^ 2 = 1 := by
    rw [← h1]
    exact sum_congr rfl fun i hi => Real.sq_sqrt (hw i hi)
  have e3 : ∑ i ∈ s, (Real.sqrt (w i) * g i) ^ 2 = ∑ i ∈ s, w i * g i ^ 2 :=
    sum_congr rfl fun i hi => by rw [mul_pow, Real.sq_sqrt (hw i hi)]
  rw [e1, e2, e3, one_mul] at h
  exact h

/-- The absolute weighted mean is at most the root of the weighted mean of the squares. -/
theorem abs_weighted_mean_le_sqrt (s : Finset ι) (w g : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (h1 : ∑ i ∈ s, w i = 1) :
    |∑ i ∈ s, w i * g i| ≤ Real.sqrt (∑ i ∈ s, w i * g i ^ 2) :=
  Real.abs_le_sqrt (sq_weighted_mean_le s w g hw h1)

/-- A weighted mean of numbers of size at most `G` has size at most `G`. -/
theorem abs_weighted_mean_le (s : Finset ι) (w g : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (h1 : ∑ i ∈ s, w i = 1) {G : ℝ} (hg : ∀ i ∈ s, |g i| ≤ G) :
    |∑ i ∈ s, w i * g i| ≤ G := by
  calc |∑ i ∈ s, w i * g i| ≤ ∑ i ∈ s, |w i * g i| := abs_sum_le_sum_abs _ _
    _ ≤ ∑ i ∈ s, w i * G := by
        refine sum_le_sum fun i hi => ?_
        rw [abs_mul, abs_of_nonneg (hw i hi)]
        exact mul_le_mul_of_nonneg_left (hg i hi) (hw i hi)
    _ = G := by rw [← sum_mul, h1, one_mul]

/-- A weighted mean of numbers at least `c` is at least `c`. -/
theorem le_weighted_mean (s : Finset ι) (w g : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (h1 : ∑ i ∈ s, w i = 1) {c : ℝ} (hg : ∀ i ∈ s, c ≤ g i) :
    c ≤ ∑ i ∈ s, w i * g i := by
  calc c = ∑ i ∈ s, w i * c := by rw [← sum_mul, h1, one_mul]
    _ ≤ ∑ i ∈ s, w i * g i :=
        sum_le_sum fun i hi => mul_le_mul_of_nonneg_left (hg i hi) (hw i hi)

end WeightedMean

/-! ### The bias-corrected exponential averages -/

/-- The recursion the code runs: `ema a g 0 = 0`, `ema a g (t+1) = a * ema a g t + (1 - a) * g t`.
This is `velocity` (with `g` the raw step) and `second_moment` (with `g` its square). -/
noncomputable def ema (a : ℝ) (g : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | t + 1 => a * ema a g t + (1 - a) * g t

/-- Closed form of the recursion. -/
theorem ema_eq (a : ℝ) (g : ℕ → ℝ) (t : ℕ) :
    ema a g t = (1 - a) * ∑ k ∈ range t, a ^ (t - 1 - k) * g k := by
  induction t with
  | zero => simp [ema]
  | succ t ih =>
    have hsum : ∑ k ∈ range t, a ^ (t - k) * g k = a * ∑ k ∈ range t, a ^ (t - 1 - k) * g k := by
      rw [mul_sum]
      refine sum_congr rfl fun k hk => ?_
      have hk' : k < t := mem_range.mp hk
      have : t - k = t - 1 - k + 1 := by omega
      rw [this, pow_succ]
      ring
    rw [show ema a g (t + 1) = a * ema a g t + (1 - a) * g t from rfl, ih, sum_range_succ]
    simp only [Nat.add_sub_cancel, Nat.sub_self, pow_zero, one_mul]
    rw [hsum]
    ring

/-- The bias-corrected weight of the contrast at position `k < t` after `t` updates at rate
`a`. The latest contrast, `k = t - 1`, has weight `(1 - a) / (1 - a ^ t)`. -/
noncomputable def weight (a : ℝ) (t k : ℕ) : ℝ := (1 - a) * a ^ (t - 1 - k) / (1 - a ^ t)

/-- Assumes `0 ≤ a < 1`. -/
theorem weight_nonneg {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a < 1) (t k : ℕ) : 0 ≤ weight a t k := by
  unfold weight
  have h1 : 0 ≤ 1 - a := by linarith
  have h2 : 0 ≤ 1 - a ^ t := by linarith [pow_le_one₀ ha0 ha1.le (n := t)]
  exact div_nonneg (mul_nonneg h1 (pow_nonneg ha0 _)) h2

/-- Assumes `0 ≤ a < 1` and `t ≠ 0`. The weights sum to one (geometric sum). -/
theorem sum_weight {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a < 1) {t : ℕ} (ht : t ≠ 0) :
    ∑ k ∈ range t, weight a t k = 1 := by
  unfold weight
  have hat : a ^ t < 1 := pow_lt_one₀ ha0 ha1 ht
  have hne : 1 - a ^ t ≠ 0 := by linarith
  have hr : ∑ k ∈ range t, a ^ (t - 1 - k) = ∑ k ∈ range t, a ^ k :=
    sum_range_reflect (fun j => a ^ j) t
  rw [← sum_div, ← mul_sum, hr, mul_neg_geom_sum, div_self hne]

/-- The bias-corrected exponential average `m_t` of the contrasts `g 0, …, g (t-1)` at rate
`a`. -/
noncomputable def firstMoment (a : ℝ) (g : ℕ → ℝ) (t : ℕ) : ℝ :=
  ∑ k ∈ range t, weight a t k * g k

/-- The bias-corrected exponential average `v_t` of the squared contrasts at rate `b`. -/
noncomputable def secondMoment (b : ℝ) (g : ℕ → ℝ) (t : ℕ) : ℝ :=
  ∑ k ∈ range t, weight b t k * g k ^ 2

/-- `m_t` is the code's `velocity` divided by its correction `1 - a ^ t`. -/
theorem firstMoment_eq_ema (a : ℝ) (g : ℕ → ℝ) (t : ℕ) :
    firstMoment a g t = ema a g t / (1 - a ^ t) := by
  unfold firstMoment weight
  rw [ema_eq, mul_sum, sum_div]
  exact sum_congr rfl fun k _ => by ring

/-- `v_t` is the code's `second_moment` divided by its correction `1 - b ^ t`. -/
theorem secondMoment_eq_ema (b : ℝ) (g : ℕ → ℝ) (t : ℕ) :
    secondMoment b g t = ema b (fun k => g k ^ 2) t / (1 - b ^ t) := by
  unfold secondMoment weight
  rw [ema_eq, mul_sum, sum_div]
  exact sum_congr rfl fun k _ => by ring

/-- The learner's step for one synapse after `t` updates:
`d_t = η m_t / (√v_t + floor)`. -/
noncomputable def step (η a b floor : ℝ) (g : ℕ → ℝ) (t : ℕ) : ℝ :=
  η * firstMoment a g t / (Real.sqrt (secondMoment b g t) + floor)

/-! ### Equal rates: the step never exceeds `η` -/

/-- Assumes `0 ≤ a < 1` and `t ≠ 0`. With equal rates, `|m_t| ≤ √v_t` for every contrast
sequence `g`. -/
theorem abs_firstMoment_le_sqrt_secondMoment {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a < 1)
    (g : ℕ → ℝ) {t : ℕ} (ht : t ≠ 0) :
    |firstMoment a g t| ≤ Real.sqrt (secondMoment a g t) :=
  abs_weighted_mean_le_sqrt _ _ _ (fun k _ => weight_nonneg ha0 ha1 t k) (sum_weight ha0 ha1 ht)

/-- Assumes `0 ≤ η`, `0 ≤ floor`, `0 ≤ a < 1` and `t ≠ 0`. With equal rates the step is at
most `η` in size, for every contrast sequence `g`. -/
theorem abs_step_le {η a floor : ℝ} (hη : 0 ≤ η) (hfloor : 0 ≤ floor) (ha0 : 0 ≤ a)
    (ha1 : a < 1) (g : ℕ → ℝ) {t : ℕ} (ht : t ≠ 0) :
    |step η a a floor g t| ≤ η := by
  unfold step
  have hm := abs_firstMoment_le_sqrt_secondMoment ha0 ha1 g ht
  have hs : 0 ≤ Real.sqrt (secondMoment a g t) := Real.sqrt_nonneg _
  rcases eq_or_lt_of_le (add_nonneg hs hfloor) with h | h
  · rw [← h, div_zero, abs_zero]
    exact hη
  · rw [abs_div, abs_mul, abs_of_nonneg hη, abs_of_pos h, div_le_iff₀ h]
    calc η * |firstMoment a g t| ≤ η * Real.sqrt (secondMoment a g t) :=
          mul_le_mul_of_nonneg_left hm hη
      _ ≤ η * (Real.sqrt (secondMoment a g t) + floor) :=
          mul_le_mul_of_nonneg_left (by linarith) hη

/-! ### Small contrasts: the step is linear in the contrast -/

/-- Assumes `0 ≤ a < 1` and `t ≠ 0`. If every contrast seen so far is at most `G` in size,
so is `m_t`. -/
theorem abs_firstMoment_le {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a < 1) (g : ℕ → ℝ) {t : ℕ}
    (ht : t ≠ 0) {G : ℝ} (hg : ∀ k < t, |g k| ≤ G) :
    |firstMoment a g t| ≤ G :=
  abs_weighted_mean_le _ _ _ (fun k _ => weight_nonneg ha0 ha1 t k) (sum_weight ha0 ha1 ht)
    fun k hk => hg k (mem_range.mp hk)

/-- Assumes `0 ≤ η`, `0 < floor`, `0 ≤ a < 1` and `t ≠ 0`; the normalising rate `b` is
arbitrary. If every contrast seen so far is at most `G` in size, the step is at most
`η G / floor`: for contrasts far below the floor the step scales linearly with the contrast. -/
theorem abs_step_le_of_abs_le {η a floor : ℝ} (hη : 0 ≤ η) (hfloor : 0 < floor)
    (ha0 : 0 ≤ a) (ha1 : a < 1) (b : ℝ) (g : ℕ → ℝ) {t : ℕ} (ht : t ≠ 0)
    {G : ℝ} (hg : ∀ k < t, |g k| ≤ G) :
    |step η a b floor g t| ≤ η * G / floor := by
  unfold step
  have hG : 0 ≤ G := le_trans (abs_nonneg _) (hg 0 (Nat.pos_of_ne_zero ht))
  have hm := abs_firstMoment_le ha0 ha1 g ht hg
  have hs : 0 ≤ Real.sqrt (secondMoment b g t) := Real.sqrt_nonneg _
  have hden : 0 < Real.sqrt (secondMoment b g t) + floor := by linarith
  rw [abs_div, abs_mul, abs_of_nonneg hη, abs_of_pos hden]
  exact div_le_div₀ (mul_nonneg hη hG) (mul_le_mul_of_nonneg_left hm hη) hfloor (by linarith)

/-! ### Large contrasts of one sign: the step is a fixed fraction of `η` -/

/-- Assumes `0 ≤ a < 1` and `t ≠ 0`. A constant contrast `c` has `m_t = c`. -/
theorem firstMoment_const {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a < 1) (c : ℝ) {t : ℕ} (ht : t ≠ 0) :
    firstMoment a (fun _ => c) t = c := by
  unfold firstMoment
  rw [← sum_mul, sum_weight ha0 ha1 ht, one_mul]

/-- Assumes `0 ≤ b < 1` and `t ≠ 0`. A constant contrast `c` has `v_t = c ^ 2`. -/
theorem secondMoment_const {b : ℝ} (hb0 : 0 ≤ b) (hb1 : b < 1) (c : ℝ) {t : ℕ} (ht : t ≠ 0) :
    secondMoment b (fun _ => c) t = c ^ 2 := by
  unfold secondMoment
  rw [← sum_mul, sum_weight hb0 hb1 ht, one_mul]

/-- Assumes `0 ≤ a < 1`, `0 ≤ b < 1`, `0 ≤ c` and `t ≠ 0`. For a constant contrast `c` the
step is exactly `η c / (c + floor)`, whatever the two rates. -/
theorem step_const {η a b floor : ℝ} (ha0 : 0 ≤ a) (ha1 : a < 1) (hb0 : 0 ≤ b) (hb1 : b < 1)
    {c : ℝ} (hc : 0 ≤ c) {t : ℕ} (ht : t ≠ 0) :
    step η a b floor (fun _ => c) t = η * c / (c + floor) := by
  unfold step
  rw [firstMoment_const ha0 ha1 c ht, secondMoment_const hb0 hb1 c ht, Real.sqrt_sq hc]

/-- Assumes `0 ≤ η`, `0 ≤ floor`, `0 ≤ a < 1`, `0 ≤ b < 1`, `0 < c₀ ≤ c` and `t ≠ 0`. For a
constant contrast `c ≥ c₀` the step is at least `η c₀ / (c₀ + floor)`, a fixed fraction of
`η` that does not shrink as `c` grows. -/
theorem step_const_ge {η a b floor : ℝ} (hη : 0 ≤ η) (hfloor : 0 ≤ floor) (ha0 : 0 ≤ a)
    (ha1 : a < 1) (hb0 : 0 ≤ b) (hb1 : b < 1) {c c₀ : ℝ} (hc₀ : 0 < c₀) (hc : c₀ ≤ c)
    {t : ℕ} (ht : t ≠ 0) :
    η * c₀ / (c₀ + floor) ≤ step η a b floor (fun _ => c) t := by
  rw [step_const ha0 ha1 hb0 hb1 (by linarith) ht]
  rw [div_le_div_iff₀ (by linarith) (by linarith)]
  nlinarith [mul_nonneg (mul_nonneg hη hfloor) (sub_nonneg.mpr hc)]

/-- Assumes `0 ≤ η`, `0 ≤ floor`, `0 ≤ a < 1`, `0 ≤ b < 1`, `0 < c` and `t ≠ 0`. If every
contrast seen so far lies in `[c, G]` (one sign, at least `c`), the step is at least
`η c / (G + floor)`. -/
theorem step_ge_of_bounds {η a b floor : ℝ} (hη : 0 ≤ η) (hfloor : 0 ≤ floor) (ha0 : 0 ≤ a)
    (ha1 : a < 1) (hb0 : 0 ≤ b) (hb1 : b < 1) (g : ℕ → ℝ) {t : ℕ} (ht : t ≠ 0) {c G : ℝ}
    (hc : 0 < c) (hg : ∀ k < t, c ≤ g k ∧ g k ≤ G) :
    η * c / (G + floor) ≤ step η a b floor g t := by
  unfold step
  have hwa : ∀ k ∈ range t, 0 ≤ weight a t k := fun k _ => weight_nonneg ha0 ha1 t k
  have hwb : ∀ k ∈ range t, 0 ≤ weight b t k := fun k _ => weight_nonneg hb0 hb1 t k
  have hcG : c ≤ G := le_trans (hg 0 (Nat.pos_of_ne_zero ht)).1 (hg 0 (Nat.pos_of_ne_zero ht)).2
  have hm : c ≤ firstMoment a g t :=
    le_weighted_mean _ _ _ hwa (sum_weight ha0 ha1 ht) fun k hk => (hg k (mem_range.mp hk)).1
  have hv_lo : c ^ 2 ≤ secondMoment b g t :=
    le_weighted_mean _ _ _ hwb (sum_weight hb0 hb1 ht) fun k hk => by
      have h := hg k (mem_range.mp hk)
      exact pow_le_pow_left₀ hc.le h.1 2
  have hv_hi : secondMoment b g t ≤ G ^ 2 := by
    unfold secondMoment
    calc ∑ k ∈ range t, weight b t k * g k ^ 2 ≤ ∑ k ∈ range t, weight b t k * G ^ 2 := by
          refine sum_le_sum fun k hk => ?_
          have h := hg k (mem_range.mp hk)
          exact mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (by linarith) h.2 2) (hwb k hk)
      _ = G ^ 2 := by rw [← sum_mul, sum_weight hb0 hb1 ht, one_mul]
  have hs_lo : c ≤ Real.sqrt (secondMoment b g t) := by
    rw [show c = Real.sqrt (c ^ 2) from (Real.sqrt_sq hc.le).symm]
    exact Real.sqrt_le_sqrt hv_lo
  have hs_hi : Real.sqrt (secondMoment b g t) ≤ G :=
    (Real.sqrt_le_left (by linarith)).mpr hv_hi
  exact div_le_div₀ (mul_nonneg hη (by linarith)) (mul_le_mul_of_nonneg_left hm hη)
    (by linarith) (by linarith)

/-! ### The rectified neuron -/

section Neuron

open Real

/-- The rest emission `σ(-k θ)`, what the raw logistic emits at potential zero. -/
noncomputable def restEmission (k θ : ℝ) : ℝ := Real.sigmoid (-(k * θ))

/-- The rectified neuron with leak zero:
`s v = max 0 (σ (k (v - θ)) - σ (-k θ)) / (1 - σ (-k θ))`. -/
noncomputable def rectified (k θ : ℝ) (v : ℝ) : ℝ :=
  max 0 (Real.sigmoid (k * (v - θ)) - restEmission k θ) / (1 - restEmission k θ)

theorem restEmission_lt_one (k θ : ℝ) : restEmission k θ < 1 := Real.sigmoid_lt_one _

theorem one_sub_restEmission_pos (k θ : ℝ) : 0 < 1 - restEmission k θ := by
  linarith [restEmission_lt_one k θ]

/-- The rectified neuron is the library's activation with leak zero. -/
theorem rectified_eq_activation (k θ v : ℝ) :
    rectified k θ v = CadenceFlagship.activation k θ 0 v := by
  unfold rectified CadenceFlagship.activation CadenceFlagship.hinge CadenceFlagship.offset
    CadenceFlagship.restEmission restEmission
  simp only [zero_div, zero_mul, add_zero]
  rw [max_comm]
  ring

/-- Assumes `0 < k`. At or below rest the neuron publishes zero. -/
theorem rectified_eq_zero {k θ : ℝ} (hk : 0 < k) {v : ℝ} (hv : v ≤ 0) : rectified k θ v = 0 := by
  unfold rectified
  have h : Real.sigmoid (k * (v - θ)) ≤ restEmission k θ := by
    unfold restEmission
    apply Real.sigmoid_le
    nlinarith [mul_nonpos_of_nonneg_of_nonpos hk.le hv]
  rw [max_eq_left (by linarith), zero_div]

/-- Assumes `0 < k`. The rectified neuron is monotone nondecreasing. -/
theorem rectified_monotone {k θ : ℝ} (hk : 0 < k) : Monotone (rectified k θ) := by
  intro x y hxy
  unfold rectified
  apply div_le_div_of_nonneg_right _ (one_sub_restEmission_pos k θ).le
  apply max_le_max le_rfl
  apply sub_le_sub_right
  apply Real.sigmoid_le
  exact mul_le_mul_of_nonneg_left (sub_le_sub_right hxy θ) hk.le

/-- Assumes `0 < k`. Below rest the neuron has derivative zero. -/
theorem hasDerivAt_rectified_of_neg {k θ : ℝ} (hk : 0 < k) {v : ℝ} (hv : v < 0) :
    HasDerivAt (rectified k θ) 0 v := by
  have hev : rectified k θ =ᶠ[nhds v] fun _ => (0 : ℝ) := by
    filter_upwards [Iio_mem_nhds hv] with x hx
    exact rectified_eq_zero hk (le_of_lt hx)
  exact (hasDerivAt_const v (0 : ℝ)).congr_of_eventuallyEq hev

/-- Assumes `0 < k`. Below rest the derivative is zero: no contrast passes through a neuron
whose potential is below rest. -/
theorem deriv_rectified_of_neg {k θ : ℝ} (hk : 0 < k) {v : ℝ} (hv : v < 0) :
    deriv (rectified k θ) v = 0 :=
  (hasDerivAt_rectified_of_neg hk hv).deriv

/-- Assumes `0 < k`. Above rest the neuron is the re-based logistic, with derivative
`k σ' (k (v - θ)) / (1 - σ (-k θ))`. -/
theorem hasDerivAt_rectified_of_pos {k θ : ℝ} (hk : 0 < k) {v : ℝ} (hv : 0 < v) :
    HasDerivAt (rectified k θ)
      (Real.sigmoid (k * (v - θ)) * (1 - Real.sigmoid (k * (v - θ))) * k
        / (1 - restEmission k θ)) v := by
  have hlin : HasDerivAt (fun x => k * (x - θ)) k v := by
    simpa using ((hasDerivAt_id v).sub_const θ).const_mul k
  have hσ : HasDerivAt (fun x => Real.sigmoid (k * (x - θ)) - restEmission k θ)
      (Real.sigmoid (k * (v - θ)) * (1 - Real.sigmoid (k * (v - θ))) * k) v :=
    ((Real.hasDerivAt_sigmoid _).comp v hlin).sub_const _
  have hbr := hσ.div_const (1 - restEmission k θ)
  have hev : rectified k θ =ᶠ[nhds v]
      fun x => (Real.sigmoid (k * (x - θ)) - restEmission k θ) / (1 - restEmission k θ) := by
    filter_upwards [Ioi_mem_nhds hv] with x hx
    unfold rectified
    have h : restEmission k θ ≤ Real.sigmoid (k * (x - θ)) := by
      unfold restEmission
      apply Real.sigmoid_le
      have hx' : 0 < x := hx
      nlinarith [mul_pos hk hx']
    rw [max_eq_right (by linarith)]
  exact hbr.congr_of_eventuallyEq hev

/-- Assumes `0 < k`. Above rest the derivative is positive. -/
theorem deriv_rectified_pos {k θ : ℝ} (hk : 0 < k) {v : ℝ} (hv : 0 < v) :
    0 < deriv (rectified k θ) v := by
  rw [(hasDerivAt_rectified_of_pos hk hv).deriv]
  have h1 := Real.sigmoid_pos (k * (v - θ))
  have h2 := Real.sigmoid_lt_one (k * (v - θ))
  have h3 := one_sub_restEmission_pos k θ
  exact div_pos (mul_pos (mul_pos h1 (by linarith)) hk) h3

end Neuron

end Cadence
