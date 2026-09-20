import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.Analysis.Calculus.MeanValue
import CadenceFlagship.Settling

/-!
# The library's neuron: a rectified, re-based sigmoid

The library's neuron publishes

  `act v = max r 0 / (1 - ρ) + ℓ · min r 0 / ρ`,   `r = σ(λ (v - θ)) - ρ`,   `ρ = σ(-λ θ)`,

with slope `λ`, threshold `θ`, leak `ℓ` and the logistic `σ`. Rest is exact zero. The lemmas
here give its Lipschitz constant `(λ/4) · max (1/(1-ρ)) (ℓ/ρ)`, its range bound `max 1 ℓ`, and
the values for the learning model (`λ = 1`, `θ = 0`, `ℓ = 1/10`): constant `1/2`, range bound
`1`. Through `settle_contracting` this certifies every net whose largest absolute incoming
effective weight sum is below `2` under the learning model.
-/

namespace CadenceFlagship

open Real

/-- The logistic function is `1/4`-Lipschitz. -/
theorem sigmoid_lipschitz : LipschitzWith (1 / 4 : NNReal) Real.sigmoid := by
  refine lipschitzWith_of_nnnorm_deriv_le differentiable_sigmoid fun x => ?_
  rw [Real.deriv_sigmoid]
  have h0 : 0 ≤ Real.sigmoid x * (1 - Real.sigmoid x) :=
    mul_nonneg (Real.sigmoid_nonneg x) (by linarith [Real.sigmoid_le_one x])
  have h1 : Real.sigmoid x * (1 - Real.sigmoid x) ≤ 1 / 4 := by
    nlinarith [sq_nonneg (2 * Real.sigmoid x - 1)]
  rw [← NNReal.coe_le_coe, coe_nnnorm, Real.norm_eq_abs, abs_of_nonneg h0]
  simpa using h1

/-- A two-slope hinge: slope `c₁` above zero and `c₂` below. -/
def hinge (c₁ c₂ : ℝ) (x : ℝ) : ℝ := c₁ * max x 0 + c₂ * min x 0

theorem hinge_lipschitz {c₁ c₂ : ℝ} (h₁ : 0 ≤ c₁) (h₂ : 0 ≤ c₂) :
    LipschitzWith (Real.toNNReal (max c₁ c₂)) (hinge c₁ c₂) := by
  refine LipschitzWith.of_dist_le_mul fun x y => ?_
  rw [Real.coe_toNNReal _ (le_max_of_le_left h₁), Real.dist_eq, Real.dist_eq]
  have hm₁ : c₁ ≤ max c₁ c₂ := le_max_left _ _
  have hm₂ : c₂ ≤ max c₁ c₂ := le_max_right _ _
  unfold hinge
  rcases le_or_gt 0 x with hx | hx <;> rcases le_or_gt 0 y with hy | hy
  · rw [max_eq_left hx, max_eq_left hy, min_eq_right hx, min_eq_right hy]
    rw [abs_le]
    constructor <;> nlinarith [abs_nonneg (x - y), le_abs_self (x - y), neg_abs_le (x - y)]
  · rw [max_eq_left hx, min_eq_right hx, max_eq_right hy.le, min_eq_left hy.le]
    rw [abs_of_nonneg (by nlinarith [mul_nonneg h₁ hx, mul_nonneg h₂ (neg_nonneg.mpr hy.le)]),
      abs_of_nonneg (by linarith)]
    nlinarith [mul_nonneg (sub_nonneg.mpr hm₁) hx, mul_nonneg (sub_nonneg.mpr hm₂) (neg_nonneg.mpr hy.le)]
  · rw [max_eq_right hx.le, min_eq_left hx.le, max_eq_left hy, min_eq_right hy]
    rw [abs_of_nonpos (by nlinarith [mul_nonneg h₂ (neg_nonneg.mpr hx.le), mul_nonneg h₁ hy]),
      abs_of_nonpos (by linarith)]
    nlinarith [mul_nonneg (sub_nonneg.mpr hm₁) hy, mul_nonneg (sub_nonneg.mpr hm₂) (neg_nonneg.mpr hx.le)]
  · rw [max_eq_right hx.le, max_eq_right hy.le, min_eq_left hx.le, min_eq_left hy.le]
    rw [abs_le]
    constructor <;> nlinarith [abs_nonneg (x - y), le_abs_self (x - y), neg_abs_le (x - y)]

/-- The rest emission `ρ = σ(-λ θ)`, subtracted so that rest publishes exactly zero. -/
noncomputable def restEmission (lam θ : ℝ) : ℝ := Real.sigmoid (-(lam * θ))

/-- The raw offset `σ(λ (v - θ)) - ρ`. -/
noncomputable def offset (lam θ : ℝ) (v : ℝ) : ℝ :=
  Real.sigmoid (lam * (v - θ)) - restEmission lam θ

/-- The library's activation: the offset, rectified above rest and leaking below it. -/
noncomputable def activation (lam θ leak : ℝ) (v : ℝ) : ℝ :=
  hinge (1 / (1 - restEmission lam θ)) (leak / restEmission lam θ) (offset lam θ v)

theorem restEmission_pos (lam θ : ℝ) : 0 < restEmission lam θ := Real.sigmoid_pos _

theorem restEmission_lt_one (lam θ : ℝ) : restEmission lam θ < 1 := Real.sigmoid_lt_one _

/-- Rest publishes exactly zero. -/
theorem activation_zero (lam θ leak : ℝ) : activation lam θ leak 0 = 0 := by
  have h : offset lam θ 0 = 0 := by
    unfold offset restEmission
    ring_nf
  simp [activation, hinge, h]

/-- The offset is `λ/4`-Lipschitz. -/
theorem offset_lipschitz (lam θ : ℝ) (hlam : 0 ≤ lam) :
    LipschitzWith (Real.toNNReal (lam / 4)) (offset lam θ) := by
  refine LipschitzWith.of_dist_le_mul fun a b => ?_
  rw [Real.coe_toNNReal _ (by positivity)]
  unfold offset
  rw [Real.dist_eq, sub_sub_sub_cancel_right, ← Real.dist_eq, Real.dist_eq a b]
  calc dist (Real.sigmoid (lam * (a - θ))) (Real.sigmoid (lam * (b - θ)))
      ≤ (1 / 4 : NNReal) * dist (lam * (a - θ)) (lam * (b - θ)) := sigmoid_lipschitz.dist_le_mul _ _
    _ = lam / 4 * |a - b| := by
        rw [Real.dist_eq, ← mul_sub, abs_mul, abs_of_nonneg hlam]
        push_cast
        ring_nf

/-- **Theorem 10a.** The library's activation is Lipschitz with constant
`(λ/4) · max (1/(1-ρ)) (ℓ/ρ)`. -/
theorem activation_lipschitz (lam θ leak : ℝ) (hlam : 0 ≤ lam) (hleak : 0 ≤ leak) :
    LipschitzWith
      (Real.toNNReal (lam / 4 * max (1 / (1 - restEmission lam θ)) (leak / restEmission lam θ)))
      (activation lam θ leak) := by
  have hρ0 := restEmission_pos lam θ
  have hρ1 := restEmission_lt_one lam θ
  have hc₁ : 0 ≤ 1 / (1 - restEmission lam θ) := by
    have : 0 < 1 - restEmission lam θ := by linarith
    positivity
  have hc₂ : 0 ≤ leak / restEmission lam θ := div_nonneg hleak hρ0.le
  have hmax : 0 ≤ max (1 / (1 - restEmission lam θ)) (leak / restEmission lam θ) :=
    le_max_of_le_left hc₁
  refine LipschitzWith.of_dist_le_mul fun a b => ?_
  rw [Real.coe_toNNReal _ (mul_nonneg (by positivity) hmax)]
  calc dist (activation lam θ leak a) (activation lam θ leak b)
      ≤ Real.toNNReal (max (1 / (1 - restEmission lam θ)) (leak / restEmission lam θ))
          * dist (offset lam θ a) (offset lam θ b) :=
        (hinge_lipschitz hc₁ hc₂).dist_le_mul _ _
    _ ≤ Real.toNNReal (max (1 / (1 - restEmission lam θ)) (leak / restEmission lam θ))
          * (Real.toNNReal (lam / 4) * dist a b) := by
        gcongr
        exact (offset_lipschitz lam θ hlam).dist_le_mul a b
    _ = lam / 4 * max (1 / (1 - restEmission lam θ)) (leak / restEmission lam θ) * dist a b := by
        rw [Real.coe_toNNReal _ hmax, Real.coe_toNNReal _ (by positivity)]
        ring

/-- **Theorem 10b.** The activation lies in `[-ℓ, 1]`, so `|act v| ≤ max 1 ℓ`. -/
theorem activation_abs_le (lam θ leak : ℝ) (hleak : 0 ≤ leak) (v : ℝ) :
    |activation lam θ leak v| ≤ max 1 leak := by
  have hρ0 := restEmission_pos lam θ
  have hρ1 := restEmission_lt_one lam θ
  have h1ρ : 0 < 1 - restEmission lam θ := by linarith
  have hσ0 : 0 ≤ Real.sigmoid (lam * (v - θ)) := Real.sigmoid_nonneg _
  have hσ1 : Real.sigmoid (lam * (v - θ)) ≤ 1 := Real.sigmoid_le_one _
  have hlo : -(restEmission lam θ) ≤ offset lam θ v := by unfold offset; linarith
  have hhi : offset lam θ v ≤ 1 - restEmission lam θ := by unfold offset; linarith
  unfold activation hinge
  rcases le_or_gt 0 (offset lam θ v) with h | h
  · rw [max_eq_left h, min_eq_right h, mul_zero, add_zero, abs_of_nonneg (by positivity)]
    calc 1 / (1 - restEmission lam θ) * offset lam θ v
        ≤ 1 / (1 - restEmission lam θ) * (1 - restEmission lam θ) :=
          mul_le_mul_of_nonneg_left hhi (by positivity)
      _ = 1 := by field_simp
      _ ≤ max 1 leak := le_max_left _ _
  · rw [max_eq_right h.le, min_eq_left h.le, mul_zero, zero_add]
    have hneg : leak / restEmission lam θ * offset lam θ v ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (div_nonneg hleak hρ0.le) h.le
    rw [abs_of_nonpos hneg]
    calc -(leak / restEmission lam θ * offset lam θ v)
        = leak / restEmission lam θ * (-(offset lam θ v)) := by ring
      _ ≤ leak / restEmission lam θ * restEmission lam θ :=
          mul_le_mul_of_nonneg_left (by linarith) (div_nonneg hleak hρ0.le)
      _ = leak := by field_simp
      _ ≤ max 1 leak := le_max_right _ _

section LearningModel

/-- The learning model: slope `1`, threshold `0`, leak `1/10`. -/
noncomputable def learningActivation : ℝ → ℝ := activation 1 0 (1 / 10)

theorem restEmission_learning : restEmission 1 0 = 1 / 2 := by
  unfold restEmission
  simp [Real.sigmoid_zero]

/-- **Theorem 10c.** The learning model's neuron is `1/2`-Lipschitz. -/
theorem learningActivation_lipschitz : LipschitzWith (1 / 2 : NNReal) learningActivation := by
  have h := activation_lipschitz 1 0 (1 / 10) (by norm_num) (by norm_num)
  rw [restEmission_learning] at h
  have hc : Real.toNNReal (1 / 4 * max (1 / (1 - 1 / 2)) (1 / 10 / (1 / 2))) = (1 / 2 : NNReal) := by
    rw [show (1 : ℝ) / 4 * max (1 / (1 - 1 / 2)) (1 / 10 / (1 / 2)) = 1 / 2 by norm_num]
    rw [← NNReal.coe_inj, Real.coe_toNNReal _ (by norm_num)]
    push_cast
    ring
  rw [hc] at h
  exact h

theorem learningActivation_abs_le (v : ℝ) : |learningActivation v| ≤ 1 := by
  have h := activation_abs_le 1 0 (1 / 10) (by norm_num) v
  rwa [show max (1 : ℝ) (1 / 10) = 1 by norm_num] at h

theorem learningActivation_zero : learningActivation 0 = 0 := activation_zero _ _ _

/-- **Corollary (design rule).** Under the learning model, any net whose absolute incoming
effective weight sums stay below `2` settles by a certified contraction, for every
`0 < dt ≤ 1`. -/
theorem learning_model_certified {n : ℕ} {dt : ℝ} {W : Matrix (Fin n) (Fin n) ℝ} {ρ : ℝ}
    {c : Fin n → ℝ} (hdt0 : 0 < dt) (hdt1 : dt ≤ 1) (hρ : RowMassLE W ρ) (hρ0 : 0 ≤ ρ)
    (hρ2 : ρ < 2) :
    ContractingWith (settleRate dt (1 / 2) ρ) (settle dt W learningActivation c) := by
  have hq : ((1 / 2 : NNReal) : ℝ) * ρ < 1 := by
    push_cast
    linarith
  exact settle_contracting hdt0 hdt1 learningActivation_lipschitz hρ hρ0 hq

end LearningModel

end CadenceFlagship
