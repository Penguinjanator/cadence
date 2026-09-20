import Mathlib.Topology.MetricSpace.Contracting
import Mathlib.Topology.MetricSpace.Pseudo.Pi
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Mul

/-!
# Settling: the certificate, adaptive depth, silence, bounded damage

A patch net with `n` neurons holds a potential vector `v : Fin n → ℝ`. One settling step
moves every potential a fraction `dt` of the way toward its synaptic input plus stimulus:

  `v i ← (1 - dt) * v i + dt * (∑ j, W i j * act (v j) + c i)`.

The bias is absorbed into the stimulus `c`. The activation `act` is any Lipschitz map; the
library's rectified sigmoid is treated in `Activation.lean`.

The metric is the sup norm on `Fin n → ℝ` (Mathlib's product metric). With `ρ` bounding every
neuron's absolute incoming weight sum and `L` the Lipschitz constant of the activation, a
settling step is a contraction with constant `q = 1 - dt + dt * L * ρ` whenever `L * ρ < 1`
and `0 < dt ≤ 1`. The remaining statements read Banach's theorem the way the library uses it:
a unique equilibrium, an a priori and an a posteriori error certificate, the equilibrium's
Lipschitz dependence on the stimulus (warm starts), rest as the equilibrium of silence, and a
bound on how far a lesion moves the equilibrium.
-/

namespace CadenceFlagship

open Matrix Finset

variable {n : ℕ}

/-- One settling step of a patch net. -/
def settle (dt : ℝ) (W : Matrix (Fin n) (Fin n) ℝ) (act : ℝ → ℝ) (c : Fin n → ℝ)
    (v : Fin n → ℝ) : Fin n → ℝ :=
  fun i => (1 - dt) * v i + dt * (∑ j, W i j * act (v j) + c i)

/-- `ρ` bounds every neuron's absolute incoming synaptic weight sum (the row mass). -/
def RowMassLE (W : Matrix (Fin n) (Fin n) ℝ) (ρ : ℝ) : Prop :=
  ∀ i, ∑ j, |W i j| ≤ ρ

/-- The contraction constant of a settling step, as a nonnegative real. -/
noncomputable def settleRate (dt L ρ : ℝ) : NNReal :=
  Real.toNNReal (1 - dt + dt * (L * ρ))

theorem settleRate_coe {dt L ρ : ℝ} (hdt0 : 0 ≤ dt) (hdt1 : dt ≤ 1) (hL : 0 ≤ L)
    (hρ : 0 ≤ ρ) : ((settleRate dt L ρ : NNReal) : ℝ) = 1 - dt + dt * (L * ρ) := by
  unfold settleRate
  rw [Real.coe_toNNReal]
  have : 0 ≤ dt * (L * ρ) := mul_nonneg hdt0 (mul_nonneg hL hρ)
  linarith

theorem settleRate_lt_one {dt L ρ : ℝ} (hdt0 : 0 < dt) (hdt1 : dt ≤ 1) (hL : 0 ≤ L)
    (hρ : 0 ≤ ρ) (hq : L * ρ < 1) : settleRate dt L ρ < 1 := by
  rw [← NNReal.coe_lt_coe, settleRate_coe hdt0.le hdt1 hL hρ, NNReal.coe_one]
  have := mul_lt_mul_of_pos_left hq hdt0
  linarith

/-- The pointwise difference of two settling steps. -/
theorem settle_sub_settle (dt : ℝ) (W : Matrix (Fin n) (Fin n) ℝ) (act : ℝ → ℝ)
    (c : Fin n → ℝ) (v w : Fin n → ℝ) (i : Fin n) :
    settle dt W act c v i - settle dt W act c w i
      = (1 - dt) * (v i - w i) + dt * ∑ j, W i j * (act (v j) - act (w j)) := by
  simp only [settle, mul_sub, Finset.sum_sub_distrib]
  ring

/-- A settling step shrinks sup-norm distances by the factor `1 - dt + dt * L * ρ`. -/
theorem dist_settle_le {dt : ℝ} {L : NNReal} {ρ : ℝ} {W : Matrix (Fin n) (Fin n) ℝ}
    {act : ℝ → ℝ} {c : Fin n → ℝ} (hdt0 : 0 ≤ dt) (hdt1 : dt ≤ 1)
    (hact : LipschitzWith L act) (hρ : RowMassLE W ρ) (hρ0 : 0 ≤ ρ) (v w : Fin n → ℝ) :
    dist (settle dt W act c v) (settle dt W act c w)
      ≤ (1 - dt + dt * ((L : ℝ) * ρ)) * dist v w := by
  have hD : 0 ≤ dist v w := dist_nonneg
  have h1dt : 0 ≤ 1 - dt := by linarith
  have hLρ : 0 ≤ (L : ℝ) * ρ := mul_nonneg L.coe_nonneg hρ0
  have hcoef : 0 ≤ 1 - dt + dt * ((L : ℝ) * ρ) := by
    have := mul_nonneg hdt0 hLρ
    linarith
  rw [dist_pi_le_iff (mul_nonneg hcoef hD)]
  intro i
  have hterm : ∀ j, |W i j * (act (v j) - act (w j))| ≤ |W i j| * ((L : ℝ) * dist v w) := by
    intro j
    rw [abs_mul]
    refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
    calc |act (v j) - act (w j)| = dist (act (v j)) (act (w j)) := (Real.dist_eq _ _).symm
      _ ≤ L * dist (v j) (w j) := hact.dist_le_mul _ _
      _ ≤ L * dist v w :=
          mul_le_mul_of_nonneg_left (dist_le_pi_dist v w j) L.coe_nonneg
  have hsum : |∑ j, W i j * (act (v j) - act (w j))| ≤ ρ * ((L : ℝ) * dist v w) := by
    calc |∑ j, W i j * (act (v j) - act (w j))|
        ≤ ∑ j, |W i j * (act (v j) - act (w j))| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ j, |W i j| * ((L : ℝ) * dist v w) := Finset.sum_le_sum (fun j _ => hterm j)
      _ = (∑ j, |W i j|) * ((L : ℝ) * dist v w) := by rw [Finset.sum_mul]
      _ ≤ ρ * ((L : ℝ) * dist v w) :=
          mul_le_mul_of_nonneg_right (hρ i) (mul_nonneg L.coe_nonneg hD)
  have hi : |v i - w i| ≤ dist v w := by
    rw [← Real.dist_eq]; exact dist_le_pi_dist v w i
  calc dist (settle dt W act c v i) (settle dt W act c w i)
      = |(1 - dt) * (v i - w i) + dt * ∑ j, W i j * (act (v j) - act (w j))| := by
        rw [Real.dist_eq, settle_sub_settle]
    _ ≤ |(1 - dt) * (v i - w i)| + |dt * ∑ j, W i j * (act (v j) - act (w j))| :=
        abs_add_le _ _
    _ = (1 - dt) * |v i - w i| + dt * |∑ j, W i j * (act (v j) - act (w j))| := by
        rw [abs_mul, abs_mul, abs_of_nonneg h1dt, abs_of_nonneg hdt0]
    _ ≤ (1 - dt) * dist v w + dt * (ρ * ((L : ℝ) * dist v w)) :=
        add_le_add (mul_le_mul_of_nonneg_left hi h1dt) (mul_le_mul_of_nonneg_left hsum hdt0)
    _ = (1 - dt + dt * ((L : ℝ) * ρ)) * dist v w := by ring

/-- **Theorem 1 (settling certificate).** With a Lipschitz activation, a bounded row mass,
`0 < dt ≤ 1` and `L * ρ < 1`, the settling step is a contraction. -/
theorem settle_contracting {dt : ℝ} {L : NNReal} {ρ : ℝ} {W : Matrix (Fin n) (Fin n) ℝ}
    {act : ℝ → ℝ} {c : Fin n → ℝ} (hdt0 : 0 < dt) (hdt1 : dt ≤ 1)
    (hact : LipschitzWith L act) (hρ : RowMassLE W ρ) (hρ0 : 0 ≤ ρ)
    (hq : (L : ℝ) * ρ < 1) :
    ContractingWith (settleRate dt L ρ) (settle dt W act c) := by
  refine ⟨settleRate_lt_one hdt0 hdt1 L.coe_nonneg hρ0 hq, ?_⟩
  refine LipschitzWith.of_dist_le_mul fun v w => ?_
  rw [settleRate_coe hdt0.le hdt1 L.coe_nonneg hρ0]
  exact dist_settle_le hdt0.le hdt1 hact hρ hρ0 v w

section Equilibrium

variable {K : NNReal} {f : (Fin n → ℝ) → (Fin n → ℝ)}

/-- The equilibrium of a contracting settling map: its unique fixed point. -/
noncomputable def equilibrium (hf : ContractingWith K f) : Fin n → ℝ :=
  ContractingWith.fixedPoint f hf

theorem equilibrium_isFixedPt (hf : ContractingWith K f) : f (equilibrium hf) = equilibrium hf :=
  hf.fixedPoint_isFixedPt

/-- **Corollary 1.** Any fixed point of a contracting settling map is the equilibrium. -/
theorem equilibrium_unique (hf : ContractingWith K f) {x : Fin n → ℝ} (hx : f x = x) :
    x = equilibrium hf :=
  hf.fixedPoint_unique hx

/-- **Theorem 2a (a priori certificate).** After `k` settling steps from `v`, the distance to
the equilibrium is at most the first movement times `K ^ k / (1 - K)`. -/
theorem apriori_bound (hf : ContractingWith K f) (v : Fin n → ℝ) (k : ℕ) :
    dist (f^[k] v) (equilibrium hf) ≤ dist v (f v) * (K : ℝ) ^ k / (1 - K) :=
  hf.apriori_dist_iterate_fixedPoint_le v k

/-- **Theorem 2b (a posteriori certificate, certified release).** The remaining distance to
the equilibrium is bounded by the last observed movement divided by `1 - K`. This is what a
tolerance-based stopping rule certifies. -/
theorem aposteriori_bound (hf : ContractingWith K f) (v : Fin n → ℝ) (k : ℕ) :
    dist (f^[k] v) (equilibrium hf) ≤ dist (f^[k] v) (f^[k + 1] v) / (1 - K) :=
  hf.aposteriori_dist_iterate_fixedPoint_le v k

end Equilibrium

section WarmStart

variable {dt : ℝ} {W : Matrix (Fin n) (Fin n) ℝ} {act : ℝ → ℝ} {K : NNReal}

/-- Two settling maps that differ only in their stimulus differ pointwise by at most
`dt` times the stimulus distance. -/
theorem dist_settle_stimulus (hdt0 : 0 ≤ dt) (c c' : Fin n → ℝ) (z : Fin n → ℝ) :
    dist (settle dt W act c z) (settle dt W act c' z) ≤ dt * dist c c' := by
  rw [dist_pi_le_iff (mul_nonneg hdt0 dist_nonneg)]
  intro i
  have h : settle dt W act c z i - settle dt W act c' z i = dt * (c i - c' i) := by
    simp only [settle]; ring
  rw [Real.dist_eq, h, abs_mul, abs_of_nonneg hdt0]
  refine mul_le_mul_of_nonneg_left ?_ hdt0
  rw [← Real.dist_eq]; exact dist_le_pi_dist c c' i

/-- **Theorem 3a.** The equilibrium is Lipschitz in the stimulus: a stimulus change of size
`δ` moves the equilibrium by at most `dt * δ / (1 - K)`. -/
theorem equilibrium_lipschitz_stimulus {c c' : Fin n → ℝ}
    (hf : ContractingWith K (settle dt W act c)) (hf' : ContractingWith K (settle dt W act c'))
    (hdt0 : 0 ≤ dt) :
    dist (equilibrium hf) (equilibrium hf') ≤ dt * dist c c' / (1 - K) :=
  hf.fixedPoint_lipschitz_in_map hf' (fun z => dist_settle_stimulus hdt0 c c' z)

/-- **Theorem 3b (adaptive depth).** Warm-starting the new stimulus's settling from the old
equilibrium, `k` steps leave an error of at most `dt * δ * K ^ k / (1 - K)`: the work a
decision costs is governed by how much the input changed, and an unchanged input costs no
settling at all. -/
theorem warm_start_bound {c c' : Fin n → ℝ}
    (hf : ContractingWith K (settle dt W act c)) (hf' : ContractingWith K (settle dt W act c'))
    (hdt0 : 0 ≤ dt) (k : ℕ) :
    dist ((settle dt W act c')^[k] (equilibrium hf)) (equilibrium hf')
      ≤ dt * dist c c' * (K : ℝ) ^ k / (1 - K) := by
  have h1 : dist (equilibrium hf) (settle dt W act c' (equilibrium hf)) ≤ dt * dist c c' := by
    have := dist_settle_stimulus (W := W) (act := act) hdt0 c c' (equilibrium hf)
    rwa [equilibrium_isFixedPt hf] at this
  have hK1 : 0 < 1 - (K : ℝ) := hf.one_sub_K_pos
  calc dist ((settle dt W act c')^[k] (equilibrium hf)) (equilibrium hf')
      ≤ dist (equilibrium hf) (settle dt W act c' (equilibrium hf)) * (K : ℝ) ^ k / (1 - K) :=
        apriori_bound hf' _ k
    _ ≤ dt * dist c c' * (K : ℝ) ^ k / (1 - K) :=
        div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right h1 (by positivity)) hK1.le

end WarmStart

section Silence

variable {dt : ℝ} {W : Matrix (Fin n) (Fin n) ℝ} {act : ℝ → ℝ} {K : NNReal}

theorem settle_zero (hact0 : act 0 = 0) : settle dt W act 0 (0 : Fin n → ℝ) = 0 := by
  funext i
  simp [settle, hact0]

/-- **Theorem 4 (silence is free).** With no stimulus and an activation that is exactly zero
at rest, rest is the equilibrium: an idle net has nothing to settle. -/
theorem silence (hf : ContractingWith K (settle dt W act 0)) (hact0 : act 0 = 0) :
    equilibrium hf = 0 :=
  (equilibrium_unique hf (settle_zero hact0)).symm

end Silence

section Lesion

variable {dt : ℝ} {W : Matrix (Fin n) (Fin n) ℝ} {act : ℝ → ℝ} {c : Fin n → ℝ} {K : NNReal}

/-- Remove the neurons in `S`: their outgoing synapses are cut, so their activation no longer
reaches anyone. -/
def lesion (W : Matrix (Fin n) (Fin n) ℝ) (S : Finset (Fin n)) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j => if j ∈ S then 0 else W i j

theorem rowMassLE_lesion {ρ : ℝ} (hρ : RowMassLE W ρ) (S : Finset (Fin n)) :
    RowMassLE (lesion W S) ρ := by
  intro i
  refine le_trans (Finset.sum_le_sum fun j _ => ?_) (hρ i)
  simp only [lesion]
  split_ifs <;> simp

theorem settle_sub_settle_lesion (S : Finset (Fin n)) (z : Fin n → ℝ) (i : Fin n) :
    settle dt W act c z i - settle dt (lesion W S) act c z i
      = dt * ∑ j ∈ S, W i j * act (z j) := by
  simp only [settle, lesion]
  have h : ∑ j, W i j * act (z j) - ∑ j, (if j ∈ S then 0 else W i j) * act (z j)
      = ∑ j ∈ S, W i j * act (z j) := by
    rw [← Finset.sum_sub_distrib]
    have hsummand : ∀ j, W i j * act (z j) - (if j ∈ S then 0 else W i j) * act (z j)
        = if j ∈ S then W i j * act (z j) else 0 := by
      intro j
      split_ifs <;> ring
    simp_rw [hsummand]
    rw [Finset.sum_ite_mem, Finset.univ_inter]
  rw [← h]; ring

/-- Cutting the neurons in `S` changes a settling step by at most `dt * μ * A`, where `μ`
bounds the synaptic mass arriving from `S` at any neuron and `A` bounds the activation. -/
theorem dist_settle_lesion (hdt0 : 0 ≤ dt) {S : Finset (Fin n)} {A μ : ℝ}
    (hA : ∀ x, |act x| ≤ A) (hμ : ∀ i, ∑ j ∈ S, |W i j| ≤ μ) (hμ0 : 0 ≤ μ) (z : Fin n → ℝ) :
    dist (settle dt W act c z) (settle dt (lesion W S) act c z) ≤ dt * (μ * A) := by
  have hA0 : 0 ≤ A := le_trans (abs_nonneg _) (hA 0)
  rw [dist_pi_le_iff (mul_nonneg hdt0 (mul_nonneg hμ0 hA0))]
  intro i
  rw [Real.dist_eq, settle_sub_settle_lesion, abs_mul, abs_of_nonneg hdt0]
  refine mul_le_mul_of_nonneg_left ?_ hdt0
  calc |∑ j ∈ S, W i j * act (z j)| ≤ ∑ j ∈ S, |W i j * act (z j)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j ∈ S, |W i j| * A := by
        refine Finset.sum_le_sum fun j _ => ?_
        rw [abs_mul]
        exact mul_le_mul_of_nonneg_left (hA _) (abs_nonneg _)
    _ = (∑ j ∈ S, |W i j|) * A := by rw [Finset.sum_mul]
    _ ≤ μ * A := mul_le_mul_of_nonneg_right (hμ i) hA0

/-- **Theorem 5 (bounded damage).** Removing the neurons in `S` moves the equilibrium by at
most `dt * μ * A / (1 - K)`: damage is bounded by the synaptic mass that was cut, before any
learning takes place. -/
theorem lesion_bound {S : Finset (Fin n)} {A μ : ℝ}
    (hf : ContractingWith K (settle dt W act c))
    (hf' : ContractingWith K (settle dt (lesion W S) act c))
    (hdt0 : 0 ≤ dt) (hA : ∀ x, |act x| ≤ A) (hμ : ∀ i, ∑ j ∈ S, |W i j| ≤ μ) (hμ0 : 0 ≤ μ) :
    dist (equilibrium hf) (equilibrium hf') ≤ dt * (μ * A) / (1 - K) :=
  hf.fixedPoint_lipschitz_in_map hf' (fun z => dist_settle_lesion hdt0 hA hμ hμ0 z)

/-- The lesioned net is certified with the same constant as the intact one. -/
theorem lesion_contracting {L : NNReal} {ρ : ℝ} (hdt0 : 0 < dt) (hdt1 : dt ≤ 1)
    (hact : LipschitzWith L act) (hρ : RowMassLE W ρ) (hρ0 : 0 ≤ ρ) (hq : (L : ℝ) * ρ < 1)
    (S : Finset (Fin n)) :
    ContractingWith (settleRate dt L ρ) (settle dt (lesion W S) act c) :=
  settle_contracting hdt0 hdt1 hact (rowMassLE_lesion hρ S) hρ0 hq

end Lesion

end CadenceFlagship
