import CadenceFlagship.Settling

/-!
# The certificate under changes of wiring

A wiring that evolves across lives is changed by a few operations: a synapse is removed, a
projection between two regions is masked out, a gain is lowered, a new region is added with
its projections. This module states that the settling certificate of `Settling.lean` is
closed under those operations.

* `rowMassLE_of_abs_le`: a wiring whose every synapse is no stronger in absolute value than
  the corresponding synapse of a certified wiring is certified with the same constant. Pruning,
  masking and weakening are instances.
* `rowMassLE_add`: adding projections of row mass `μ` to a wiring of row mass `ρ` gives a
  wiring of row mass at most `ρ + μ`. Growth stays certified while `L (ρ + μ) < 1`.
* `mutation_contracting`, `growth_contracting`: the two closure statements for the settling
  map itself.
* `rowMassLE_smul`: scaling every synapse by a gain `g` with `|g| ≤ 1` keeps the certificate.
-/

namespace CadenceFlagship

open Finset

variable {n : ℕ}

/-- A wiring no stronger than a certified one, synapse by synapse, is certified with the same
row mass. Pruning a synapse (setting it to zero), masking a projection, and lowering a gain
are instances. -/
theorem rowMassLE_of_abs_le {W W' : Matrix (Fin n) (Fin n) ℝ} {ρ : ℝ}
    (hρ : RowMassLE W ρ) (h : ∀ i j, |W' i j| ≤ |W i j|) : RowMassLE W' ρ := by
  intro i
  exact le_trans (sum_le_sum fun j _ => h i j) (hρ i)

/-- Scaling every synapse by a gain of absolute value at most one keeps the certificate. -/
theorem rowMassLE_smul {W : Matrix (Fin n) (Fin n) ℝ} {ρ g : ℝ}
    (hρ : RowMassLE W ρ) (hg : |g| ≤ 1) : RowMassLE (g • W) ρ := by
  refine rowMassLE_of_abs_le hρ fun i j => ?_
  simp only [Matrix.smul_apply, smul_eq_mul, abs_mul]
  calc |g| * |W i j| ≤ 1 * |W i j| :=
        mul_le_mul_of_nonneg_right hg (abs_nonneg _)
    _ = |W i j| := one_mul _

/-- Adding projections of row mass `μ` to a wiring of row mass `ρ` gives row mass at most
`ρ + μ`. -/
theorem rowMassLE_add {W V : Matrix (Fin n) (Fin n) ℝ} {ρ μ : ℝ}
    (hW : RowMassLE W ρ) (hV : RowMassLE V μ) : RowMassLE (W + V) (ρ + μ) := by
  intro i
  calc ∑ j, |(W + V) i j| = ∑ j, |W i j + V i j| := by simp [Matrix.add_apply]
    _ ≤ ∑ j, (|W i j| + |V i j|) := sum_le_sum fun j _ => abs_add_le _ _
    _ = ∑ j, |W i j| + ∑ j, |V i j| := sum_add_distrib
    _ ≤ ρ + μ := add_le_add (hW i) (hV i)

/-- Row mass is a bound, so any larger bound is also a bound. -/
theorem rowMassLE_mono {W : Matrix (Fin n) (Fin n) ℝ} {ρ ρ' : ℝ}
    (hW : RowMassLE W ρ) (h : ρ ≤ ρ') : RowMassLE W ρ' :=
  fun i => le_trans (hW i) h

section Certificate

variable {dt : ℝ} {L : NNReal} {act : ℝ → ℝ} {c : Fin n → ℝ}

/-- A mutation that weakens, prunes or masks synapses of a certified wiring leaves the
settling map a contraction with the same constant. -/
theorem mutation_contracting {ρ : ℝ} {W W' : Matrix (Fin n) (Fin n) ℝ}
    (hdt0 : 0 < dt) (hdt1 : dt ≤ 1) (hact : LipschitzWith L act)
    (hρ : RowMassLE W ρ) (hρ0 : 0 ≤ ρ) (hq : (L : ℝ) * ρ < 1)
    (h : ∀ i j, |W' i j| ≤ |W i j|) :
    ContractingWith (settleRate dt L ρ) (settle dt W' act c) :=
  settle_contracting hdt0 hdt1 hact (rowMassLE_of_abs_le hρ h) hρ0 hq

/-- Growth: adding projections of row mass `μ` to a certified wiring of row mass `ρ` leaves
the settling map a contraction with constant `settleRate dt L (ρ + μ)`, provided the grown
row mass satisfies `L (ρ + μ) < 1`. -/
theorem growth_contracting {ρ μ : ℝ} {W V : Matrix (Fin n) (Fin n) ℝ}
    (hdt0 : 0 < dt) (hdt1 : dt ≤ 1) (hact : LipschitzWith L act)
    (hW : RowMassLE W ρ) (hV : RowMassLE V μ) (hρ0 : 0 ≤ ρ) (hμ0 : 0 ≤ μ)
    (hq : (L : ℝ) * (ρ + μ) < 1) :
    ContractingWith (settleRate dt L (ρ + μ)) (settle dt (W + V) act c) :=
  settle_contracting hdt0 hdt1 hact (rowMassLE_add hW hV) (add_nonneg hρ0 hμ0) hq

/-- Every member of a population whose wirings share one row-mass bound settles by one
certificate: the constant depends on the bound, not on the individual. -/
theorem population_contracting {ρ : ℝ} {ι : Type*} (G : ι → Matrix (Fin n) (Fin n) ℝ)
    (hdt0 : 0 < dt) (hdt1 : dt ≤ 1) (hact : LipschitzWith L act)
    (hG : ∀ g, RowMassLE (G g) ρ) (hρ0 : 0 ≤ ρ) (hq : (L : ℝ) * ρ < 1) (g : ι) :
    ContractingWith (settleRate dt L ρ) (settle dt (G g) act c) :=
  settle_contracting hdt0 hdt1 hact (hG g) hρ0 hq

end Certificate

end CadenceFlagship
