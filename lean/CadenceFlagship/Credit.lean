import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic

/-!
# Credit by two equilibria: locality of the contrast

The centered learner compares two states: the negative-nudge endpoint `sm` and the
positive-nudge endpoint `sp`. Its raw contrast at the contact between `i` and `j` is

  `(sp i * sp j - sm i * sm j) / (2 β)`.

It is a function of the two endpoint activations in the two phases and of nothing else. The
lemmas below record this locality, the symmetry that lets a reciprocal pair share one
parameter, and the fact that identical endpoints give zero raw contrast. Momentum,
decay and other parameter transformations can still produce a nonzero applied update.
These are algebraic identities for two arrays. They do not establish phase convergence,
stability, a task gradient, or the one-sided learner's different denominator `beta`.
-/

namespace CadenceFlagship

variable {n : ℕ}

/-- The centered endpoint contrast, one entry per synapse. -/
noncomputable def contrast (β : ℝ) (sp sm : Fin n → ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j => (sp i * sp j - sm i * sm j) / (2 * β)

/-- **Lemma 6a (locality).** The update of synapse `(i, j)` is a fixed function of the four
numbers the two endpoint neurons published in the two phases. -/
theorem contrast_local (β : ℝ) :
    ∃ g : ℝ → ℝ → ℝ → ℝ → ℝ,
      ∀ (sp sm : Fin n → ℝ) (i j : Fin n), contrast β sp sm i j = g (sp i) (sp j) (sm i) (sm j) :=
  ⟨fun a b a' b' => (a * b - a' * b') / (2 * β), fun _ _ _ _ => rfl⟩

/-- **Lemma 6b (symmetry).** The contrast is symmetric, so a reciprocal synapse pair can share
a single parameter. -/
theorem contrast_symm (β : ℝ) (sp sm : Fin n → ℝ) (i j : Fin n) :
    contrast β sp sm i j = contrast β sp sm j i := by
  simp only [contrast, mul_comm]

/-- Identical endpoints give zero raw contrast, before optimizer history and decay. -/
theorem contrast_eq_zero_of_agree (β : ℝ) (sp sm : Fin n → ℝ) (h : sp = sm) :
    contrast β sp sm = 0 := by
  subst h
  ext i j
  simp [contrast]

end CadenceFlagship
