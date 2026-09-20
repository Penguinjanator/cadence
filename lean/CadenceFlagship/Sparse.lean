import CadenceFlagship.Memory
import Mathlib.Analysis.Normed.Module.Basic

/-!
# Pattern separation: disjoint supports do not interfere

The interference law (`read_write`) says a write at key `k` moves the read at key `q` by
`η (q ⬝ᵥ k)` times the correction. Keys whose supports are disjoint have dot product zero,
so they never interfere, whatever their values on their own supports. This is the reason a
memory should sparsify its keys before writing: a k-winner code with distinct winners turns
correlated cues into orthogonal addresses, and `read_writeUpTo` then stores every one of them
exactly. The dentate gyrus does the same thing for the hippocampus.
-/

namespace CadenceFlagship.Memory

open Matrix

variable {R : Type*} [CommRing R] {d m : ℕ}

/-- Two keys with disjoint supports are orthogonal. -/
theorem dotProduct_eq_zero_of_disjoint_support (u v : Fin d → R)
    (h : ∀ i, u i = 0 ∨ v i = 0) : u ⬝ᵥ v = 0 := by
  unfold dotProduct
  refine Finset.sum_eq_zero fun i _ => ?_
  rcases h i with hu | hv
  · simp [hu]
  · simp [hv]

/-- **Theorem (pattern separation).** A family of unit keys with pairwise disjoint supports is
stored exactly by sequential residual writes at rate one, from any initial record. -/
theorem read_writeUpTo_of_disjoint (M0 : Record d m R) (keys : ℕ → Fin d → R)
    (vals : ℕ → Fin m → R) (N : ℕ)
    (hdisj : ∀ s t, s < N → t < N → s ≠ t → ∀ i, keys s i = 0 ∨ keys t i = 0)
    (hunit : ∀ t, t < N → keys t ⬝ᵥ keys t = 1) :
    ∀ t, t < N → read (writeUpTo M0 keys vals N) (keys t) = vals t :=
  read_writeUpTo M0 keys vals N
    (fun s t hs ht hst => dotProduct_eq_zero_of_disjoint_support _ _ (hdisj s t hs ht hst))
    hunit

/-- A write at a key whose support is disjoint from the query's leaves the query's read
unchanged: interference is exactly zero, not merely small. -/
theorem read_write_of_disjoint (M : Record d m R) {k q : Fin d → R} (y : Fin m → R) (η : R)
    (h : ∀ i, q i = 0 ∨ k i = 0) : read (write M k y η) q = read M q :=
  read_write_orth M y η (dotProduct_eq_zero_of_disjoint_support q k h)

/-- **How many separated keys a code of a given width holds.** If every key's support has at
least `active` cells and the supports are pairwise disjoint, then the number of keys times the
active count is at most the width of the code: a code of `D` cells with `k` winners admits at
most `⌊D / k⌋` pairwise non-interfering addresses. -/
theorem disjoint_family_card_le {N D active : ℕ} (supp : Fin N → Finset (Fin D))
    (hdisj : ∀ i j, i ≠ j → Disjoint (supp i) (supp j))
    (hactive : ∀ i, active ≤ (supp i).card) :
    N * active ≤ D := by
  classical
  have hsum : ∑ i : Fin N, (supp i).card = (Finset.univ.biUnion supp).card :=
    (Finset.card_biUnion fun i _ j _ hij => hdisj i j hij).symm
  have hle : (Finset.univ.biUnion supp).card ≤ D := by
    have := Finset.card_le_card (Finset.subset_univ (Finset.univ.biUnion supp))
    simpa using this
  have hlow : N * active ≤ ∑ i : Fin N, (supp i).card := by
    calc N * active = ∑ _i : Fin N, active := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
      _ ≤ ∑ i : Fin N, (supp i).card := Finset.sum_le_sum fun i _ => hactive i
  omega

/-- **The exact size of one write's reach.** Over the reals, a write at key `k` moves the read
at key `q` by `|η (q ⬝ᵥ k)|` times the size of the correction, in the sup norm. Sharing no
active cell makes this exactly zero; sharing everything makes it the whole correction. -/
theorem read_write_dist {d m : ℕ} (M : Record d m ℝ) (k : Fin d → ℝ) (y : Fin m → ℝ) (η : ℝ)
    (q : Fin d → ℝ) :
    ‖read (write M k y η) q - read M q‖ = |η * (q ⬝ᵥ k)| * ‖y - read M k‖ := by
  rw [read_write, add_sub_cancel_left, norm_smul, Real.norm_eq_abs]

/-- The bound a designer uses: an overlap of at most `ε` confines the interference. -/
theorem read_write_dist_le {d m : ℕ} (M : Record d m ℝ) (k : Fin d → ℝ) (y : Fin m → ℝ)
    (η : ℝ) (q : Fin d → ℝ) {ε : ℝ} (hq : |q ⬝ᵥ k| ≤ ε) :
    ‖read (write M k y η) q - read M q‖ ≤ |η| * ε * ‖y - read M k‖ := by
  rw [read_write_dist, abs_mul]
  have hnorm : 0 ≤ ‖y - read M k‖ := norm_nonneg _
  have habs : 0 ≤ |η| := abs_nonneg _
  have : |η| * |q ⬝ᵥ k| ≤ |η| * ε := mul_le_mul_of_nonneg_left hq habs
  exact mul_le_mul_of_nonneg_right this hnorm

end CadenceFlagship.Memory
