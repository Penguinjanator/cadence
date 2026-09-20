import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Dimension.Finite
import Mathlib.LinearAlgebra.Dimension.Constructions
import Mathlib.LinearAlgebra.FiniteDimensional.Defs
import Mathlib.LinearAlgebra.Matrix.ToLin

/-!
# Fast synapses: one-shot memory, and attention as one read

A fast-synapse record is a matrix `M : Matrix (Fin d) (Fin m) R` between `d` key neurons and
`m` value neurons. A read at key `q` is the vector `q ᵥ* M`. A residual (delta-rule) write at
rate `η` moves the record toward the observed value:

  `M ← M + η • vecMulVec k (y - k ᵥ* M)`.

The theorems: the written key reads exactly its value at rate one; a different key's read
changes by `(q ⬝ᵥ k)` times the correction, so orthogonal keys are untouched; a repeated
correct observation changes nothing; an orthonormal family is stored exactly from any initial
record; at most `d` keys can be orthonormal. The Hebbian (additive) record reads as
unnormalised linear attention, and one-hot keys give exact recall at every length up to the
vocabulary.
-/

namespace CadenceFlagship.Memory

open Matrix

variable {R : Type*} [CommRing R] {d m : ℕ}

/-- A fast-synapse record between `d` key neurons and `m` value neurons. -/
abbrev Record (d m : ℕ) (R : Type*) := Matrix (Fin d) (Fin m) R

/-- Reading the record at a key. -/
def read (M : Record d m R) (q : Fin d → R) : Fin m → R := q ᵥ* M

/-- A residual write: correct the record toward the observed value at rate `η`. -/
def write (M : Record d m R) (k : Fin d → R) (y : Fin m → R) (η : R) : Record d m R :=
  M + η • vecMulVec k (y - read M k)

/-- **Theorem 7a (interference law).** A write at key `k` changes the read at key `q` by
`η * (q ⬝ᵥ k)` times the correction. -/
theorem read_write (M : Record d m R) (k : Fin d → R) (y : Fin m → R) (η : R)
    (q : Fin d → R) :
    read (write M k y η) q = read M q + (η * (q ⬝ᵥ k)) • (y - read M k) := by
  unfold read write
  rw [Matrix.vecMul_add, Matrix.vecMul_smul, Matrix.vecMul_vecMulVec, smul_smul]
  rfl

/-- **Theorem 7b (one-shot write).** At rate one, a unit key reads exactly the value just
written, from any record. -/
theorem read_write_self (M : Record d m R) {k : Fin d → R} (y : Fin m → R) (hk : k ⬝ᵥ k = 1) :
    read (write M k y 1) k = y := by
  rw [read_write, hk]
  simp

/-- **Theorem 7c (orthogonal invariance).** A write at `k` leaves the read at any key
orthogonal to `k` unchanged. -/
theorem read_write_orth (M : Record d m R) {k q : Fin d → R} (y : Fin m → R) (η : R)
    (hqk : q ⬝ᵥ k = 0) : read (write M k y η) q = read M q := by
  rw [read_write, hqk]
  simp

/-- **Theorem 7d (idempotence).** Writing the same correct observation twice changes nothing:
repeated evidence does not keep growing a record that is already right. -/
theorem repeat_write_stable (M : Record d m R) {k : Fin d → R} (y : Fin m → R)
    (hk : k ⬝ᵥ k = 1) : write (write M k y 1) k y 1 = write M k y 1 := by
  show write M k y 1 + (1 : R) • vecMulVec k (y - read (write M k y 1) k) = write M k y 1
  rw [read_write_self M y hk]
  simp only [sub_self, one_smul, add_eq_left]
  ext i j
  simp [Matrix.vecMulVec_apply]

/-- Sequential residual writes at rate one. -/
def writeUpTo (M0 : Record d m R) (keys : ℕ → Fin d → R) (vals : ℕ → Fin m → R) :
    ℕ → Record d m R
  | 0 => M0
  | t + 1 => write (writeUpTo M0 keys vals t) (keys t) (vals t) 1

/-- **Theorem 8 (exact storage of an orthonormal family).** After writing `N` pairwise
orthogonal unit keys in sequence, every one of them reads its own value, whatever the initial
record was. -/
theorem read_writeUpTo (M0 : Record d m R) (keys : ℕ → Fin d → R) (vals : ℕ → Fin m → R)
    (N : ℕ) (horth : ∀ s t, s < N → t < N → s ≠ t → keys s ⬝ᵥ keys t = 0)
    (hunit : ∀ t, t < N → keys t ⬝ᵥ keys t = 1) :
    ∀ t, t < N → read (writeUpTo M0 keys vals N) (keys t) = vals t := by
  induction N with
  | zero => intro t ht; exact absurd ht (Nat.not_lt_zero _)
  | succ N ih =>
    intro t ht
    simp only [writeUpTo]
    rcases Nat.lt_succ_iff_lt_or_eq.mp ht with h | h
    · rw [read_write_orth _ _ _ (horth t N (by omega) (by omega) (by omega))]
      exact ih (fun s t hs ht hst => horth s t (by omega) (by omega) hst)
        (fun t ht => hunit t (by omega)) t h
    · subst h
      exact read_write_self _ _ (hunit t (by omega))

/-- The Hebbian (additive) record of `N` key/value pairs. -/
def hebb {N : ℕ} (keys : Fin N → Fin d → R) (vals : Fin N → Fin m → R) : Record d m R :=
  ∑ t, vecMulVec (keys t) (vals t)

/-- **Theorem 9a (attention is one read).** The Hebbian record reads as unnormalised linear
attention: the sum of the values weighted by the query/key dot products. -/
theorem read_hebb {N : ℕ} (keys : Fin N → Fin d → R) (vals : Fin N → Fin m → R)
    (q : Fin d → R) : read (hebb keys vals) q = ∑ t, (q ⬝ᵥ keys t) • vals t := by
  unfold read hebb
  ext j
  simp only [Matrix.vecMul, dotProduct, Matrix.sum_apply, vecMulVec_apply, Finset.sum_apply,
    Pi.smul_apply, smul_eq_mul, Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun i _ => ?_
  ring

/-- **Theorem 9b (zero-parameter recall).** With one-hot keys at distinct addresses, the
Hebbian record recalls every stored value exactly, at any length up to the vocabulary. -/
theorem one_hot_recall {N : ℕ} (a : Fin N → Fin d) (ha : Function.Injective a)
    (vals : Fin N → Fin m → R) (s : Fin N) :
    read (hebb (fun t => Pi.single (a t) (1 : R)) vals) (Pi.single (a s) 1) = vals s := by
  rw [read_hebb]
  have hdot : ∀ t, (Pi.single (a s) (1 : R) ⬝ᵥ Pi.single (a t) (1 : R))
      = if s = t then (1 : R) else 0 := by
    intro t
    rw [single_dotProduct, one_mul, Pi.single_apply]
    by_cases h : s = t
    · subst h; simp
    · have : a s ≠ a t := fun e => h (ha e)
      simp [h, this]
  simp_rw [hdot]
  simp [ite_smul, Finset.sum_ite_eq]

section Capacity

variable {F : Type*} [Field F]

/-- Pairwise orthogonal unit keys are linearly independent. -/
theorem orthonormal_linearIndependent {N : ℕ} (keys : Fin N → Fin d → F)
    (horth : ∀ s t, keys s ⬝ᵥ keys t = if s = t then (1 : F) else 0) :
    LinearIndependent F keys := by
  rw [Fintype.linearIndependent_iff]
  intro g hg t
  have h := congrArg (fun v => v ⬝ᵥ keys t) hg
  simp only [sum_dotProduct, smul_dotProduct, horth, smul_eq_mul, mul_ite,
    mul_one, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, if_true,
    zero_dotProduct] at h
  exact h

/-- **Theorem 8' (capacity).** At most `d` keys can be pairwise orthogonal unit vectors in a
`d`-dimensional key space, so exact one-shot storage is bounded by the number of key neurons. -/
theorem capacity_le {N : ℕ} (keys : Fin N → Fin d → F)
    (horth : ∀ s t, keys s ⬝ᵥ keys t = if s = t then (1 : F) else 0) : N ≤ d := by
  have h := LinearIndependent.fintype_card_le_finrank (orthonormal_linearIndependent keys horth)
  simpa using h

end Capacity

end CadenceFlagship.Memory
