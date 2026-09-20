import Mathlib.Data.Real.Basic
import Mathlib.Logic.Equiv.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

/-!
# Exact boundaries for emergent patch specialization

A deterministic update commuting with a state symmetry preserves symmetric
states under a symmetry-invariant input stream. In particular, exchangeable
owners with identical initial states and identical inputs cannot differentiate
under a permutation-equivariant rule. The state must include every relevant
parameter, port, scheduler choice and random seed for these hypotheses to hold.
Unequal exposure, independent random draws or asymmetric schedules can violate
the hypotheses; this is not an impossibility theorem for stochastic development.

A two-owner example separately shows that a permutation-invariant objective
can favor differentiated configurations. An asymmetric objective is therefore
not necessary for useful role separation; a symmetry-breaking trajectory and
task value are different obligations. The supplied Boolean objective is a
mathematical example, not evidence that a learned patch net discovers roles.

Finally, minimizing a nonnegative weighted mismatch plus positive contact cost
over an unconstrained nonnegative contact strength uniquely selects zero.
This is a scalar free-energy boundary, not a result about task-coupled
free/nudged learning, mandatory connection budgets or structural growth.
-/

set_option autoImplicit false

namespace CadenceMission.SpecializationBoundary

universe u v

/-- A finite deterministic rollout cannot break a symmetry that fixes its
initial state and every input and commutes with the update. -/
theorem equivariant_rollout_stays_fixed {S : Type u} {I : Type v}
    (step : S → I → S) (stateSymmetry : S → S) (inputSymmetry : I → I)
    (hequivariant : ∀ state input,
      stateSymmetry (step state input) = step (stateSymmetry state) (inputSymmetry input))
    (inputs : List I) (initial : S)
    (hinitial : stateSymmetry initial = initial)
    (hinputs : ∀ input ∈ inputs, inputSymmetry input = input) :
    stateSymmetry (inputs.foldl step initial) = inputs.foldl step initial := by
  induction inputs generalizing initial with
  | nil => simpa using hinitial
  | cons input rest ih =>
    have hinput : inputSymmetry input = input := hinputs input (by simp)
    have hnext : stateSymmetry (step initial input) = step initial input := by
      rw [hequivariant, hinitial, hinput]
    have hrest : ∀ item ∈ rest, inputSymmetry item = item := by
      intro item hitem
      exact hinputs item (List.mem_cons_of_mem input hitem)
    simpa only [List.foldl_cons] using ih (step initial input) hnext hrest

/-- Relabeling owners relabels the state array. -/
def permuteOwners {n : Nat} {S : Type u} (permutation : Equiv.Perm (Fin n))
    (states : Fin n → S) : Fin n → S :=
  fun owner => states (permutation owner)

/-- A shared input stream and identical complete initial owner states leave
all owners identical under a deterministic owner-equivariant update. -/
theorem homogeneous_owner_rollout {n : Nat} {S : Type u} {I : Type v}
    (step : (Fin n → S) → I → (Fin n → S))
    (hequivariant : ∀ permutation states input,
      permuteOwners permutation (step states input) =
        step (permuteOwners permutation states) input)
    (inputs : List I) (initial : S) (left right : Fin n) :
    (inputs.foldl step (fun _ => initial)) left =
      (inputs.foldl step (fun _ => initial)) right := by
  have hfixed := equivariant_rollout_stays_fixed step
    (permuteOwners (Equiv.swap left right)) id
    (fun states input => hequivariant (Equiv.swap left right) states input)
    inputs (fun _ => initial) rfl (fun _ _ => rfl)
  have hequal := congrFun hfixed left
  simpa [permuteOwners] using hequal.symm

/-! A symmetric objective can favor complementary owners without assigning
which owner must take which role. There are two equally good permutations. -/

def swapOwners (state : Bool × Bool) : Bool × Bool := (state.2, state.1)

def complementaryUtility (state : Bool × Bool) : Nat :=
  if state.1 = state.2 then 0 else 1

theorem complementary_utility_is_symmetric (state : Bool × Bool) :
    complementaryUtility (swapOwners state) = complementaryUtility state := by
  rcases state with ⟨left, right⟩
  cases left <;> cases right <;> decide

/-- The two global maximizers differentiate the owners, despite the
objective itself being invariant under their exchange. -/
theorem complementary_utility_maximizers (state : Bool × Bool) :
    complementaryUtility state ≤ 1 ∧
      (complementaryUtility state = 1 ↔
        state = (false, true) ∨ state = (true, false)) := by
  rcases state with ⟨left, right⟩
  cases left <;> cases right <;> decide

/-- A symmetric rollout cannot reach those complementary optima from a
homogeneous start unless an equivariance or input hypothesis is broken. -/
theorem symmetric_rollout_has_zero_complementarity {I : Type u}
    (step : (Bool × Bool) → I → (Bool × Bool))
    (hequivariant : ∀ state input,
      swapOwners (step state input) = step (swapOwners state) input)
    (inputs : List I) (initial : Bool) :
    complementaryUtility (inputs.foldl step (initial, initial)) = 0 := by
  have hfixed := equivariant_rollout_stays_fixed step swapOwners id
    hequivariant inputs (initial, initial) rfl (fun _ _ => rfl)
  have hequal := congrArg Prod.fst hfixed
  change (inputs.foldl step (initial, initial)).2 =
    (inputs.foldl step (initial, initial)).1 at hequal
  simp [complementaryUtility, hequal]

/-! Scalar connection-strength boundary. The mismatch and contact cost are
held fixed, strength zero is feasible, and no required-connectivity constraint
or separate task objective is included. -/

def wireObjective (strength mismatch cost : ℝ) : ℝ :=
  strength * (mismatch ^ 2 + cost)

/-- With nonnegative contact cost, increasing strength cannot lower this
free mismatch objective at a fixed mismatch. -/
theorem wire_objective_monotone {weak strong mismatch cost : ℝ}
    (hstrength : weak ≤ strong) (hcost : 0 ≤ cost) :
    wireObjective weak mismatch cost ≤ wireObjective strong mismatch cost := by
  exact mul_le_mul_of_nonneg_right hstrength (add_nonneg (sq_nonneg mismatch) hcost)

/-- Positive contact cost makes zero the unique optimal nonnegative
strength. Pure free-mismatch minimization cannot justify a useful wire. -/
theorem positive_cost_selects_zero {strength mismatch cost : ℝ}
    (hstrength : 0 ≤ strength) (hcost : 0 < cost) :
    wireObjective 0 mismatch cost ≤ wireObjective strength mismatch cost ∧
      (wireObjective strength mismatch cost = wireObjective 0 mismatch cost ↔ strength = 0) := by
  have hcoefficient : 0 < mismatch ^ 2 + cost := by
    have hsquare := sq_nonneg mismatch
    linarith
  constructor
  · exact wire_objective_monotone hstrength hcost.le
  · simp only [wireObjective, zero_mul]
    constructor
    · intro hzero
      exact (mul_eq_zero.mp hzero).resolve_right (ne_of_gt hcoefficient)
    · intro hzero
      rw [hzero, zero_mul]

end CadenceMission.SpecializationBoundary
