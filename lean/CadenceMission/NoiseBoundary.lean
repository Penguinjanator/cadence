import Mathlib.Data.Real.Basic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Linarith

/-!
# Noise, protected records, and indistinguishable change

Two elementary boundaries for the proposed persistent predictive patch:

1. Repair preserving a protected record is a different operation from writing
   a new protected record. Exact settling cannot undo the new datum while also
   preserving it. A one-step exact-equilibrium example makes this explicit.
2. A lone conflicting observation can be consistent with either transient
   measurement error or a genuine change. With identical available evidence,
   every deterministic predictor gives the same estimate in both worlds and
   incurs squared error at least 1/4 in one of them for targets 0 and 1.

No noise law, prior, change hazard, or evidence-acceptance rule is selected by
these theorems. The two-world result is a worst-case identification boundary,
not an expected-risk statement under a specified probability model. It does
not exclude useful statistical filtering or adaptation after more evidence.
-/

set_option autoImplicit false

namespace CadenceMission.NoiseBoundary

universe u v

/-- A patch exposes a protected record and a repairable interior. -/
def repair {R : Type u} {Z : Type v} (inner : R → Z → Z)
    (state : R × Z) : R × Z :=
  (state.1, inner state.1 state.2)

/-- An externally accepted write changes the record; its acceptance is not
proved or chosen here. -/
def writeRecord {R : Type u} {Z : Type v} (record : R)
    (state : R × Z) : R × Z :=
  (record, state.2)

/-- Finitely many repair steps, with a fixed repair rule. -/
def settle {R : Type u} {Z : Type v} (inner : R → Z → Z) :
    Nat → R × Z → R × Z
  | 0, state => state
  | n + 1, state => settle inner n (repair inner state)

theorem one_repair_preserves_record
    {R : Type u} {Z : Type v} (inner : R → Z → Z) (state : R × Z) :
    (repair inner state).1 = state.1 := rfl

theorem finite_repair_preserves_record
    {R : Type u} {Z : Type v} (inner : R → Z → Z) (n : Nat)
    (state : R × Z) : (settle inner n state).1 = state.1 := by
  induction n generalizing state with
  | zero => rfl
  | succ n ih =>
    change (settle inner n (repair inner state)).1 = state.1
    rw [ih]
    rfl

/-- After a write, preserving the boundary preserves the new record. -/
theorem repair_after_write_preserves_new_record
    {R : Type u} {Z : Type v} (inner : R → Z → Z) (n : Nat)
    (record : R) (state : R × Z) :
    (settle inner n (writeRecord record state)).1 = record := by
  rw [finite_repair_preserves_record]
  rfl

/-- Record-preserving repair cannot recover the previous record from an
accepted replacement without violating its preservation contract. -/
theorem changed_write_survives_arbitrarily_much_repair
    {R : Type u} {Z : Type v} (inner : R → Z → Z) (n : Nat)
    (record : R) (state : R × Z) (hchanged : record ≠ state.1) :
    (settle inner n (writeRecord record state)).1 ≠ state.1 := by
  rw [repair_after_write_preserves_new_record]
  exact hchanged

/-! A perfectly convergent solver can exactly reproduce a mistaken boundary.
This repair replaces its interior by its record and settles in one step. -/

def literalRepair : ℝ × ℝ → ℝ × ℝ := repair (fun record _ => record)

theorem literal_repair_is_idempotent (state : ℝ × ℝ) :
    literalRepair (literalRepair state) = literalRepair state := rfl

theorem every_literal_record_has_exact_equilibrium (record : ℝ) :
    literalRepair (record, record) = (record, record) := rfl

theorem literal_repair_tracks_new_write (record : ℝ) (state : ℝ × ℝ) :
    literalRepair (writeRecord record state) = (record, record) := rfl

/-- Zero repair residual is compatible with an arbitrary error relative to a
separately specified true value. The chosen record may be wrong. -/
theorem exact_equilibrium_can_have_arbitrary_readout_error (error : ℝ) :
    literalRepair (error, error) = (error, error) ∧
      (literalRepair (error, error)).2 - 0 = error := by
  constructor
  · rfl
  · simp [literalRepair, repair]

/-! ## The first conflicting observation does not identify its cause -/

theorem two_target_squared_errors (estimate : ℝ) :
    (1 / 2 : ℝ) ≤ estimate ^ 2 + (estimate - 1) ^ 2 := by
  nlinarith [sq_nonneg (estimate - 1 / 2)]

theorem one_target_squared_error_at_least_quarter (estimate : ℝ) :
    (1 / 4 : ℝ) ≤ estimate ^ 2 ∨ (1 / 4 : ℝ) ≤ (estimate - 1) ^ 2 := by
  by_cases h : (1 / 4 : ℝ) ≤ estimate ^ 2
  · exact Or.inl h
  · right
    have hs := two_target_squared_errors estimate
    have hn := lt_of_not_ge h
    linarith

theorem worst_target_squared_error_at_least_quarter (estimate : ℝ) :
    (1 / 4 : ℝ) ≤ max (estimate ^ 2) ((estimate - 1) ^ 2) := by
  rcases one_target_squared_error_at_least_quarter estimate with h | h
  · exact h.trans (le_max_left _ _)
  · exact h.trans (le_max_right _ _)

theorem minimax_bound_is_attained :
    max ((1 / 2 : ℝ) ^ 2) (((1 / 2 : ℝ) - 1) ^ 2) = 1 / 4 := by
  norm_num

/-- The same reports: two old zero readings followed by a one. In world
`false`, the final report is transient error and the required latent value
remains zero. In world `true`, the latent value has changed to one. -/
def availableReports (_world : Bool) : List ℝ := [0, 0, 1]

def requiredLatent (world : Bool) : ℝ := if world then 1 else 0

theorem both_worlds_supply_identical_evidence :
    availableReports false = availableReports true := rfl

/-- No predictor using only these reports can be both perfectly immune to the
outlier world and instantly exact in the changed world. At least one world's
squared error is at least 1/4. The worlds are not assigned probabilities. -/
theorem indistinguishable_noise_or_change_forces_error
    (predict : List ℝ → ℝ) :
    (1 / 4 : ℝ) ≤
      (predict (availableReports false) - requiredLatent false) ^ 2 ∨
    (1 / 4 : ℝ) ≤
      (predict (availableReports true) - requiredLatent true) ^ 2 := by
  simpa [availableReports, requiredLatent] using
    one_target_squared_error_at_least_quarter (predict [0, 0, 1])

/-- A generic evidence-collision version: richer context helps only when it
actually distinguishes the two worlds. -/
theorem same_available_state_forces_noise_change_tradeoff
    {M : Type u} (state : Bool → M) (predict : M → ℝ)
    (he : state false = state true) :
    (1 / 4 : ℝ) ≤ (predict (state false)) ^ 2 ∨
    (1 / 4 : ℝ) ≤ (predict (state true) - 1) ^ 2 := by
  rw [← he]
  exact one_target_squared_error_at_least_quarter (predict (state false))

#print axioms finite_repair_preserves_record
#print axioms changed_write_survives_arbitrarily_much_repair
#print axioms literal_repair_is_idempotent
#print axioms exact_equilibrium_can_have_arbitrary_readout_error
#print axioms two_target_squared_errors
#print axioms worst_target_squared_error_at_least_quarter
#print axioms indistinguishable_noise_or_change_forces_error
#print axioms same_available_state_forces_noise_change_tradeoff

end CadenceMission.NoiseBoundary
