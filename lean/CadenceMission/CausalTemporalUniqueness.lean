import Init

/-!
A zero-residual finite causal chain has exactly its forward recurrence.

The transition and readback may be nonlinear and time-dependent. This is an
algebraic statement about fixed parameters and fixed supplied boundaries. It
does not say that arbitrary stationary points of an energy are defect-free,
that the transition forgets, or that observations/goal constraints cannot
change inference. In the temporal prototypes, initial = B * cue,
step t h = A * phi h, and readback t h = C * phi h.

Only Lean's bundled Init is required; no new external dependency is used.
-/

namespace CadenceMission.CausalTemporalUniqueness

universe u v

variable {State : Type u} {Output : Type v}

/-- The unique executable forward sequence for a fixed initial boundary. -/
def forward (step : Nat → State → State) (initial : State) : Nat → State
  | 0 => initial
  | n + 1 => step n (forward step initial n)

/-- Vanishing initial and adjacent defects through the finite frontier. -/
def DefectFree (step : Nat → State → State) (initial : State)
    (path : Nat → State) (frontier : Nat) : Prop :=
  path 0 = initial ∧ ∀ n, n < frontier → path (n + 1) = step n (path n)

theorem forward_defectFree (step : Nat → State → State) (initial : State)
    (frontier : Nat) : DefectFree step initial (forward step initial) frontier := by
  constructor
  · rfl
  · intro n _
    rfl

/-- Two defect-free paths with the same boundary agree on every owned moment. -/
theorem defectFree_unique (step : Nat → State → State) (initial : State)
    (left right : Nat → State) (frontier : Nat)
    (hl : DefectFree step initial left frontier)
    (hr : DefectFree step initial right frontier) :
    ∀ n, n ≤ frontier → left n = right n := by
  intro n
  induction n with
  | zero =>
      intro _
      exact hl.1.trans hr.1.symm
  | succ n ih =>
      intro bound
      have earlier : n < frontier := Nat.lt_of_succ_le bound
      have same : left n = right n := ih (Nat.le_of_lt earlier)
      rw [hl.2 n earlier, hr.2 n earlier, same]

/-- Joint zero-residual repair adds no alternative free trajectory. -/
theorem eq_forward (step : Nat → State → State) (initial : State)
    (path : Nat → State) (frontier : Nat)
    (h : DefectFree step initial path frontier) :
    ∀ n, n ≤ frontier → path n = forward step initial n :=
  defectFree_unique step initial path (forward step initial) frontier h
    (forward_defectFree step initial frontier)

/-- A deterministic defect-free readback is the forward model's readback. -/
theorem readback_eq_forward (step : Nat → State → State) (initial : State)
    (readback : Nat → State → Output) (path : Nat → State)
    (published : Nat → Output) (frontier : Nat)
    (h : DefectFree step initial path frontier)
    (readback_eq : ∀ n, n ≤ frontier → published n = readback n (path n)) :
    ∀ n, n ≤ frontier → published n = readback n (forward step initial n) := by
  intro n bound
  rw [readback_eq n bound, eq_forward step initial path frontier h n bound]

end CadenceMission.CausalTemporalUniqueness

#print axioms CadenceMission.CausalTemporalUniqueness.forward_defectFree
#print axioms CadenceMission.CausalTemporalUniqueness.defectFree_unique
#print axioms CadenceMission.CausalTemporalUniqueness.eq_forward
#print axioms CadenceMission.CausalTemporalUniqueness.readback_eq_forward
