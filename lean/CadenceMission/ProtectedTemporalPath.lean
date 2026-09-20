import CadenceMission.CausalTemporalUniqueness

/-!
Conditional path retention after a parameter update. New transition/readback
maps need agree only on the protected old path, not on every possible state.
This theorem assumes exact map agreement. It does not prove the floating-point
SVD projector, importance selection, rank capacity or retention of other cues.
-/
namespace CadenceMission.ProtectedTemporalPath

open CadenceMission.CausalTemporalUniqueness

universe u v
variable {State : Type u} {Output : Type v}

/-- Local agreement makes the old forward path defect-free for the new map. -/
theorem protected_defectFree (oldStep newStep : Nat → State → State)
    (initial : State) (frontier : Nat)
    (agreement : ∀ n, n < frontier →
      newStep n (forward oldStep initial n) = oldStep n (forward oldStep initial n)) :
    DefectFree newStep initial (forward oldStep initial) frontier := by
  constructor
  · rfl
  · intro n hn
    exact (agreement n hn).symm

/-- Preserving local maps along one old path preserves that finite path. -/
theorem protected_forward (oldStep newStep : Nat → State → State)
    (initial : State) (frontier : Nat)
    (agreement : ∀ n, n < frontier →
      newStep n (forward oldStep initial n) = oldStep n (forward oldStep initial n)) :
    ∀ n, n ≤ frontier → forward oldStep initial n = forward newStep initial n :=
  eq_forward newStep initial (forward oldStep initial) frontier
    (protected_defectFree oldStep newStep initial frontier agreement)

/-- Readback agreement on the protected path preserves its published values. -/
theorem protected_readback (oldStep newStep : Nat → State → State)
    (oldRead newRead : Nat → State → Output) (initial : State) (frontier : Nat)
    (agreement : ∀ n, n < frontier →
      newStep n (forward oldStep initial n) = oldStep n (forward oldStep initial n))
    (readAgreement : ∀ n, n ≤ frontier →
      newRead n (forward oldStep initial n) = oldRead n (forward oldStep initial n)) :
    ∀ n, n ≤ frontier →
      newRead n (forward newStep initial n) = oldRead n (forward oldStep initial n) := by
  intro n hn
  rw [← protected_forward oldStep newStep initial frontier agreement n hn]
  exact readAgreement n hn

end CadenceMission.ProtectedTemporalPath

#print axioms CadenceMission.ProtectedTemporalPath.protected_defectFree
#print axioms CadenceMission.ProtectedTemporalPath.protected_forward
#print axioms CadenceMission.ProtectedTemporalPath.protected_readback
