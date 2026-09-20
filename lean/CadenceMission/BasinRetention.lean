import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

/-!
# Record-basin retention under bounded changes of the repair energy

The local move relation, a record basin, its observable readout, and a
reference-energy exit margin are supplied hypotheses. If changing the energy
by at most `epsilon` at each state leaves a strictly positive exit margin,
accepted non-increasing-energy repair cannot leave the basin. The energy may
change at every step of a finite trace. Strict-descent repair is included.

This is a nonconvex retention contract: interior states and their energies may
change, and several microscopic endpoints may encode the same record. No
termination, global minimization, confluence, basin discovery, useful record
selection, or successful revision is asserted. The move relation is fixed:
changing topology or permitting larger jumps requires a new margin check.
The state space need not be finite; only the trace is finite. In particular,
a continuous potential barrier alone does not establish the exit hypothesis
for moves that can jump over it. All energies are mathematical real numbers;
no floating-point implementation or empirical margin is certified here.
-/

set_option autoImplicit false

namespace CadenceMission.BasinRetention

universe u v

/-- The permitted local exits from the basin cost at least `margin` under
the reference energy. This includes the support and size of allowed moves. -/
def ExitMargin {S : Type u} (localMove : S → S → Prop) (basin : Set S)
    (energy : S → ℝ) (margin : ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ basin → y ∉ basin → localMove x y →
    margin ≤ energy y - energy x

/-- A uniform bound relative to one reference energy, not merely a bound
on successive increments that could accumulate without limit. -/
def EnergyBound {S : Type u} (energy reference : S → ℝ)
    (epsilon : ℝ) : Prop :=
  ∀ x, |energy x - reference x| ≤ epsilon

/-- Acceptance is allowed to include equal-energy moves. The retention
theorem therefore also covers strictly decreasing accepted moves. -/
def AcceptedStep {S : Type u} (localMove : S → S → Prop)
    (energy : S → ℝ) (x y : S) : Prop :=
  localMove x y ∧ energy y ≤ energy x

/-- A finite trace with its own energy at each step. Its states and energy
sequence are supplied; no optimizer or scheduler is selected here. -/
def FiniteTrace {S : Type u} (localMove : S → S → Prop)
    (energies : Nat → S → ℝ) (states : Nat → S) (length : Nat) : Prop :=
  ∀ t, t < length → AcceptedStep localMove (energies t) (states t) (states (t + 1))

/-- Each endpoint can consume at most `epsilon` of the old exit margin. -/
theorem perturbed_exit_gap {S : Type u}
    {localMove : S → S → Prop} {basin : Set S}
    {reference energy : S → ℝ} {margin epsilon : ℝ}
    (hmargin : ExitMargin localMove basin reference margin)
    (hbound : EnergyBound energy reference epsilon)
    {x y : S} (hx : x ∈ basin) (hy : y ∉ basin) (hmove : localMove x y) :
    margin - 2 * epsilon ≤ energy y - energy x := by
  have hgap := hmargin hx hy hmove
  have hxBound := (abs_le.mp (hbound x)).2
  have hyBound := (abs_le.mp (hbound y)).1
  linarith

/-- Bounded energy changes cannot make an allowed basin exit acceptable
while the perturbed exit margin remains strictly positive. -/
theorem accepted_step_preserves_basin {S : Type u}
    {localMove : S → S → Prop} {basin : Set S}
    {reference energy : S → ℝ} {margin epsilon : ℝ}
    (hmargin : ExitMargin localMove basin reference margin)
    (hbound : EnergyBound energy reference epsilon)
    (hroom : 2 * epsilon < margin)
    {x y : S} (hx : x ∈ basin) (hstep : AcceptedStep localMove energy x y) :
    y ∈ basin := by
  by_contra hy
  have hgap := perturbed_exit_gap hmargin hbound hx hy hstep.1
  have haccepted := hstep.2
  linarith

/-- Every state in a finite trace remains in the record basin. The energy
may vary with time, provided every step stays within the same reference
margin budget. No fairness or microscopic endpoint uniqueness is needed. -/
theorem finite_trace_preserves_basin {S : Type u}
    {localMove : S → S → Prop} {basin : Set S}
    {reference : S → ℝ} {energies : Nat → S → ℝ}
    {states : Nat → S} {length : Nat} {margin epsilon : ℝ}
    (hmargin : ExitMargin localMove basin reference margin)
    (hbound : ∀ t, t < length → EnergyBound (energies t) reference epsilon)
    (hroom : 2 * epsilon < margin)
    (hstart : states 0 ∈ basin)
    (htrace : FiniteTrace localMove energies states length) :
    ∀ t, t ≤ length → states t ∈ basin := by
  intro t
  induction t with
  | zero => intro _; exact hstart
  | succ t ih =>
    intro ht
    have hlt : t < length := Nat.lt_of_succ_le ht
    exact accepted_step_preserves_basin hmargin (hbound t hlt) hroom
      (ih (Nat.le_of_succ_le ht)) (htrace t hlt)

/-- A record constant on the protected basin survives the complete finite
trace, although its hidden representative can move and fail to be unique. -/
theorem finite_trace_preserves_readout {S : Type u} {R : Type v}
    {localMove : S → S → Prop} {basin : Set S}
    {reference : S → ℝ} {energies : Nat → S → ℝ}
    {states : Nat → S} {length : Nat} {margin epsilon : ℝ}
    (hmargin : ExitMargin localMove basin reference margin)
    (hbound : ∀ t, t < length → EnergyBound (energies t) reference epsilon)
    (hroom : 2 * epsilon < margin)
    (hstart : states 0 ∈ basin)
    (htrace : FiniteTrace localMove energies states length)
    (readout : S → R) (record : R)
    (hrecord : ∀ x ∈ basin, readout x = record) :
    ∀ t, t ≤ length → readout (states t) = record := by
  intro t ht
  exact hrecord (states t)
    (finite_trace_preserves_basin hmargin hbound hroom hstart htrace t ht)

/-! Two-state sharpness checks. The baseline basin is `{false}`; exiting it
costs 2. Detuning raises its energy by `epsilon` and lowers the other by the
same amount. At `epsilon = 1` an equal-energy exit becomes acceptable; above
that threshold even a strictly decreasing exit is possible. -/

def twoStateReference (state : Bool) : ℝ := if state then 2 else 0

def twoStateDetuned (epsilon : ℝ) (state : Bool) : ℝ :=
  if state then 2 - epsilon else epsilon

/-- The strict safety inequality cannot be weakened to equality for an
acceptance rule that permits equal-energy moves. -/
theorem threshold_allows_equal_energy_exit :
    ExitMargin (fun _ _ : Bool => True) {false} twoStateReference 2 ∧
    EnergyBound (twoStateDetuned 1) twoStateReference 1 ∧
    AcceptedStep (fun _ _ : Bool => True) (twoStateDetuned 1) false true ∧
    false ∈ ({false} : Set Bool) ∧ true ∉ ({false} : Set Bool) := by
  constructor
  · intro x y hx hy _
    have hx' : x = false := hx
    subst x
    cases y with
    | false => exact False.elim (hy (by simp))
    | true => norm_num [twoStateReference]
  constructor
  · intro x
    cases x <;> norm_num [twoStateDetuned, twoStateReference]
  norm_num [AcceptedStep, twoStateDetuned]

/-- Beyond the half-margin bound, the same finite fixture admits a strict
energy-decreasing exit, with the stated perturbation size verified. -/
theorem above_threshold_allows_strict_exit {epsilon : ℝ} (hepsilon : 1 < epsilon) :
    EnergyBound (twoStateDetuned epsilon) twoStateReference epsilon ∧
    twoStateDetuned epsilon true < twoStateDetuned epsilon false := by
  have hepsilon0 : 0 ≤ epsilon := by linarith
  constructor
  · intro x
    cases x with
    | false => simp [twoStateDetuned, twoStateReference, abs_of_nonneg hepsilon0]
    | true =>
      change |2 - epsilon - 2| ≤ epsilon
      have hnegative : 2 - epsilon - 2 = -epsilon := by linarith
      rw [hnegative]
      simp [abs_of_nonneg hepsilon0]
  · change 2 - epsilon < epsilon
    linarith

end CadenceMission.BasinRetention
