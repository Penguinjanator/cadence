import Mathlib.Data.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
An exact scalar boundary message for a fixed quadratic temporal model.

Eliminating an older latent state preserves every later objective that reads
only the retained boundary. Two quadratic coefficients suffice for later
minimization; an additive constant is needed to preserve absolute energy.
The model coefficients are fixed. This does not prove sufficient memory for
arbitrary data, nonlinear models, later parameter changes or semantic facts.
It does not infer sensor precision, source independence or importance.
-/

namespace CadenceMission.QuadraticBoundary

noncomputable section

def precision (p q a : ℝ) : ℝ := p + q * a ^ 2

def oldMinimum (p q a b u y : ℝ) : ℝ :=
  (b + q * a * (y - u)) / precision p q a

def fullEnergy (p q a b u x y : ℝ) (later : ℝ → ℝ) : ℝ :=
  p * x ^ 2 - 2 * b * x + q * (y - a * x - u) ^ 2 + later y

def reducedEnergy (p q a b u y : ℝ) (later : ℝ → ℝ) : ℝ :=
  q * (y - u) ^ 2 - (b + q * a * (y - u)) ^ 2 / precision p q a + later y

theorem precision_pos (p q a : ℝ) (hp : 0 < p) (hq : 0 ≤ q) :
    0 < precision p q a := by
  unfold precision
  have h := mul_nonneg hq (sq_nonneg a)
  linarith

theorem elimination_identity (p q a b u x y : ℝ) (later : ℝ → ℝ)
    (h : precision p q a ≠ 0) :
    fullEnergy p q a b u x y later =
      precision p q a * (x - oldMinimum p q a b u y) ^ 2 +
        reducedEnergy p q a b u y later := by
  unfold fullEnergy oldMinimum reducedEnergy
  field_simp
  unfold precision at *
  field_simp at *
  nlinarith [sq_nonneg (p + q * a ^ 2)]

theorem reduced_attained (p q a b u y : ℝ) (later : ℝ → ℝ)
    (h : precision p q a ≠ 0) :
    fullEnergy p q a b u (oldMinimum p q a b u y) y later =
      reducedEnergy p q a b u y later := by
  rw [elimination_identity p q a b u _ y later h]
  simp

theorem reduced_lower_bound (p q a b u x y : ℝ) (later : ℝ → ℝ)
    (hp : 0 < p) (hq : 0 ≤ q) :
    reducedEnergy p q a b u y later ≤ fullEnergy p q a b u x y later := by
  have hd := precision_pos p q a hp hq
  rw [elimination_identity p q a b u x y later (ne_of_gt hd)]
  have h := mul_nonneg (le_of_lt hd) (sq_nonneg (x - oldMinimum p q a b u y))
  linarith

theorem boundary_coefficients (p q a b u y : ℝ) (later : ℝ → ℝ)
    (h : precision p q a ≠ 0) :
    reducedEnergy p q a b u y later =
      (q * p / precision p q a) * y ^ 2 -
      2 * (q * (p * u + a * b) / precision p q a) * y +
      (q * p * u ^ 2 + 2 * b * q * a * u - b ^ 2) / precision p q a + later y := by
  unfold reducedEnergy
  field_simp
  unfold precision
  ring

theorem admit_observation (p b c r z y : ℝ) :
    p * y ^ 2 - 2 * b * y + c + r * (y - z) ^ 2 =
      (p + r) * y ^ 2 - 2 * (b + r * z) * y + (c + r * z ^ 2) := by
  ring

end
end CadenceMission.QuadraticBoundary

#print axioms CadenceMission.QuadraticBoundary.precision_pos
#print axioms CadenceMission.QuadraticBoundary.elimination_identity
#print axioms CadenceMission.QuadraticBoundary.reduced_attained
#print axioms CadenceMission.QuadraticBoundary.reduced_lower_bound
#print axioms CadenceMission.QuadraticBoundary.boundary_coefficients
#print axioms CadenceMission.QuadraticBoundary.admit_observation
