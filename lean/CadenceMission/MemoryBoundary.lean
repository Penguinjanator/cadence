import Mathlib.Data.Fintype.Card
import Mathlib.Logic.Function.Basic

/-!
# Minimal memory: exact representation and update boundaries

These are representation contracts, not a learned-memory algorithm. The query
family is explicit. It may consist of present-value questions, historical
questions, or continuations whose answers include future behaviour. No theorem
asserts that an implementation discovers the relevant query family, sufficient
state, correct writes, or important facts.

The first results characterize precisely when an encoding retains enough
information for every declared query. The later finite examples show why a
causal order or a list of current writer IDs cannot substitute for retained
payloads, and why current-value sufficiency does not imply history sufficiency.

No new axioms and no admitted proofs are used. Classical choice appears only
in the existence of a decoder, not in a claimed executable learning algorithm.
-/

set_option autoImplicit false

namespace CadenceMission.MemoryBoundary

universe u v w z

/-- An encoding is sufficient when histories it merges agree on every query. -/
def Sufficient {H : Type u} {M : Type v} {Q : Type w} {A : Type z}
    (encode : H → M) (answer : H → Q → A) : Prop :=
  ∀ h k, encode h = encode k → ∀ q, answer h q = answer k q

/-- A decoder answers every declared query from the encoded state alone. -/
def Decodes {H : Type u} {M : Type v} {Q : Type w} {A : Type z}
    (encode : H → M) (answer : H → Q → A) (decode : M → Q → A) : Prop :=
  ∀ h q, decode (encode h) q = answer h q

/-- Exact decoding is possible iff no merged histories require different
answers. `Inhabited A` fills unreachable memory states only. The existence
direction uses choice and supplies no efficient decoder construction. -/
theorem decoder_exists_iff {H : Type u} {M : Type v} {Q : Type w}
    {A : Type z} [Inhabited A] (encode : H → M) (answer : H → Q → A) :
    (∃ decode, Decodes encode answer decode) ↔ Sufficient encode answer := by
  constructor
  · rintro ⟨decode, hd⟩ h k he q
    rw [← hd h q, ← hd k q, he]
  · intro hs
    classical
    let decode : M → Q → A := fun m q =>
      if hm : ∃ h, encode h = m then answer (Classical.choose hm) q else default
    refine ⟨decode, ?_⟩
    intro h q
    have hm : ∃ k, encode k = encode h := ⟨h, rfl⟩
    dsimp [decode]
    rw [dif_pos hm]
    exact hs _ _ (Classical.choose_spec hm) q

/-- Any one distinguishing query forbids merging those histories. -/
theorem distinguishing_query_requires_distinct_memory
    {H : Type u} {M : Type v} {Q : Type w} {A : Type z}
    {encode : H → M} {answer : H → Q → A}
    (hs : Sufficient encode answer) {h k : H}
    (hd : ∃ q, answer h q ≠ answer k q) : encode h ≠ encode k := by
  intro he
  obtain ⟨q, hq⟩ := hd
  exact hq (hs h k he q)

/-- If two merged histories require different answers, every decoder is
wrong on at least one of them. This conclusion does not assume sufficiency. -/
theorem encoding_collision_forces_error
    {H : Type u} {M : Type v} {Q : Type w} {A : Type z}
    (encode : H → M) (answer : H → Q → A) (decode : M → Q → A)
    {h k : H} {q : Q} (he : encode h = encode k)
    (hd : answer h q ≠ answer k q) :
    decode (encode h) q ≠ answer h q ∨ decode (encode k) q ≠ answer k q := by
  classical
  by_cases hh : decode (encode h) q = answer h q
  · right
    intro hk
    apply hd
    calc
      answer h q = decode (encode h) q := hh.symm
      _ = decode (encode k) q := congrArg (fun m => decode m q) he
      _ = answer k q := hk
  · exact Or.inl hh

/-- If the declared queries distinguish every history, exact memory must be
injective. No fixed finite store can evade this condition by changing topology. -/
theorem injective_of_all_histories_distinguishable
    {H : Type u} {M : Type v} {Q : Type w} {A : Type z}
    {encode : H → M} {answer : H → Q → A}
    (hs : Sufficient encode answer)
    (hsep : ∀ h k, h ≠ k → ∃ q, answer h q ≠ answer k q) :
    Function.Injective encode := by
  intro h k he
  by_contra hne
  exact distinguishing_query_requires_distinct_memory hs (hsep h k hne) he

/-- Exact finite capacity bound. The units are distinguishable memory states,
not parameters, floating point coordinates, or a calibrated physical entropy. -/
theorem finite_capacity_bound
    {H : Type u} {M : Type v} {Q : Type w} {A : Type z}
    [Fintype H] [Fintype M]
    {encode : H → M} {answer : H → Q → A}
    (hs : Sufficient encode answer)
    (hsep : ∀ h k, h ≠ k → ∃ q, answer h q ≠ answer k q) :
    Fintype.card H ≤ Fintype.card M :=
  Fintype.card_le_of_injective encode
    (injective_of_all_histories_distinguishable hs hsep)

/-- A lower bound needs only a distinguishable family of representatives,
not distinguishability of every raw history. This is the finite behavioural
quotient bound, expressed without requiring a chosen quotient representation. -/
theorem distinguishable_family_capacity_bound
    {H : Type u} {M : Type v} {Q : Type w} {A : Type z}
    {R : Type*} [Fintype R] [Fintype M]
    {encode : H → M} {answer : H → Q → A}
    (hs : Sufficient encode answer) (represent : R → H)
    (hsep : Function.Injective (fun r => answer (represent r))) :
    Fintype.card R ≤ Fintype.card M := by
  apply Fintype.card_le_of_injective (fun r => encode (represent r))
  intro r s he
  apply hsep
  funext q
  exact hs _ _ he q

/-- The full query-answer signature is sufficient. This is a mathematical
reference representation; it can be infinite or computationally inaccessible. -/
theorem signature_is_sufficient {H : Type u} {Q : Type w} {A : Type z}
    (answer : H → Q → A) : Sufficient answer answer := by
  intro h k he q
  exact congrFun he q

/-- Every sufficient memory determines the same behavioural equivalence class.
Equal signatures permit merging histories; sufficiency alone does not force
the encoding to remove irrelevant distinctions. -/
theorem sufficient_refines_behavior
    {H : Type u} {M : Type v} {Q : Type w} {A : Type z}
    {encode : H → M} {answer : H → Q → A}
    (hs : Sufficient encode answer) {h k : H} (he : encode h = encode k) :
    answer h = answer k := by
  funext q
  exact hs h k he q

/-! ## Retained state must also support the next update

Present-query sufficiency does not imply that a proposed compressed state is
closed under future inputs. The next theorem states the additional obligation
exactly. The rollout theorem transports it to every finite continuation. -/

/-- A state-only update exists iff merged histories remain merged after every
admitted next input. This is the exact Markov/sufficient-boundary condition;
no theorem here asserts that a learned encoding satisfies it. -/
theorem state_update_exists_iff
    {H : Type u} {M : Type v} {I : Type w} [Inhabited M]
    (encode : H → M) (step : H → I → H) :
    (∃ update : M → I → M,
      ∀ h input, update (encode h) input = encode (step h input)) ↔
      (∀ h k, encode h = encode k →
        ∀ input, encode (step h input) = encode (step k input)) :=
  decoder_exists_iff encode (fun h input => encode (step h input))

/-- When one-step state updates commute with encoding, so does every finite
input sequence. The theorem is exact and assumes that one-step contract. -/
theorem update_rollout_commutes
    {H : Type u} {M : Type v} {I : Type w}
    (encode : H → M) (step : H → I → H) (update : M → I → M)
    (hu : ∀ h input, update (encode h) input = encode (step h input))
    (inputs : List I) (h : H) :
    inputs.foldl update (encode h) = encode (inputs.foldl step h) := by
  induction inputs generalizing h with
  | nil => rfl
  | cons input rest ih =>
    simp only [List.foldl_cons, hu]
    exact ih (step h input)

/-- An initially exact readout and an exact encoded update reproduce readout
after every admitted finite continuation, not just the present observation. -/
theorem rollout_readout_is_exact
    {H : Type u} {M : Type v} {I : Type w} {O : Type z}
    (encode : H → M) (step : H → I → H) (update : M → I → M)
    (observe : H → O) (decode : M → O)
    (hu : ∀ h input, update (encode h) input = encode (step h input))
    (hd : ∀ h, decode (encode h) = observe h)
    (inputs : List I) (h : H) :
    decode (inputs.foldl update (encode h)) = observe (inputs.foldl step h) := by
  rw [update_rollout_commutes encode step update hu, hd]

/-- Present observations can be decodable even though no state-only next-step
update exists. Here the currently hidden previous bit becomes the next output. -/
theorem current_value_does_not_determine_next_update :
    ¬ ∃ update : Bool → Unit → Bool,
      ∀ h : Bool × Bool, update h.2 () = h.1 := by
  rintro ⟨update, hu⟩
  have hf := hu (false, false)
  have ht := hu (true, false)
  exact Bool.false_ne_true (hf.symm.trans ht)

/-! ## The canonical full future signature

This solves the abstract state-identification problem: histories are equivalent
exactly when every finite future input continuation has the same readout. A
signature has a definitional shift update and its present readout is the empty
continuation. Every exact dynamic representation must distinguish unequal
signatures.

The function space of signatures can be infinite. Equality of signatures can
be undecidable, and evaluating one can be expensive. In particular, a closure
implementing a signature might retain the whole original history internally.
This construction is therefore not a finite-storage implementation, learned
representation, efficient minimization algorithm, or autonomous importance
selector. Its minimality is minimality of the equivalence relation only.
-/

/-- The readout after every finite continuation, including the empty one. -/
def futureSignature {H : Type u} {I : Type v} {O : Type w}
    (step : H → I → H) (observe : H → O) (h : H) : List I → O :=
  fun inputs => observe (inputs.foldl step h)

/-- A definitional update on a supplied signature function: prepend the
witnessed input to any later query. This does not require choosing a history. -/
def shiftSignature {I : Type u} {O : Type v}
    (signature : List I → O) (input : I) : List I → O :=
  fun continuation => signature (input :: continuation)

/-- Present readout is the query with no remaining inputs. -/
def signatureReadout {I : Type u} {O : Type v}
    (signature : List I → O) : O := signature []

theorem future_signature_present_readout
    {H : Type u} {I : Type v} {O : Type w}
    (step : H → I → H) (observe : H → O) (h : H) :
    signatureReadout (futureSignature step observe h) = observe h := rfl

/-- The canonical signature construction intertwines the original update and
the shift update exactly. No choice or learned decoder is used. -/
theorem future_signature_one_step
    {H : Type u} {I : Type v} {O : Type w}
    (step : H → I → H) (observe : H → O) (h : H) (input : I) :
    shiftSignature (futureSignature step observe h) input =
      futureSignature step observe (step h input) := rfl

/-- Behaviourally equivalent histories stay equivalent after the same input. -/
theorem future_equivalence_is_step_stable
    {H : Type u} {I : Type v} {O : Type w}
    (step : H → I → H) (observe : H → O) {h k : H}
    (he : futureSignature step observe h = futureSignature step observe k)
    (input : I) :
    futureSignature step observe (step h input) =
      futureSignature step observe (step k input) :=
  congrArg (fun signature => shiftSignature signature input) he

/-- The supplied signature update remains exact for any finite continuation. -/
theorem future_signature_rollout
    {H : Type u} {I : Type v} {O : Type w}
    (step : H → I → H) (observe : H → O) (inputs : List I) (h : H) :
    inputs.foldl shiftSignature (futureSignature step observe h) =
      futureSignature step observe (inputs.foldl step h) :=
  update_rollout_commutes (futureSignature step observe) step shiftSignature
    (future_signature_one_step step observe) inputs h

theorem future_signature_rollout_readout
    {H : Type u} {I : Type v} {O : Type w}
    (step : H → I → H) (observe : H → O) (inputs : List I) (h : H) :
    signatureReadout (inputs.foldl shiftSignature (futureSignature step observe h)) =
      observe (inputs.foldl step h) :=
  rollout_readout_is_exact (futureSignature step observe) step shiftSignature
    observe signatureReadout (future_signature_one_step step observe)
    (future_signature_present_readout step observe) inputs h

/-- Every exact dynamic memory is sufficient for the full future query family.
The hypotheses concern the original system's every state and every input, not
only finitely many observed training examples. -/
theorem exact_dynamic_memory_is_future_sufficient
    {H : Type u} {M : Type v} {I : Type w} {O : Type z}
    (encode : H → M) (step : H → I → H) (update : M → I → M)
    (observe : H → O) (decode : M → O)
    (hu : ∀ h input, update (encode h) input = encode (step h input))
    (hd : ∀ h, decode (encode h) = observe h) :
    Sufficient encode (futureSignature step observe) := by
  intro h k he inputs
  change observe (inputs.foldl step h) = observe (inputs.foldl step k)
  rw [← rollout_readout_is_exact encode step update observe decode hu hd inputs h,
      ← rollout_readout_is_exact encode step update observe decode hu hd inputs k, he]

/-- Canonical minimality: every encoding with exact state-only updates and
readout refines future-behaviour equality. The future signature itself attains
that equality and has the shift update above. This is not a byte-size bound. -/
theorem every_exact_dynamic_encoding_refines_future_signature
    {H : Type u} {M : Type v} {I : Type w} {O : Type z}
    (encode : H → M) (step : H → I → H) (update : M → I → M)
    (observe : H → O) (decode : M → O)
    (hu : ∀ h input, update (encode h) input = encode (step h input))
    (hd : ∀ h, decode (encode h) = observe h)
    {h k : H} (he : encode h = encode k) :
    futureSignature step observe h = futureSignature step observe k :=
  sufficient_refines_behavior
    (exact_dynamic_memory_is_future_sufficient encode step update observe decode hu hd) he

/-! ## A fixed two-event chain with variable payload

Both histories below have exactly the same node IDs, strict order, and maximal
node. Only the payload differs. The order is a two-element strict chain. -/

def chainBefore (a b : Bool) : Prop := a = false ∧ b = true

theorem chainBefore_irreflexive (a : Bool) : ¬ chainBefore a a := by
  cases a <;> simp [chainBefore]

theorem chainBefore_transitive :
    ∀ a b c, chainBefore a b → chainBefore b c → chainBefore a c := by
  intro a b c hab hbc
  exact False.elim (Bool.false_ne_true (hbc.1.symm.trans hab.2))

/-- The sole maximal node of the fixed strict chain is `true`. -/
theorem chain_maximal_iff (a : Bool) :
    (∀ b, ¬ chainBefore a b) ↔ a = true := by
  cases a <;> simp [chainBefore]

def topologyEncoding (_payload : Bool) : Bool → Bool → Prop := chainBefore
def frontierIDs (_payload : Bool) : List Bool := [true]
def payloadQuery (payload : Bool) (_q : Unit) : Bool := payload

theorem topology_only_is_insufficient :
    ¬ Sufficient topologyEncoding payloadQuery := by
  intro hs
  have h := hs false true rfl ()
  simp [payloadQuery] at h

theorem maximal_ids_only_are_insufficient :
    ¬ Sufficient frontierIDs payloadQuery := by
  intro hs
  have h := hs false true rfl ()
  simp [payloadQuery] at h

/-- Adding the retained payload makes this particular frontier sufficient. -/
def valuedFrontier (payload : Bool) : List Bool × Bool := ([true], payload)

theorem valued_frontier_is_sufficient :
    Sufficient valuedFrontier payloadQuery := by
  intro h k he q
  exact congrArg Prod.snd he

/-! ## Current-value state and its precise query boundary -/

/-- The current finite register map answers current-value queries exactly. -/
theorem current_register_map_is_sufficient {K : Type u} {V : Type v} :
    Sufficient (fun state : K → V => state) (fun state key => state key) := by
  intro h k he q
  exact congrFun he q

/-- A history here consists of an earlier and a current bit. The current bit
alone suffices for current queries, but cannot answer arbitrary past queries. -/
theorem current_value_is_insufficient_for_past :
    ¬ Sufficient (fun h : Bool × Bool => h.2) (fun h (_q : Unit) => h.1) := by
  intro hs
  have h := hs (false, false) (true, false) rfl ()
  simp at h

/-! ## Independent record replacement has a small exact merge law -/

/-- Replacements at distinct keys commute. This is a property of the actual
update rule and disjoint keys, not a consequence of causal incomparability. -/
theorem distinct_key_updates_commute {K : Type u} {V : Type v}
    [DecidableEq K] (state : K → V) {i j : K} (hij : i ≠ j) (x y : V) :
    Function.update (Function.update state i x) j y =
      Function.update (Function.update state j y) i x := by
  exact Function.update_comm hij x y state

/-- Competing replacements at one key retain the last payload. A policy or
conflict rule is necessary; treating them as independent is false. -/
theorem same_key_updates_do_not_commute :
    Function.update (Function.update (fun _ : Unit => false) () false) () true ≠
      Function.update (Function.update (fun _ : Unit => false) () true) () false := by
  intro h
  have hv := congrFun h ()
  simp at hv

#print axioms decoder_exists_iff
#print axioms distinguishing_query_requires_distinct_memory
#print axioms encoding_collision_forces_error
#print axioms finite_capacity_bound
#print axioms distinguishable_family_capacity_bound
#print axioms signature_is_sufficient
#print axioms state_update_exists_iff
#print axioms update_rollout_commutes
#print axioms rollout_readout_is_exact
#print axioms current_value_does_not_determine_next_update
#print axioms future_signature_present_readout
#print axioms future_signature_one_step
#print axioms future_equivalence_is_step_stable
#print axioms future_signature_rollout
#print axioms future_signature_rollout_readout
#print axioms exact_dynamic_memory_is_future_sufficient
#print axioms every_exact_dynamic_encoding_refines_future_signature
#print axioms topology_only_is_insufficient
#print axioms maximal_ids_only_are_insufficient
#print axioms valued_frontier_is_sufficient
#print axioms current_value_is_insufficient_for_past
#print axioms distinct_key_updates_commute
#print axioms same_key_updates_do_not_commute

end CadenceMission.MemoryBoundary
