"""Adversarial checks of proof inventory and the axiom audit's failure paths."""

import unittest

import check


class AuditTests(unittest.TestCase):
    def test_nested_comments_cannot_add_fake_declarations(self):
        source = """/- theorem fake /- axiom nested -/ sorry -/
namespace Test
noncomputable section
theorem actual : True := by trivial
end
end Test
"""
        self.assertEqual(check.theorem_names(source), ["Test.actual"])

    def test_admission_and_new_axiom_are_rejected(self):
        for source in ("theorem bad : False := by sorry", "axiom bad : False",
                       "private axiom bad : False"):
            with self.assertRaises(ValueError):
                check.theorem_names(source)

    def test_unsupported_declaration_syntax_cannot_silently_skip_a_theorem(self):
        for source in ("@[simp] theorem unseen : True := by trivial",
                       "private theorem unseen : True := by trivial"):
            with self.assertRaises(ValueError):
                check.theorem_names(source)

    def test_lean_foundations_are_allowed_but_sorry_and_new_axioms_fail(self):
        allowed = "'Test.actual' depends on axioms: [propext, Classical.choice, Quot.sound]"
        self.assertEqual(len(check.parse_axioms(allowed, ["Test.actual"])), 1)
        for axiom in ("sorryAx", "Test.unproved_assumption"):
            with self.assertRaises(ValueError):
                check.parse_axioms(f"'Test.actual' depends on axioms: [{axiom}]", ["Test.actual"])

    def test_missing_or_duplicate_audit_output_is_rejected(self):
        line = "'Test.actual' does not depend on any axioms"
        for output in ("", line + "\n" + line):
            with self.assertRaises(ValueError):
                check.parse_axioms(output, ["Test.actual"])

    def test_actual_lean_version_must_match_the_pinned_toolchain(self):
        pinned = "leanprover/lean4:v4.29.1"
        check.check_toolchain_version("Lean (version 4.29.1, x86_64-unknown-linux-gnu)", pinned)
        for output in ("Lean (version 4.28.0, x86_64-unknown-linux-gnu)",
                       "Lean (version 4.29.1-rc1, x86_64-unknown-linux-gnu)", "unknown"):
            with self.assertRaises(ValueError):
                check.check_toolchain_version(output, pinned)


if __name__ == "__main__":
    unittest.main()
