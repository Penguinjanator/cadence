"""A ring of neurons, a declared protocol, a shuffled control, a conformance check, a receipt.

Run:  python examples/ring_protocol.py
"""

from __future__ import annotations

from pathlib import Path

import cadence as cd

n = 12
connectome = cd.Connectome.from_synapses(
    n,
    pre=list(range(n)),
    post=[(i + 1) % n for i in range(n)],
    count=[60] * n,
    populations={"head": [0], "middle": [n // 2], "tail": [n - 1]},
)
neuron_model = cd.NeuronModel()
protocol = cd.Protocol(
    stimuli={"rest": (), "poke": ("head",)},
    training=[("poke", "tail", "active")],
    rows=[
        cd.Row("R1", "rest", "tail", "inactive", "nothing in, nothing out"),
        cd.Row(
            "R2", "poke", "tail", "reduced", "the ring is cut in the middle", ablate=("middle",)
        ),
        cd.Row("R3", "poke", "middle", "active", "the middle lights before the tail"),
    ],
    steps=120,
)

gain, table = cd.select_gain(
    lambda g: cd.Brain(connectome, neuron_model.replace(gain=g)),
    protocol,
    (0.01, 0.02, 0.03, 0.05),
    sparsity_cap=None,  # a twelve-neuron ring lights entirely; the 5% cap is for large nets
)
brain = cd.Brain(connectome, neuron_model.replace(gain=gain))
score = protocol.score(brain)
control = protocol.score(cd.Brain(cd.shuffled(connectome, seed=0), neuron_model.replace(gain=gain)))
conformance = cd.conformance(brain, connectome.members("head"), steps=120)

print(f"gain {gain}: connectome {score['passed']}/{score['total']}, ", end="")
print(f"shuffled {control['passed']}/{control['total']}")
print(f"conformance deviation {conformance['max_abs_deviation']:.1e}, ", end="")
print(f"ledger clean {conformance['ledger']['clean']}")

receipt = cd.Receipt.build(
    "examples/ring-protocol/v1",
    {
        "connectome": connectome.summary(),
        "rule": brain.neuron_model.to_dict(),
        "gain_selection": {"selected": gain, "table": table},
        "protocol": protocol.to_dict(),
        "score": score,
        "control": control,
        "conformance": conformance,
        "boundary": {"toy": True, "no_claim_beyond_scored_predicates": True},
    },
    sources=[("examples/ring_protocol.py", Path(__file__))],
)
path = receipt.write(Path("ring_protocol_receipt.json"))


def check(body: dict) -> str | None:  # type: ignore[type-arg]
    for row in body["score"]["rows"]:
        if row["passed"] != cd.evaluate_predicate(
            row["predicate"], row["reading"], row["reference"]
        ):
            return f"row {row['id']} pass flag does not follow from its readings"
    return None


print(
    path,
    cd.Receipt.verify(path, sources=[("examples/ring_protocol.py", Path(__file__))], check=check),
)
