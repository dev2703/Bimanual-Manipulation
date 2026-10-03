from bimanual.evaluation.aloha_predicates import handoff_succeeded


def test_handoff_gate_requires_receiver_support_after_carrier_release():
    assert handoff_succeeded(0.02, 0.13, 0.10, 0.11, 0.037, 0.008, 30)
    assert not handoff_succeeded(0.02, 0.13, 0.02, 0.02, 0.037, 0.008, 30)
    assert not handoff_succeeded(0.02, 0.13, 0.10, 0.11, 0.037, 0.008, 0)
