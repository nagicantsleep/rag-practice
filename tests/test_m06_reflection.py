from rag_practice.adaptive.reflection import ActiveRetrievalPolicy, ReflectionCritic


def test_active_retrieval_policy_uses_explicit_confidence_threshold() -> None:
    policy = ActiveRetrievalPolicy(confidence_threshold=0.6)
    assert policy.should_retrieve(0.59)
    assert not policy.should_retrieve(0.60)


def test_reflection_critic_separates_relevance_support_and_retrieve() -> None:
    critic = ReflectionCritic(
        relevance_threshold=0.1,
        support_threshold=0.8,
        active_policy=ActiveRetrievalPolicy(confidence_threshold=0.7),
    )
    signals = critic.reflect(
        question="Which protocol does Vega use?",
        answer="Vega uses Raft",
        contexts=["Vega database replicas use the Raft consensus protocol."],
        generation_confidence=0.5,
    )
    assert signals.retrieve
    assert signals.relevant
    assert signals.supported
    assert signals.utility > 0.9


def test_reflection_critic_detects_unsupported_answer() -> None:
    critic = ReflectionCritic(support_threshold=0.8)
    signals = critic.reflect(
        question="Which protocol does Vega use?",
        answer="Vega uses Paxos",
        contexts=["Vega database replicas use the Raft consensus protocol."],
        generation_confidence=0.9,
    )
    assert signals.relevant
    assert not signals.supported
    assert not signals.retrieve
