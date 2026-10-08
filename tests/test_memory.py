from heavden_api.chat.memory import ConversationStore, as_messages, tool_note


def _store(**kwargs):
    now = [0.0]
    store = ConversationStore(clock=lambda: now[0], **kwargs)
    return store, now


def test_unknown_or_malformed_ids_start_a_new_conversation():
    store, _ = _store()
    cid, turns = store.resolve(None)
    assert ConversationStore.valid_id(cid) and turns == []
    for bad in ["", "not-hex" * 5, "A" * 32, "a" * 31]:
        new, turns = store.resolve(bad)
        assert new != bad and turns == []


def test_turns_are_remembered_and_capped():
    store, _ = _store(max_turns=2)
    cid, _ = store.resolve(None)
    for i in range(3):
        store.remember(cid, f"q{i}", f"a{i}", [])
    same, turns = store.resolve(cid)
    assert same == cid and [t.question for t in turns] == ["q1", "q2"]


def test_idle_conversations_expire():
    store, now = _store(ttl_seconds=60)
    cid, _ = store.resolve(None)
    store.remember(cid, "q", "a", [])
    now[0] = 59
    assert store.resolve(cid)[0] == cid  # touching it keeps it alive
    now[0] = 59 + 61
    assert store.resolve(cid)[0] != cid


def test_least_recently_used_conversation_is_dropped_first():
    store, now = _store(max_conversations=2)
    ids = []
    for i in range(3):
        now[0] = i
        cid, _ = store.resolve(None)
        store.remember(cid, "q", "a", [])
        ids.append(cid)
    assert len(store) == 2
    assert store.resolve(ids[0])[0] != ids[0] and store.resolve(ids[2])[0] == ids[2]


def test_tool_note_keeps_what_follow_ups_need():
    steps = [
        {
            "tool": "query_gold",
            "purpose": "alerts by unit",
            "sql": "SELECT unit_id,\n  count(*) FROM gold.alerts_fact GROUP BY 1",
            "columns": ["unit_id", "n"],
            "rows": [["SITE_A-GENERAL", 58]],
        },
        {"tool": "query_gold", "sql": "DROP", "error": "rejected"},  # failed queries are left out
        {"tool": "search_documents", "query": "High band response"},
        {
            "tool": "explain_patient",
            "result": {
                "patient_label": "P-1735",
                "encounter_id": "E-100941",
                "unit_id": "SITE_C-GENERAL",
                "risk": 0.04,
                "risk_band": "High",
            },
        },
    ]
    note = tool_note(steps)
    assert "SELECT unit_id, count(*) FROM gold.alerts_fact GROUP BY 1" in note
    assert "SITE_A-GENERAL" in note and "DROP" not in note
    assert "High band response" in note and "P-1735" in note and "E-100941" in note


def test_messages_carry_the_note_after_the_answer():
    store, _ = _store()
    cid, _ = store.resolve(None)
    store.remember(
        cid, "Who is highest risk?", "P-1735.", [{"tool": "search_documents", "query": "x"}]
    )
    messages = as_messages(store.resolve(cid)[1])
    assert messages[0] == {"role": "user", "content": "Who is highest risk?"}
    assert (
        messages[1]["content"].startswith("P-1735.")
        and "Context from this turn" in messages[1]["content"]
    )
