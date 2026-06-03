import importlib


def _fresh_store(path):
    import recent_store
    importlib.reload(recent_store)
    recent_store._STORE_PATH = path  # point at a temp file
    return recent_store


def test_add_and_list_newest_first(tmp_path):
    s = _fresh_store(str(tmp_path / "recent.json"))
    a = s.add_recent({"question": "q1", "kind": "equity"})
    b = s.add_recent({"question": "q2", "kind": "raw"})
    items = s.list_recent()
    assert [i["question"] for i in items] == ["q2", "q1"]
    assert a["id"] != b["id"] and "ts" in a


def test_dedup_same_question_and_kind(tmp_path):
    s = _fresh_store(str(tmp_path / "recent.json"))
    s.add_recent({"question": "same", "kind": "equity"})
    s.add_recent({"question": "same", "kind": "equity"})
    assert len(s.list_recent()) == 1


def test_cap_at_60(tmp_path):
    s = _fresh_store(str(tmp_path / "recent.json"))
    for i in range(70):
        s.add_recent({"question": f"q{i}", "kind": "equity"})
    items = s.list_recent()
    assert len(items) == 60
    assert items[0]["question"] == "q69"  # newest kept
