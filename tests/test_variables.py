import copy

import variables as v


def test_variables_are_valid():
    assert v.validate(v.load()) == []


def test_slot_times():
    assert v.slot_times("A") == {
        "briefing": ("09:15", "09:55"),
        "lab": ("09:55", "11:05"),
        "debrief": ("11:05", "11:15"),
    }
    assert v.slot_times("B")["lab"] == ("13:10", "14:20")
    assert v.slot_times("C")["debrief"] == ("16:35", "16:45")
    assert v.capstone_times()["share-outs"] == ("16:15", "16:45")


def test_validator_catches_problems():
    bad = copy.deepcopy(v.load())
    bad["modules"]["m01"]["exercises"].append({"n": 9, "title": "x", "minutes": 30, "writes": []})
    bad["modules"]["m02"]["slot"] = "C"
    bad["clock"]["blocks"][3]["end"] = "11:40"
    problems = " ".join(v.validate(bad))
    assert "over the slot-A limit" in problems
    assert "says slot C" in problems
    assert "gap or overlap" in problems
