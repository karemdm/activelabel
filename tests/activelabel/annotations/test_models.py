import json

from activelabel.annotations.models import ParseResult, Problem
from activelabel.labels import Label


def test_problem_counts_and_keeps_a_few_distinct_examples():
    problem = Problem("warning", "x", "message")
    for i in range(30):
        problem.add(f"e{i % 8}")
    problem.add(None, 2)
    assert problem.count == 32 and len(problem.examples) <= 20
    assert len(set(problem.examples)) == len(problem.examples)
    assert json.dumps(problem.as_dict())


def test_parse_result_summary():
    result = ParseResult("yolo", {"1": [Label("cat", bbox=[0, 0, 1, 1])], "2": []}, ["cat"], ["a.txt"], 3)
    assert result.images_matched == 2 and len(result.all_labels()) == 1
    summary = result.as_dict()
    assert summary["images_total"] == 3 and summary["labels_total"] == 1 and summary["classes"] == ["cat"]
    json.dumps(summary)
