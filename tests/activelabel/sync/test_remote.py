from activelabel.labels import Label
from activelabel.sync.remote import RemoteInfo, RemoteItem, RemoteJob


def test_remote_records_have_sensible_defaults():
    assert RemoteInfo("1", "P", "http://x/1").images == 0
    item = RemoteItem("7:0", "a.png", 10, 10)
    assert item.labels == [] and item.key.partition(":")[0] == "7"
    item.labels.append(Label("cat"))
    assert RemoteItem("7:1", "b.png", 10, 10).labels == []  # no list shared between items
    job = RemoteJob("701", 1, 0, 9)
    assert (job.state, job.stage, job.assignee) == ("", "", "")
