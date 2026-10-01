import pytest

from transferlab.config import VastConfig
from transferlab.io import read_json
from transferlab.vast import (
    CleanupError,
    RemoteOps,
    VastError,
    cleanup,
    recover_results,
    reserve_budget,
    run_remote,
    select_offers,
)


def offer(**kwargs):
    result = {
        "id": 123,
        "gpu_name": "test-gpu",
        "gpu_ram": 49152,
        "num_gpus": 1,
        "dph_total": 0.5,
        "reliability": 0.99,
        "dlperf": 100,
        "rentable": True,
        "rented": False,
        "verification": "verified",
        "is_bid": False,
        "disk_space": 200,
        "direct_port_count": 1,
        "storage_cost": 0.01,
        "internet_up_cost_per_tb": 1,
        "internet_down_cost_per_tb": 1,
        "cuda_max_good": 13.0,
    }
    return result | kwargs


@pytest.mark.parametrize(
    "change",
    [
        {"gpu_ram": 24000},
        {"reliability": 0.5},
        {"dph_total": 10},
        {"num_gpus": 2},
        {"verification": "unverified"},
        {"is_bid": True},
        {"rented": True},
        {"dph_total": float("nan")},
        {"direct_port_count": 0},
    ],
)
def test_offer_rejections(change):
    assert not select_offers([offer(**change)], VastConfig())


def test_estimated_total_cost_and_ranking():
    assert not select_offers([offer()], VastConfig(max_cost=1))
    offers = select_offers([offer(id=1, dlperf=50), offer(id=2, dlperf=100)], VastConfig())
    assert offers[0].id == 2


def test_budget_reserves_category_and_total(tmp_path):
    path = tmp_path / "budget.json"
    reserve_budget(path, "a", 6, 10, "pilot", 8)
    with pytest.raises(VastError):
        reserve_budget(path, "b", 3, 10, "pilot", 8)
    with pytest.raises(VastError):
        reserve_budget(path, "c", 5, 10, "training", 100)


class FakeAPI:
    def __init__(self, fail_destroy=False, ambiguous=False):
        self.created = []
        self.destroyed = []
        self.live = None
        self.fail_destroy = fail_destroy
        self.ambiguous = ambiguous

    def offers(self, config):
        return [offer()]

    def create(self, ident, config, label):
        self.created.append(ident)
        self.live = {
            "id": 456,
            "label": label,
            "actual_status": "running",
            "ssh_host": "test",
            "ssh_port": 22,
        }
        if self.ambiguous:
            raise VastError("creation response lost")
        return 456

    def instance(self, ident):
        return self.live

    def instances(self):
        return [self.live] if self.live else []

    def destroy(self, ident):
        self.destroyed.append(ident)
        if self.fail_destroy:
            raise VastError("provider unavailable")
        self.live = None

    def stop(self, ident):
        self.live["actual_status"] = "stopped"

    def start(self, ident):
        self.live["actual_status"] = "running"


class FakeOps:
    def __init__(self, failure=None):
        self.failure = failure
        self.collected = 0

    def connect(self, instance):
        return True

    def upload(self, *args):
        if self.failure == "upload":
            raise OSError("upload failed")

    def execute(self, *args):
        if self.failure == "command":
            raise VastError("training failed")
        if self.failure == "timeout":
            raise TimeoutError("runtime cap")
        if self.failure == "interrupt":
            raise KeyboardInterrupt

    def collect(self, *args):
        self.collected += 1
        if self.failure == "collect":
            raise OSError("download failed")


def invoke(tmp_path, api, ops, dry_run=False, restore=None):
    (tmp_path / "project").mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    return run_remote(
        api,
        VastConfig(image="test/image", image_digest="sha256:" + "a" * 64),
        project=tmp_path / "project",
        data=tmp_path / "data",
        output=tmp_path / "output",
        state_dir=tmp_path / "state",
        command=["python", "train.py"],
        ceiling=1000,
        allocation=100,
        category="pilot",
        ops=ops,
        dry_run=dry_run,
        sleep=lambda _: None,
        restore=restore,
    )


@pytest.mark.parametrize("failure", [None, "upload", "command", "timeout", "interrupt", "collect"])
def test_cleanup_on_every_stage(tmp_path, failure):
    api, ops = FakeAPI(), FakeOps(failure)
    if failure:
        with pytest.raises((OSError, VastError, TimeoutError, KeyboardInterrupt)):
            invoke(tmp_path, api, ops)
    else:
        assert invoke(tmp_path, api, ops)["status"] == "destroyed"
    if failure == "collect":
        assert not api.destroyed
        assert api.live["actual_status"] == "stopped"
        assert read_json(tmp_path / "state/active.json")["needs_recovery"]
        assert ops.collected == 3
        with pytest.raises(CleanupError, match="Uncollected"):
            cleanup(api, tmp_path / "state", sleep=lambda _: None)
        recovered = recover_results(
            api, VastConfig(), tmp_path / "state", ops=FakeOps(), sleep=lambda _: None
        )
        assert recovered["collected"]
        assert api.destroyed == [456]
        return
    assert api.destroyed == [456]
    assert ops.collected == 1
    assert not (tmp_path / "state/active.json").exists()


def test_dry_run_never_creates(tmp_path):
    api = FakeAPI()
    assert invoke(tmp_path, api, FakeOps(), True)["dry_run"]
    assert not api.created
    assert not (tmp_path / "state/active.json").exists()


def test_failed_cleanup_retains_recovery_state(tmp_path):
    api = FakeAPI(fail_destroy=True)
    with pytest.raises(CleanupError):
        invoke(tmp_path, api, FakeOps())
    assert read_json(tmp_path / "state/active.json")["instance_id"] == 456
    assert len(api.destroyed) == 4
    with pytest.raises(VastError, match="unresolved"):
        invoke(tmp_path, api, FakeOps())
    api.fail_destroy = False
    assert cleanup(api, tmp_path / "state", sleep=lambda _: None)["status"] == "destroyed"


def test_creation_response_loss_reconciles_by_label(tmp_path):
    api = FakeAPI(ambiguous=True)
    with pytest.raises(VastError, match="lost"):
        invoke(tmp_path, api, FakeOps())
    assert api.destroyed == [456]
    assert not (tmp_path / "state/active.json").exists()


def test_creation_response_loss_and_delayed_visibility_retains_record(tmp_path):
    class HiddenAPI(FakeAPI):
        visible = False

        def instances(self):
            return super().instances() if self.visible else []

    api = HiddenAPI(ambiguous=True)
    with pytest.raises(CleanupError, match="outcome unknown"):
        invoke(tmp_path, api, FakeOps())
    assert api.live is not None
    assert (tmp_path / "state/active.json").exists()
    with pytest.raises(CleanupError):
        cleanup(api, tmp_path / "state", sleep=lambda _: None)
    assert (tmp_path / "state/active.json").exists()
    api.visible = True
    cleanup(api, tmp_path / "state", sleep=lambda _: None)
    assert api.destroyed == [456]


def test_transient_final_collection_failure_retries_without_losing_results(tmp_path):
    class TransientOps(FakeOps):
        def collect(self, *args):
            self.collected += 1
            if self.collected == 1:
                raise OSError("temporary network loss")

    ops = TransientOps()
    result = invoke(tmp_path, FakeAPI(), ops)
    assert result["collected"]
    assert ops.collected == 2


def test_restore_uploads_into_results_before_execution(tmp_path):
    from transferlab.io import write_json

    restore = tmp_path / "previous"
    write_json(restore / "math/run.json", {"status": "failed"})
    config = VastConfig()
    ops = RemoteOps(config, tmp_path / "new")
    ops.endpoint = ("example", 22)
    calls = []
    ops._rsync = lambda *args: calls.append(args)
    ops.restore(restore, 123)
    assert calls[0][0] == str(restore.resolve()) + "/"
    assert calls[0][1].endswith(":/workspace/results/")
    assert calls[0][2] == 123
    order = []

    class RestoringOps(FakeOps):
        def upload(self, *args):
            order.append("upload")

        def restore(self, source, remaining):
            assert source == restore
            order.append("restore")

        def execute(self, *args):
            order.append("execute")

    invoke(tmp_path, FakeAPI(), RestoringOps(), restore=restore)
    assert order == ["upload", "restore", "execute"]


def test_periodic_collection_failure_does_not_abort_remote_process(tmp_path, monkeypatch):
    import subprocess
    import sys

    class LocalOps(RemoteOps):
        def _ssh(self):
            return []

        def collect(self, timeout):
            raise OSError("temporary rsync failure")

    original = subprocess.Popen
    monkeypatch.setattr(
        "transferlab.vast.subprocess.Popen",
        lambda *args, **kwargs: original(
            [sys.executable, "-c", "import time; print('working',flush=True); time.sleep(2.2)"],
            **kwargs,
        ),
    )
    ops = LocalOps(VastConfig(collection_seconds=1), tmp_path)
    ops.execute(["ignored"], 10)
    assert "working" in (tmp_path / "remote.log").read_text()
    assert (tmp_path / "collection-errors.jsonl").exists()
