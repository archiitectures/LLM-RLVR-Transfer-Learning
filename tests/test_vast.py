import pytest

from transferlab.config import VastConfig
from transferlab.io import read_json
from transferlab.vast import (
    CleanupError,
    VastError,
    cleanup,
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


def invoke(tmp_path, api, ops, dry_run=False):
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
    )


@pytest.mark.parametrize("failure", [None, "upload", "command", "timeout", "interrupt", "collect"])
def test_cleanup_on_every_stage(tmp_path, failure):
    api, ops = FakeAPI(), FakeOps(failure)
    if failure:
        with pytest.raises((OSError, VastError, TimeoutError, KeyboardInterrupt)):
            invoke(tmp_path, api, ops)
    else:
        assert invoke(tmp_path, api, ops)["status"] == "destroyed"
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
