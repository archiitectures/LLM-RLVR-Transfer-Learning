from __future__ import annotations

import fcntl
import math
import os
import selectors
import shlex
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from .config import VastConfig
from .io import append_jsonl, read_json, timestamp, write_json


class VastError(RuntimeError):
    pass


class CleanupError(VastError):
    pass


class VastAPI:
    def __init__(self):
        self._key = os.environ.get("VAST_API_KEY")
        if not self._key:
            raise VastError("Set VAST_API_KEY; it is never sent to the training instance")

    def _request(self, method: str, endpoint: str, payload=None):
        try:
            response = httpx.request(
                method,
                "https://console.vast.ai/api/v0/" + endpoint,
                json=payload,
                headers={"Authorization": "Bearer " + self._key},
                timeout=30,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            value = response.json()
        except httpx.HTTPStatusError as exc:
            # Do not echo HTTP bodies: external error messages can contain submitted secrets.
            raise VastError(
                f"Vast API {method} {endpoint}: HTTP {exc.response.status_code}"
            ) from None
        except httpx.RequestError:
            raise VastError(f"Vast API {method} {endpoint}: network failure") from None
        if isinstance(value, dict) and value.get("success") is False:
            raise VastError(f"Vast API {method} {endpoint}: unsuccessful response")
        return value

    def offers(self, config: VastConfig) -> list[dict]:
        result = self._request(
            "POST",
            "bundles/",
            {
                "type": "on-demand",
                "limit": 100,
                "verified": {"eq": True},
                "rentable": {"eq": True},
                "rented": {"eq": False},
                "num_gpus": {"eq": 1},
                "gpu_ram": {"gte": config.min_vram_gb * 1024},
                "reliability": {"gte": config.min_reliability},
                "dph_total": {"lte": config.max_rate},
                "disk_space": {"gte": config.disk_gb},
                "direct_port_count": {"gte": 1},
                "cuda_max_good": {"gte": config.min_cuda_version},
                "order": [["dlperf_per_dphtotal", "desc"]],
            },
        )
        return result["offers"] if result else []

    def create(self, offer_id: int, config: VastConfig, label: str) -> int:
        if not config.image or not config.image_digest:
            raise VastError(
                "Set vast.image and its sha256 image_digest after a verified GPU/sandbox pilot"
            )
        image = config.image.split("@")[0] + "@" + config.image_digest
        result = self._request(
            "PUT",
            f"asks/{offer_id}/",
            {
                "client_id": "me",
                "image": image,
                "disk": config.disk_gb,
                "runtype": "ssh_direct",
                "label": label,
                "cancel_unavail": True,
            },
        )
        if not result or "new_contract" not in result:
            raise VastError("Creation returned no instance ID; reconcile the persisted job label")
        return int(result["new_contract"])

    def instances(self) -> list[dict]:
        result = self._request("GET", "instances/")
        return result["instances"]

    def instance(self, instance_id: int) -> dict | None:
        return next((r for r in self.instances() if int(r["id"]) == instance_id), None)

    def destroy(self, instance_id: int) -> None:
        self._request("DELETE", f"instances/{instance_id}/")


@dataclass
class Offer:
    id: int
    gpu: str
    vram_gb: float
    hourly: float
    reliability: float
    performance: float
    exposure_usd: float
    transfer_rate_unknown: bool


def select_offers(rows: list[dict], config: VastConfig) -> list[Offer]:
    candidates = []
    for row in rows:
        try:
            vram = float(row["gpu_ram"]) / 1024
            reliability = float(row["reliability"])
            # storage_cost is USD per GB per month. Include a conservative allocated-disk charge.
            hourly = (
                float(row["dph_total"]) + float(row.get("storage_cost", 0)) * config.disk_gb / 720
            )
            performance = float(row.get("dlperf", 0) or 0)
            up, down = row.get("internet_up_cost_per_tb"), row.get("internet_down_cost_per_tb")
            unknown = up is None or down is None
            # Reserve 100 GB in each direction, plus setup/collection headroom; this is an estimate.
            transfer = 0.1 * (float(up or 0) + float(down or 0))
            exposure = hourly * config.max_hours + transfer + 3.0
            values = (vram, reliability, hourly, performance, exposure)
            if not all(math.isfinite(v) for v in values) or hourly <= 0:
                continue
            if (
                vram < config.min_vram_gb
                or reliability < config.min_reliability
                or hourly > config.max_rate
                or exposure > config.max_cost
                or row.get("num_gpus") != 1
                or not row.get("rentable")
                or row.get("rented")
                or row.get("verification") != "verified"
                or row.get("is_bid")
                or float(row.get("disk_space", 0)) < config.disk_gb
                or int(row.get("direct_port_count", 0)) < 1
                or float(row.get("cuda_max_good", 0)) < config.min_cuda_version
            ):
                continue
            candidates.append(
                Offer(
                    int(row["id"]),
                    str(row["gpu_name"]),
                    vram,
                    hourly,
                    reliability,
                    performance,
                    exposure,
                    unknown,
                )
            )
        except (ValueError, TypeError, KeyError):
            continue
    return sorted(
        candidates, key=lambda x: (-(x.performance / x.hourly), -x.reliability, x.hourly, x.id)
    )


@contextmanager
def state_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "controller.lock").open("w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise VastError("Another controller is active") from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def reserve_budget(
    path: Path, job: str, exposure: float, ceiling: float, category: str, allocation: float
) -> None:
    ledger = read_json(path) if path.exists() else {"jobs": {}}
    if job in ledger["jobs"]:
        raise VastError("Duplicate spending reservation")
    committed = sum(j["reserved_usd"] for j in ledger["jobs"].values())
    category_committed = sum(
        j["reserved_usd"] for j in ledger["jobs"].values() if j["category"] == category
    )
    if committed + exposure > ceiling or category_committed + exposure > allocation:
        raise VastError("Study/category spending ceiling would be exceeded")
    ledger["jobs"][job] = {"reserved_usd": exposure, "category": category, "status": "reserved"}
    write_json(path, ledger)


def verify_destroyed(api, instance_id: int, sleep=time.sleep) -> None:
    last_error = None
    for attempt in range(4):
        try:
            api.destroy(instance_id)
            if api.instance(instance_id) is None:
                return
        except Exception as exc:
            last_error = exc
        sleep(min(2**attempt, 8))
    raise CleanupError(
        f"Could not verify destruction of instance {instance_id}; reservation retained. Run vast cleanup. Last error: {type(last_error).__name__}"
    )


class RemoteOps:
    def __init__(self, config: VastConfig, local_output: Path):
        self.config = config
        self.output = local_output
        self.endpoint = None

    def connect(self, instance: dict) -> bool:
        host, port = instance.get("ssh_host"), instance.get("ssh_port")
        if not host or not port:
            return False
        self.endpoint = (str(host), int(port))
        result = subprocess.run(self._ssh() + ["true"], capture_output=True, timeout=20)
        return result.returncode == 0

    def _ssh(self) -> list[str]:
        if self.endpoint is None:
            raise VastError("SSH endpoint not established")
        host, port = self.endpoint
        options = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-p",
            str(port),
        ]
        if self.config.ssh_key:
            options += ["-i", self.config.ssh_key]
        return options + ["root@" + host]

    def _rsync(self, source: str, target: str, timeout: float, excludes: list[str] = ()) -> None:
        if timeout <= 0:
            raise TimeoutError("No transfer time remains inside rental deadline")
        ssh = self._ssh()[:-1]
        subprocess.run(
            ["rsync", "-a", "--partial", "--safe-links", "--timeout=60", "-e", shlex.join(ssh)]
            + [f"--exclude={x}" for x in excludes]
            + [source, target],
            check=True,
            timeout=max(1, timeout),
        )

    def upload(self, project: Path, data: Path, remaining: float) -> None:
        subprocess.run(
            self._ssh() + ["mkdir -p /workspace/project /workspace/data /workspace/results"],
            check=True,
            timeout=20,
        )
        excludes = [
            ".git",
            ".venv",
            "__pycache__",
            ".env*",
            ".aws",
            ".codex",
            ".agents",
            ".transferlab",
            "runs",
            "data",
            "reports",
            ".pytest_cache",
            ".ruff_cache",
            ".cache",
            ".ssh",
            "*.pem",
            "*.key",
        ]
        started = time.monotonic()
        self._rsync(
            str(project.resolve()) + "/",
            self._ssh()[-1] + ":/workspace/project/",
            remaining,
            excludes,
        )
        self._rsync(
            str(data.resolve()) + "/",
            self._ssh()[-1] + ":/workspace/data/",
            remaining - (time.monotonic() - started),
        )

    def execute(self, command: list[str], remaining: float) -> None:
        # Remote GNU timeout enforces the command cap even if the controller disconnects.
        wrapped = (
            "cd /workspace/project && timeout --signal=TERM --kill-after=30s "
            + str(max(1, int(remaining)))
            + "s "
            + shlex.join(command)
        )
        process = subprocess.Popen(
            self._ssh() + [wrapped], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        deadline = time.monotonic() + remaining
        next_collect = time.monotonic() + self.config.collection_seconds
        self.output.mkdir(parents=True, exist_ok=True)
        try:
            with (
                (self.output / "remote.log").open("ab") as log,
                selectors.DefaultSelector() as selector,
            ):
                selector.register(process.stdout, selectors.EVENT_READ)
                while process.poll() is None:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Local rental watchdog expired")
                    for key, _ in selector.select(timeout=0.5):
                        block = key.fileobj.read1(4096)
                        if block:
                            log.write(block)
                            log.flush()
                            print(block.decode(errors="replace"), end="", flush=True)
                    if time.monotonic() >= next_collect:
                        self.collect(min(60, max(1, deadline - time.monotonic())))
                        next_collect = time.monotonic() + self.config.collection_seconds
                tail = process.stdout.read()
                log.write(tail)
            if process.returncode:
                raise VastError(f"Remote command failed with exit code {process.returncode}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            process.wait(timeout=5)
            process.stdout.close()

    def collect(self, timeout: float = 120) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        self._rsync(
            self._ssh()[-1] + ":/workspace/results/", str(self.output.resolve()) + "/", timeout
        )


def run_remote(
    api,
    config: VastConfig,
    *,
    project: Path,
    data: Path,
    output: Path,
    state_dir: Path,
    command: list[str],
    ceiling: float,
    allocation: float,
    category: str,
    dry_run: bool = False,
    ops=None,
    sleep=time.sleep,
) -> dict:
    if not project.is_dir() or not data.is_dir() or not command:
        raise VastError("Project/data paths must exist and command must be nonempty")
    offers = select_offers(api.offers(config), config)
    if not offers:
        raise VastError(
            "No compatible offers satisfy resource, price, reliability, and exposure limits"
        )
    selected = offers[0]
    if dry_run:
        return {
            "dry_run": True,
            "selected": asdict(selected),
            "candidates": [asdict(o) for o in offers[:5]],
        }
    if selected.transfer_rate_unknown:
        raise VastError("Selected offer has unknown transfer fees; cannot reserve bounded exposure")
    if not config.image or not config.image_digest or not config.image_digest.startswith("sha256:"):
        raise VastError("A verified image and sha256 digest are required before renting")
    if not config.controller_required:
        raise VastError("v1 requires an online controller for billing cleanup")
    operations = ops or RemoteOps(config, output)
    with state_lock(state_dir):
        active_path = state_dir / "active.json"
        if active_path.exists():
            raise VastError("An unresolved rental exists; run vast status/cleanup before renting")
        label = "transferlab-" + uuid.uuid4().hex
        reserve_budget(
            state_dir / "budget.json", label, selected.exposure_usd, ceiling, category, allocation
        )
        state = {
            "job": label,
            "instance_id": None,
            "created": timestamp(),
            "offer": asdict(selected),
            "output": str(output.resolve()),
            "status": "creation_requested",
        }
        write_json(active_path, state)
        started = time.monotonic()
        deadline = started + config.max_hours * 3600
        instance_id = None
        error = None
        collected = False
        try:
            instance_id = api.create(selected.id, config, label)
            state.update(instance_id=instance_id, status="provisioning")
            write_json(active_path, state)
            readiness_deadline = min(deadline, started + config.readiness_seconds)
            while time.monotonic() < readiness_deadline:
                instance = api.instance(instance_id)
                if instance is None or instance.get("actual_status") in {
                    "exited",
                    "offline",
                    "unknown",
                    "stopped",
                }:
                    raise VastError(
                        "Instance entered a failed/disappeared state during provisioning"
                    )
                if instance.get("actual_status") == "running":
                    try:
                        if operations.connect(instance):
                            break
                    except (OSError, subprocess.SubprocessError):
                        pass
                sleep(5)
            else:
                raise TimeoutError("SSH readiness timeout")
            state["status"] = "uploading"
            write_json(active_path, state)
            operations.upload(project, data, deadline - time.monotonic() - 120)
            state["status"] = "running"
            write_json(active_path, state)
            if deadline - time.monotonic() <= 120:
                raise TimeoutError("No job time remains after setup and collection reservation")
            operations.execute(command, deadline - time.monotonic() - 120)
        except BaseException as exc:
            error = exc
        finally:
            if instance_id is None:
                # A lost creation response must not lead to duplicate rentals or an invisible orphan.
                try:
                    matches = [r for r in api.instances() if r.get("label") == label]
                except Exception as exc:
                    raise CleanupError(
                        f"Creation outcome unknown for job {label}; recovery state and spending reservation retained"
                    ) from exc
                if len(matches) > 1:
                    raise CleanupError(
                        "Multiple instances match creation label; reconcile explicitly"
                    )
                if matches:
                    instance_id = int(matches[0]["id"])
                    state["instance_id"] = instance_id
                    write_json(active_path, state)
            if instance_id is not None:
                try:
                    operations.collect(max(1, min(120, deadline - time.monotonic())))
                    collected = True
                except BaseException as exc:
                    if error is None:
                        error = exc
                state["status"] = "cleanup"
                write_json(active_path, state)
                verify_destroyed(api, instance_id, sleep)
            state.update(
                status="destroyed" if instance_id is not None else "not_created",
                collected=collected,
                seconds=time.monotonic() - started,
                finished=timestamp(),
                error=type(error).__name__ if error else None,
            )
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "rental.json", state)
            append_jsonl(state_dir / "lifecycle.jsonl", state)
            ledger = read_json(state_dir / "budget.json")
            ledger["jobs"][label]["status"] = state["status"]
            # Keep worst-case reservations rather than freeing money based on approximate billing.
            write_json(state_dir / "budget.json", ledger)
            active_path.unlink()
        if error:
            raise error
        return state


def cleanup(api, state_dir: Path, *, sleep=time.sleep) -> dict:
    with state_lock(state_dir):
        path = state_dir / "active.json"
        if not path.exists():
            return {"status": "no_tracked_rental"}
        state = read_json(path)
        matches = [r for r in api.instances() if r.get("label") == state["job"]]
        ids = {int(r["id"]) for r in matches}
        if state["instance_id"] is not None:
            ids.add(state["instance_id"])
        for ident in ids:
            verify_destroyed(api, ident, sleep)
        state["status"] = "destroyed"
        append_jsonl(state_dir / "lifecycle.jsonl", state)
        path.unlink()
        return state
