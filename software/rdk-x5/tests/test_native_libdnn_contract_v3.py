from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any

import numpy as np
import pytest


ROOTSCOPE = Path(__file__).resolve().parents[1]
WORKSPACE = ROOTSCOPE.parent
if str(ROOTSCOPE) not in sys.path:
    sys.path.insert(0, str(ROOTSCOPE))

from app.runtime_v3 import native_libdnn_adapter as native


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class LockedAssets:
    model: Path
    model_sha256: str
    worker: Path
    worker_sha256: str
    source: Path
    source_sha256: str
    contract: Path


@pytest.fixture
def locked_assets(tmp_path: Path) -> LockedAssets:
    model = tmp_path / "model.bin"
    worker = tmp_path / "rootscope-native-libdnn-worker"
    source = tmp_path / "rootscope_libdnn_worker.cpp"
    contract = tmp_path / "compile_contract_x5.v1.json"
    model.write_bytes(b"fake hash-bound model bytes")
    worker.write_bytes(b"fake hash-bound worker bytes")
    source.write_text("// fake worker source\n", encoding="utf-8")
    model.chmod(0o444)
    worker.chmod(0o555)
    source.chmod(0o444)
    model_sha256 = _sha256(model)
    worker_sha256 = _sha256(worker)
    source_sha256 = _sha256(source)
    contract.write_text(
        json.dumps(
            {
                "schema": "rootscope.native-libdnn.x5-compile-contract.v1",
                "status": "PASS_REPRODUCIBLE_TWO_BUILD",
                "target_arch": "aarch64",
                "protocol": native.PROTOCOL_SCHEMA,
                "source": {
                    "path": source.name,
                    "sha256": source_sha256,
                },
                "reproducibility": {
                    "independent_build_count": 2,
                    "build_1_sha256": worker_sha256,
                    "build_2_sha256": worker_sha256,
                    "byte_identical": True,
                },
                "binary": {
                    "package_path": "bin/rootscope-native-libdnn-worker",
                    "bytes": worker.stat().st_size,
                    "sha256": worker_sha256,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    assets = LockedAssets(
        model=model,
        model_sha256=model_sha256,
        worker=worker,
        worker_sha256=worker_sha256,
        source=source,
        source_sha256=source_sha256,
        contract=contract,
    )
    try:
        yield assets
    finally:
        # Windows maps chmod to the read-only attribute; restore write access so
        # pytest can remove the temporary directory.
        for path in (model, worker, source):
            if path.exists():
                path.chmod(0o666)


class RecordingStdin:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False
        self.flush_count = 0

    def write(self, value: bytes) -> int:
        if self.closed:
            raise BrokenPipeError("synthetic stdin is closed")
        self.data.extend(value)
        return len(value)

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.closed = True


class FakeWorkerProcess:
    """Pipe-compatible subprocess double; it never executes a host program."""

    def __init__(self, response: bytes) -> None:
        read_fd, write_fd = os.pipe()
        try:
            if response:
                os.write(write_fd, response)
        finally:
            os.close(write_fd)
        self.stdin = RecordingStdin()
        self.stdout = os.fdopen(read_fd, "rb", buffering=0)
        self.pid = 424242
        self.returncode: int | None = None
        self.wait_observed_stdin_eof = False
        self.wait_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.wait_calls += 1
        self.wait_observed_stdin_eof = self.stdin.closed
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


def _valid_response(
    *,
    magic: bytes = native.RESPONSE_MAGIC,
    version: int = native.PROTOCOL_VERSION,
    response_id: int = 1,
    status: int = 0,
    payload_size: int = native.LOGITS.size,
    logits: tuple[float, float, float, float] = (0.1, 0.2, 0.9, -0.4),
) -> bytes:
    return native.RESPONSE.pack(
        magic,
        version,
        response_id,
        status,
        2_500_000,
        payload_size,
    ) + native.LOGITS.pack(*logits)


def _launch_fake(
    monkeypatch: pytest.MonkeyPatch,
    assets: LockedAssets,
    response: bytes,
) -> tuple[native.PersistentNativeLibdnnR7Adapter, FakeWorkerProcess, dict[str, Any]]:
    process = FakeWorkerProcess(response)
    captured: dict[str, Any] = {}

    def fake_popen(command: list[str], **kwargs: Any) -> FakeWorkerProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(native.subprocess, "Popen", fake_popen)
    adapter = native.PersistentNativeLibdnnR7Adapter(
        assets.model,
        assets.model_sha256,
        worker_path=assets.worker,
        compile_contract_path=assets.contract,
    )
    return adapter, process, captured


def test_packaged_compile_contract_binds_source_and_two_identical_binaries() -> None:
    contract_path = (
        ROOTSCOPE / "app/runtime_v3/native/compile_contract_x5.v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    source = WORKSPACE / contract["source"]["path"]
    binary = WORKSPACE / contract["binary"]["source_path"]
    binary_sha256 = _sha256(binary)

    assert contract["schema"] == (
        "rootscope.native-libdnn.x5-compile-contract.v1"
    )
    assert contract["status"] == "PASS_REPRODUCIBLE_TWO_BUILD"
    assert contract["target_arch"] == "aarch64"
    assert contract["protocol"] == native.PROTOCOL_SCHEMA
    assert source.is_file()
    assert _sha256(source) == contract["source"]["sha256"]
    assert binary.is_file()
    assert binary.stat().st_size == contract["binary"]["bytes"]
    assert binary_sha256 == contract["binary"]["sha256"]
    assert contract["binary"]["package_path"] == (
        "bin/rootscope-native-libdnn-worker"
    )
    reproducibility = contract["reproducibility"]
    assert reproducibility["independent_build_count"] == 2
    assert reproducibility["byte_identical"] is True
    assert reproducibility["build_1_sha256"] == binary_sha256
    assert reproducibility["build_2_sha256"] == binary_sha256


def test_contract_or_worker_tamper_fails_before_process_launch(
    monkeypatch: pytest.MonkeyPatch,
    locked_assets: LockedAssets,
) -> None:
    launched = False

    def forbidden_popen(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        nonlocal launched
        launched = True
        raise AssertionError("tampered artifact must fail before launch")

    monkeypatch.setattr(native.subprocess, "Popen", forbidden_popen)
    payload = json.loads(locked_assets.contract.read_text(encoding="utf-8"))
    payload["binary"]["sha256"] = "0" * 64
    locked_assets.contract.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(native.NativeLibdnnContractError, match="SHA-256 mismatch"):
        native.PersistentNativeLibdnnR7Adapter(
            locked_assets.model,
            locked_assets.model_sha256,
            worker_path=locked_assets.worker,
            compile_contract_path=locked_assets.contract,
        )
    assert launched is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema", "rootscope.native-libdnn.x5-compile-contract.v0"),
        ("status", "UNVERIFIED"),
        ("target_arch", "x86_64"),
        ("protocol", "rootscope.native-libdnn.protocol.v0"),
    ),
)
def test_compile_contract_metadata_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    locked_assets: LockedAssets,
    field: str,
    value: str,
) -> None:
    monkeypatch.setattr(
        native.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("must not launch a worker"),
    )
    payload = json.loads(locked_assets.contract.read_text(encoding="utf-8"))
    payload[field] = value
    locked_assets.contract.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(native.NativeLibdnnContractError, match=field):
        native.PersistentNativeLibdnnR7Adapter(
            locked_assets.model,
            locked_assets.model_sha256,
            worker_path=locked_assets.worker,
            compile_contract_path=locked_assets.contract,
        )


def test_strict_request_response_protocol_and_clean_stdin_eof(
    monkeypatch: pytest.MonkeyPatch,
    locked_assets: LockedAssets,
) -> None:
    adapter, process, captured = _launch_fake(
        monkeypatch, locked_assets, _valid_response()
    )
    tensor = np.arange(native.INPUT_BYTES, dtype=np.uint8).reshape(
        native.INPUT_SHAPE
    )[:, :, :, ::-1]
    expected_payload = np.ascontiguousarray(tensor).tobytes(order="C")
    try:
        result = adapter.infer_uint8(tensor)
        request_bytes = bytes(process.stdin.data)
        header = request_bytes[: native.REQUEST.size]
        payload = request_bytes[native.REQUEST.size :]
        magic, version, request_id, payload_size = native.REQUEST.unpack(header)

        assert magic == native.REQUEST_MAGIC
        assert version == native.PROTOCOL_VERSION
        assert request_id == 1
        assert payload_size == native.INPUT_BYTES
        assert payload == expected_payload
        assert result["logits"] == pytest.approx([0.1, 0.2, 0.9, -0.4])
        assert result["top1_index"] == 2
        assert result["worker_inference_ms"] == 2.5
        assert result["inference_count_since_load"] == 1
        assert result["persistent_model"] is True
        assert result["cold_load_per_inference"] is False
        assert result["external_tensor_sha256"] == hashlib.sha256(
            expected_payload
        ).hexdigest()
        assert all(value is False for value in result["authority"].values())

        command = captured["command"]
        assert command == [
            str(locked_assets.worker.resolve()),
            "--model",
            str(locked_assets.model.resolve()),
            "--model-sha256",
            locked_assets.model_sha256,
            "--expected-model-name",
            native.EXPECTED_MODEL_NAME,
        ]
        assert captured["kwargs"]["shell"] is False
        assert captured["kwargs"]["close_fds"] is True
        assert captured["kwargs"]["env"]["PATH"] == (
            "/usr/sbin:/usr/bin:/sbin:/bin"
        )
    finally:
        exit_code = adapter.close()

    assert exit_code == 0
    assert adapter.closed is True
    assert process.stdin.closed is True
    assert process.wait_observed_stdin_eof is True
    assert process.wait_calls == 1
    assert process.terminate_calls == 0
    assert process.kill_calls == 0


@pytest.mark.parametrize(
    "tensor",
    (
        np.zeros((1, 3, 224, 223), dtype=np.uint8),
        np.zeros(native.INPUT_SHAPE, dtype=np.float32),
        np.zeros((3, 224, 224), dtype=np.uint8),
    ),
)
def test_tensor_shape_and_dtype_are_exact_and_fail_before_protocol_write(
    monkeypatch: pytest.MonkeyPatch,
    locked_assets: LockedAssets,
    tensor: np.ndarray,
) -> None:
    adapter, process, _ = _launch_fake(
        monkeypatch, locked_assets, _valid_response()
    )
    try:
        with pytest.raises(
            native.NativeLibdnnContractError,
            match="external tensor must be uint8",
        ):
            adapter.infer_uint8(tensor)
        assert bytes(process.stdin.data) == b""
        assert adapter.inference_count == 0
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("response", "message"),
    (
        (
            _valid_response(magic=b"BADMAGIC"),
            "response header changed",
        ),
        (
            _valid_response(version=2),
            "response header changed",
        ),
        (
            _valid_response(response_id=99),
            "response id mismatch",
        ),
        (
            _valid_response(status=1),
            "response status/shape mismatch",
        ),
        (
            _valid_response(payload_size=8),
            "response status/shape mismatch",
        ),
        (
            _valid_response(logits=(0.1, float("nan"), 0.2, 0.3)),
            "non-finite logits",
        ),
    ),
)
def test_malformed_worker_response_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    locked_assets: LockedAssets,
    response: bytes,
    message: str,
) -> None:
    adapter, _, _ = _launch_fake(monkeypatch, locked_assets, response)
    try:
        with pytest.raises(native.NativeLibdnnContractError, match=message):
            adapter.infer_uint8(
                np.zeros(native.INPUT_SHAPE, dtype=np.uint8)
            )
        assert adapter.inference_count == 0
    finally:
        adapter.close()


def test_unexpected_worker_eof_fails_closed_and_close_remains_clean(
    monkeypatch: pytest.MonkeyPatch,
    locked_assets: LockedAssets,
) -> None:
    adapter, process, _ = _launch_fake(monkeypatch, locked_assets, b"")
    try:
        with pytest.raises(
            native.NativeLibdnnContractError,
            match="closed its response pipe",
        ):
            adapter.infer_uint8(
                np.zeros(native.INPUT_SHAPE, dtype=np.uint8)
            )
    finally:
        assert adapter.close() == 0
    assert process.wait_observed_stdin_eof is True
    assert process.stdin.closed is True
