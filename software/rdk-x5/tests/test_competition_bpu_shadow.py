from __future__ import annotations

import hashlib
import socket
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from app.competition_runtime.bpu_shadow_client import BpuShadowClient
from app.competition_runtime.bpu_shadow_protocol import (
    OUTPUT_SHAPE,
    R7_REFERENCE_SHA256,
    TENSOR_SHAPE,
    ZERO_AUTHORITY,
    BpuShadowProtocolError,
    make_request,
    pack_frame,
    recv_frame,
    send_frame,
)
from app.competition_runtime.bpu_shadow_worker import (
    HashBoundR7BpuBackend,
    LegacyPyeasyR7BpuBackend,
    UnixBpuShadowWorker,
)
from app.edge.bpu_seed17 import Seed17BpuContractError


class _Properties:
    def __init__(self, shape: tuple[int, ...], *, is_input: bool) -> None:
        self.name = "image" if is_input else "logits"
        self.shape = list(shape)
        self.validShape = list(shape)
        self.alignedShape = list(shape)
        self.layout = "HB_DNN_LAYOUT_NCHW" if is_input else "HB_DNN_LAYOUT_NC"
        self.dtype = "uint8" if is_input else "float32"
        self.tensor_type = (
            "HB_DNN_IMG_TYPE_RGB" if is_input else "HB_DNN_TENSOR_TYPE_F32"
        )


class _Tensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        is_input: bool,
        buffer_dtype: np.dtype | None = None,
    ) -> None:
        self.properties = _Properties(shape, is_input=is_input)
        if buffer_dtype is None:
            buffer_dtype = np.dtype(np.uint8 if is_input else np.float32)
        self.buffer = np.zeros(
            shape, dtype=buffer_dtype
        )


class _Value:
    def __init__(self, buffer: np.ndarray) -> None:
        self.buffer = buffer


class _FakeModel:
    def __init__(
        self,
        *,
        delay_s: float = 0.0,
        output_shape: tuple[int, ...] = OUTPUT_SHAPE,
        input_buffer_dtype: np.dtype = np.dtype(np.uint8),
    ) -> None:
        self.inputs = [
            _Tensor(
                TENSOR_SHAPE,
                is_input=True,
                buffer_dtype=input_buffer_dtype,
            )
        ]
        self.outputs = [_Tensor(output_shape, is_input=False)]
        self.delay_s = delay_s
        self.output_shape = output_shape
        self.forward_calls = 0

    def forward(self, tensor: np.ndarray):
        self.forward_calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        seed = float(np.mean(tensor, dtype=np.float64) / 255.0)
        logits = np.asarray(
            [[seed, seed + 1.0, seed - 1.0, 0.25]],
            dtype=np.float32,
        ).reshape(self.output_shape)
        return [_Value(logits)]


class _FakeDnn:
    def __init__(self, model: _FakeModel) -> None:
        self.model = model
        self.load_calls: list[str] = []

    def load(self, path: str):
        self.load_calls.append(path)
        return [self.model]


_FAKE_HRT_MODEL_INFO = """
This model file has 1 model:
[rootscope_seed17_resnet18_224x224_rgb_ddr_r7_default_int16_all_nodes]
[model name]: rootscope_seed17_resnet18_224x224_rgb_ddr_r7_default_int16_all_nodes
input[0]:
name: image
valid shape: (1,3,224,224,)
aligned shape: (1,3,224,256,)
aligned byte size: 172032
tensor type: HB_DNN_IMG_TYPE_RGB
tensor layout: HB_DNN_LAYOUT_NCHW
stride: (172032,57344,256,1,)
output[0]:
name: logits
valid shape: (1,4,1,1,)
aligned shape: (1,4,1,1,)
aligned byte size: 16
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NCHW
stride: (16,4,4,4,)
"""


class _FakeHrtRunner:
    def __init__(
        self,
        *,
        corrupt_input_dump: bool = False,
        output_bytes: int = 16,
    ) -> None:
        self.calls: list[list[str]] = []
        self.infer_calls = 0
        self.raw_inputs: list[bytes] = []
        self.corrupt_input_dump = corrupt_input_dump
        self.output_bytes = output_bytes

    def __call__(self, command, **_kwargs):
        command = [str(item) for item in command]
        self.calls.append(command)
        if command[1:] == ["--version"]:
            return subprocess.CompletedProcess(command, 0, "1.24.5\n", "")
        if command[1] == "model_info":
            return subprocess.CompletedProcess(command, 0, _FAKE_HRT_MODEL_INFO, "")
        assert command[1] == "infer"
        self.infer_calls += 1
        input_path = Path(command[command.index("--input_file") + 1])
        dump_path = Path(command[command.index("--dump_path") + 1])
        raw = input_path.read_bytes()
        self.raw_inputs.append(raw)
        tensor = np.frombuffer(raw, dtype=np.uint8).reshape(TENSOR_SHAPE)
        seed = float(np.mean(tensor, dtype=np.float64) / 255.0)
        logits = np.asarray(
            [seed, seed + 1.0, seed - 1.0, 0.25],
            dtype="<f4",
        )
        dumped_input = bytearray(raw)
        if self.corrupt_input_dump:
            dumped_input[0] ^= 0x80
        (dump_path / "model_infer_input_0_image.bin").write_bytes(dumped_input)
        output = logits.tobytes()
        if self.output_bytes != len(output):
            output = output[: self.output_bytes].ljust(self.output_bytes, b"\0")
        (dump_path / "model_infer_output_0_logits.bin").write_bytes(output)
        return subprocess.CompletedProcess(command, 0, "Infer time: 0.1 ms\n", "")


def _tensor(seed: int) -> np.ndarray:
    return np.full(TENSOR_SHAPE, seed, dtype=np.uint8)


def _cpu_outputs(tensors):
    return [
        np.asarray([[0.0, 0.0, 0.0, float(index)]], dtype=np.float32)
        for index, _tensor_value in enumerate(tensors)
    ]


def _backend(directory: Path):
    model_path = directory / "r7.bin"
    model_path.write_bytes(b"fake-r7-bin-for-protocol-tests")
    executable_path = directory / "hrt_model_exec"
    executable_path.write_bytes(b"fake-hrt-model-exec")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    runner = _FakeHrtRunner()
    backend = HashBoundR7BpuBackend(
        model_path,
        digest,
        executable_path=executable_path,
        work_root=directory / "hrt-work",
        command_runner=runner,
    )
    return backend, runner, digest


def _run_one_connection(worker: UnixBpuShadowWorker):
    server, client = socket.socketpair()
    thread = threading.Thread(
        target=lambda: _serve_and_close(worker, server),
        daemon=True,
    )
    thread.start()
    return client, thread


def _serve_and_close(worker: UnixBpuShadowWorker, server: socket.socket) -> None:
    with server:
        worker.serve_connection(server)


def test_fake_hrt_protocol_supports_one_and_four_tensors_with_truthful_backend():
    with tempfile.TemporaryDirectory() as directory_string:
        backend, runner, digest = _backend(Path(directory_string))
        worker = UnixBpuShadowWorker(
            Path(directory_string) / "unused.sock",
            backend,
        )

        for count in (1, 4):
            client_socket, thread = _run_one_connection(worker)
            client = BpuShadowClient(
                Path(directory_string) / "unused.sock",
                expected_model_sha256=digest,
                connector=lambda _timeout, sock=client_socket: sock,
            )
            result = client.infer_or_cpu(
                [_tensor(index + 1) for index in range(count)],
                cpu_fallback=_cpu_outputs,
            )
            thread.join(timeout=2.0)
            assert not thread.is_alive()
            assert result["status"] == "BPU_SHADOW_OK"
            assert result["used_cpu_fallback"] is False
            assert result["backend_actual"] == "FAKE_HRT_MODEL_EXEC_UNIT_TEST_ONLY"
            assert len(result["logits"]) == count
            assert result["authority"] == ZERO_AUTHORITY
            assert result["zero_authority"] is True
            assert result["model"]["qualification"] == "SHADOW_CANDIDATE_NOT_DEFAULT"
            assert result["model"]["selected_bin_changed"] is False
        assert runner.infer_calls == 5


def test_canonical_hrt_metadata_and_1x4x1x1_output_are_strictly_canonicalized():
    with tempfile.TemporaryDirectory() as directory_string:
        directory = Path(directory_string)
        backend, _runner, digest = _backend(directory)
        worker = UnixBpuShadowWorker(directory / "unused.sock", backend)
        client_socket, thread = _run_one_connection(worker)
        client = BpuShadowClient(
            directory / "unused.sock",
            expected_model_sha256=digest,
            connector=lambda _timeout: client_socket,
        )
        result = client.infer_tensors([_tensor(12)], cpu_fallback=_cpu_outputs)
        thread.join(timeout=2.0)
        assert result["status"] == "BPU_SHADOW_OK"
        assert len(result["logits"][0]) == 4
        assert result["bpu_results"][0]["logits"] == result["logits"][0]
        assert result["backend"]["actual_metadata"]["compiled_output_shape"] == [
            1,
            4,
            1,
            1,
        ]
        assert result["backend"]["actual_metadata"]["canonical_logits_shape"] == [1, 4]
        assert backend.compiled_output_shape == (1, 4, 1, 1)
        assert backend.output_metadata["valid_shape"] == [1, 4, 1, 1]


def test_canonical_hrt_keeps_uint8_valid_shape_and_vendor_owns_centering_alignment():
    with tempfile.TemporaryDirectory() as directory_string:
        directory = Path(directory_string)
        backend, runner, _digest = _backend(directory)
        worker = UnixBpuShadowWorker(directory / "unused.sock", backend)
        source = _tensor(240)
        response = worker.process_message(make_request("canonical-uint8", [source]))
        metadata = response["backend"]["actual_metadata"]
        assert response["status"] == "OK_SHADOW_ONLY"
        assert metadata["wire_input_dtype"] == "uint8"
        assert metadata["declared_input_dtype"] == "uint8"
        assert metadata["runtime_input_buffer_dtype"] == "uint8"
        assert metadata["accepted_runtime_input_buffer_dtypes"] == ["uint8"]
        assert metadata["input_adapter"] == (
            "HRT_MODEL_EXEC_VALID_UINT8_VENDOR_OWNS_RGB128_AND_ALIGNMENT"
        )
        assert metadata["host_must_not_center_or_pad"] is True
        assert metadata["cold_load_per_inference"] is True
        assert metadata["real_time_qualified"] is False
        assert runner.raw_inputs == [source.tobytes(order="C")]
        assert len(runner.raw_inputs[0]) == 150_528
        assert response["backend"]["actual_metadata"]["input"]["aligned_shape"] == [
            1,
            3,
            224,
            256,
        ]
        assert response["interface"]["input_dtype"] == "uint8"


def test_explicit_legacy_pyeasy_rejects_unobserved_input_backing_dtype():
    with tempfile.TemporaryDirectory() as directory_string:
        directory = Path(directory_string)
        model_path = directory / "r7.bin"
        model_path.write_bytes(b"fake-r7-float-backing-buffer")
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        with pytest.raises(
            Seed17BpuContractError,
            match="backing buffer must be uint8",
        ):
            LegacyPyeasyR7BpuBackend(
                model_path,
                digest,
                dnn_module=_FakeDnn(
                    _FakeModel(input_buffer_dtype=np.dtype(np.float32))
                ),
            )


def test_explicit_legacy_pyeasy_rejects_any_other_output_shape():
    with tempfile.TemporaryDirectory() as directory_string:
        directory = Path(directory_string)
        model_path = directory / "r7.bin"
        model_path.write_bytes(b"fake-r7-invalid-output-metadata")
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        with pytest.raises(
            Seed17BpuContractError,
            match=r"exactly \(1, 4\) or \(1, 4, 1, 1\)",
        ):
            LegacyPyeasyR7BpuBackend(
                model_path,
                digest,
                dnn_module=_FakeDnn(_FakeModel(output_shape=(1, 4, 2, 1))),
            )


def test_worker_returns_fail_closed_error_for_truncated_frame():
    with tempfile.TemporaryDirectory() as directory_string:
        backend, _runner, _digest = _backend(Path(directory_string))
        worker = UnixBpuShadowWorker(
            Path(directory_string) / "unused.sock",
            backend,
        )
        server, client = socket.socketpair()
        thread = threading.Thread(
            target=lambda: _serve_and_close(worker, server),
            daemon=True,
        )
        thread.start()
        with client:
            client.sendall(struct.pack(">I", 200) + b'{"schema":"cut')
            client.shutdown(socket.SHUT_WR)
            response = recv_frame(client)
        thread.join(timeout=2.0)
        assert response["status"] == "ERROR_FAIL_CLOSED"
        assert response["error"]["code"] == "TRUNCATED_FRAME"
        assert response["zero_authority"] is True
        assert response["authority"] == ZERO_AUTHORITY


def test_worker_rejects_wrong_tensor_shape_before_fake_forward():
    with tempfile.TemporaryDirectory() as directory_string:
        backend, runner, _digest = _backend(Path(directory_string))
        worker = UnixBpuShadowWorker(
            Path(directory_string) / "unused.sock",
            backend,
        )
        request = dict(make_request("wrong-shape", [_tensor(3)]))
        tensor_payload = dict(request["tensors"][0])
        tensor_payload["shape"] = [1, 3, 224, 223]
        request["tensors"] = [tensor_payload]
        server, client = socket.socketpair()
        thread = threading.Thread(
            target=lambda: _serve_and_close(worker, server),
            daemon=True,
        )
        thread.start()
        with client:
            send_frame(client, request)
            response = recv_frame(client)
        thread.join(timeout=2.0)
        assert response["status"] == "ERROR_FAIL_CLOSED"
        assert response["error"]["code"] == "INVALID_TENSOR_SHAPE"
        assert runner.infer_calls == 0


def test_client_short_timeout_invokes_cpu_fallback():
    server, client_socket = socket.socketpair()
    fallback_calls: list[int] = []

    def slow_server() -> None:
        with server:
            recv_frame(server)
            time.sleep(0.15)

    def fallback(tensors):
        fallback_calls.append(len(tensors))
        return _cpu_outputs(tensors)

    thread = threading.Thread(target=slow_server, daemon=True)
    thread.start()
    client = BpuShadowClient(
        Path(tempfile.gettempdir()) / "unused.sock",
        expected_model_sha256=R7_REFERENCE_SHA256,
        timeout_s=0.03,
        connector=lambda _timeout: client_socket,
    )
    result = client.infer_or_cpu([_tensor(9)], cpu_fallback=fallback)
    thread.join(timeout=1.0)
    assert result["status"] == "CPU_FALLBACK_OK"
    assert result["used_cpu_fallback"] is True
    assert result["backend_actual"] == "CPU_FALLBACK"
    assert fallback_calls == [1]
    assert result["authority"] == ZERO_AUTHORITY


def test_model_hash_is_checked_before_fake_hrt_runner_is_called():
    with tempfile.TemporaryDirectory() as directory_string:
        directory = Path(directory_string)
        model_path = directory / "r7.bin"
        model_path.write_bytes(b"fake-r7-bin-for-hash-test")
        executable_path = directory / "hrt_model_exec"
        executable_path.write_bytes(b"fake-hrt")
        runner = _FakeHrtRunner()
        with pytest.raises(Seed17BpuContractError, match="SHA-256 mismatch"):
            HashBoundR7BpuBackend(
                model_path,
                "0" * 64,
                executable_path=executable_path,
                work_root=directory / "work",
                command_runner=runner,
            )
        assert runner.calls == []


def test_client_rejects_worker_model_hash_mismatch_and_uses_cpu():
    with tempfile.TemporaryDirectory() as directory_string:
        backend, _runner, actual_digest = _backend(Path(directory_string))
        worker = UnixBpuShadowWorker(
            Path(directory_string) / "unused.sock",
            backend,
        )
        client_socket, thread = _run_one_connection(worker)
        wrong_expected = "a" * 64
        assert wrong_expected != actual_digest
        client = BpuShadowClient(
            Path(directory_string) / "unused.sock",
            expected_model_sha256=wrong_expected,
            connector=lambda _timeout: client_socket,
        )
        result = client.infer_or_cpu([_tensor(4)], cpu_fallback=_cpu_outputs)
        thread.join(timeout=2.0)
        assert result["status"] == "CPU_FALLBACK_OK"
        assert "MODEL_SHA256_MISMATCH" not in result["fallback"]["bpu_error"]
        assert "sha256 differs" in result["fallback"]["bpu_error"]


def test_client_rejects_any_nonzero_authority_and_uses_cpu():
    server, client_socket = socket.socketpair()
    tensor = _tensor(5)

    def malicious_server() -> None:
        with server:
            request = recv_frame(server)
            request_id = request["request_id"]
            request_sha = request["tensors"][0]["sha256"]
            authority = dict(ZERO_AUTHORITY)
            authority["serial_write"] = True
            send_frame(
                server,
                {
                    "schema": "rootscope.bpu-shadow-response.v1",
                    "protocol": "rootscope.bpu-shadow-unix.v1",
                    "request_id": request_id,
                    "status": "OK_SHADOW_ONLY",
                    "shadow_only": True,
                    "zero_authority": True,
                    "authority": authority,
                    "model": {
                        "release_id": "rootscope-seed17-r7-default-int16-all-nodes",
                        "sha256": R7_REFERENCE_SHA256,
                        "qualification": "SHADOW_CANDIDATE_NOT_DEFAULT",
                        "selected_bin_changed": False,
                    },
                    "backend": {
                        "mode": "BPU_SHADOW",
                        "actual": "MALICIOUS_TEST",
                        "evidence_scope": "TEST",
                        "injected_test_backend": True,
                    },
                    "interface": {},
                    "batch": {
                        "count": 1,
                        "latency_ms": 0.1,
                        "sequential_fixed_batch1_forwards": 1,
                    },
                    "results": [
                        {
                            "index": 0,
                            "input_sha256": request_sha,
                            "logits": [0.0, 1.0, 2.0, 3.0],
                            "latency_ms": 0.1,
                        }
                    ],
                    "error": None,
                },
            )

    thread = threading.Thread(target=malicious_server, daemon=True)
    thread.start()
    client = BpuShadowClient(
        Path(tempfile.gettempdir()) / "unused.sock",
        connector=lambda _timeout: client_socket,
    )
    result = client.infer_or_cpu([tensor], cpu_fallback=_cpu_outputs)
    thread.join(timeout=2.0)
    assert result["status"] == "CPU_FALLBACK_OK"
    assert "zero-authority contract" in result["fallback"]["bpu_error"]
    assert result["authority"] == ZERO_AUTHORITY


def test_protocol_rejects_more_than_four_tensors_locally():
    with pytest.raises(BpuShadowProtocolError, match="1-4 tensors"):
        make_request("too-many", [_tensor(1)] * 5)
