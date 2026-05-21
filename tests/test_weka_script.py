import importlib.util
import subprocess
from pathlib import Path


def load_weka_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "weka.py"
    spec = importlib.util.spec_from_file_location("weka_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeStdout:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeFindProcess:
    def __init__(self):
        self.stdout = FakeStdout()

    def wait(self):
        return 0


def test_warmup_datasets_passes_paths_as_argv(monkeypatch):
    weka = load_weka_module()
    popen_calls = []
    run_calls = []
    find_process = FakeFindProcess()

    def fake_popen(cmd, stdout, text):
        popen_calls.append({"cmd": cmd, "stdout": stdout, "text": text})
        return find_process

    def fake_run(cmd, stdin, check, text):
        run_calls.append({"cmd": cmd, "stdin": stdin, "check": check, "text": text})
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(weka.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(weka.subprocess, "run", fake_run)

    folder_path = "/fsx/datasets/name; touch /tmp/should-not-run"
    weka.warmup_datasets([folder_path])

    assert popen_calls == [{"cmd": ["find", "-L", folder_path, "-type", "f"], "stdout": subprocess.PIPE, "text": True}]
    assert run_calls == [
        {
            "cmd": ["xargs", "-d", "\n", "-r", "-n512", "-P64", "weka", "fs", "tier", "fetch"],
            "stdin": find_process.stdout,
            "check": True,
            "text": True,
        }
    ]
    assert find_process.stdout.closed
