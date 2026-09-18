#!/usr/bin/env python3
"""Offline tests for the build step (`cmd_build`) of the launcher.

No Docker, SSH or network: the only thing driven here is the exclude list of the
rsync that stages the tree on the workers, because that one runs with `--delete`
against a copy of the repository and a wrong pattern removes files from it.
"""
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAVE_RSYNC = shutil.which("rsync") is not None


def build_source():
    """The body of cmd_build() from start.sh."""
    source = (ROOT / "start.sh").read_text()
    build = source[source.index("cmd_build()"):]
    return build[:build.index("\n}\n")]


def rsync_excludes():
    return re.findall(r"--exclude '([^']+)'", build_source())


def rsync(source, destination):
    args = ["rsync", "-aH", "--delete"]
    for pattern in rsync_excludes():
        args += ["--exclude", pattern]
    args += [f"{source}/", f"{destination}/"]
    result = subprocess.run(args, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@unittest.skipUnless(HAVE_RSYNC, "rsync is not installed")
class RsyncExcludesTests(unittest.TestCase):
    """cmd_build must not exclude the staged SGLang tree from the workers."""

    def test_the_staged_tree_reaches_the_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp, "src"), Path(tmp, "dst")
            staged = "runtime/sglang-canary/python/sglang/srt/models"
            (src / staged).mkdir(parents=True)
            (src / staged / "deepseek_v4.py").write_text("branch")
            (dst / staged).mkdir(parents=True)
            (dst / staged / "deepseek_v4.py").write_text("stale")
            (src / "models").mkdir()
            (src / "models" / "shard.safetensors").write_text("weights")

            rsync(src, dst)

            self.assertEqual((dst / staged / "deepseek_v4.py").read_text(), "branch",
                             "the staged SGLang tree never reached the worker")
            self.assertFalse((dst / "models").exists(),
                             "the checkpoint directory at the repository root was copied")

    def test_every_exclude_is_anchored(self):
        """A bare name matches any path component; all five are root-level artifacts."""
        for pattern in rsync_excludes():
            if pattern in (".env", ".env.tp4"):
                continue  # exact filenames, meant to be protected everywhere
            self.assertTrue(pattern.startswith("/"), f"unanchored exclude: {pattern}")
