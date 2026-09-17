#!/usr/bin/env python3
"""Offline tests for the opt-in switchless ring (files/nccl.sh + the launcher hooks).

No Docker, SSH, sysfs or model paths are touched: `files/nccl.sh` is sourced
directly, and `start.sh` is sliced from its variable defaults to the command
dispatch so the real `docker_common_args` / `worker_env_lines` can be driven.

Default-off is the property that matters most here: with NCCL_SWITCHLESS_RING_ONLY
unset every generated argument list must be byte-identical to before this change.
"""
import os
import subprocess

import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NCCL_SH = ROOT / "files" / "nccl.sh"


def launcher_source():
    """start.sh from the variable defaults through the helper functions, no dispatch."""
    source = (ROOT / "start.sh").read_text()
    start = source.index("HEAD_IP=")
    end = source.index('case "$CMD" in')
    return "set -euo pipefail\n" + source[start:end]


BASE_SETTINGS = {
    "NCCL_HOST_DIR": "/nonexistent",
    "NNODES": "4",
    "TP_SIZE": "4",
    "EP_SIZE": "2",
    "NCCL_NET": "IB",
    "NCCL_IB_DISABLE": "0",
    "IB_HCA": "rocep1s0f0,rocep1s0f1",
}


def shell(code, directory, settings=None):
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "ROOT": str(ROOT)}
    for name in ("MODEL_DIR", "COMMON_MODEL", "SSH_IDENTITY", "NCCL_HOST_DIR",
                 "STATE_DIR", "LOG_DIR", "WORKER_DIR", "WORKER_ENGRAM_DIR", "ENGRAM_DIR"):
        env[name] = str(directory / name)
    Path(env["MODEL_DIR"]).mkdir(exist_ok=True)
    Path(env["MODEL_DIR"], "config.json").write_text("{}")
    # A four-node launcher needs three workers; the repo default is the 3-Spark fleet.
    env.update({
        "HEAD_IP": "10.0.0.1",
        "WORKER_IPS": "10.0.0.2 10.0.0.3 10.0.0.4",
        "WORKER_HOSTS": "spark2 spark3 spark4",
    })
    env.update(settings or {})
    return subprocess.run(
        ["bash", "-c", launcher_source() + "\n" + code],
        env=env, cwd=ROOT, text=True, capture_output=True, timeout=30,
    )


def nccl_shell(code, settings=None):
    """Run a snippet with files/nccl.sh sourced (no launcher)."""
    env = dict(os.environ)
    env.update(BASE_SETTINGS)
    env.update(settings or {})
    return subprocess.run(
        ["bash", "-c", 'set -uo pipefail\nsource "$0"\n' + code, str(NCCL_SH)],
        env=env, text=True, capture_output=True, timeout=30,
    )


def check(result):
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def nfs_shell(code, directory, settings=None):
    """Run a snippet with files/nfs-share.sh sourced (sourcing it is side-effect free)."""
    env = {
        "PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "ROOT": str(ROOT),
        "MODEL_DIR": str(directory), "EXPECTED_SHARDS": "2",
    }
    env.update(settings or {})
    return subprocess.run(
        ["bash", "-c", 'set -uo pipefail\nsource "$ROOT/files/nfs-share.sh"\n' + code],
        env=env, text=True, capture_output=True, timeout=30,
    )


def lines_after(out, tag):
    """Tokens printed one-per-line by `printf '<tag>%s\\n' "${array[@]}"`."""
    return [line[len(tag):] for line in out.splitlines() if line.startswith(tag)]


class SwitchOffByDefaultTests(unittest.TestCase):
    """With the switch unset nothing may change."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tests")

    def tearDown(self):
        self.tmp.cleanup()

    def test_enabled_predicate_defaults_to_false(self):
        self.assertIn("enabled=0", check(nccl_shell(
            'if switchless_ring_enabled; then echo enabled=1; else echo enabled=0; fi')))
        self.assertIn("enabled=0", check(nccl_shell(
            'NCCL_SWITCHLESS_RING_ONLY=0; if switchless_ring_enabled; then echo enabled=1; else echo enabled=0; fi')))

    def test_ring_args_and_env_string_are_empty(self):
        out = check(nccl_shell('''
          a=(); switchless_ring_args a; echo "count=${#a[@]}"
          echo "envstr=[$(switchless_ring_env_string)]"
        '''))
        self.assertIn("count=0", out)
        self.assertIn("envstr=[]", out)

    def test_validation_passes_without_the_switch(self):
        # NNODES=3 / a switched fabric is a perfectly good non-ring deployment.
        check(nccl_shell("nccl_validate_config", {"NNODES": "3", "TP_SIZE": "3"}))

    def test_worker_env_has_no_stray_whitespace(self):
        """Regression guard: the ring hook must not leave `  \\` behind when off.

        The first version appended the ring env to the heredoc as `$_extra_env
        $_ring_env`, which is byte-identical except for a trailing space whenever
        the switch is off. Harmless to bash, but "off changes nothing" has to mean
        the bytes too, so the separator now comes from the launcher.
        """
        out = check(shell('worker_env_lines "$HEAD_IP" 3 1', Path(self.tmp.name), BASE_SETTINGS))
        for line in out.splitlines():
            self.assertFalse(line.endswith(" "), repr(line))
            self.assertNotIn("  \\", line, repr(line))
        self.assertIn("-e DSV41_EXTRA_ENV=1 \\", out)

    def test_head_and_worker_arguments_are_unchanged(self):
        out = check(shell('''
          a=(); docker_common_args a "$HEAD_IP" 3
          printf '%s\\n' "${a[@]}"
        ''', Path(self.tmp.name), BASE_SETTINGS))
        self.assertNotIn("NCCL_SWITCHLESS_RING_ONLY", out)
        self.assertNotIn("NCCL_ALGO", out)
        self.assertNotIn("NCCL_PIP_SO", out)


class SwitchOnTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.on = dict(BASE_SETTINGS)
        self.on.update({
            "NCCL_SWITCHLESS_RING_ONLY": "1",
            "NCCL_OVERLAY_PIP": "1",
            "NCCL_HOST_DIR": "/nonexistent",
        })

    def tearDown(self):
        self.tmp.cleanup()

    def test_ring_args_carry_the_ring_environment(self):
        out = check(nccl_shell('''
          a=(); switchless_ring_args a; printf '%s\\n' "${a[@]}"
        ''', self.on))
        for expected in ("NCCL_SWITCHLESS_RING_ONLY=1", "NCCL_ALGO=Ring",
                         "NCCL_SKIP_TREE_CONNECT=1", "NCCL_IB_SUBNET_PREFIX_LEN=24",
                         "NCCL_MIN_NCHANNELS=4", "NCCL_P2P_LEVEL=SYS"):
            self.assertIn(expected, out)

    def test_env_string_and_array_agree(self):
        """The worker heredoc and the head array must inject the same variables."""
        out = check(nccl_shell('''
          a=(); switchless_ring_args a
          printf 'ARGV %s\\n' "${a[@]}"
          printf 'STRV %s\\n' $(switchless_ring_env_string)
        ''', self.on))
        argv = sorted(lines_after(out, "ARGV "))
        strv = sorted(lines_after(out, "STRV "))
        self.assertEqual(argv, strv)
        self.assertIn("NCCL_SWITCHLESS_RING_ONLY=1", strv)

    def test_env_string_survives_quoted_values(self):
        out = check(nccl_shell('''
          eval "set -- $(switchless_ring_env_string)"
          printf '[%s]\\n' "$@"
        ''', self.on))
        self.assertIn("[NCCL_ALGO=Ring]", out)

    def test_worker_env_mirrors_the_ring_environment(self):
        out = check(shell('worker_env_lines "$HEAD_IP" 3 1', Path(self.tmp.name), self.on))
        for expected in ("NCCL_SWITCHLESS_RING_ONLY=1", "NCCL_ALGO=Ring",
                         "NCCL_SKIP_TREE_CONNECT=1", "NCCL_P2P_LEVEL=SYS"):
            self.assertIn(expected, out)

    def test_head_uses_the_overlay_mount_not_ld_library_path(self):
        nccl_dir = Path(self.tmp.name) / "nccl"
        nccl_dir.mkdir()
        (nccl_dir / "libnccl.so.2.30.7").write_text("SWITCHLESS_RING_ONLY\n")
        out = check(shell('''
          a=(); docker_common_args a "$HEAD_IP" 3
          printf '%s\\n' "${a[@]}"
        ''', Path(self.tmp.name), dict(self.on, NCCL_HOST_DIR=str(nccl_dir))))
        self.assertIn(":/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2:ro", out)
        self.assertNotIn("LD_LIBRARY_PATH", out)

    def test_mount_refuses_a_missing_library_when_the_switch_is_on(self):
        result = nccl_shell("a=(); nccl_mount_args a", dict(self.on, NCCL_HOST_DIR="/nonexistent"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no readable library", result.stderr)


class ValidationTests(unittest.TestCase):
    def run_validation(self, settings):
        return nccl_shell("nccl_validate_config", dict(BASE_SETTINGS, **settings))

    def test_accepts_ep_from_one_to_tp(self):
        for ep in ("1", "2", "4"):
            with self.subTest(ep=ep):
                check(self.run_validation({"NCCL_SWITCHLESS_RING_ONLY": "1",
                                           "NCCL_OVERLAY_PIP": "1", "EP_SIZE": ep}))

    def test_rejects_ep_above_tp_and_zero(self):
        for ep in ("5", "0", "x"):
            with self.subTest(ep=ep):
                result = self.run_validation({"NCCL_SWITCHLESS_RING_ONLY": "1",
                                              "NCCL_OVERLAY_PIP": "1", "EP_SIZE": ep})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("1 <= EP_SIZE <= TP_SIZE", result.stderr)

    def test_rejects_topology_that_cannot_carry_the_ring(self):
        for settings in ({"NNODES": "3"}, {"TP_SIZE": "3"}):
            with self.subTest(**settings):
                result = self.run_validation(dict({"NCCL_SWITCHLESS_RING_ONLY": "1",
                                                   "NCCL_OVERLAY_PIP": "1"}, **settings))
                self.assertNotEqual(result.returncode, 0)

    def test_rejects_transport_that_is_not_the_ring(self):
        for settings in ({"NCCL_ALGO": "Tree"}, {"NCCL_NET": "Socket"},
                         {"NCCL_IB_DISABLE": "1"}, {"NCCL_OVERLAY_PIP": "0"}):
            with self.subTest(**settings):
                result = self.run_validation(dict({"NCCL_SWITCHLESS_RING_ONLY": "1",
                                                   "NCCL_OVERLAY_PIP": "1"}, **settings))
                self.assertNotEqual(result.returncode, 0)

    def test_rejects_channel_range_that_is_empty(self):
        result = self.run_validation({"NCCL_SWITCHLESS_RING_ONLY": "1", "NCCL_OVERLAY_PIP": "1",
                                      "NCCL_MIN_NCHANNELS": "8", "NCCL_MAX_NCHANNELS": "4"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NCCL_MIN_NCHANNELS", result.stderr)

    def test_rejects_non_boolean_switches(self):
        for flag in ("NCCL_SWITCHLESS_RING_ONLY", "NCCL_OVERLAY_PIP"):
            with self.subTest(flag=flag):
                result = self.run_validation({flag: "yes"})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be 0 or 1", result.stderr)


class GidIndexTests(unittest.TestCase):
    """ring_gid_index against a synthetic sysfs tree."""

    def build(self, directory, gids, types=None):
        """gids: {port_hca: {index: gid}}; ports are always ACTIVE."""
        for hca, table in gids.items():
            base = directory / hca / "ports" / "1"
            (base / "gid_attrs" / "types").mkdir(parents=True)
            (base / "gids").mkdir(parents=True)
            (base / "state").write_text("4: ACTIVE\n")
            for index, gid in table.items():
                (base / "gids" / index).write_text(gid + "\n")
                kind = (types or {}).get((hca, index), "RoCE v2")
                (base / "gid_attrs" / "types" / index).write_text(kind + "\n")
        return str(directory)

    IPV4 = "0000:0000:0000:0000:0000:ffff:c0a8:32a8"
    IPV6 = "fe80:0000:0000:0000:0000:0000:0000:0001"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tests")

    def tearDown(self):
        self.tmp.cleanup()

    def test_finds_the_common_ipv4_index(self):
        sysfs = self.build(Path(self.tmp.name),
                           {"rocep1s0f0": {"0": self.IPV6, "3": self.IPV4},
                            "rocep1s0f1": {"0": self.IPV6, "3": self.IPV4}})
        out = check(nccl_shell(f'ring_gid_index {sysfs}'))
        self.assertEqual(out.strip(), "3")

    def test_rejects_ports_without_a_shared_index(self):
        sysfs = self.build(Path(self.tmp.name),
                           {"rocep1s0f0": {"3": self.IPV4},
                            "rocep1s0f1": {"5": self.IPV4}})
        result = nccl_shell(f'ring_gid_index {sysfs}')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no common IPv4 RoCE v2 GID", result.stderr)

    def test_rejects_a_zero_gid(self):
        sysfs = self.build(Path(self.tmp.name),
                           {"rocep1s0f0": {"3": "0000:0000:0000:0000:0000:ffff:0000:0000"}})
        self.assertNotEqual(nccl_shell(f'ring_gid_index {sysfs}').returncode, 0)

    def test_honours_an_explicit_override(self):
        sysfs = self.build(Path(self.tmp.name),
                           {"rocep1s0f0": {"3": self.IPV4, "5": self.IPV4}})
        out = check(nccl_shell(f'ring_gid_index {sysfs}',
                               {"NCCL_IB_GID_INDEX": "5", "IB_HCA": "rocep1s0f0"}))
        self.assertEqual(out.strip(), "5")

    def test_rejects_an_inactive_port(self):
        sysfs = self.build(Path(self.tmp.name), {"rocep1s0f0": {"3": self.IPV4}})
        Path(sysfs, "rocep1s0f0", "ports", "1", "state").write_text("1: DOWN\n")
        result = nccl_shell(f'ring_gid_index {sysfs}')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not ACTIVE", result.stderr)

    def test_rejects_a_malformed_hca_name(self):
        result = nccl_shell(f'ring_gid_index {self.tmp.name}',
                            {"IB_HCA": "rocep1s0f0;rm -rf"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exact HCA names", result.stderr)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tests")

    def tearDown(self):
        self.tmp.cleanup()

    def library(self, content):
        directory = Path(self.tmp.name)
        directory.mkdir(exist_ok=True)
        (directory / "libnccl.so.2.30.7").write_text(content)
        return str(directory)

    def test_rejects_a_stock_library(self):
        result = nccl_shell("nccl_preflight",
                            {"NCCL_SWITCHLESS_RING_ONLY": "1", "NCCL_OVERLAY_PIP": "0",
                             "NCCL_HOST_DIR": self.library("stock nccl")})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("lacks SWITCHLESS_RING_ONLY", result.stderr)

    def test_rejects_a_missing_library(self):
        result = nccl_shell("nccl_preflight",
                            {"NCCL_SWITCHLESS_RING_ONLY": "1", "NCCL_OVERLAY_PIP": "0",
                             "NCCL_HOST_DIR": "/nonexistent"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no readable library", result.stderr)

    def test_is_free_when_the_switch_is_off(self):
        """A switched fabric may legitimately have no patched library at all."""
        check(nccl_shell("nccl_preflight",
                         {"NCCL_SWITCHLESS_RING_ONLY": "0", "NCCL_HOST_DIR": "/nonexistent"}))

    def test_worker_settings_maps_the_home_relative_path(self):
        out = check(nccl_shell('nccl_worker_settings | head -1',
                              {"NCCL_HOST_DIR": os.environ["HOME"] + "/nccl-2.30.7"}))
        self.assertIn('NCCL_HOST_DIR="$HOME"/nccl-2.30.7', out)

    def test_worker_settings_honours_an_absolute_override(self):
        out = check(nccl_shell('nccl_worker_settings | head -1',
                               {"NCCL_WORKER_DIR": "/opt/nccl"}))
        self.assertIn("NCCL_HOST_DIR=/opt/nccl", out)

    def test_worker_settings_exports_the_ring_functions(self):
        out = check(nccl_shell("nccl_worker_settings"))
        for function in ("nccl_library", "nccl_mount_args", "ring_gid_index", "nccl_preflight"):
            self.assertIn(f"{function} ()", out)


class SourceHygieneTests(unittest.TestCase):
    def test_nccl_sh_is_side_effect_free(self):
        """Sourcing it must not probe sysfs, docker or the filesystem."""
        result = subprocess.run(
            ["bash", "-c", 'set -euo pipefail\nsource "$0"\necho sourced-ok', str(NCCL_SH)],
            text=True, capture_output=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sourced-ok", result.stdout)

    def test_switch_variables_have_defaults_in_start_sh(self):
        source = (ROOT / "start.sh").read_text()
        for name in ("NCCL_SWITCHLESS_RING_ONLY", "NCCL_OVERLAY_PIP", "NCCL_PIP_SO"):
            self.assertIn(f'{name}="${{{name}:-', source, name)

    def test_every_ring_hook_is_behind_the_switch(self):
        """The ring must not run on a switched fabric by accident."""
        source = (ROOT / "start.sh").read_text()
        head = source[source.index("docker_common_args()"):]
        head = head[:head.index("\n}\n")]
        self.assertIn('if [[ "$NCCL_SWITCHLESS_RING_ONLY" == "1" ]]', head)
        self.assertIn("switchless_ring_args _a", head)
        self.assertIn("nccl_mount_args _a", head)

    def test_serve_gates_the_ring_before_replacing_containers(self):
        source = (ROOT / "start.sh").read_text()
        serve = source[source.index("cmd_serve()"):]
        serve = serve[:serve.index("\n}\n")]
        gate = serve.index("nccl_validate_config")
        self.assertIn("switchless_ring_enabled", serve)
        # The gate must precede the first container replacement in serve.
        for later in ("docker rm", "docker run -d", "start_worker"):
            if later in serve:
                self.assertLess(gate, serve.index(later), later)


class LocalWeightsTests(unittest.TestCase):
    """local_model_has_weights: the NFS_SHARE=0 guard before four containers go down."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.model = Path(self.tmp.name, "model")
        self.model.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def complete(self, shards=2):
        (self.model / "config.json").write_text('{"model_type": "deepseek_v4"}')
        for i in range(shards):
            (self.model / f"model-{i:05d}-of-{shards:05d}.safetensors").write_text("x" * 32)

    def test_accepts_a_complete_checkpoint(self):
        self.complete()
        check(nfs_shell("local_model_has_weights", self.model))

    def test_rejects_a_missing_config(self):
        self.complete()
        (self.model / "config.json").unlink()
        self.assertNotEqual(nfs_shell("local_model_has_weights", self.model).returncode, 0)

    def test_rejects_an_empty_config(self):
        """A truncated copy leaves a zero-byte file, which the old check accepted."""
        self.complete()
        (self.model / "config.json").write_text("")
        self.assertNotEqual(nfs_shell("local_model_has_weights", self.model).returncode, 0)

    def test_rejects_too_few_shards(self):
        self.complete(shards=1)
        self.assertNotEqual(nfs_shell("local_model_has_weights", self.model).returncode, 0)

    def test_rejects_an_empty_shard(self):
        self.complete()
        (self.model / "model-00000-of-00002.safetensors").write_text("")
        self.assertNotEqual(nfs_shell("local_model_has_weights", self.model).returncode, 0)

    def test_worker_probe_does_not_pull_or_create_a_volume(self):
        """The old probe pulled alpine and `-v` would create an empty volume."""
        source = (ROOT / "files" / "nfs-share.sh").read_text()
        probe = source[source.index("nfs_worker_has_model()"):]
        probe = probe[:probe.index("\n}\n")]
        self.assertIn("--pull=never", probe)
        self.assertIn("--network none", probe)
        self.assertIn("docker volume inspect", probe)
        self.assertNotIn("alpine", probe)
        self.assertNotIn("-v \'${NFS_VOLUME}", probe)


class NfsShareOffTests(unittest.TestCase):
    """NFS_SHARE=0 is the local-weights profile the ring needs."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tests")

    def tearDown(self):
        self.tmp.cleanup()

    def test_share_is_a_noop(self):
        """serve calls cmd_share unconditionally; with NFS_SHARE=0 it must not touch NFS."""
        out = check(shell('''
          nfs_ensure_server() { echo "NFS-STARTED"; return 1; }
          nfs_publish_model()  { echo "NFS-PUBLISHED"; }
          nfs_ensure_worker_volume() { echo "NFS-VOLUME"; }
          cmd_share
        ''', Path(self.tmp.name), dict(BASE_SETTINGS, NFS_SHARE="0")))
        self.assertIn("keeping local worker volumes", out)
        for forbidden in ("NFS-STARTED", "NFS-PUBLISHED", "NFS-VOLUME"):
            self.assertNotIn(forbidden, out)

    def test_share_still_runs_when_enabled(self):
        out = check(shell('''
          nfs_ensure_server() { echo "NFS-STARTED"; }
          nfs_publish_model()  { echo "NFS-PUBLISHED"; }
          nfs_ensure_worker_volume() { :; }
          nfs_worker_has_model() { return 0; }
          ensure_ssh_keys() { :; }
          touch "$MODEL_DIR/config.json"
          cmd_share
        ''', Path(self.tmp.name), dict(BASE_SETTINGS, NFS_SHARE="1")))
        self.assertIn("NFS-STARTED", out)
        self.assertIn("NFS-PUBLISHED", out)

    def test_serve_validates_local_weights_before_replacing_containers(self):
        source = (ROOT / "start.sh").read_text()
        serve = source[source.index("cmd_serve()"):]
        serve = serve[:serve.index("\n}\n")]
        branch = serve.index('if [[ "$NFS_SHARE" == "1" ]]')
        self.assertIn("local_model_has_weights", serve[branch:])
        self.assertIn("disables NFS setup", serve[branch:])
        for later in ("docker rm", "docker run -d", "start_worker"):
            if later in serve:
                self.assertLess(branch, serve.index(later), later)

    def test_status_reads_the_volume_not_the_head_symlink(self):
        source = (ROOT / "start.sh").read_text()
        status = source[source.index("cmd_status()"):]
        status = status[:status.index("\n}\n")]
        worker = status[status.index("for h in"):]
        self.assertNotIn("$COMMON_MODEL/config.json", worker)
        self.assertIn("nfs_worker_has_model", worker)

    def test_example_configures_local_weights_for_the_ring(self):
        example = (ROOT / ".env.tp4.example").read_text()
        self.assertIn("NFS_SHARE=0", example)


if __name__ == "__main__":
    unittest.main(verbosity=2)
