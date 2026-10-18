"""rdna3/serve.sh on a fake machine: a KFD topology (RX 7900 XTX + XT), a meminfo and docker / getent stubs.

Nothing here touches Docker or a GPU; the RAM arithmetic and the commands serve.sh suggests or prints are checked.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SERVE = ROOT / "rdna3" / "serve.sh"
GIB = 1 << 30

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

DOCKER_STUB = r"""#!/usr/bin/env bash
# fake docker: the STUB_* variables say what this machine has
case "$1" in
  info)
    [ -z "${STUB_NO_DAEMON:-}" ] || exit 1
    [[ "$*" == *SecurityOptions* ]] && echo "${STUB_SECURITY:-[\"name=seccomp,profile=builtin\"]}"
    exit 0 ;;
  image)
    [ -n "${STUB_IMAGE_ID:-}" ] || exit 1
    echo "$STUB_IMAGE_ID"; exit 0 ;;
  run)
    [ -n "${STUB_IMAGE_ID:-}" ] || exit 125   # the real one would try to pull the tag
    [[ "$*" == *PROVENANCE* ]] && { printf '%s' "${STUB_PROV:-}"; exit 0; }
    exit 0 ;;                                  # grep -q --default-temperature: found
  ps)
    [[ "$*" == *ancestor=* ]] && exit 0         # the running server is of an older image than the tag
    [ -z "${STUB_RUNNING:-}" ] || printf 'c0ffee\nbadf00d\n'
    exit 0 ;;
  inspect)                                     # the server with the GPUs, and a container without them
    echo "/${STUB_RUNNING} /dev/kfd /dev/dri "; echo "/webapp "; exit 0 ;;
esac
exit 0
"""

GETENT_STUB = r"""#!/usr/bin/env bash
case "$1 $2" in
  "group video") echo "video:x:44:" ;;
  "group render") echo "render:x:992:" ;;
  *) exit 2 ;;
esac
"""


def _gpu_node(nodes: Path, i: int, simd: int, vram: int, bus: int) -> None:
    d = nodes / str(i)
    (d / "mem_banks" / "0").mkdir(parents=True)
    (d / "properties").write_text(
        f"simd_count {simd}\nlocation_id {bus << 8}\ndomain 0\ngfx_target_version 110000\n")
    (d / "mem_banks" / "0" / "properties").write_text(f"size_in_bytes {vram}\n")


CARDS = {"xt": (168, 20 * GIB), "xtx": (192, 24 * GIB)}


@pytest.fixture
def machine(tmp_path):
    """Returns run(args, ram_gib, cards, serve, **stub_env) -> combined output of serve.sh on the fake machine."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("docker", DOCKER_STUB), ("getent", GETENT_STUB)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    model = tmp_path / "Qwen3.8 NVFP4"  # a space in the path on purpose
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type": "qwen4_exp", "quantization_config": {"quant_algo": "NVFP4"}}')
    meminfo = tmp_path / "meminfo"

    def run(args, ram_gib=125.7, cards=("xt", "xtx"), serve=SERVE, **stub):
        nodes = tmp_path / ("kfd-" + "-".join(cards))
        if not nodes.exists():
            (nodes / "0").mkdir(parents=True)
            (nodes / "0" / "properties").write_text("simd_count 0\n")  # the CPU node
            for i, c in enumerate(cards, 1):  # buses no real device uses
                _gpu_node(nodes, i, *CARDS[c], 0xF0 + i)
        kib = int(ram_gib * (1 << 20))
        meminfo.write_text(f"MemTotal: {kib} kB\nMemAvailable: {kib - 10 * (1 << 20)} kB\n")
        env = {k: v for k, v in os.environ.items() if not k.startswith("STUB_")}
        env.update(PATH=f"{bin_dir}:{env['PATH']}", SERVE_SH_MEMINFO=str(meminfo), SERVE_SH_KFD_NODES=str(nodes))
        env.update({f"STUB_{k.upper()}": v for k, v in stub.items()})
        args = [str(model) if a == "MODEL" else a for a in args]
        r = subprocess.run(["bash", str(serve), *args], env=env, capture_output=True, text=True, timeout=120)
        return r.stdout + r.stderr

    return run


@pytest.mark.parametrize("ram", [125.7, 133.2, 140.0, 160.3])
def test_offered_pools_are_judged_the_same(machine, ram):
    # every pool --check offers without a condition must not turn "tight" or "over" when asked for
    out = machine(["--check", "--model", "MODEL"], ram_gib=ram)
    offers = 0
    m = re.search(r"or one conversation of up to \d+k tokens, ~\d+ GiB[^:\n]*: xtx-xt-1m --ctx (\d+)", out)
    if m and "little else" not in m.group(0):
        offers += 1
        again = machine(["--check", "--model", "MODEL", "--ctx", m.group(1), "xtx-xt-1m"], ram_gib=ram)
        assert "xtx-xt-1m with NVFP4" not in again, again
    for m in re.finditer(r"K/V in RAM: (\d) conversations? of 262k[^\n]*: xtx-xt-4x262k([^\n]*)", out):
        if "little else" in m.group(0):
            continue
        offers += 1
        pool = str(int(m.group(1)) * 262500)
        again = machine(["--check", "--model", "MODEL", "xtx-xt-4x262k", "--", "--kv-reserve-tokens", pool], ram_gib=ram)
        assert "xtx-xt-4x262k with NVFP4" not in again, again
    assert offers, out


def test_ram_fallback_skips_untested_profiles(machine):
    # 100 GiB: no measured profile keeps 25 GiB free with NVFP4; gre (a 16 GB card's) must not be offered instead
    out = machine(["--check", "--model", "MODEL"], ram_gib=100)
    assert "for this RAM with NVFP4: gre" not in out, out
    assert not re.search(r"^Next: rdna3/serve.sh gre ", out, re.M), out


def test_two_xt_keep_their_two_card_profile(machine):
    # xt-xt is untested but the only two-card profile there: the RAM fallback must not swap it for one card
    out = machine(["--check", "--model", "MODEL"], cards=("xt", "xt"))
    assert "suggested: xt-xt" in out and "for this RAM" not in out, out


def test_running_server_of_an_older_image_is_seen(machine):
    out = machine(["--check"], image_id="sha256:" + "1" * 64, running="llm-freetoken")
    assert "a container with the GPUs is running (llm-freetoken)" in out, out  # webapp has no GPU


def test_start_note_checks_the_pool_asked_for(machine):
    # 3 x 262k with NVFP4 on 125.7 GiB is "tight": the note must suggest the check of that pool, not the profile's 4
    out = machine(["xtx-xt-4x262k", "--model", "MODEL", "--dry-run", "--", "--kv-reserve-tokens", "787500"],
                  image_id="sha256:" + "1" * 64)
    note = next(line for line in out.splitlines() if line.startswith("note: this server will lock"))
    assert "--check --model " in note and "Qwen3.8\\ NVFP4 xtx-xt-4x262k -- --kv-reserve-tokens 787500" in note, note


def test_dry_run_prints_the_rootless_command(machine):
    out = machine(["xtx-xt", "--model", "MODEL", "--dry-run"], image_id="sha256:" + "1" * 64,
                  security='["name=seccomp,profile=builtin","name=rootless"]')
    assert "ft serve" in out and "--oom-score-adj" not in out, out
    out = machine(["xtx-xt", "--model", "MODEL", "--dry-run"], image_id="sha256:" + "1" * 64)
    assert "--oom-score-adj -999" in out, out


def test_dry_run_without_an_image_keeps_the_temperature(machine):
    out = machine(["xtx-xt", "--model", "MODEL", "--dry-run"])
    assert "--default-temperature 0.6" in out and "predates" not in out, out


@pytest.fixture
def repo(tmp_path):
    """A clean git repository holding a copy of rdna3/serve.sh and its profiles: (serve.sh, HEAD, HEAD's tree)."""
    if shutil.which("git") is None:
        pytest.skip("needs git")
    r = tmp_path / "repo"
    shutil.copytree(ROOT / "rdna3" / "profiles", r / "rdna3" / "profiles")
    shutil.copy(SERVE, r / "rdna3" / "serve.sh")
    git = lambda *a: subprocess.run(["git", "-C", str(r), *a], capture_output=True, text=True, check=True).stdout.strip()
    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "t")
    return r / "rdna3" / "serve.sh", git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")


IMG = "sha256:" + "1" * 64


def test_image_of_a_clean_head_is_ok(machine, repo):
    serve, head, tree = repo
    out = machine(["--check"], serve=serve, image_id=IMG, prov=f"commit {head}\ntree   {tree}\n")
    assert f"built from this checkout's commit ({head[:7]})" in out and "local changes" not in out, out


def test_image_with_files_not_in_a_clean_head_is_flagged(machine, repo):
    serve, head, _ = repo
    out = machine(["--check"], serve=serve, image_id=IMG, prov=f"commit {head}\ntree   {'0' * 40}\n")
    assert "with changes not in it: rebuild" in out, out


def test_dirty_checkout_built_from_its_head_is_not_told_to_rebuild(machine, repo):
    # an image built from these very local changes records another tree than HEAD's: only the note
    serve, head, _ = repo
    (serve.parent / "profiles" / "xtx.env").write_text("# a local change\n", encoding="utf-8")
    out = machine(["--check"], serve=serve, image_id=IMG, prov=f"commit {head}\ntree   {'0' * 40}\n")
    assert "the checkout has local changes" in out and "rebuild" not in out, out


def test_provenance_without_a_tree_compares_the_commit(machine, repo):
    serve, head, _ = repo
    out = machine(["--check"], serve=serve, image_id=IMG, prov=f"commit {'1' * 40}\n")
    assert f"was built from commit 1111111, this checkout is at {head[:7]}" in out, out
