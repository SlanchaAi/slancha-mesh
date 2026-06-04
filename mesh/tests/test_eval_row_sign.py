"""Eval-row HMAC signing + gate enforcement (#104)."""

from __future__ import annotations

import json
from pathlib import Path


from mesh.eval.row_sign import eval_row_hmac, verify_eval_row


def test_sign_verify_roundtrip():
    key = b"eval-signing-key"
    row = {"router_version": "v2", "mean_score": 4.1, "meta_stub": False, "n_eval": 100}
    row["row_hmac"] = eval_row_hmac(row, key)
    assert verify_eval_row(row, key) is True
    # tamper any field → verification fails
    row["meta_stub"] = True
    assert verify_eval_row(row, key) is False
    # wrong key fails
    row["meta_stub"] = False
    assert verify_eval_row(row, b"other") is False
    # missing hmac fails
    assert verify_eval_row({"router_version": "v2"}, key) is False


def test_append_pass_signs_when_key_set(tmp_path: Path, monkeypatch):
    from mesh.eval.runner import append_pass

    # Minimal EvalPass via the runner's own constructor path is heavy; build a
    # tiny stand-in with to_row().
    class _EP:
        def to_row(self):
            return {"router_version": "v9", "mean_score": 3.0, "meta_stub": False}

    out = tmp_path / "eval_results.jsonl"
    append_pass(out, _EP(), hmac_key=b"k")
    row = json.loads(out.read_text().splitlines()[0])
    assert "row_hmac" in row and verify_eval_row(row, b"k")
    # unsigned when no key
    append_pass(out, _EP())
    row2 = json.loads(out.read_text().splitlines()[1])
    assert "row_hmac" not in row2


def test_gate_refuses_forged_row_when_required(tmp_path: Path, monkeypatch):
    """End-to-end: an unsigned (forged) eval row is refused by the gate under
    --require-signed-rows; a properly signed one passes verification."""
    from mesh.eval.gate import main as gate_main

    key = "trusted-eval-key"
    monkeypatch.setenv("SLANCHA_EVAL_HMAC_KEY", key)

    def _row(version, stub, signed):
        r = {"router_version": version, "ts": f"2026-06-04T00:00:0{version[-1]}Z",
             "mean_score": 4.0, "n_eval": 100, "judge_model": "j",
             "per_domain_mean": {"code": 4.0}, "meta_stub": stub}
        if signed:
            r["row_hmac"] = eval_row_hmac(r, key.encode())
        return r

    p = tmp_path / "eval_results.jsonl"
    # champion signed; challenger FORGED (unsigned, meta_stub flipped to look real)
    p.write_text("\n".join(json.dumps(r) for r in (
        _row("v1", False, signed=True), _row("v2", False, signed=False))) + "\n")
    rc = gate_main(["--eval-jsonl", str(p), "--champion", "v1", "--challenger", "v2",
                    "--require-signed-rows", "--promotions-log", str(tmp_path / "promo.jsonl")])
    assert rc == 2  # forged challenger row refused
