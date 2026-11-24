"""Generalist rollout telemetry analyzer, eval logger, and pretrained config-sync helper.

Designed to work across any LeRobot robot embodiment, DoF count, task prompt, or policy
architecture (ACT, Diffusion, SmolVLA, X-VLA, etc.) without project-specific hardcoding.
"""

from __future__ import annotations

import argparse
from dataclasses import fields as dataclass_fields, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

# Safe architectural/inference config fields per policy type when overriding --policy.type + --policy.pretrained_path.
# Excludes training/optimizer fields, device, hub metadata, and unknown 0.6.1-only fields (pretrained_revision, compile_model).
POLICY_SYNC_FIELDS: dict[str, tuple[str, ...]] = {
    "act": (
        "chunk_size",
        "n_action_steps",
    ),
    "diffusion": (
        "n_obs_steps",
        "horizon",
        "n_action_steps",
        "noise_scheduler_type",
        "num_inference_steps",
        "crop_shape",
        "resize_shape",
        "use_separate_rgb_encoder_per_camera",
        "spatial_softmax_num_keypoints",
        "vision_backbone",
        "prediction_type",
        "clip_sample",
        "clip_sample_range",
    ),
    "smolvla": (
        "chunk_size",
        "n_action_steps",
        "num_steps",
        "resize_imgs_with_padding",
        "empty_cameras",
        "adapt_to_pi_aloha",
        "use_delta_joint_actions_aloha",
        "tokenizer_max_length",
        "pad_language_to",
        "num_vlm_layers",
        "num_expert_layers",
        "self_attn_every_n_layers",
        "expert_width_multiplier",
        "add_image_special_tokens",
        "prefix_length",
        "attention_mode",
    ),
    "xvla": (
        "chunk_size",
        "n_action_steps",
        "dtype",
        "action_mode",
        "num_denoising_steps",
        "use_proprio",
        "max_state_dim",
        "max_action_dim",
        "resize_imgs_with_padding",
        "num_image_views",
        "empty_cameras",
        "tokenizer_max_length",
        "pad_language_to",
        "hidden_size",
        "depth",
        "num_heads",
        "mlp_ratio",
        "num_domains",
        "len_soft_prompts",
        "use_hetero_proj",
    ),
}

COMMON_SYNC_FIELDS: tuple[str, ...] = (
    "n_obs_steps",
    "chunk_size",
    "n_action_steps",
    "horizon",
)


def _format_cli_value(val: Any) -> str:
    """Format a Python value cleanly for draccus/lerobot CLI parsing."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (list, tuple, dict)):
        return json.dumps(val, separators=(",", ":"))
    return str(val)


def load_pretrained_config_dict(pretrained_path: str) -> dict[str, Any]:
    """Load config.json from a local directory or Hugging Face Hub model repository."""
    local_cfg = Path(pretrained_path) / "config.json"
    if local_cfg.is_file():
        return json.loads(local_cfg.read_text(encoding="utf-8"))

    # Also support direct path to config.json
    direct_path = Path(pretrained_path)
    if direct_path.is_file() and direct_path.name == "config.json":
        return json.loads(direct_path.read_text(encoding="utf-8"))

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id=pretrained_path, filename="config.json")
    return json.loads(Path(downloaded).read_text(encoding="utf-8"))


def _get_valid_config_fields(policy_type: str | None) -> set[str] | None:
    """Return dataclass field names valid in the currently installed LeRobot PreTrainedConfig subclass."""
    if not policy_type:
        return None
    try:
        from lerobot.configs.policies import PreTrainedConfig

        cls = PreTrainedConfig.get_choice_class(policy_type)
        if is_dataclass(cls):
            return {f.name for f in dataclass_fields(cls)}
    except Exception:
        return None
    return None


def _extract_cli_override_keys(extra_cli_args: list[str]) -> set[str]:
    """Collect `--policy.<field>` names already passed explicitly by the user."""
    overrides: set[str] = set()
    for arg in extra_cli_args:
        if arg.startswith("--policy."):
            key = arg[len("--policy.") :].split("=", 1)[0].strip()
            if key:
                overrides.add(key)
    return overrides


def get_synced_policy_cli_args(
    pretrained_path: str,
    extra_cli_args: list[str] | None = None,
    policy_type: str | None = None,
) -> list[str]:
    """Build `--policy.<key>=<val>` flags matching `pretrained_path`'s config.json without overriding CLI args."""
    cfg_dict = load_pretrained_config_dict(pretrained_path)
    resolved_type = policy_type or cfg_dict.get("type")
    candidate_fields = POLICY_SYNC_FIELDS.get(str(resolved_type), COMMON_SYNC_FIELDS)
    valid_fields = _get_valid_config_fields(str(resolved_type) if resolved_type else None)
    user_overrides = _extract_cli_override_keys(extra_cli_args or [])

    cli_args: list[str] = []
    for key in candidate_fields:
        if key in user_overrides or key not in cfg_dict:
            continue
        if valid_fields is not None and key not in valid_fields:
            continue
        val = cfg_dict[key]
        cli_args.append(f"--policy.{key}={_format_cli_value(val)}")
    return cli_args


def sync_policy_config_from_pretrained(policy_cfg: Any, cli_args: list[str] | None = None) -> list[str]:
    """Synchronize safe architecture fields onto an instantiated PolicyConfig before `make_policy`."""
    pretrained_path = getattr(policy_cfg, "pretrained_path", None)
    if not pretrained_path:
        return []
    cfg_dict = load_pretrained_config_dict(str(pretrained_path))
    resolved_type = getattr(policy_cfg, "type", None) or cfg_dict.get("type")
    candidate_fields = POLICY_SYNC_FIELDS.get(str(resolved_type), COMMON_SYNC_FIELDS)
    valid_fields = {f.name for f in dataclass_fields(policy_cfg)} if is_dataclass(policy_cfg) else set()
    user_overrides = _extract_cli_override_keys(sys.argv[1:] if cli_args is None else cli_args)

    updated: list[str] = []
    for key in candidate_fields:
        if key in user_overrides or key not in cfg_dict:
            continue
        if valid_fields and key not in valid_fields:
            continue
        val = cfg_dict[key]
        if getattr(policy_cfg, key, None) != val:
            setattr(policy_cfg, key, val)
            updated.append(key)
    return updated


def _read_parquet_columns(dataset_root: Path) -> dict[str, np.ndarray]:
    """Read action, observation.state, timestamp, episode_index across all Parquet shards without decoding video."""
    parquet_files = sorted(dataset_root.glob("data/**/*.parquet"))
    if not parquet_files:
        parquet_files = sorted(dataset_root.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found under {dataset_root}")

    cols_wanted = ["observation.state", "action", "timestamp", "episode_index", "frame_index"]
    try:
        import pyarrow.parquet as pq

        tables = []
        for pf in parquet_files:
            schema_names = set(pq.read_schema(pf).names)
            present = [c for c in cols_wanted if c in schema_names]
            tables.append(pq.read_table(pf, columns=present))
        import pyarrow as pa

        combined = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        raw = combined.to_pydict()
    except ImportError:
        import pandas as pd

        dfs = [pd.read_parquet(pf) for pf in parquet_files]
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        raw = {c: df[c].tolist() for c in cols_wanted if c in df.columns}

    out: dict[str, np.ndarray] = {}
    for k, v in raw.items():
        dtype = np.int64 if k in ("episode_index", "frame_index") else np.float32
        out[k] = np.asarray(v, dtype=dtype)
    return out


def _resolve_joint_names(info: dict[str, Any], dim: int, feature_key: str) -> list[str]:
    """Resolve human-readable joint/action dimension names dynamically from meta/info.json."""
    feat = info.get("features", {}).get(feature_key, {})
    names = feat.get("names")
    if isinstance(names, dict) and "motors" in names:
        names = names["motors"]
    if isinstance(names, list) and len(names) == dim:
        return [str(n) for n in names]
    return [f"dim_{i}" for i in range(dim)]


def _count_ee_cycles(series: np.ndarray) -> int:
    """Count open->close cycles on the last DoF (typically gripper) via adaptive min/max hysteresis."""
    if len(series) < 4:
        return 0
    lo, hi = float(np.percentile(series, 5)), float(np.percentile(series, 95))
    span = hi - lo
    if span < 5.0:
        return 0
    close_thresh = lo + 0.35 * span
    open_thresh = lo + 0.65 * span
    cycles = 0
    is_closed = bool(series[0] <= close_thresh)
    for val in series[1:]:
        if not is_closed and val <= close_thresh:
            cycles += 1
            is_closed = True
        elif is_closed and val >= open_thresh:
            is_closed = False
    return cycles


def _smoothness_metrics(arr: np.ndarray) -> dict[str, float]:
    """Compute velocity d1, acceleration d2, and jerk d3 means & p95 across frames and joints."""
    if len(arr) < 4:
        return {"d1_mean": 0.0, "d1_p95": 0.0, "d2_mean": 0.0, "d2_p95": 0.0, "d3_mean": 0.0, "d3_p95": 0.0}
    d1 = np.abs(np.diff(arr, n=1, axis=0))
    d2 = np.abs(np.diff(arr, n=2, axis=0))
    d3 = np.abs(np.diff(arr, n=3, axis=0))
    return {
        "d1_mean": float(d1.mean()),
        "d1_p95": float(np.percentile(d1, 95)),
        "d2_mean": float(d2.mean()),
        "d2_p95": float(np.percentile(d2, 95)),
        "d3_mean": float(d3.mean()),
        "d3_p95": float(np.percentile(d3, 95)),
    }


def _compute_rollout_markdown(record: dict[str, Any]) -> str:
    """Render single-run rollout analysis into a clean Markdown report."""
    timing = record["timing"]
    act_sm = record["action_smoothness"]
    st_sm = record["state_smoothness"]
    lag = record["tracking_lag"]
    kin = record["kinematics"]
    lines = [
        f"# Physical Policy Rollout Analysis: `{record['run_id']}`\n",
        f"- **Timestamp (UTC):** `{record['timestamp_utc']}`",
        f"- **Dataset Repo / Root:** `{record['dataset_repo_id']}` (`{record['dataset_root']}`)",
        f"- **Policy:** `{record['policy_path']}` (`type={record['policy_type']}`, `n_action_steps={record['n_action_steps']}`, `chunk_size={record['chunk_size']}`)",
        f"- **Task Prompt:** `{record['task']}`",
        f"- **Outcome / Notes:** `{record['outcome']}` — {record['notes'] or '—'}",
        f"- **Extra CLI Args:** `{record['cli_args'] or '—'}`\n",
        "## Control Loop & Smoothness\n",
        f"- **Episodes / Frames / Duration:** `{timing['num_episodes']}` ep / `{timing['num_frames']}` frames / `{timing['duration_s']:.2f} s`",
        f"- **Control Rate:** `{timing['fps_median']:.2f} Hz` median (`{timing['fps_mean']:.2f} Hz` mean), `dt p95 = {timing['dt_p95_ms']:.1f} ms`, `max = {timing['dt_max_ms']:.1f} ms`, slow steps (>50 ms) = `{timing['slow_frames_gt_50ms']}`",
        f"- **Action Smoothness (`d1 / d2 / d3` mean):** `{act_sm['d1_mean']:.2f} / {act_sm['d2_mean']:.2f} / {act_sm['d3_mean']:.2f}` (`d2 p95 = {act_sm['d2_p95']:.2f}`)",
        f"- **State Smoothness (`d1 / d2 / d3` mean):** `{st_sm['d1_mean']:.2f} / {st_sm['d2_mean']:.2f} / {st_sm['d3_mean']:.2f}`",
        f"- **Chunk Boundary vs Interior Jump (`n={record['n_action_steps']}`):** boundary `{kin['boundary_jump_mean']:.2f}` (max `{kin['boundary_jump_max']:.2f}`) vs interior `{kin['interior_step_mean']:.2f}` (`ratio = {kin['boundary_to_interior_ratio']:.2f}x`)",
        f"- **Kinematics:** Initial stationary frames `{kin['initial_idle_frames']}`, max excursion `{kin['max_excursion_overall']:.2f}` (`frozen_arm={kin['is_frozen_arm']}`), last-DoF cycles `{kin['commanded_ee_cycles']}` commanded / `{kin['measured_ee_cycles']}` measured\n",
        "## Tracking Lag (`|action - observation.state|`) Per Dimension\n",
        "| Dimension | Mean Lag | Max Lag | Mean `|Δaction|` |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for name in lag["joint_names"]:
        lines.append(
            f"| `{name}` | `{lag['per_joint_mean'][name]:.2f}` | `{lag['per_joint_max'][name]:.2f}` | `{kin['per_joint_action_d1'][name]:.2f}` |"
        )
    lines.append(f"\n- **Overall Tracking Lag:** mean `{lag['overall_mean']:.2f}`, p95 `{lag['overall_p95']:.2f}`, max `{lag['overall_max']:.2f}`\n")
    return "\n".join(lines)


def _append_summary_table(log_path: Path, record: dict[str, Any]) -> None:
    """Create or append a Markdown summary row to `<log_dir>/EVAL_RESULTS_LOG.md`."""
    header = (
        "# Physical Robot Evaluation Results Log\n\n"
        "| Run ID | Policy (`type`) | `n_act` | Frames (`s`) | FPS med (`dt max`) | Lag mean | Action `d2` (`bnd/int`) | Excursion (`frozen`) | Cycles (cmd/meas) | Outcome | Notes | Dataset Root |\n"
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |\n"
    )
    if not log_path.exists():
        log_path.write_text(header, encoding="utf-8")

    t = record["timing"]
    sm = record["action_smoothness"]
    lag = record["tracking_lag"]
    k = record["kinematics"]
    notes_clean = (record["notes"] or "—").replace("|", "/").replace("\n", " ")
    policy_short = str(record["policy_path"] or "—")
    row = (
        f"| `{record['run_id']}` | `{policy_short}` (`{record['policy_type']}`) | `{record['n_action_steps']}` | "
        f"`{t['num_frames']}` (`{t['duration_s']:.1f}s`) | `{t['fps_median']:.1f} Hz` (`{t['dt_max_ms']:.0f}ms`) | "
        f"`{lag['overall_mean']:.2f}` | `{sm['d2_mean']:.2f}` (`{k['boundary_to_interior_ratio']:.2f}x`) | "
        f"`{k['max_excursion_overall']:.1f}` (`{k['is_frozen_arm']}`) | `{k['commanded_ee_cycles']}/{k['measured_ee_cycles']}` | "
        f"`{record['outcome']}` | {notes_clean} | `{record['dataset_root']}` |\n"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(row)


def analyze_and_log_rollout(
    dataset_root: str | Path,
    *,
    dataset_repo_id: str | None = None,
    policy_path: str | None = None,
    policy_type: str | None = None,
    n_action_steps: int | None = None,
    chunk_size: int | None = None,
    task: str | None = None,
    log_dir: str | Path = "eval_logs",
    cli_args: str = "",
    outcome: str = "unlabeled",
    notes: str = "",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Analyze a recorded LeRobot rollout directory and store structured JSON + Markdown logs."""
    root = Path(dataset_root).expanduser().resolve()
    info: dict[str, Any] = {}
    info_file = root / "meta" / "info.json"
    if info_file.is_file():
        info = json.loads(info_file.read_text(encoding="utf-8"))

    # Resolve policy config metadata when available
    resolved_chunk = chunk_size
    resolved_n_act = n_action_steps
    resolved_type = policy_type
    if policy_path and (resolved_chunk is None or resolved_n_act is None or resolved_type is None):
        try:
            pcfg = load_pretrained_config_dict(policy_path)
            resolved_type = resolved_type or pcfg.get("type")
            if resolved_chunk is None:
                resolved_chunk = pcfg.get("chunk_size") or pcfg.get("horizon")
            if resolved_n_act is None:
                resolved_n_act = pcfg.get("n_action_steps")
        except Exception:
            pass

    # Override n_action_steps if explicitly supplied in cli_args
    if cli_args:
        for tok in cli_args.split():
            if tok.startswith("--policy.n_action_steps="):
                try:
                    resolved_n_act = int(tok.split("=", 1)[1])
                except ValueError:
                    pass

    cols = _read_parquet_columns(root)
    states = cols.get("observation.state")
    actions = cols.get("action")
    if actions is None or len(actions) == 0:
        raise ValueError(f"Dataset under {root} has no action frames to analyze.")
    if states is None or len(states) != len(actions):
        states = actions.copy()

    timestamps = cols.get("timestamp")
    ep_indices = cols.get("episode_index", np.zeros(len(actions), dtype=np.int64))
    unique_eps = np.unique(ep_indices)

    dt_all: list[np.ndarray] = []
    act_d1_list, act_d2_list, act_d3_list = [], [], []
    st_d1_list, st_d2_list, st_d3_list = [], [], []
    boundary_jumps, interior_jumps = [], []

    step_period = int(resolved_n_act) if resolved_n_act and resolved_n_act > 1 else None

    for ep in unique_eps:
        mask = ep_indices == ep
        ep_act = actions[mask]
        ep_st = states[mask]
        if timestamps is not None:
            ep_ts = timestamps[mask]
            ep_dt = np.diff(ep_ts)
            ep_dt = ep_dt[ep_dt > 0]
            if len(ep_dt) > 0:
                dt_all.append(ep_dt)

        if len(ep_act) >= 2:
            d1 = np.abs(np.diff(ep_act, axis=0))
            act_d1_list.append(d1)
            st_d1_list.append(np.abs(np.diff(ep_st, axis=0)))
            if step_period and len(d1) >= step_period:
                step_l1 = d1.mean(axis=1)
                idx = np.arange(1, len(step_l1) + 1)
                bnd_mask = (idx % step_period) == 0
                if np.any(bnd_mask):
                    boundary_jumps.extend(step_l1[bnd_mask].tolist())
                if np.any(~bnd_mask):
                    interior_jumps.extend(step_l1[~bnd_mask].tolist())
        if len(ep_act) >= 3:
            act_d2_list.append(np.abs(np.diff(ep_act, n=2, axis=0)))
            st_d2_list.append(np.abs(np.diff(ep_st, n=2, axis=0)))
        if len(ep_act) >= 4:
            act_d3_list.append(np.abs(np.diff(ep_act, n=3, axis=0)))
            st_d3_list.append(np.abs(np.diff(ep_st, n=3, axis=0)))

    if dt_all:
        dt_cat = np.concatenate(dt_all)
        duration_s = float(dt_cat.sum())
        fps_mean = float(1.0 / np.mean(dt_cat))
        fps_median = float(1.0 / np.median(dt_cat))
        dt_p95_ms = float(np.percentile(dt_cat * 1000.0, 95))
        dt_max_ms = float(np.max(dt_cat * 1000.0))
        slow_frames = int(np.sum(dt_cat > 0.050))
    else:
        fps_cfg = float(info.get("fps") or 30.0)
        duration_s = float(len(actions) / fps_cfg)
        fps_mean = fps_cfg
        fps_median = fps_cfg
        dt_p95_ms = 1000.0 / fps_cfg
        dt_max_ms = dt_p95_ms
        slow_frames = 0

    dim_common = min(actions.shape[1], states.shape[1])
    joint_names = _resolve_joint_names(info, dim_common, "action")
    lag_mat = np.abs(actions[:, :dim_common] - states[:, :dim_common])
    act_d1_cat = np.concatenate(act_d1_list, axis=0) if act_d1_list else np.zeros((1, dim_common))

    first_ep_mask = ep_indices == unique_eps[0]
    first_st = states[first_ep_mask, :dim_common]
    first_act = actions[first_ep_mask, :dim_common]
    st_deltas = np.abs(np.diff(first_st, axis=0)).max(axis=1) if len(first_st) > 1 else np.array([0.0])
    non_idle = np.where(st_deltas >= 0.3)[0]
    initial_idle = int(non_idle[0]) if len(non_idle) > 0 else int(len(st_deltas))
    max_excursion = float(np.abs(first_st - first_st[0:1]).max()) if len(first_st) > 0 else 0.0

    bnd_mean = float(np.mean(boundary_jumps)) if boundary_jumps else float("nan")
    bnd_max = float(np.max(boundary_jumps)) if boundary_jumps else float("nan")
    int_mean = float(np.mean(interior_jumps)) if interior_jumps else float(act_d1_cat.mean())
    bnd_ratio = float(bnd_mean / max(1e-6, int_mean)) if boundary_jumps else 1.0

    act_sm = _smoothness_metrics(actions)
    st_sm = _smoothness_metrics(states)

    now_utc = datetime.now(timezone.utc)
    resolved_run_id = run_id or f"{root.name}_{now_utc.strftime('%Y%m%dT%H%M%SZ')}"

    record: dict[str, Any] = {
        "run_id": resolved_run_id,
        "timestamp_utc": now_utc.isoformat(),
        "dataset_repo_id": dataset_repo_id or root.name,
        "dataset_root": str(root),
        "policy_path": policy_path or "unknown",
        "policy_type": resolved_type or "unknown",
        "chunk_size": resolved_chunk,
        "n_action_steps": resolved_n_act,
        "task": task or "",
        "outcome": outcome,
        "notes": notes,
        "cli_args": cli_args,
        "timing": {
            "num_episodes": int(len(unique_eps)),
            "num_frames": int(len(actions)),
            "duration_s": duration_s,
            "fps_mean": fps_mean,
            "fps_median": fps_median,
            "dt_p95_ms": dt_p95_ms,
            "dt_max_ms": dt_max_ms,
            "slow_frames_gt_50ms": slow_frames,
        },
        "action_smoothness": act_sm,
        "state_smoothness": st_sm,
        "tracking_lag": {
            "joint_names": joint_names,
            "overall_mean": float(lag_mat.mean()),
            "overall_p95": float(np.percentile(lag_mat, 95)),
            "overall_max": float(lag_mat.max()),
            "per_joint_mean": {joint_names[i]: float(lag_mat[:, i].mean()) for i in range(dim_common)},
            "per_joint_max": {joint_names[i]: float(lag_mat[:, i].max()) for i in range(dim_common)},
        },
        "kinematics": {
            "initial_idle_frames": initial_idle,
            "max_excursion_overall": max_excursion,
            "is_frozen_arm": bool(max_excursion < 5.0),
            "boundary_jump_mean": bnd_mean if not np.isnan(bnd_mean) else 0.0,
            "boundary_jump_max": bnd_max if not np.isnan(bnd_max) else 0.0,
            "interior_step_mean": int_mean,
            "boundary_to_interior_ratio": bnd_ratio,
            "commanded_ee_cycles": _count_ee_cycles(first_act[:, -1]),
            "measured_ee_cycles": _count_ee_cycles(first_st[:, -1]),
            "per_joint_action_d1": {joint_names[i]: float(act_d1_cat[:, i].mean()) for i in range(dim_common)},
        },
    }

    md_report = _compute_rollout_markdown(record)

    # Save beside dataset
    (root / "rollout_analysis.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (root / "rollout_analysis.md").write_text(md_report, encoding="utf-8")

    # Save to centralized eval_logs directory
    log_base = Path(log_dir).expanduser().resolve()
    run_out_dir = log_base / "runs" / resolved_run_id
    run_out_dir.mkdir(parents=True, exist_ok=True)
    (run_out_dir / "rollout_analysis.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (run_out_dir / "rollout_analysis.md").write_text(md_report, encoding="utf-8")

    jsonl_path = log_base / "eval_history.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as jf:
        jf.write(json.dumps(record) + "\n")

    _append_summary_table(log_base / "EVAL_RESULTS_LOG.md", record)
    print(md_report)
    print(f"Saved rollout analysis -> {root / 'rollout_analysis.json'} & {log_base / 'EVAL_RESULTS_LOG.md'}")
    return record


def update_run_outcome(
    *,
    log_dir: str | Path = "eval_logs",
    outcome: str,
    notes: str = "",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Update outcome and notes for the latest (or specified) logged rollout across JSON, JSONL, and Markdown."""
    log_base = Path(log_dir).expanduser().resolve()
    jsonl_path = log_base / "eval_history.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"No evaluation history found at {jsonl_path}")

    lines = [ln for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Evaluation history at {jsonl_path} is empty.")

    records = [json.loads(ln) for ln in lines]
    target_idx = len(records) - 1 if run_id is None else next(
        (i for i, r in enumerate(records) if r.get("run_id") == run_id), -1
    )
    if target_idx < 0:
        raise KeyError(f"Run ID '{run_id}' not found in {jsonl_path}")

    rec = records[target_idx]
    rec["outcome"] = outcome
    if notes:
        rec["notes"] = notes
    md_report = _compute_rollout_markdown(rec)

    # Update JSONL atomically
    jsonl_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    # Update per-run files and dataset copy
    run_dir = log_base / "runs" / rec["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rollout_analysis.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    (run_dir / "rollout_analysis.md").write_text(md_report, encoding="utf-8")
    ds_root = Path(rec["dataset_root"])
    if ds_root.is_dir():
        (ds_root / "rollout_analysis.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        (ds_root / "rollout_analysis.md").write_text(md_report, encoding="utf-8")

    # Rebuild EVAL_RESULTS_LOG.md cleanly from updated records
    log_md = log_base / "EVAL_RESULTS_LOG.md"
    if log_md.exists():
        log_md.unlink()
    for r in records:
        _append_summary_table(log_md, r)

    print(f"Updated rollout '{rec['run_id']}' -> outcome={outcome}, notes={rec['notes']!r}")
    return rec


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generalist LeRobot rollout analyzer, logger, and pretrained config-sync CLI"
    )
    parser.add_argument(
        "--print-policy-args",
        type=str,
        default=None,
        help="Resolve `--policy.<field>=<val>` CLI args from a pretrained model's config.json and print one per line.",
    )
    parser.add_argument(
        "--update-last",
        action="store_true",
        help="Update `--outcome` and `--notes` for the most recent (or `--run-id`) entry in `--log-dir`.",
    )
    parser.add_argument("--dataset-root", type=str, default=None, help="Path to recorded LeRobot dataset directory")
    parser.add_argument("--dataset-repo-id", type=str, default=None)
    parser.add_argument("--policy-path", type=str, default=None)
    parser.add_argument("--policy-type", type=str, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default="eval_logs")
    parser.add_argument("--extra-args", type=str, default="")
    parser.add_argument("--outcome", type=str, default="unlabeled")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("remainder", nargs=argparse.REMAINDER, help="Extra CLI args after `--` for override detection")

    args = parser.parse_args(argv)

    if args.print_policy_args:
        extra = [a for a in args.remainder if a != "--"]
        flags = get_synced_policy_cli_args(
            pretrained_path=args.print_policy_args,
            extra_cli_args=extra,
            policy_type=args.policy_type,
        )
        for flag in flags:
            print(flag)
        return

    if args.update_last:
        update_run_outcome(
            log_dir=args.log_dir,
            outcome=args.outcome,
            notes=args.notes,
            run_id=args.run_id,
        )
        return

    if not args.dataset_root:
        parser.error("Either --print-policy-args, --update-last, or --dataset-root must be provided.")

    analyze_and_log_rollout(
        dataset_root=args.dataset_root,
        dataset_repo_id=args.dataset_repo_id,
        policy_path=args.policy_path,
        policy_type=args.policy_type,
        n_action_steps=args.n_action_steps,
        chunk_size=args.chunk_size,
        task=args.task,
        log_dir=args.log_dir,
        cli_args=args.extra_args,
        outcome=args.outcome,
        notes=args.notes,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
