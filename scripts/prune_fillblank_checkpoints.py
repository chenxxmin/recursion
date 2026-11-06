#!/usr/bin/env python3
"""Audit or prune the approved fill_blank checkpoints.

The default is a dry-run: it only writes a JSON manifest.  Deletion requires an
explicit --apply and uses only the exact paths recorded in that manifest.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path("/data/cxm/models")
DEFAULT_MANIFEST = REPO_ROOT / "reports/data/fillblank_checkpoint_prune_manifest.json"
RESULT_PATH = REPO_ROOT / "reports/data/fillblank_checkpoint_prune_result.json"
STRIDE_ROOT = MODEL_ROOT / "stride5_len5to65_3coef_3seed_v1"
THRESHOLD = 0.99

EARLY_DIRS = tuple(MODEL_ROOT / name for name in (
    "fill_blank_arith_o1_P10007_sft",
    "fill_blank_o2_10coef_d512l2_P127_tr16",
    "fill_blank_o2_8coefB_d512l2_P127_tr16",
    "fill_blank_o2_a1b1_d512l2_P127_tr16_h4",
    "variant_arith_o1_P1009_sft",
    "variant_geo_o1_c6_P10009_sft",
    "variant_o2_a11b97_d512l2_P127_tr16",
))
NOAUX_PREFIX = "variant_o2_fillblank_noaux_randpos_d512l4_P127"
NOAUX_DIRS = (
    MODEL_ROOT / f"{NOAUX_PREFIX}_len8_wd2_initLen35",
    MODEL_ROOT / f"{NOAUX_PREFIX}_len8_wd2_initLen35_rest",
    *(MODEL_ROOT / f"{NOAUX_PREFIX}_len{n}_wd2_initLen{n - 1}"
      for n in range(9, 65)),
)
DONOR_DIRS = (
    MODEL_ROOT / f"{NOAUX_PREFIX}_len3to5_wd2_ckpt",
    MODEL_ROOT / f"{NOAUX_PREFIX}_len3to5_wd2_ckpt_rest",
)
EXPECTED_COUNTS = {
    ("early", "delete"): 92,
    ("noaux", "keep"): 174,
    ("noaux", "delete"): 350,
    ("stride", "keep"): 39,
    ("stride", "delete"): 326,
    ("donor", "keep"): 24,
}
STRIDE_NAME = re.compile(
    r"^fb_a(?P<a>\d+)b(?P<b>\d+)_r(?P<replica>\d+)_"
    r"len(?P<length>\d+)_a(?P<attempt>\d+)_s(?P<seed>\d+)$"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checked_path(path):
    path = Path(path)
    require(path.is_absolute(), f"not an absolute path: {path}")
    resolved = path.resolve(strict=False)
    require(resolved.is_relative_to(MODEL_ROOT), f"outside {MODEL_ROOT}: {path}")
    require(path.suffix == ".pth", f"not a .pth file: {path}")
    require(not any(part.startswith("cf_") for part in path.parts),
            f"cf_* path is forbidden: {path}")
    require("corruptfix" not in path.name.lower(),
            f"corrupt_fix checkpoint is forbidden: {path}")
    return path


def checkpoint_metadata(path):
    """Load one checkpoint safely and return only JSON-compatible metadata."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(obj, dict), f"checkpoint is not a dict: {path}")
    config = obj.get("config")
    require(isinstance(config, dict), f"checkpoint has no config dict: {path}")
    require(config.get("variant") == "fill_blank",
            f"not a fill_blank checkpoint: {path}")
    train_len = config.get("train_len")
    require(isinstance(train_len, int) and not isinstance(train_len, bool),
            f"invalid config train_len: {path}: {train_len!r}")
    raw_rules = config.get("rules")
    require(isinstance(raw_rules, list) and raw_rules,
            f"invalid config rules: {path}: {raw_rules!r}")
    rules = []
    for rule in raw_rules:
        require(isinstance(rule, (list, tuple)) and rule,
                f"invalid rule: {path}: {rule!r}")
        rules.append([int(value) for value in rule])
    accuracy = obj.get("best_accuracy")
    accuracy_source = "best_accuracy"
    if accuracy is None:
        accuracy = obj.get("best_acc")
        accuracy_source = "best_acc"
    require(isinstance(accuracy, (int, float)) and not isinstance(accuracy, bool),
            f"checkpoint has no numeric best accuracy: {path}")
    return {
        "best_accuracy": float(accuracy),
        "accuracy_source": accuracy_source,
        "train_len": train_len,
        "rules": rules,
    }


def config_key(metadata):
    return metadata["train_len"], tuple(tuple(rule) for rule in metadata["rules"])


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_kind(path):
    if path.name.endswith("_best.pth"):
        return "best"
    if path.name.endswith("_latest.pth"):
        return "latest"
    if path.name.endswith("_resume.pth"):
        return "resume"
    return "final"


def make_entry(path, action, category, reason, hash_all, metadata=None):
    path = checked_path(path)
    require(path.is_file() and not path.is_symlink(),
            f"missing, non-file, or symlink: {path}")
    size = path.stat().st_size
    if metadata is None:
        metadata = checkpoint_metadata(path)
    sha256 = None
    if action == "keep" or hash_all:
        sha256 = sha256_file(path)
    require(path.stat().st_size == size, f"file changed during dry-run: {path}")
    entry = {
        "action": action,
        "category": category,
        "path": str(path),
        "size": size,
        **metadata,
        "kind": checkpoint_kind(path),
        "reason": reason,
        "sha256": sha256,
    }
    return entry


def summarize(entries):
    summary = {
        "keep": 0,
        "delete": 0,
        "estimated_free_bytes": 0,
        "groups": {},
    }
    for category in ("early", "noaux", "stride", "donor"):
        subset = [entry for entry in entries if entry["category"] == category]
        group = {
            "keep": sum(entry["action"] == "keep" for entry in subset),
            "delete": sum(entry["action"] == "delete" for entry in subset),
            "keep_bytes": sum(entry["size"] for entry in subset
                              if entry["action"] == "keep"),
            "delete_bytes": sum(entry["size"] for entry in subset
                                if entry["action"] == "delete"),
        }
        summary["groups"][category] = group
        summary["keep"] += group["keep"]
        summary["delete"] += group["delete"]
        summary["estimated_free_bytes"] += group["delete_bytes"]
    return summary


def direct_pth_files(directory):
    require(directory.is_dir() and not directory.is_symlink(),
            f"missing or symlink directory: {directory}")
    files = sorted(directory.rglob("*.pth"))
    require(all(path.parent == directory for path in files),
            f"unexpected nested checkpoint under: {directory}")
    return files


def stride_directory_metadata(directory):
    match = STRIDE_NAME.fullmatch(directory.name)
    require(match is not None, f"unexpected stride fb directory: {directory}")
    values = {key: int(value) for key, value in match.groupdict().items()}
    return values


def build_manifest(hash_all=False):
    entries = []
    selections = []
    early_corruptfix = []

    def add(path, action, category, reason, metadata=None):
        entry = make_entry(path, action, category, reason, hash_all, metadata)
        entries.append(entry)
        if len(entries) % 50 == 0:
            print(f"[dry-run] processed {len(entries)} checkpoints", flush=True)
        return entry

    print("[dry-run] scanning early fill_blank checkpoints", flush=True)
    for directory in EARLY_DIRS:
        files = direct_pth_files(directory)
        fillblank = [path for path in files if "_fillblank_" in path.name]
        corruptfix = [path for path in files if "_corruptfix_" in path.name]
        require(len(files) == len(fillblank) + len(corruptfix),
                f"unexpected checkpoint name in early directory: {directory}")
        early_corruptfix.extend(str(path) for path in corruptfix)
        for path in fillblank:
            add(path, "delete", "early", "approved early fill_blank checkpoint")

    print("[dry-run] scanning noaux len8 main/rest and len9..64", flush=True)
    require(len(NOAUX_DIRS) == 58, "internal noaux directory list must have 58 entries")
    noaux_keys = set()
    for directory in NOAUX_DIRS:
        match = re.search(r"_len(\d+)_", directory.name)
        require(match is not None, f"cannot parse noaux length: {directory}")
        expected_len = int(match.group(1))
        files = direct_pth_files(directory)
        require(sum(path.name.endswith("_best.pth") for path in files) == 3,
                f"expected exactly three noaux best checkpoints: {directory}")
        for path in files:
            kind = checkpoint_kind(path)
            if kind == "best":
                metadata = checkpoint_metadata(path)
                require(metadata["best_accuracy"] >= THRESHOLD,
                        f"noaux best below {THRESHOLD}: {path}")
                require(metadata["train_len"] == expected_len,
                        f"noaux config train_len mismatch: {path}")
                key = config_key(metadata)
                require(key not in noaux_keys,
                        f"duplicate noaux config train_len/rules: {key}: {path}")
                noaux_keys.add(key)
                add(path, "keep", "noaux",
                    "noaux best; accuracy>=0.99; unique config train_len/rules",
                    metadata)
            else:
                add(path, "delete", "noaux", f"noaux non-best {kind} checkpoint")
    require(len(noaux_keys) == 174, "noaux config keys are not 174 unique pairs")

    print("[dry-run] loading all stride best checkpoints and selecting winners", flush=True)
    stride_dirs = sorted(path for path in STRIDE_ROOT.iterdir()
                         if path.is_dir() and path.name.startswith("fb_"))
    require(len(stride_dirs) == 127,
            f"expected 127 stride fb directories, found {len(stride_dirs)}")
    candidates = []
    stride_files = {}
    for directory in stride_dirs:
        name_meta = stride_directory_metadata(directory)
        files = direct_pth_files(directory)
        stride_files[directory] = files
        for path in files:
            if path.name.endswith("_best.pth"):
                metadata = checkpoint_metadata(path)
                require(metadata["best_accuracy"] >= THRESHOLD,
                        f"stride best below {THRESHOLD}: {path}")
                require(metadata["train_len"] == name_meta["length"],
                        f"stride config train_len mismatch: {path}")
                require(metadata["rules"] == [[name_meta["a"], name_meta["b"]]],
                        f"stride config rules mismatch: {path}")
                candidates.append({
                    "path": path,
                    "metadata": metadata,
                    **name_meta,
                })
    require(len(candidates) == 117,
            f"expected 117 stride best checkpoints, found {len(candidates)}")
    groups = defaultdict(list)
    for candidate in candidates:
        groups[config_key(candidate["metadata"])].append(candidate)
    expected_groups = {
        (length, ((a, b),))
        for length in range(5, 66, 5)
        for a, b in ((25, 56), (56, 93), (62, 59))
    }
    require(set(groups) == expected_groups, "stride config groups are not the approved 39")
    winners = {}
    for key, group in sorted(groups.items()):
        ranked = sorted(group, key=lambda item: (
            -item["metadata"]["best_accuracy"],
            item["replica"],
            item["attempt"],
            0 if item["attempt"] == 0 else 1,
            str(item["path"]),
        ))
        winner = ranked[0]
        winners[winner["path"]] = key
        selections.append({
            "train_len": key[0],
            "rules": [list(rule) for rule in key[1]],
            "candidate_count": len(group),
            "selected_path": str(winner["path"]),
            "best_accuracy": winner["metadata"]["best_accuracy"],
            "replica": winner["replica"],
            "attempt": winner["attempt"],
        })
    require(len(winners) == 39 and len(selections) == 39,
            "stride must select exactly 39 winners")

    for directory in stride_dirs:
        name_meta = stride_directory_metadata(directory)
        for path in stride_files[directory]:
            kind = checkpoint_kind(path)
            candidate = next((item for item in candidates if item["path"] == path), None)
            metadata = candidate["metadata"] if candidate is not None else None
            if kind == "best" and path in winners:
                entry = add(path, "keep", "stride",
                            "highest-accuracy stride best in its config group; "
                            "ties: replica, attempt, a0, path",
                            metadata)
            elif kind == "best":
                group = groups[config_key(metadata)]
                winner = min(group, key=lambda item: (
                    -item["metadata"]["best_accuracy"], item["replica"],
                    item["attempt"], 0 if item["attempt"] == 0 else 1,
                    str(item["path"])))
                entry = add(path, "delete", "stride",
                            f"non-selected stride best; winner={winner['path']}",
                            metadata)
            else:
                entry = add(path, "delete", "stride",
                            f"stride non-best {kind} checkpoint", metadata)
            require(entry["train_len"] == name_meta["length"] and
                    entry["rules"] == [[name_meta["a"], name_meta["b"]]],
                    f"stride non-best config mismatch: {path}")

    print("[dry-run] hashing and recording fully preserved len3to5 donors", flush=True)
    for directory in DONOR_DIRS:
        children = sorted(directory.iterdir())
        require(all(path.is_file() and not path.is_symlink() for path in children),
                f"unexpected non-file in donor directory: {directory}")
        require(all(path.suffix == ".pth" for path in children),
                f"unexpected non-pth donor file: {directory}")
        for path in children:
            add(path, "keep", "donor", "fully preserved len3to5 wd2 donor")

    entries.sort(key=lambda entry: (entry["category"], entry["path"], entry["action"]))
    summary = summarize(entries)
    manifest = {
        "schema_version": 1,
        "mode": "dry-run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_root": str(MODEL_ROOT),
        "selection_policy": (
            "maximum best_accuracy; ties by replica r0<r1<r2, numeric attempt, "
            "a0 first, then absolute path lexicographic order"
        ),
        "summary": summary,
        "inventory": {
            "early_directories": len(EARLY_DIRS),
            "early_corruptfix_preserved": sorted(early_corruptfix),
            "noaux_directories": len(NOAUX_DIRS),
            "noaux_unique_config_pairs": len(noaux_keys),
            "stride_fb_directories": len(stride_dirs),
            "stride_initially_empty_directories": sum(not stride_files[d] for d in stride_dirs),
            "stride_best_candidates": len(candidates),
            "stride_config_groups": len(groups),
            "donor_directories": len(DONOR_DIRS),
        },
        "stride_selections": selections,
        "empty_directory_candidates": [str(path) for path in (*EARLY_DIRS, *stride_dirs)],
        "entries": entries,
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    require(manifest.get("schema_version") == 1, "unsupported manifest schema")
    require(manifest.get("model_root") == str(MODEL_ROOT), "wrong manifest model root")
    entries = manifest.get("entries")
    require(isinstance(entries, list), "manifest entries must be a list")
    counts = Counter((entry.get("category"), entry.get("action")) for entry in entries)
    require(set(counts) == set(EXPECTED_COUNTS), "unexpected category/action in manifest")
    for key, expected in EXPECTED_COUNTS.items():
        require(counts[key] == expected, f"wrong count for {key}: {counts[key]} != {expected}")
    require(len(entries) == 1005, f"expected 1005 entries, found {len(entries)}")

    paths = set()
    keep_paths = set()
    delete_paths = set()
    noaux_keys = set()
    stride_keep_keys = set()
    stride_best_count = 0
    donor_kinds = Counter()
    required_fields = {
        "action", "category", "path", "size", "best_accuracy", "train_len",
        "rules", "kind", "reason", "sha256", "accuracy_source",
    }
    for entry in entries:
        require(required_fields <= set(entry), f"incomplete manifest entry: {entry}")
        path = checked_path(entry["path"])
        require(str(path) == entry["path"], f"non-canonical path: {entry['path']}")
        require(path not in paths, f"duplicate manifest path: {path}")
        paths.add(path)
        require(isinstance(entry["size"], int) and entry["size"] >= 0,
                f"invalid size: {path}")
        require(isinstance(entry["best_accuracy"], (int, float)),
                f"invalid best_accuracy: {path}")
        require(isinstance(entry["train_len"], int) and entry["rules"],
                f"invalid config metadata: {path}")
        require(entry["kind"] in {"best", "final", "latest", "resume"},
                f"invalid checkpoint kind: {path}")
        require(entry["reason"], f"missing reason: {path}")
        if entry["action"] == "keep":
            keep_paths.add(path)
            if entry["kind"] == "best":
                require(entry["best_accuracy"] >= THRESHOLD,
                        f"kept best checkpoint has accuracy below {THRESHOLD}: {path}")
            require(isinstance(entry["sha256"], str) and
                    len(entry["sha256"]) == 64 and
                    all(char in "0123456789abcdef" for char in entry["sha256"]),
                    f"kept checkpoint lacks SHA256: {path}")
        elif entry["action"] == "delete":
            delete_paths.add(path)
            require("len3to5" not in str(path), f"donor path scheduled for delete: {path}")
        else:
            raise RuntimeError(f"invalid action: {path}")

        category = entry["category"]
        if category == "early":
            require(path.parent in EARLY_DIRS and "_fillblank_" in path.name,
                    f"unexpected early path: {path}")
        elif category == "noaux":
            require(path.parent in NOAUX_DIRS, f"unexpected noaux path: {path}")
            require("len3to5" not in str(path), f"donor leaked into noaux: {path}")
            if entry["action"] == "keep":
                require(entry["kind"] == "best", f"noaux non-best kept: {path}")
                key = entry["train_len"], tuple(tuple(rule) for rule in entry["rules"])
                require(key not in noaux_keys, f"duplicate kept noaux config: {key}")
                noaux_keys.add(key)
            else:
                require(entry["kind"] != "best", f"noaux best scheduled for delete: {path}")
        elif category == "stride":
            require(path.parent.parent == STRIDE_ROOT and path.parent.name.startswith("fb_"),
                    f"unexpected stride path: {path}")
            if entry["kind"] == "best":
                stride_best_count += 1
            if entry["action"] == "keep":
                require(entry["kind"] == "best", f"stride non-best kept: {path}")
                key = entry["train_len"], tuple(tuple(rule) for rule in entry["rules"])
                require(key not in stride_keep_keys, f"duplicate stride winner: {key}")
                stride_keep_keys.add(key)
        elif category == "donor":
            require(path.parent in DONOR_DIRS and entry["action"] == "keep",
                    f"unexpected donor entry: {path}")
            require("len3to5" in str(path), f"non-len3to5 donor: {path}")
            donor_kinds[entry["kind"]] += 1
        else:
            raise RuntimeError(f"unexpected category: {category}")

    require(keep_paths.isdisjoint(delete_paths), "keep/delete intersection is not empty")
    require(len(keep_paths) == 237 and len(delete_paths) == 768,
            f"wrong keep/delete totals: {len(keep_paths)}/{len(delete_paths)}")
    require(len(noaux_keys) == 174, "kept noaux configs are not 174 unique pairs")
    require(len(stride_keep_keys) == 39 and stride_best_count == 117,
            "stride manifest does not contain 39 winners among 117 best files")
    require(donor_kinds == Counter({"best": 6, "final": 6, "latest": 6, "resume": 6}),
            f"donor manifest is not the complete 24-file set: {donor_kinds}")
    require(manifest.get("summary") == summarize(entries), "manifest summary is inconsistent")

    selections = manifest.get("stride_selections")
    require(isinstance(selections, list) and len(selections) == 39,
            "manifest must contain 39 stride selections")
    selected_paths = {Path(item["selected_path"]) for item in selections}
    require(selected_paths == {Path(entry["path"]) for entry in entries
                               if entry["category"] == "stride" and entry["action"] == "keep"},
            "stride selection list does not match kept files")
    inventory = manifest.get("inventory", {})
    require(inventory.get("early_directories") == 7 and
            inventory.get("noaux_directories") == 58 and
            inventory.get("stride_best_candidates") == 117 and
            inventory.get("stride_config_groups") == 39 and
            inventory.get("donor_directories") == 2,
            "manifest inventory counts are inconsistent")
    excluded = {Path(path) for path in inventory.get("early_corruptfix_preserved", [])}
    require(len(excluded) == 12 and excluded.isdisjoint(paths),
            "expected 12 out-of-scope early corrupt_fix files, disjoint from manifest")
    candidates = manifest.get("empty_directory_candidates")
    require(isinstance(candidates, list) and len(candidates) == 134,
            "expected seven early plus 127 stride empty-directory candidates")
    for raw_path in candidates:
        path = Path(raw_path)
        require(path.is_absolute() and path.resolve(strict=False).is_relative_to(MODEL_ROOT),
                f"empty-directory candidate outside model root: {path}")
        require(path in EARLY_DIRS or
                (path.parent == STRIDE_ROOT and path.name.startswith("fb_")),
                f"unexpected empty-directory candidate: {path}")


def verify_keeps(entries):
    keeps = [entry for entry in entries if entry["action"] == "keep"]
    require(len(keeps) == 237, f"expected 237 keeps, found {len(keeps)}")
    for index, entry in enumerate(keeps, 1):
        path = checked_path(entry["path"])
        require(path.is_file() and not path.is_symlink(), f"kept file missing: {path}")
        require(path.stat().st_size == entry["size"], f"kept file size changed: {path}")
        metadata = checkpoint_metadata(path)
        for key in ("best_accuracy", "train_len", "rules"):
            require(metadata[key] == entry[key],
                    f"kept file {key} changed: {path}: {metadata[key]!r} != {entry[key]!r}")
        if entry["kind"] == "best":
            require(metadata["best_accuracy"] >= THRESHOLD,
                    f"kept best file accuracy below {THRESHOLD}: {path}")
        if index % 50 == 0:
            print(f"verified {index}/237 kept checkpoints", flush=True)
    print("verified all 237 kept checkpoints", flush=True)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def print_summary(manifest, manifest_path):
    summary = manifest["summary"]
    print("\nDry-run complete; no checkpoint or model directory was removed.")
    for category in ("early", "noaux", "stride", "donor"):
        group = summary["groups"][category]
        print(f"{category:6s}: keep={group['keep']:3d} delete={group['delete']:3d} "
              f"delete_bytes={group['delete_bytes']}")
    print(f"TOTAL : keep={summary['keep']} delete={summary['delete']}")
    print(f"Estimated space to free: {summary['estimated_free_bytes']} bytes "
          f"({summary['estimated_free_bytes'] / (1024 ** 3):.3f} GiB)")
    inventory = manifest["inventory"]
    print(f"Out-of-scope early corrupt_fix files preserved: "
          f"{len(inventory['early_corruptfix_preserved'])}")
    print(f"Initially empty stride retry directories (not file deletions): "
          f"{inventory['stride_initially_empty_directories']}")
    print("\nStride selections (one per config train_len/rules):")
    for item in manifest["stride_selections"]:
        print(f"  len={item['train_len']:2d} rules={item['rules']} "
              f"acc={item['best_accuracy']:.9f} r={item['replica']} "
              f"a={item['attempt']} candidates={item['candidate_count']} "
              f"path={item['selected_path']}")
    print(f"\nManifest: {manifest_path}")


def apply_manifest(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("mode") == "dry-run", "apply requires a dry-run manifest")
    validate_manifest(manifest)
    entries = manifest["entries"]

    print("Apply preflight: checking all 768 delete paths and sizes", flush=True)
    for entry in (entry for entry in entries if entry["action"] == "delete"):
        path = checked_path(entry["path"])
        require(path.is_file() and not path.is_symlink(),
                f"delete path missing, non-file, or symlink: {path}")
        require(path.stat().st_size == entry["size"],
                f"delete path size differs from manifest: {path}")
    print("Apply preflight: reloading all kept checkpoints", flush=True)
    verify_keeps(entries)

    # No assertion is intentionally introduced into the deletion loop: every
    # required safety assertion has completed before the first unlink.
    deleted = 0
    freed_bytes = 0
    for entry in (entry for entry in entries if entry["action"] == "delete"):
        path = Path(entry["path"])
        path.unlink()
        deleted += 1
        freed_bytes += entry["size"]
        if deleted % 100 == 0:
            print(f"deleted {deleted}/768 files", flush=True)

    removed_directories = 0
    for raw_path in manifest["empty_directory_candidates"]:
        path = Path(raw_path)
        if path.exists() and path.is_dir() and not path.is_symlink():
            if not any(path.iterdir()):
                path.rmdir()
                removed_directories += 1

    print("Apply post-check: reloading kept checkpoints and checking deleted paths", flush=True)
    verify_keeps(entries)
    for entry in (entry for entry in entries if entry["action"] == "delete"):
        require(not Path(entry["path"]).exists(),
                f"deleted path still exists: {entry['path']}")
    require(deleted == 768 and freed_bytes == manifest["summary"]["estimated_free_bytes"],
            "actual deletion count/bytes do not match manifest")
    result = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "deleted_files": deleted,
        "freed_bytes": freed_bytes,
        "removed_empty_directories": removed_directories,
        "kept_files": 237,
        "postcheck": "all keeps exist and reload; all deletes are absent",
    }
    write_json(RESULT_PATH, result)
    print(json.dumps(result, indent=2), flush=True)
    print(f"Result: {RESULT_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="build the manifest without deleting anything (default)")
    mode.add_argument("--apply", action="store_true",
                      help="delete exactly the paths in an existing dry-run manifest")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help=f"manifest path (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--hash-all", action="store_true",
                        help="dry-run: hash deletes too; keeps are always hashed")
    args = parser.parse_args()
    if args.apply:
        require(not args.hash_all, "--hash-all is a dry-run-only option")
        apply_manifest(args.manifest)
    else:
        manifest = build_manifest(hash_all=args.hash_all)
        write_json(args.manifest, manifest)
        print_summary(manifest, args.manifest)


if __name__ == "__main__":
    main()
