"""Gather the episodes contributed to a community dataset into one raw dataset.

A community dataset holds no recordings. Each accepted contribution is one small
JSON under ``contrib/<contributor>/<episode_id>.json`` pointing at the episodes,
which live in the CONTRIBUTOR's own raw dataset. That split is what keeps the
Hub's two hard constraints out of the way: no unreviewed gigabyte on the owner's
storage quota, and no two contributors ever writing the same file (so their pull
requests merge in any order).

Turning that into a LeRobot dataset is two steps, and only the first is here:

  1. **consolidate** — copy the episodes named by the merged manifests into ONE
     raw dataset in the owner's namespace, in the ``{episode}/{role}`` layout the
     converter expects. A single writer, so nothing can conflict;
  2. **convert** — point the SLAM Space at that raw dataset. It runs the
     completeness checks, SLAM, the build and the push, exactly as it does for a
     dataset recorded by one person.

Episode directories are renamed ``{episode_id}__{contributor}``: episode ids are
UTC stamps at one-second resolution, so two contributors CAN produce the same one,
and the converter only uses the directory name as a label (the roles are the
subdirectories). Collisions are therefore impossible, and provenance stays legible
in the raw dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

CONTRIB_DIR = "contrib"

# The reason a contribution is skipped because it is already in the raw dataset.
# A constant because callers COUNT it separately from the reasons that need
# reading — "8 already gathered" is reassurance, "1 missing an arm" is a problem —
# and keying that split on a prose string typed twice would break silently.
ALREADY_GATHERED = "already gathered"


@dataclass
class Contribution:
    """One accepted episode, as its manifest describes it."""
    contributor: str
    episode_id: str
    raw_repo: str
    roles: list[str]
    task: str = ""
    path: str = ""  # where the manifest itself lives, for error messages

    @property
    def dir_name(self) -> str:
        return episode_dir_name(self.episode_id, self.contributor)


@dataclass
class Plan:
    """What a gathering run will do, and what it will leave out."""
    gather: list[Contribution] = field(default_factory=list)
    skipped: list[tuple[Contribution, str]] = field(default_factory=list)
    # Manifests that could not be read as a contribution at all.
    malformed: list[tuple[str, str]] = field(default_factory=list)


def episode_dir_name(episode_id: str, contributor: str) -> str:
    """Directory an episode gets in the consolidated raw dataset.

    Carries the contributor because episode ids are UTC ``YYYYmmdd_HHMMSS``
    stamps: two people recording in the same second would otherwise overwrite
    each other. The converter treats this name as a label only, so it is free to
    say something useful.
    """
    return f"{episode_id}__{contributor}"


def parse_manifest(path: str, data: dict) -> tuple[Optional[Contribution], str]:
    """Read one contrib/<user>/<episode>.json. Returns (contribution, reason).

    A manifest missing any of the four things a copy needs is refused rather than
    guessed at: without them there is nothing to fetch, or nowhere to put it.
    """
    if not isinstance(data, dict):
        return None, "not a JSON object"
    contributor = str(data.get("contributor") or "").strip()
    episode_id = str(data.get("episode_id") or "").strip()
    raw_repo = str(data.get("raw_repo") or "").strip()
    roles = [str(r) for r in (data.get("roles") or []) if str(r).strip()]
    missing = [n for n, v in (("contributor", contributor), ("episode_id", episode_id),
                              ("raw_repo", raw_repo), ("roles", roles)) if not v]
    if missing:
        return None, f"missing {', '.join(missing)}"
    if "/" not in raw_repo:
        return None, f"raw_repo '{raw_repo}' is not a <namespace>/<name> repo id"
    return Contribution(contributor=contributor, episode_id=episode_id,
                        raw_repo=raw_repo, roles=roles,
                        task=str(data.get("task_name") or data.get("task") or ""),
                        path=path), ""


def plan_gathering(contributions: Iterable[Contribution], *,
                   only_contributors: Optional[Iterable[str]] = None,
                   present: Iterable[str] = (),
                   require_roles: Optional[Iterable[str]] = None) -> Plan:
    """Decide what to copy. Pure: no Hub, no disk.

    present: directory names already in the consolidated raw dataset, so a re-run
        copies only what is new. Gathering is meant to be repeatable — a
        contribution merged today must not mean re-uploading everything.

    require_roles: the task's device signature. A contribution that does not carry
        every role is left out AND named: the converter drops a recording missing
        an arm anyway, and an omission nobody sees is what makes a dataset
        untrustworthy.
    """
    only = {c for c in (only_contributors or [])}
    have = set(present)
    need = {r for r in (require_roles or [])}
    plan = Plan()
    seen: set[str] = set()
    for c in contributions:
        if only and c.contributor not in only:
            continue
        if c.dir_name in seen:
            plan.skipped.append((c, "duplicate manifest for this episode"))
            continue
        seen.add(c.dir_name)
        if c.dir_name in have:
            plan.skipped.append((c, ALREADY_GATHERED))
            continue
        if need and not need.issubset(set(c.roles)):
            plan.skipped.append(
                (c, f"carries {'+'.join(sorted(c.roles))}, the task needs "
                    f"{'+'.join(sorted(need))}"))
            continue
        plan.gather.append(c)
    return plan


def skipped_lines(skipped, *, names: int = 6) -> list[str]:
    """One line per REASON, with a few names each. For the log.

    Grouped rather than listed one episode per line: a routine re-run skips
    everything already gathered, and a hundred identical lines would bury the
    reasons that actually need reading. Counted first, named after — and the
    count is always exact even when the names are cut short.
    """
    by_reason: dict[str, list[str]] = {}
    for c, why in skipped:
        by_reason.setdefault(why, []).append(f"{c.episode_id} ({c.contributor})")
    out = []
    for why, who in sorted(by_reason.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        shown = ", ".join(who[:names])
        more = f", +{len(who) - names} more" if len(who) > names else ""
        out.append(f"  – {len(who)} {why}: {shown}{more}")
    return out


def read_contributions(community_repo: str, *, token: Optional[str] = None,
                       revision: str = "main") -> tuple[list[Contribution], list[tuple[str, str]]]:
    """Read every merged manifest from a community dataset.

    Returns (contributions, malformed) — the second being (path, reason) pairs, so
    a bad file is reported instead of silently reducing the harvest.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    files = [f for f in api.list_repo_files(community_repo, repo_type="dataset",
                                            revision=revision)
             if f.startswith(f"{CONTRIB_DIR}/") and f.endswith(".json")]
    out: list[Contribution] = []
    bad: list[tuple[str, str]] = []
    for path in sorted(files):
        local = hf_hub_download(community_repo, path, repo_type="dataset",
                                revision=revision, token=token)
        try:
            data = json.loads(Path(local).read_text())
        except Exception as e:  # noqa: BLE001
            bad.append((path, f"unreadable: {e}"))
            continue
        contribution, reason = parse_manifest(path, data)
        if contribution is None:
            bad.append((path, reason))
        else:
            out.append(contribution)
    return out, bad


def gathered_dirs(raw_repo: str, *, token: Optional[str] = None) -> set[str]:
    """Top-level episode directories already in the consolidated raw dataset.

    An empty set when the repo does not exist yet (the first run), and when no
    repo is named at all — a dry run may be asked without a target, and it then
    simply cannot know what is already there.
    """
    from huggingface_hub import HfApi

    if not raw_repo:
        return set()
    try:
        files = HfApi(token=token).list_repo_files(raw_repo, repo_type="dataset")
    except Exception:  # noqa: BLE001 — not created yet
        return set()
    return {f.split("/", 1)[0] for f in files if "/" in f}


def gather(community_repo: str, raw_repo: str, *, token: Optional[str] = None,
           only_contributors: Optional[Iterable[str]] = None,
           require_roles: Optional[Iterable[str]] = None,
           dry_run: bool = False, log=print, on_plan=None) -> Plan:
    """Copy the contributed episodes into `raw_repo`. Returns what was done.

    raw_repo may be empty ONLY with dry_run: a preview writes nothing, so it needs
    no target — it just cannot then say what is already gathered.

    on_plan(plan) is called once the plan is known and BEFORE any copying, so a
    caller can show the counts while the copy runs rather than only after it. The
    plan it receives is not final: episodes whose source turns out to be missing
    are moved to `skipped` as the copy proceeds.

    One commit per episode, which makes a re-run resumable: an interrupted or
    partial gathering is fixed by running it again, since an episode already
    there is skipped rather than re-uploaded.

    An episode whose source no longer holds every role it promised is skipped and
    named. That is not hypothetical: the recordings live in the contributor's own
    repo, which they can change or delete at any time — the reason to copy what
    you keep rather than depend on it.
    """
    from huggingface_hub import HfApi, snapshot_download

    if not raw_repo and not dry_run:
        raise ValueError("a raw dataset to write is required unless dry_run is set")
    api = HfApi(token=token)
    contributions, malformed = read_contributions(community_repo, token=token)
    plan = plan_gathering(contributions, only_contributors=only_contributors,
                          present=gathered_dirs(raw_repo, token=token),
                          require_roles=require_roles)
    plan.malformed = malformed
    for path, reason in malformed:
        log(f"  ⚠️ {path}: {reason}")
    log(f"{len(contributions)} contribution(s) in {community_repo}; "
        f"{len(plan.gather)} to gather, {len(plan.skipped)} skipped")
    if not raw_repo:
        log("  ⚠️ no raw dataset named: this plan cannot account for episodes "
            "already gathered")
    for line in skipped_lines(plan.skipped):
        log(line)
    if on_plan is not None:
        on_plan(plan)
    if dry_run or not plan.gather:
        return plan

    api.create_repo(raw_repo, repo_type="dataset", exist_ok=True)
    # Group by source repo so each contributor's episodes come down in one call.
    by_repo: dict[str, list[Contribution]] = {}
    for c in plan.gather:
        by_repo.setdefault(c.raw_repo, []).append(c)
    done: list[Contribution] = []
    for source, group in by_repo.items():
        log(f"\n{source}: fetching {len(group)} episode(s)…")
        snap = Path(snapshot_download(
            source, repo_type="dataset", token=token,
            allow_patterns=[f"{c.episode_id}/**" for c in group]))
        for c in group:
            ep_dir = snap / c.episode_id
            absent = [r for r in c.roles if not (ep_dir / r).is_dir()]
            if not ep_dir.is_dir() or absent:
                why = ("the source no longer holds it" if not ep_dir.is_dir()
                       else f"the source is missing its {'+'.join(absent)} data")
                log(f"  ⚠️ skipping {c.episode_id} from {c.contributor}: {why}")
                plan.skipped.append((c, why))
                continue
            log(f"  ↑ {c.dir_name}")
            api.upload_folder(folder_path=str(ep_dir), repo_id=raw_repo,
                              repo_type="dataset", path_in_repo=c.dir_name,
                              commit_message=f"Add {c.episode_id} from {c.contributor}")
            done.append(c)
    plan.gather = done
    log(f"\n{len(done)} episode(s) now in https://huggingface.co/datasets/{raw_repo}")
    return plan
