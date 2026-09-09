"""Gathering contributed episodes into one raw dataset.

The manifests come from strangers and point at repos those strangers control, so
the decisions here are all about what to REFUSE: a malformed manifest, an episode
that no longer exists, a take missing an arm, the same episode twice. Every
refusal is named, because a dataset quietly built from half the contributions is
worse than one that says what it left out.
"""
from grabette_postprocess.community import (
    Contribution,
    episode_dir_name,
    parse_manifest,
    plan_gathering,
)

EP1, EP2 = "20260907_100000", "20260907_100500"


def manifest(**kw):
    base = {"contributor": "alice", "episode_id": EP1,
            "raw_repo": "alice/grabette-community-fold-a-towel-raw",
            "roles": ["left", "right"], "task_name": "Fold a towel"}
    base.update(kw)
    return base


def contribution(**kw):
    c, reason = parse_manifest("contrib/alice/x.json", manifest(**kw))
    assert c is not None, reason
    return c


# ── the episode directory carries the contributor ───────────────────────────

def test_two_contributors_recording_in_the_same_second_do_not_collide():
    # Episode ids are UTC stamps at one-second resolution, so this is reachable,
    # and the loser would silently overwrite the winner.
    assert episode_dir_name(EP1, "alice") != episode_dir_name(EP1, "bob")
    assert episode_dir_name(EP1, "alice").startswith(EP1)  # still sorts by time


# ── reading a manifest ──────────────────────────────────────────────────────

def test_a_complete_manifest_is_read():
    c = contribution()
    assert (c.contributor, c.episode_id, c.roles) == ("alice", EP1, ["left", "right"])
    assert c.raw_repo == "alice/grabette-community-fold-a-towel-raw"


def test_a_manifest_missing_what_a_copy_needs_is_refused():
    for field in ("contributor", "episode_id", "raw_repo"):
        c, reason = parse_manifest("p", manifest(**{field: ""}))
        assert c is None and field in reason
    c, reason = parse_manifest("p", manifest(roles=[]))
    assert c is None and "roles" in reason


def test_a_manifest_naming_something_that_is_not_a_repo_is_refused():
    c, reason = parse_manifest("p", manifest(raw_repo="just-a-name"))
    assert c is None and "repo id" in reason


def test_a_manifest_that_is_not_an_object_is_refused():
    c, reason = parse_manifest("p", ["nope"])
    assert c is None and reason


# ── planning what to copy ───────────────────────────────────────────────────

def test_everything_new_is_gathered():
    plan = plan_gathering([contribution(), contribution(episode_id=EP2)])
    assert [c.episode_id for c in plan.gather] == [EP1, EP2]
    assert plan.skipped == []


def test_what_is_already_there_is_not_copied_again():
    # Merging one new pull request must not mean re-uploading the whole dataset.
    c1, c2 = contribution(), contribution(episode_id=EP2)
    plan = plan_gathering([c1, c2], present=[c1.dir_name])
    assert [c.episode_id for c in plan.gather] == [EP2]
    assert plan.skipped == [(c1, "already gathered")]


def test_a_contribution_missing_a_required_role_is_left_out_and_named():
    # The converter drops a recording missing an arm anyway; saying so here is
    # what keeps the omission visible.
    c = contribution(roles=["left"])
    plan = plan_gathering([c], require_roles=["left", "right"])
    assert plan.gather == []
    assert "the task needs left+right" in plan.skipped[0][1]


def test_a_contribution_with_more_roles_than_required_is_kept():
    plan = plan_gathering([contribution(roles=["left", "right", "casquette"])],
                          require_roles=["left", "right"])
    assert len(plan.gather) == 1


def test_only_the_named_contributors_are_gathered():
    plan = plan_gathering([contribution(), contribution(contributor="bob")],
                          only_contributors=["bob"])
    assert [c.contributor for c in plan.gather] == ["bob"]


def test_the_same_episode_manifested_twice_is_copied_once():
    plan = plan_gathering([contribution(), contribution()])
    assert len(plan.gather) == 1
    assert "duplicate" in plan.skipped[0][1]


def test_the_same_episode_id_from_two_people_is_two_episodes():
    plan = plan_gathering([contribution(), contribution(contributor="bob")])
    assert len(plan.gather) == 2


# ── a dry run needs no target ───────────────────────────────────────────────

def test_a_plan_can_be_made_without_knowing_the_target():
    # A dry run writes nothing, so it may be asked before a target is chosen. It
    # then cannot subtract what is already gathered, which the caller is told.
    from grabette_postprocess.community import gathered_dirs
    assert gathered_dirs("") == set()


def test_a_real_gather_still_demands_a_target():
    import pytest as _pytest

    from grabette_postprocess.community import gather
    with _pytest.raises(ValueError, match="required unless dry_run"):
        gather("org/community", "", dry_run=False)


# ── what was left out, in the log ───────────────────────────────────────────

def test_the_reasons_are_grouped_and_counted():
    from grabette_postprocess.community import skipped_lines
    c1, c2 = contribution(), contribution(episode_id=EP2)
    lines = skipped_lines([(c1, "already gathered"), (c2, "already gathered")])
    assert len(lines) == 1
    assert lines[0].startswith("  – 2 already gathered:")
    assert EP1 in lines[0] and EP2 in lines[0]


def test_the_commonest_reason_comes_first():
    from grabette_postprocess.community import skipped_lines
    skipped = [(contribution(episode_id=f"2026090{i}_100000"), "already gathered")
               for i in range(3)]
    skipped.append((contribution(episode_id=EP2), "missing an arm"))
    lines = skipped_lines(skipped)
    assert lines[0].startswith("  – 3 already gathered")
    assert lines[1].startswith("  – 1 missing an arm")


def test_a_long_list_is_counted_in_full_but_named_in_part():
    # A routine re-run skips everything already gathered; a hundred identical
    # lines would bury the reasons that need reading.
    from grabette_postprocess.community import skipped_lines
    skipped = [(contribution(episode_id=f"2026090{i}_100000"), "already gathered")
               for i in range(9)]
    line = skipped_lines(skipped, names=6)[0]
    assert line.startswith("  – 9 already gathered")   # the count stays exact
    assert "+3 more" in line


def test_already_gathered_has_a_name_callers_can_count_on():
    # The split between "reassurance" and "a problem" is counted by callers, so
    # the reason cannot be a prose string typed twice.
    from grabette_postprocess.community import ALREADY_GATHERED
    c = contribution()
    plan = plan_gathering([c], present=[c.dir_name])
    assert plan.skipped == [(c, ALREADY_GATHERED)]
