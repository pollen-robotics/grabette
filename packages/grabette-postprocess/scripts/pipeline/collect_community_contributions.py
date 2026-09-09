#!/usr/bin/env python3
"""Gather the episodes contributed to a community dataset into one raw dataset.

Step 1 of two. A community dataset holds only manifests: one small JSON per
accepted episode under contrib/<contributor>/, pointing at recordings that live
in each contributor's own raw dataset. This copies the ones you merged into a
single raw dataset of your own, in the {episode}/{role} layout the converter
expects.

Step 2 is the SLAM Space, pointed at the raw dataset this produces — it runs the
checks, SLAM, the build and the push.

Re-runnable: an episode already gathered is skipped, so merging a new pull
request means running this again, not rebuilding from scratch.

Requires: huggingface-cli login (or HF_TOKEN env var).
"""

import sys
from pathlib import Path

import click

try:
    from grabette_postprocess.community import gather, skipped_lines
except ModuleNotFoundError:
    # Run straight from a checkout, without installing the package. Worth the
    # four lines: this step needs `huggingface_hub` and `click` and nothing else,
    # so requiring the full postprocess install (lerobot, opencv, av) to copy
    # files between two Hub repos would be absurd.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from grabette_postprocess.community import gather, skipped_lines


@click.command()
@click.option("--community-repo", required=True,
              help="The community dataset holding contrib/ (e.g. "
                   "'pollen-robotics/grabette-community-fold-a-towel')")
@click.option("--raw-repo", required=True,
              help="YOUR raw dataset (created if needed). Gatherings ADD to it: "
                   "an episode already there is skipped, nothing is replaced")
@click.option("--contributor", "contributors", multiple=True,
              help="Only gather from these contributors (repeatable)")
@click.option("--role", "roles", multiple=True,
              help="Roles the task requires, e.g. --role left --role right. "
                   "A contribution missing one is skipped and named.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Say what would be gathered, copy nothing")
def main(community_repo, raw_repo, contributors, roles, dry_run):
    plan = gather(community_repo, raw_repo,
                  only_contributors=contributors or None,
                  require_roles=roles or None, dry_run=dry_run)
    # The reasons already went through the log via gather(); repeat the grouped
    # view at the end so a long run needs no scrolling back.
    if plan.skipped:
        click.echo("\nLeft out:")
        for line in skipped_lines(plan.skipped, names=20):
            click.echo(line)
    if not dry_run and plan.gather:
        click.echo("\nNext: convert it with the SLAM Space —")
        click.echo(f"  source repo: {raw_repo}")
        if roles:
            click.echo(f"  roles:       {', '.join(roles)}")
        click.echo("  target repo: <your namespace>/<lerobot dataset name>")


if __name__ == "__main__":
    main()
