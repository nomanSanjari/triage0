"""Run the git history helpers on real curl.

The unit tests use a tiny made-up repo; only this shows the helpers agree with
the real history the golden fixtures were labelled from. Needs internet:
TRIAGE0_NETWORK_TESTS=1 (clones curl, about 140 MB).
"""

from __future__ import annotations

import os

import pytest

from triage0.checkers.tools.git_history import (
    blame_lines,
    last_change,
    line_history,
    parents,
    path_in_tree,
    symbol_in_tree,
)
from triage0.checkers.tools.git_tools import run_git
from triage0.storage.repo_cache import RepoCache

pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAGE0_NETWORK_TESTS") != "1",
    reason="set TRIAGE0_NETWORK_TESTS=1 (clones curl)",
)

URL = "https://github.com/curl/curl"
# "curlx: move into to curlx/" renamed Curl_dyn_* to curlx_dyn_* (see the
# renamed_symbol golden fixture); looked up in the real history first.
RENAME_COMMIT = "255aac56f9773d7c138816bfe0390293f7da33a5"


@pytest.fixture(scope="module")
def curl(repos: RepoCache) -> tuple[RepoCache, str]:
    repo = repos.ensure_clone(URL)
    version = run_git(["rev-parse", "curl-8_23_0^{commit}"], repo=repo).stdout.strip()
    return repos, version


def test_renamed_symbol_is_found_gone_and_its_removal_verified(
    curl: tuple[RepoCache, str],
) -> None:
    repos, version = curl
    repo = repos.ensure_clone(URL)

    assert symbol_in_tree(repo, version, "Curl_dyn_add") is False
    assert symbol_in_tree(repo, version, "curlx_dyn_add") is True
    assert last_change(repo, "Curl_dyn_add", version) == RENAME_COMMIT
    # The whole-word check that makes the commit above trustworthy.
    (before,) = parents(repo, RENAME_COMMIT)
    assert symbol_in_tree(repo, before, "Curl_dyn_add") is True
    assert symbol_in_tree(repo, RENAME_COMMIT, "Curl_dyn_add") is False


def test_a_fragment_of_a_real_name_hits_but_fails_the_whole_word_check(
    curl: tuple[RepoCache, str],
) -> None:
    # `git log -S` is a text match, so it finds an unrelated commit for
    # "curlx_dyn_ad"; only the whole-word check shows it was never a word.
    repos, version = curl
    repo = repos.ensure_clone(URL)

    hit = last_change(repo, "curlx_dyn_ad", version)

    assert hit is not None
    (before,) = parents(repo, hit)
    assert symbol_in_tree(repo, before, "curlx_dyn_ad") is False


def test_made_up_symbol_never_appears_in_any_ref(curl: tuple[RepoCache, str]) -> None:
    repos, _ = curl
    assert last_change(repos.ensure_clone(URL), "Curl_cookie_parse_fast") is None


def test_files_lines_and_blame_match_the_fixture_labels(curl: tuple[RepoCache, str]) -> None:
    repos, version = curl
    repo = repos.ensure_clone(URL)

    assert path_in_tree(repo, version, "lib/curlx/dynbuf.c") is True
    assert path_in_tree(repo, version, "lib/dynbuf.c") is False  # moved by the rename
    assert line_history(repo, version, "lib/curlx/dynbuf.c", 173, 173)[0] == RENAME_COMMIT
    assert blame_lines(repo, version, "lib/curlx/dynbuf.c", 173, 173) == {173: RENAME_COMMIT}
