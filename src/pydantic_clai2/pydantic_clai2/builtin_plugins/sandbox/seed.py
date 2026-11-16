"""Copy the local checkout into a new sandbox: its branch, its commits, and its uncommitted changes.

`snapshot` reads the checkout without changing it: the uncommitted files go into a commit made with a
temporary index, which no branch points to. A GitHub checkout is cloned in the sandbox, through the proxy
that holds the token, and only the commits GitHub may lack travel from here, in a Git bundle; any other
checkout travels whole. `seed` then rebuilds the same branch and files in the sandbox.
"""

import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic_ai.workspaces import SupportsCommands, SupportsFilesystem, WorkspaceBackend

BUNDLE_PATH = '/tmp/clai-seed.bundle'
_SEED_REF = 'refs/clai-seed/work'
_GITHUB = re.compile(
    r'^(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)([\w.-]+/[\w.-]+?)(?:\.git)?/?$'
)
_SEED_TIMEOUT = 900.0
"""Seconds the sandbox may take to clone and rebuild the checkout; a large repository clones slowly."""


class SandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem, Protocol):
    """A sandbox's workspace: it runs commands and reads and writes files, so `seed` can copy into it."""


@dataclass(frozen=True, kw_only=True)
class Snapshot:
    """What `seed` rebuilds in the sandbox; all `None` for a folder outside Git."""

    clone_url: str | None = None
    """The `https://github.com/...` URL the sandbox clones, from `origin`; `None` when it is not on GitHub."""
    branch: str | None = None
    """The checked-out branch; `None` when `HEAD` is detached."""
    head: str | None = None
    """The commit checked out; `None` in a repository with no commits yet."""
    work: str | None = None
    """The commit holding the uncommitted files too; `head` when there are none."""
    bundle: bytes | None = None
    """The commits the sandbox cannot get from GitHub; `None` when it needs none."""
    author_name: str | None = None
    """Git's `user.name` here, for the sandbox's commits."""
    author_email: str | None = None
    """Git's `user.email` here, for the sandbox's commits."""

    @property
    def changed(self) -> bool:
        """Whether there were uncommitted changes, untracked files included."""
        return self.work != self.head


def github_clone_url(remote: str) -> str | None:
    """The HTTPS URL of a GitHub remote, as the proxy authenticates only HTTPS; `None` for any other remote."""
    found = _GITHUB.match(remote.strip())
    return f'https://github.com/{found.group(1)}.git' if found else None


def _git(root: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ['git', '-C', str(root), *args],
        capture_output=True,
        text=True,
        encoding='utf-8',
        env=env,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f'git {args[0]} failed: {result.stderr.strip() or result.stdout.strip()}')
    return result.stdout.strip()


def _try_git(root: Path, *args: str) -> str | None:
    try:
        return _git(root, *args) or None
    except ValueError:
        return None


def snapshot(directory: Path) -> Snapshot:
    """Read the checkout at `directory`, or an empty `Snapshot` outside Git; `ValueError` when Git fails."""
    top = _try_git(directory, 'rev-parse', '--show-toplevel')
    if top is None:
        return Snapshot()
    root = Path(top)
    head = _try_git(root, 'rev-parse', '--verify', '--quiet', 'HEAD^{commit}')
    remote = _try_git(root, 'remote', 'get-url', 'origin')
    clone_url = github_clone_url(remote) if remote else None
    branch = _try_git(root, 'symbolic-ref', '--short', '-q', 'HEAD')
    name, email = _try_git(root, 'config', 'user.name'), _try_git(root, 'config', 'user.email')
    work = _work_commit(root, head)
    if work is None:
        return Snapshot(clone_url=clone_url, branch=branch, author_name=name, author_email=email)
    _git(root, 'update-ref', _SEED_REF, work)
    try:
        bundle = _bundle(root, clone_url=clone_url)
    finally:
        _git(root, 'update-ref', '-d', _SEED_REF)
    return Snapshot(
        clone_url=clone_url, branch=branch, head=head, work=work, bundle=bundle, author_name=name, author_email=email
    )


def _work_commit(root: Path, head: str | None) -> str | None:
    """`head`, or a commit on top of it holding every file Git would add, without touching the real index."""
    with tempfile.TemporaryDirectory() as scratch:
        env = {
            **os.environ,
            'GIT_INDEX_FILE': str(Path(scratch) / 'index'),
            'GIT_AUTHOR_NAME': 'clai',
            'GIT_AUTHOR_EMAIL': 'clai@localhost',
            'GIT_COMMITTER_NAME': 'clai',
            'GIT_COMMITTER_EMAIL': 'clai@localhost',
        }
        if head is not None:
            _git(root, 'read-tree', head, env=env)
        _git(root, 'add', '--all', env=env)
        tree = _git(root, 'write-tree', env=env)
        if head is not None and tree == _git(root, 'rev-parse', f'{head}^{{tree}}'):
            return head
        if head is None and not _git(root, 'ls-files', env=env):
            return None
        parent = ('-p', head) if head is not None else ()
        return _git(root, 'commit-tree', tree, *parent, '-m', 'Uncommitted changes', env=env)


def _bundle(root: Path, *, clone_url: str | None) -> bytes | None:
    """The seed commit's history, less what GitHub has when the sandbox clones from it; `None` when that is nothing."""
    # A clone has every branch on GitHub, which `origin`'s remote-tracking branches name.
    exclude = ('--not', '--remotes=origin') if clone_url is not None else ()
    if _git(root, 'rev-list', '--count', _SEED_REF, *exclude) == '0':
        return None
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / 'seed.bundle'
        _git(root, 'bundle', 'create', '--quiet', str(path), _SEED_REF, *exclude)
        return path.read_bytes()


def seed_script(snapshot: Snapshot) -> str:
    """The shell script that rebuilds `snapshot` in the sandbox's working directory, which must be empty."""
    q = shlex.quote
    lines = [
        'set -eu',
        # SSH remotes carry no credential the proxy can set; HTTPS ones do.
        'git config --global url.https://github.com/.insteadOf git@github.com:',
        'git config --global --add url.https://github.com/.insteadOf ssh://git@github.com/',
        # git sends `GH_TOKEN`'s stand-in; the provider's proxy replaces it or the whole header.
        'git config --global credential.https://github.com.helper '
        + shlex.quote('!f() { echo username=x-access-token; echo "password=$GH_TOKEN"; }; f'),
    ]
    if snapshot.author_name:
        lines.append(f'git config --global user.name {q(snapshot.author_name)}')
    if snapshot.author_email:
        lines.append(f'git config --global user.email {q(snapshot.author_email)}')
    if snapshot.clone_url is not None:
        lines.append(f'git clone --quiet --no-checkout {q(snapshot.clone_url)} .')
    else:
        lines.append('git init --quiet')
    if snapshot.bundle is not None:
        lines += [f'git fetch --quiet {BUNDLE_PATH} {_SEED_REF}:{_SEED_REF}', f'rm -f {BUNDLE_PATH}']
    branch = q(snapshot.branch) if snapshot.branch else None
    if snapshot.head is not None:
        lines.append(
            f'git checkout --quiet -B {branch} {snapshot.head}'
            if branch
            else f'git checkout --quiet --detach {snapshot.head}'
        )
        if branch and snapshot.clone_url is not None:
            lines.append(f'git branch --quiet --set-upstream-to=origin/{branch} 2>/dev/null || true')
    elif branch:
        lines.append(f'git symbolic-ref HEAD refs/heads/{branch}')
    if snapshot.changed and snapshot.work is not None:
        # The uncommitted files, as they were: changed in the working tree, not staged.
        lines += (
            [f'git read-tree -u -m HEAD {snapshot.work}', 'git reset --quiet']
            if snapshot.head is not None
            else [f'git read-tree -u --reset {snapshot.work}', 'git read-tree --empty']
        )
    if snapshot.bundle is not None:
        lines.append(f'git update-ref -d {_SEED_REF}')
    return '\n'.join(lines)


async def seed(backend: SandboxBackend, snapshot: Snapshot) -> None:
    """Rebuild `snapshot` in `backend`'s working directory; `ValueError` with Git's output when that fails."""
    if snapshot.bundle is not None:
        await backend.write_bytes(BUNDLE_PATH, snapshot.bundle)
    result = await backend.run(['bash', '-c', seed_script(snapshot)], timeout=_SEED_TIMEOUT)
    if result.exit_code != 0:
        raise ValueError(f'Could not copy the checkout into the sandbox: {(result.stderr or result.stdout).strip()}')
