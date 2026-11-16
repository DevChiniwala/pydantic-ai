"""`/sandbox` moves a new conversation into a cloud sandbox holding a copy of the checkout.

A fake provider makes each sandbox a local folder, so the real seed script rebuilds the checkout there
with real Git. Its global Git config sends the GitHub URL to a local bare repository, as the provider's
proxy would send it to GitHub.
"""

import base64
import importlib.machinery
import io
import itertools
import subprocess
import sys
import types
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pytest
from anyio import to_thread
from rich.console import Console

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend, WorkspaceRef
from pydantic_clai2 import Session
from pydantic_clai2.builtin_plugins import sandbox
from pydantic_clai2.builtin_plugins.sandbox import SandboxPlugin
from pydantic_clai2.builtin_plugins.sandbox.providers import (
    GITHUB_HOSTS,
    GitHubCredential,
    boxd as boxd_provider,
    e2b as e2b_provider,
    modal as modal_provider,
)
from pydantic_clai2.builtin_plugins.sandbox.providers.base import SANDBOX_ENV
from pydantic_clai2.builtin_plugins.sandbox.providers.e2b import E2BProvider
from pydantic_clai2.builtin_plugins.sandbox.seed import Snapshot, github_clone_url, seed, snapshot
from pydantic_clai2.config.api_keys import save_key
from pydantic_clai2.plugins import (
    ConversationChanged,
    LoadedPlugin,
    PluginHost,
    SessionEnd,
    Transcript,
    TurnStart,
    load_plugin,
)

if TYPE_CHECKING or find_spec('e2b') is not None:  # pragma: no branch - the dev environment has every extra
    from tests.harness.e2b_sandbox.fake_e2b import FakeE2B
if TYPE_CHECKING or find_spec('boxd') is not None:  # pragma: no branch - the dev environment has every extra
    from boxd import Machine

    from tests.harness.boxd_sandbox.fake_boxd import FakeBoxd, FakeClient
if TYPE_CHECKING or find_spec('modal') is not None:  # pragma: no branch - the dev environment has every extra
    from tests.harness.modal_sandbox.fake_modal import FakeImage, FakeModal

CLONE_URL = 'https://github.com/owner/repo.git'


def git(directory: Path, *args: str) -> str:
    result = subprocess.run(['git', '-C', str(directory), *args], capture_output=True, text=True, check=True)
    return result.stdout.rstrip()


class FakeSandboxBackend(LocalWorkspaceBackend):
    """A local folder that reports the fake provider's reference, as a cloud sandbox's backend would."""

    def __init__(self, ref: WorkspaceRef, directory: Path, *, env: Mapping[str, str]) -> None:
        super().__init__(directory, env=env)
        self._sandbox_ref = ref

    @property
    def ref(self) -> WorkspaceRef:
        return self._sandbox_ref


@dataclass
class FakeProvider:
    """Sandboxes as folders under `root`, whose Git sends `https://github.com/owner/` to `upstream`'s folder."""

    root: Path
    upstream: Path
    name: str = 'fake'
    title: str = 'Fake'
    working_dir: str = '/home/user/workspace'
    requirement: str | None = None
    created: list[GitHubCredential | None] = field(default_factory=list[GitHubCredential | None])
    reopened: list[WorkspaceRef] = field(default_factory=list[WorkspaceRef])
    deleted: list[WorkspaceRef] = field(default_factory=list[WorkspaceRef])

    def missing(self) -> str | None:
        return self.requirement

    async def create(self, *, github: GitHubCredential | None) -> tuple[WorkspaceRef, FakeSandboxBackend]:
        self.created.append(github)
        env = {**SANDBOX_ENV, 'GH_TOKEN': 'stand-in'} if github is not None else SANDBOX_ENV
        ref = WorkspaceRef(provider=self.name, id=f'sbx-{len(self.created)}')
        folder = anyio.Path(self.folder(ref))
        await folder.mkdir()
        # A fresh global config per sandbox, as a new sandbox has.
        rewrite = f'[url "{self.upstream.parent}/"]\n\tinsteadOf = https://github.com/owner/\n'
        await anyio.Path(f'{folder}.gitconfig').write_text(rewrite, encoding='utf-8')
        return ref, self._backend(ref, env)

    def reopen(self, ref: WorkspaceRef) -> FakeSandboxBackend:
        self.reopened.append(ref)
        return self._backend(ref, {})

    async def delete(self, ref: WorkspaceRef) -> None:
        self.deleted.append(ref)

    def folder(self, ref: WorkspaceRef) -> Path:
        return self.root / ref.id

    def _backend(self, ref: WorkspaceRef, env: Mapping[str, str]) -> FakeSandboxBackend:
        git_env = {'GIT_CONFIG_GLOBAL': f'{self.folder(ref)}.gitconfig', 'GIT_CONFIG_NOSYSTEM': '1'}
        return FakeSandboxBackend(ref, self.folder(ref), env={**env, **git_env})


@pytest.fixture
def upstream(tmp_path: Path) -> Path:
    """A bare repository standing in for `CLONE_URL` on GitHub, with one commit on `main`."""
    seedling = tmp_path / 'seedling'
    seedling.mkdir()
    git(seedling, 'init', '--quiet', '--initial-branch=main')
    (seedling / 'README.md').write_text('hello\n', encoding='utf-8')
    git(seedling, 'add', '.')
    git(seedling, '-c', 'user.name=Up', '-c', 'user.email=up@example.com', 'commit', '--quiet', '-m', 'Initial')
    bare = tmp_path / 'github' / 'repo.git'
    git(tmp_path, 'clone', '--quiet', '--bare', str(seedling), str(bare))
    return bare


@pytest.fixture
def checkout(tmp_path: Path, upstream: Path) -> Path:
    """A clone of `upstream` whose `origin` names GitHub, with an unpushed commit and uncommitted files."""
    directory = tmp_path / 'checkout'
    git(tmp_path, 'clone', '--quiet', str(upstream), str(directory))
    git(directory, 'remote', 'set-url', 'origin', 'git@github.com:owner/repo.git')
    git(directory, 'config', 'user.name', 'Ada')
    git(directory, 'config', 'user.email', 'ada@example.com')
    git(directory, 'checkout', '--quiet', '-b', 'feature')
    (directory / 'committed.txt').write_text('committed\n', encoding='utf-8')
    git(directory, 'add', 'committed.txt')
    git(directory, 'commit', '--quiet', '-m', 'Unpushed')
    (directory / 'README.md').write_text('hello, edited\n', encoding='utf-8')
    (directory / 'untracked.txt').write_text('new\n', encoding='utf-8')
    return directory


@pytest.fixture
def provider(tmp_path: Path, upstream: Path, monkeypatch: pytest.MonkeyPatch) -> FakeProvider:
    sandboxes = tmp_path / 'sandboxes'
    sandboxes.mkdir()
    fake = FakeProvider(sandboxes, upstream)
    monkeypatch.setattr(sandbox, 'PROVIDERS', {fake.name: fake})

    def gh_token(hostname: str) -> str:
        return 'gho_secret'

    def gh_user(hostname: str) -> str:
        return 'Ada'

    monkeypatch.setattr(sandbox, 'gh_token', gh_token)
    monkeypatch.setattr(sandbox, 'gh_user', gh_user)
    return fake


def load(directory: Path, conversation: Transcript | None = None) -> LoadedPlugin[None]:
    host = PluginHost[None](
        name='sandbox',
        console=Console(file=io.StringIO()),
        settings={},
        conversation=conversation or Transcript(),
        workspace=lambda: directory,
    )
    return load_plugin(SandboxPlugin, host)


def plugin_of(loaded: LoadedPlugin[None]) -> SandboxPlugin:
    assert isinstance(loaded.plugin, SandboxPlugin)
    return loaded.plugin


def worked_in(ref: WorkspaceRef) -> list[ModelRequest | ModelResponse]:
    """A history whose one turn ran in the workspace `ref` names."""
    return [
        ModelRequest(parts=[UserPromptPart('go')]),
        ModelResponse(parts=[TextPart('done')], workspace_ref=ref),
    ]


def status(directory: Path) -> str:
    return git(directory, 'status', '--porcelain', '--branch')


class TestSeed:
    @pytest.mark.parametrize(
        ('remote', 'expected'),
        [
            ('https://github.com/owner/repo.git', CLONE_URL),
            ('https://github.com/owner/repo', CLONE_URL),
            ('git@github.com:owner/repo.git', CLONE_URL),
            ('ssh://git@github.com/owner/repo.git/', CLONE_URL),
            ('https://gitlab.com/owner/repo.git', None),
            ('https://github.com/owner', None),
        ],
    )
    def test_github_clone_url(self, remote: str, expected: str | None) -> None:
        assert github_clone_url(remote) == expected

    async def test_outside_git(self, tmp_path: Path) -> None:
        assert await to_thread.run_sync(snapshot, tmp_path) == Snapshot()

    async def test_github_checkout(self, checkout: Path, provider: FakeProvider) -> None:
        before = status(checkout)
        taken = await to_thread.run_sync(snapshot, checkout)
        assert taken.clone_url == CLONE_URL
        assert (taken.branch, taken.author_name, taken.author_email) == ('feature', 'Ada', 'ada@example.com')
        assert taken.changed
        ref, backend = await provider.create(github=None)
        await seed(backend, taken)

        copy = provider.folder(ref)
        assert git(copy, 'rev-parse', 'HEAD') == git(checkout, 'rev-parse', 'HEAD')
        assert git(copy, 'status', '--porcelain') == ' M README.md\n?? untracked.txt'
        assert (copy / 'README.md').read_text(encoding='utf-8') == 'hello, edited\n'
        assert git(copy, 'remote', 'get-url', 'origin') == CLONE_URL
        assert git(copy, 'config', '--file', f'{copy}.gitconfig', 'user.name') == 'Ada'
        assert git(copy, 'for-each-ref', 'refs/clai-seed') == ''
        # Reading the checkout changed nothing there.
        assert status(checkout) == before
        assert git(checkout, 'for-each-ref', 'refs/clai-seed') == ''

    async def test_pushed_and_clean(self, checkout: Path, provider: FakeProvider) -> None:
        git(checkout, 'checkout', '--quiet', '.')
        (checkout / 'untracked.txt').unlink()
        git(checkout, 'checkout', '--quiet', 'main')
        taken = await to_thread.run_sync(snapshot, checkout)
        assert (taken.bundle, taken.changed) == (None, False)
        ref, backend = await provider.create(github=None)
        await seed(backend, taken)
        assert status(provider.folder(ref)) == '## main...origin/main'

    async def test_not_on_github(self, checkout: Path, provider: FakeProvider) -> None:
        git(checkout, 'remote', 'remove', 'origin')
        git(checkout, 'checkout', '--quiet', '--detach')
        taken = await to_thread.run_sync(snapshot, checkout)
        assert (taken.clone_url, taken.branch) == (None, None)
        assert taken.bundle is not None
        ref, backend = await provider.create(github=None)
        await seed(backend, taken)
        copy = provider.folder(ref)
        assert git(copy, 'rev-parse', 'HEAD') == git(checkout, 'rev-parse', 'HEAD')
        assert git(copy, 'status', '--porcelain') == ' M README.md\n?? untracked.txt'
        assert git(copy, 'remote') == ''

    async def test_no_commits_yet(self, tmp_path: Path, provider: FakeProvider) -> None:
        directory = tmp_path / 'fresh'
        directory.mkdir()
        git(directory, 'init', '--quiet', '--initial-branch=start')
        assert await to_thread.run_sync(snapshot, directory) == Snapshot(branch='start')
        (directory / 'notes.txt').write_text('draft\n', encoding='utf-8')
        taken = await to_thread.run_sync(snapshot, directory)
        assert (taken.head, taken.branch) == (None, 'start')
        ref, backend = await provider.create(github=None)
        await seed(backend, taken)
        copy = provider.folder(ref)
        assert status(copy) == '## No commits yet on start\n?? notes.txt'
        assert (copy / 'notes.txt').read_text(encoding='utf-8') == 'draft\n'

    async def test_failure_names_git_output(self, checkout: Path, provider: FakeProvider) -> None:
        git(checkout, 'remote', 'set-url', 'origin', 'https://github.com/owner/missing.git')
        ref, backend = await provider.create(github=None)
        del ref
        with pytest.raises(ValueError, match=r'Could not copy the checkout into the sandbox: .*missing\.git'):
            await seed(backend, await to_thread.run_sync(snapshot, checkout))


def test_github_credential() -> None:
    github = GitHubCredential(login='Ada', token='gho_secret')
    assert base64.b64decode(github.basic.removeprefix('Basic ')) == b'x-access-token:gho_secret'
    assert github.bearer == 'token gho_secret'
    assert [github.authorization(host) for host in GITHUB_HOSTS] == [github.basic, github.bearer, github.bearer]
    assert 'gho_secret' not in repr(github)


class TestCommand:
    async def test_start_in_a_new_conversation(self, checkout: Path, provider: FakeProvider) -> None:
        loaded = load(checkout)
        plugin = plugin_of(loaded)
        assert await plugin.command([]) == (
            'This conversation runs on this machine. /sandbox fake moves a new conversation into a cloud sandbox.'
        )
        message = await plugin.command(['fake'])
        head = git(checkout, 'rev-parse', '--short=7', 'HEAD')
        assert message == (
            f'This conversation now runs in Fake sandbox sbx-1. /home/user/workspace holds feature at {head}, '
            'with your uncommitted changes.\n'
            'git and gh there act as @Ada, on every repository that login reaches. '
            "Fake's proxy, outside the sandbox, holds the token until the sandbox is deleted.\n"
            'Your files here stay as they are: ask the agent to push, then pull here. ! commands still run here.'
        )
        assert provider.created == [GitHubCredential(login='Ada', token='gho_secret')]
        assert [segment() for segment in loaded.status_segments] == ['fake sandbox']
        assert await plugin.command([]) == (
            'This conversation runs in Fake sandbox sbx-1, in /home/user/workspace. /sandbox delete deletes it.'
        )
        assert await plugin.command(['fake']) == 'This conversation already runs in Fake sandbox sbx-1.'

    async def test_refused_once_the_conversation_has_run(self, checkout: Path, provider: FakeProvider) -> None:
        history = worked_in(WorkspaceRef(provider='local', id=str(checkout)))
        plugin = plugin_of(load(checkout, Transcript(messages=history)))
        with pytest.raises(ValueError, match=r'already run on this machine.*Run /new, then /sandbox fake'):
            await plugin.command(['fake'])
        assert provider.created == []

    async def test_refused_in_another_providers_sandbox(
        self, checkout: Path, provider: FakeProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        other = FakeProvider(tmp_path, provider.upstream, name='other', title='Other')
        monkeypatch.setattr(sandbox, 'PROVIDERS', {provider.name: provider, other.name: other})
        plugin = plugin_of(load(checkout))
        await plugin.command(['fake'])
        with pytest.raises(ValueError, match=r'^This conversation runs in Fake sandbox sbx-1\. Run /new first\.$'):
            await plugin.command(['other'])

    async def test_missing_requirement(self, checkout: Path, provider: FakeProvider) -> None:
        provider.requirement = 'Add your Fake key in /keys.'
        with pytest.raises(ValueError, match=r'^Add your Fake key in /keys\.$'):
            await plugin_of(load(checkout)).command(['fake'])

    @pytest.mark.parametrize('args', [['nope'], ['fake', 'extra']])
    async def test_usage(self, checkout: Path, provider: FakeProvider, args: list[str]) -> None:
        with pytest.raises(ValueError, match=r'^Usage: /sandbox \[fake \| local \| delete\]$'):
            await plugin_of(load(checkout)).command(args)

    def test_complete(self, checkout: Path, provider: FakeProvider) -> None:
        plugin = plugin_of(load(checkout))
        assert list(plugin.complete([])) == list(plugin.complete(['f'])) == ['fake', 'local', 'delete']
        assert list(plugin.complete(['fake', ''])) == []

    async def test_without_a_github_login(
        self, checkout: Path, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_gh(hostname: str) -> str:
            raise UserError('gh is not installed')

        monkeypatch.setattr(sandbox, 'gh_token', no_gh)
        git(checkout, 'remote', 'remove', 'origin')
        message = await plugin_of(load(checkout)).command(['fake'])
        assert 'gh has no GitHub login here, so the sandbox cannot push or reach private repositories.' in message
        assert provider.created == [None]

    async def test_origin_not_on_github(self, checkout: Path, provider: FakeProvider) -> None:
        git(checkout, 'remote', 'set-url', 'origin', 'https://gitlab.com/owner/repo.git')
        message = await plugin_of(load(checkout)).command(['fake'])
        assert 'origin is not on GitHub, so the sandbox has no credentials to push it.' in message

    async def test_outside_git(self, tmp_path: Path, provider: FakeProvider) -> None:
        folder = tmp_path / 'plain'
        folder.mkdir()
        message = await plugin_of(load(folder)).command(['fake'])
        assert 'holds an empty folder: this folder is not in a Git repository with files.' in message

    async def test_detached_and_clean(self, checkout: Path, provider: FakeProvider) -> None:
        git(checkout, 'stash', '--quiet', '--include-untracked')
        git(checkout, 'checkout', '--quiet', '--detach')
        message = await plugin_of(load(checkout)).command(['fake'])
        assert '/home/user/workspace holds this checkout.\n' in message

    async def test_failed_copy_deletes_the_sandbox(self, checkout: Path, provider: FakeProvider) -> None:
        git(checkout, 'remote', 'set-url', 'origin', 'https://github.com/owner/missing.git')
        plugin = plugin_of(load(checkout))
        with pytest.raises(ValueError, match='Could not copy the checkout'):
            await plugin.command(['fake'])
        assert provider.deleted == [WorkspaceRef(provider='fake', id='sbx-1')]
        assert plugin.current() is None


class TestLifecycle:
    async def test_local_undoes_an_unused_sandbox(self, checkout: Path, provider: FakeProvider) -> None:
        plugin = plugin_of(load(checkout))
        assert await plugin.command(['local']) == 'This conversation already runs on this machine.'
        assert await plugin.command(['delete']) == 'This conversation has no sandbox.'
        await plugin.command(['fake'])
        assert await plugin.command(['local']) == 'Deleted the Fake sandbox. This conversation runs on this machine.'
        assert provider.deleted == [WorkspaceRef(provider='fake', id='sbx-1')]
        assert plugin.current() is None
        await plugin.command(['fake'])
        assert await plugin.command(['delete']) == 'Deleted the Fake sandbox. This conversation runs on this machine.'

    async def test_a_used_sandbox_stays(self, checkout: Path, provider: FakeProvider) -> None:
        loaded = load(checkout)
        plugin = plugin_of(loaded)
        await plugin.command(['fake'])
        await loaded.dispatch(TurnStart(text='go'))
        with pytest.raises(ValueError, match=r'has worked in Fake sandbox sbx-1, so it stays there\. Run /new'):
            await plugin.command(['local'])
        await loaded.dispatch(ConversationChanged(conversation_id='other', title=None))
        await loaded.dispatch(SessionEnd(reason='exit'))
        assert provider.deleted == []

    async def test_delete_fails_the_next_run_closed(self, checkout: Path, provider: FakeProvider) -> None:
        loaded = load(checkout)
        plugin = plugin_of(loaded)
        await plugin.command(['fake'])
        await loaded.dispatch(TurnStart(text='go'))
        assert await plugin.command(['delete']) == (
            'Deleted Fake sandbox sbx-1. Run /new to start another conversation.'
        )
        assert provider.deleted == [WorkspaceRef(provider='fake', id='sbx-1')]
        assert await plugin.command([]) == (
            'This conversation runs in Fake sandbox sbx-1, deleted, in /home/user/workspace. '
            '/sandbox delete deletes it.'
        )
        current = plugin.current()
        assert current is not None
        with pytest.raises(
            UserError, match=r"^This conversation's Fake sandbox was deleted\. Run /new to start another\.$"
        ):
            current.open()

    @pytest.mark.parametrize('leave', ['change', 'end'])
    async def test_leaving_deletes_an_unused_sandbox(self, checkout: Path, provider: FakeProvider, leave: str) -> None:
        conversation = Transcript(conversation_id='first')
        loaded = load(checkout, conversation)
        await plugin_of(loaded).command(['fake'])
        if leave == 'change':
            # The shell swaps the conversation before it tells plugins.
            loaded.host.conversation = Transcript(conversation_id='second')
            await loaded.dispatch(ConversationChanged(conversation_id='second', title=None))
        else:
            await loaded.dispatch(SessionEnd(reason='exit'))
        assert provider.deleted == [WorkspaceRef(provider='fake', id='sbx-1')]

    async def test_history_reopens_its_sandbox(self, checkout: Path, provider: FakeProvider) -> None:
        ref = WorkspaceRef(provider='fake', id='sbx-9')
        provider.folder(ref).mkdir()
        loaded = load(checkout, Transcript(messages=worked_in(ref)))
        plugin = plugin_of(loaded)
        assert [segment() for segment in loaded.status_segments] == ['fake sandbox']
        current = plugin.current()
        assert current is not None
        backend = current.open()
        assert current.open() is backend
        assert backend.ref == ref
        assert provider.reopened == [ref]
        await loaded.dispatch(SessionEnd(reason='exit'))
        assert provider.deleted == []

    async def test_a_turn_here_keeps_running_here(self, checkout: Path, provider: FakeProvider) -> None:
        loaded = load(checkout)
        await loaded.dispatch(TurnStart(text='go'))
        assert plugin_of(loaded).current() is None

    async def test_history_in_an_unknown_workspace_runs_here(self, checkout: Path, provider: FakeProvider) -> None:
        loaded = load(checkout, Transcript(messages=worked_in(WorkspaceRef(provider='modal', id='sb-1'))))
        assert plugin_of(loaded).current() is None
        assert [segment() for segment in loaded.status_segments] == ['']

    async def test_reopen_needs_the_requirement(self, checkout: Path, provider: FakeProvider) -> None:
        provider.requirement = 'Add your Fake key in /keys.'
        plugin = plugin_of(load(checkout, Transcript(messages=worked_in(WorkspaceRef(provider='fake', id='x')))))
        current = plugin.current()
        assert current is not None
        with pytest.raises(
            UserError, match=r'^This conversation runs in a Fake sandbox\. Add your Fake key in /keys\.$'
        ):
            current.open()


async def test_runs_use_the_sandbox(checkout: Path, provider: FakeProvider) -> None:
    working_dirs: list[str] = []

    class Probe(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            working_dirs.append(await ctx.workspace.working_dir())

    loaded = load(checkout)
    plugin = plugin_of(loaded)
    [capability] = loaded.capabilities
    agent = Agent(TestModel(call_tools=[], custom_output_text='done'), deps_type=type(None))
    session = Session(agent, deps=None, plugins=[capability, Probe()], workspace=checkout)
    loaded.host.conversation = session
    await session.prompt('here')
    # A conversation with history refuses `/sandbox`, so start the sandboxed one fresh.
    session.replace_messages([])
    await plugin.command(['fake'])
    result = await session.prompt('there')
    assert working_dirs == [str(checkout.resolve()), str(provider.folder(WorkspaceRef(provider='fake', id='sbx-1')))]
    assert result.response.workspace_ref == WorkspaceRef(provider='fake', id='sbx-1')
    [request] = [message for message in result.new_messages() if isinstance(message, ModelRequest)]
    assert 'Your shell and file tools run in a Fake sandbox.' in (request.instructions or '')


@pytest.fixture
def fake_e2b(monkeypatch: pytest.MonkeyPatch) -> Iterator['FakeE2B']:
    """The harness's fake E2B SDK as `e2b`; each client it makes is named by its API key."""
    control = FakeE2B()
    module = control.module
    module.__spec__ = importlib.machinery.ModuleSpec('e2b', None)
    module.E2B = lambda api_key: control.client(api_key)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'e2b', module)
    monkeypatch.setattr('pydantic_ai_harness.e2b_sandbox._backend.e2b', module)
    monkeypatch.delenv(e2b_provider.KEY_NAME, raising=False)
    try:
        yield control
    finally:
        control.close()


@pytest.mark.skipif(find_spec('e2b') is None, reason='e2b not installed')
class TestE2B:
    """E2B, through the SDK fake the harness's own E2B tests use: a live run needs a billed account."""

    def test_missing(self, fake_e2b: 'FakeE2B', monkeypatch: pytest.MonkeyPatch) -> None:
        e2b = E2BProvider()
        assert e2b.missing() == (
            'Add your E2B API key in /keys as E2B_API_KEY. Get one at https://e2b.dev/dashboard?tab=keys.'
        )
        monkeypatch.setenv('E2B_API_KEY', 'from-env')
        assert e2b.missing() is None

        def not_installed(name: str) -> None:
            return None

        monkeypatch.setattr(e2b_provider, 'find_spec', not_installed)
        assert e2b.missing() == (
            "E2B support is not installed. Install CLAI with it: uv tool install 'pydantic-clai2[e2b]'."
        )

    async def test_create_reopen_delete(self, fake_e2b: 'FakeE2B', monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('E2B_API_KEY', 'from-env')
        save_key(name='E2B_API_KEY', value='from-keys')
        e2b = E2BProvider()
        github = GitHubCredential(login='Ada', token='gho_x')
        ref, backend = await e2b.create(github=github)
        assert ref == backend.ref == WorkspaceRef(provider='e2b', id='sbx-1')
        [call] = fake_e2b.create_calls
        assert call.envs == {'GIT_TERMINAL_PROMPT': '0', 'GH_TOKEN': 'placeholder'}
        # The token is only in this sandbox's rules, so it ends with the sandbox.
        assert call.network == {
            'rules': {
                'github.com': [{'transform': {'headers': {'Authorization': github.basic}}}],
                'api.github.com': [{'transform': {'headers': {'Authorization': github.bearer}}}],
                'uploads.github.com': [{'transform': {'headers': {'Authorization': github.bearer}}}],
            }
        }
        await e2b.create(github=None)
        assert fake_e2b.create_calls[-1].network is None
        reopened = e2b.reopen(ref)
        assert reopened.ref == ref
        await e2b.delete(ref)
        assert fake_e2b.sandboxes[0].killed
        # `/keys` wins over the environment, for every call.
        assert set(fake_e2b.clients) == {'from-keys'}


@dataclass(frozen=True)
class FakePolicy:
    """Modal's `OutboundPolicy`, keeping each header replacement as `(domain, headers)`."""

    replacements: tuple[tuple[str, dict[str, str]], ...] = ()

    def with_header_replacement(self, *, domain: str, headers: dict[str, str]) -> 'FakePolicy':
        return FakePolicy((*self.replacements, (domain, headers)))


@pytest.fixture
def fake_modal(monkeypatch: pytest.MonkeyPatch) -> Iterator['FakeModal']:
    """The harness's fake Modal SDK as `modal`, with a `FakePolicy` as `modal.experimental.OutboundPolicy`."""
    control = FakeModal()
    # The harness accepts only a `modal.Image`, so the fake's images must be its `Image`.
    monkeypatch.setattr(FakeImage, 'debian_slim', staticmethod(FakeImage), raising=False)
    control.module.Image = FakeImage  # type: ignore[attr-defined]
    experimental = types.ModuleType('modal.experimental')
    experimental.OutboundPolicy = FakePolicy  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'modal', control.module)
    monkeypatch.setitem(sys.modules, 'modal.experimental', experimental)
    yield control


@pytest.mark.skipif(find_spec('modal') is None, reason='modal not installed')
class TestModal:
    """Modal, through the SDK fake the harness's own Modal tests use: a live run needs a billed account."""

    def test_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from modal.config import config

        settings: dict[str, str] = {}

        def get(key: str) -> str | None:
            return settings.get(key)

        monkeypatch.setattr(config, 'get', get)
        modal = modal_provider.ModalProvider()
        assert modal.missing() == (
            'Sign in to Modal: run uvx modal token new, or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.'
        )
        settings.update(token_id='ak-1', token_secret='as-1')
        assert modal.missing() is None

        def not_installed(name: str) -> None:
            return None

        monkeypatch.setattr(modal_provider, 'find_spec', not_installed)
        assert modal.missing() == (
            "Modal support is not installed. Install CLAI with it: uv tool install 'pydantic-clai2[modal]'."
        )

    async def test_create_reopen_delete(self, fake_modal: 'FakeModal') -> None:
        modal = modal_provider.ModalProvider()
        github = GitHubCredential(login='Ada', token='gho_x')
        ref, backend = await modal.create(github=github)
        assert ref == backend.ref == WorkspaceRef(provider='modal', id='sb-owned')
        [call] = fake_modal.create_kwargs
        assert call['env'] == {'GIT_TERMINAL_PROMPT': '0', 'GH_TOKEN': 'placeholder'}
        assert call['workdir'] == '/home/user/workspace'
        image = call['image']
        assert isinstance(image, FakeImage)
        assert {'git', 'gh'} <= set(image.apt_packages)
        policy = call['outbound_policy']
        assert isinstance(policy, FakePolicy)
        assert list(policy.replacements) == [
            ('github.com', {'Authorization': github.basic}),
            ('api.github.com', {'Authorization': github.bearer}),
            ('uploads.github.com', {'Authorization': github.bearer}),
        ]
        await modal.create(github=None)
        assert fake_modal.create_kwargs[-1]['outbound_policy'] is None
        assert modal.reopen(ref).ref == ref
        await modal.delete(ref)
        assert fake_modal.sandboxes[0].shutting_down


@dataclass
class FakeBoxdAccount:
    """boxd's secrets and machine specs, which the harness's boxd fake leaves out."""

    secrets: dict[str, tuple[str, str, list[str]]] = field(default_factory=dict[str, tuple[str, str, list[str]]])
    specs: dict[str, dict[str, object]] = field(default_factory=dict[str, dict[str, object]])

    async def set(self, name: str, value: str, *, scope: str, domains: list[str]) -> None:
        self.secrets[name] = (value, scope, domains)

    async def delete(self, name: str, *, scope: str) -> None:
        assert self.secrets.pop(name)[1] == scope

    async def update_spec(self, machine_id: str, patch: dict[str, object]) -> None:
        self.specs[machine_id] = patch


@pytest.fixture
async def fake_boxd(monkeypatch: pytest.MonkeyPatch) -> 'AsyncIterator[tuple[FakeBoxd, FakeBoxdAccount]]':
    """The harness's boxd fake, whose commands run here, with `FakeBoxdAccount` on every client."""
    import boxd

    fake, account = FakeBoxd(), FakeBoxdAccount()

    def client(api_key: str | None = None) -> 'FakeClient':
        made = fake.client()
        made.secrets = account  # type: ignore[attr-defined]
        made.machines.update_spec = account.update_spec  # type: ignore[attr-defined]
        get = made.machines.get

        async def get_with_spec(machine_id: str) -> 'Machine':
            # The fake's records leave out the spec a machine was created with.
            return (await get(machine_id)).model_copy(update={'spec': fake.find(machine_id).create_kwargs['spec']})

        made.machines.get = get_with_spec
        return made

    monkeypatch.setattr('pydantic_ai_harness.boxd_sandbox._backend.boxd', boxd)
    monkeypatch.setattr(boxd, 'AsyncBoxd', client)
    monkeypatch.setattr('pydantic_ai_harness.boxd_sandbox._backend._POLL_INTERVAL', 0)
    yield fake, account
    await fake.reap()


@pytest.mark.skipif(find_spec('boxd') is None, reason='boxd not installed')
class TestBoxd:
    """boxd, through the SDK fake the harness's own boxd tests use; the swap itself was proven live."""

    def test_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(boxd_provider.KEY_NAME, raising=False)
        provider = boxd_provider.BoxdProvider()
        assert (
            provider.missing() == 'Add your boxd API key in /keys as BOXD_API_KEY. Make one with boxd api-keys create.'
        )
        monkeypatch.setenv('BOXD_API_KEY', 'bk-1')
        assert provider.missing() is None

        def not_installed(name: str) -> None:
            return None

        monkeypatch.setattr(boxd_provider, 'find_spec', not_installed)
        assert provider.missing() == (
            "boxd support is not installed. Install CLAI with it: uv tool install 'pydantic-clai2[boxd]'."
        )

    async def test_create_reopen_delete(
        self, fake_boxd: 'tuple[FakeBoxd, FakeBoxdAccount]', monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake, account = fake_boxd
        fake.host_root = tmp_path.resolve()
        secret = 'CLAI_GITHUB_' + '0' * 32
        numbers = itertools.count()
        monkeypatch.setattr(uuid, 'uuid4', lambda: uuid.UUID(int=next(numbers)))  # the secret's name is the first
        # boxd sets the secret's variable to its stand-in; the fake's commands run here, so this stands in for that.
        monkeypatch.setenv(secret, 'bxds_stand-in')
        provider = boxd_provider.BoxdProvider(working_dir=str(tmp_path / 'workspace'))
        ref, backend = await provider.create(github=GitHubCredential(login='Ada', token='gho_x'))
        assert account.secrets == {secret: ('gho_x', 'private-unattached', list(GITHUB_HOSTS))}
        [machine] = fake.machines.values()
        assert machine.isolated
        assert machine.create_kwargs['spec']['secrets'] == [secret]
        # The spec keeps the stand-in as `GH_TOKEN` for every later command, after a reopen too.
        assert account.specs == {ref.id: {'env': {'GH_TOKEN': 'bxds_stand-in'}}}
        assert backend.ref == provider.reopen(ref).ref == ref
        assert (await provider.reopen(ref).run(['printenv', 'GIT_TERMINAL_PROMPT'])).stdout == '0\n'
        # Deleting the machine deletes its secret, so the token ends with it.
        await provider.delete(ref)
        assert machine.status == 'destroyed'
        assert account.secrets == {}
        # Without a GitHub login: no secret, and nothing for the spec to keep.
        ref, _ = await provider.create(github=None)
        assert account.secrets == {}
        assert ref.id not in account.specs
        await provider.delete(ref)
