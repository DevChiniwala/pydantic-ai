"""`/sandbox PROVIDER`: run a new conversation in a cloud sandbox holding a copy of this checkout.

The sandbox gets the branch, its unpushed commits, and the uncommitted files (`seed`). Its `git` and `gh`
act as the user's `gh` login: the provider's proxy adds the token to requests for GitHub, so the token never
enters the sandbox. A conversation stays where its first turn ran: its history refers to files there.

Which conversations run in a sandbox is read from their history, whose responses name the sandbox
(`ModelResponse.workspace_ref`), so a resumed or forked conversation reopens it. One made by `/sandbox` before
its first prompt is known only here, and is deleted when the session leaves the conversation without using it.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import anyio
from anyio import to_thread

from pydantic_ai import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_clai2.builtin_plugins.sandbox.providers import PROVIDERS, GitHubCredential, SandboxProvider
from pydantic_clai2.builtin_plugins.sandbox.seed import SandboxBackend, Snapshot, seed, snapshot
from pydantic_clai2.commands import Command
from pydantic_clai2.gh_cli import gh_token, gh_user
from pydantic_clai2.plugins import ConversationChanged, NoSettings, Plugin, PluginHost, SessionEnd, TurnStart
from pydantic_clai2.runtime.session import SessionWorkspace
from pydantic_clai2.ui.rendering.status import StatusSegment

__all__ = ('SandboxPlugin',)


def history_sandbox(messages: Sequence[ModelMessage]) -> WorkspaceRef | None:
    """The sandbox the history's latest response worked in, when a `/sandbox` provider owns it."""
    response = next((message for message in reversed(messages) if isinstance(message, ModelResponse)), None)
    ref = response.workspace_ref if response is not None else None
    return ref if ref is not None and ref.provider in PROVIDERS else None


@dataclass(kw_only=True)
class _Sandbox:
    """A conversation's sandbox, opened once per session."""

    provider: SandboxProvider
    ref: WorkspaceRef
    backend: SandboxBackend | None = None
    unused: bool = False
    """Made by `/sandbox` and no prompt sent yet, so leaving the conversation deletes it."""
    deleted: bool = False

    def open(self) -> SandboxBackend:
        """The workspace, attached on first use; `UserError` when it cannot be, so the run fails closed."""
        if self.deleted:
            raise UserError(
                f"This conversation's {self.provider.title} sandbox was deleted. Run /new to start another."
            )
        if self.backend is None:
            if (missing := self.provider.missing()) is not None:
                raise UserError(f'This conversation runs in a {self.provider.title} sandbox. {missing}')
            self.backend = self.provider.reopen(self.ref)
        return self.backend


@dataclass
class _ConversationSandbox(SessionWorkspace[None]):
    """Supplies the run's workspace while the session's conversation runs in a sandbox."""

    plugin: 'SandboxPlugin' = field(repr=False)

    def supplies(self) -> bool:
        return self.plugin.current() is not None

    def get_instructions(self) -> Callable[[RunContext[None]], str]:
        return self._instructions

    def _instructions(self, ctx: RunContext[None]) -> str:
        sandbox = self.plugin.current()
        if sandbox is None:
            return ''
        return (
            f'Your shell and file tools run in a {sandbox.provider.title} sandbox. {sandbox.provider.working_dir} '
            "is a copy of the user's checkout, made when this conversation started; the user sees changes there "
            'only once they are pushed. git and gh are signed in to GitHub as the user through a proxy outside the '
            'sandbox, so GH_TOKEN holds a stand-in the proxy replaces. Never run `gh auth login`.'
        )

    def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        del ctx, ref
        sandbox = self.plugin.current()
        return None if sandbox is None else sandbox.open()


class SandboxPlugin(Plugin):
    """`/sandbox PROVIDER` moves a new conversation into a cloud sandbox; `/sandbox` says where it runs."""

    def __init__(self, host: PluginHost[None], settings: NoSettings) -> None:
        super().__init__(host, settings)
        self._sandboxes: dict[str, _Sandbox | None] = {}
        """Each conversation seen, by ID, with its sandbox or `None` for one that runs on this machine."""

    def get_capabilities(self) -> Sequence[_ConversationSandbox]:
        return (_ConversationSandbox(self),)

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='sandbox',
                description=f'Run a new conversation in a cloud sandbox ({", ".join(PROVIDERS)}), or show where it runs',
                handler=self.command,
                complete=self.complete,
            ),
        )

    def get_status_segments(self) -> Sequence[StatusSegment]:
        return (self._segment,)

    def _segment(self) -> str:
        sandbox = self.current()
        return '' if sandbox is None else f'{sandbox.provider.name} sandbox'

    def current(self) -> _Sandbox | None:
        """The sandbox the session's conversation runs in; `None` when it runs on this machine."""
        conversation = self.host.conversation
        key = conversation.conversation_id
        if key not in self._sandboxes:
            ref = history_sandbox(conversation.messages)
            self._sandboxes[key] = None if ref is None else _Sandbox(provider=PROVIDERS[ref.provider], ref=ref)
        return self._sandboxes[key]

    def complete(self, args: list[str]) -> Iterable[str]:
        return (*PROVIDERS, 'local', 'delete') if len(args) <= 1 else ()

    async def command(self, args: list[str]) -> str:
        match args:
            case []:
                return self._where()
            case ['local']:
                return await self._local()
            case ['delete']:
                return await self._delete()
            case [name] if name in PROVIDERS:
                return await self._start(PROVIDERS[name])
            case _:
                raise ValueError(f'Usage: /sandbox [{" | ".join(PROVIDERS)} | local | delete]')

    def _where(self) -> str:
        sandbox = self.current()
        if sandbox is None:
            offer = ' or '.join(f'/sandbox {name}' for name in PROVIDERS)
            return f'This conversation runs on this machine. {offer} moves a new conversation into a cloud sandbox.'
        state = ', deleted' if sandbox.deleted else ''
        return (
            f'This conversation runs in {sandbox.provider.title} sandbox {sandbox.ref.id}{state}, '
            f'in {sandbox.provider.working_dir}. '
            '/sandbox delete deletes it.'
        )

    async def _start(self, provider: SandboxProvider) -> str:
        current = self.current()
        if current is not None:
            if current.provider is provider:
                return f'This conversation already runs in {provider.title} sandbox {current.ref.id}.'
            raise ValueError(
                f'This conversation runs in {current.provider.title} sandbox {current.ref.id}. Run /new first.'
            )
        if self.host.conversation.messages:
            raise ValueError(
                'This conversation has already run on this machine, and its history refers to files here. '
                f'Run /new, then /sandbox {provider.name}, to start one in {provider.title}.'
            )
        if (missing := provider.missing()) is not None:
            raise ValueError(missing)
        conversation = self.host.conversation.conversation_id
        github = await to_thread.run_sync(_github)
        checkout = await to_thread.run_sync(snapshot, self.host.workspace)
        ref, backend = await provider.create(github=github)
        try:
            await seed(backend, checkout)
        except BaseException:
            with anyio.CancelScope(shield=True):
                await provider.delete(ref)
            raise
        self._sandboxes[conversation] = _Sandbox(provider=provider, ref=ref, backend=backend, unused=True)
        return _started(provider, ref, checkout, github=github)

    async def _local(self) -> str:
        sandbox = self.current()
        if sandbox is None:
            return 'This conversation already runs on this machine.'
        if not sandbox.unused:
            raise ValueError(
                f'This conversation has worked in {sandbox.provider.title} sandbox {sandbox.ref.id}, so it stays there. '
                'Run /new to work on this machine; /sandbox delete deletes the sandbox.'
            )
        await sandbox.provider.delete(sandbox.ref)
        self._sandboxes[self.host.conversation.conversation_id] = None
        return f'Deleted the {sandbox.provider.title} sandbox. This conversation runs on this machine.'

    async def _delete(self) -> str:
        sandbox = self.current()
        if sandbox is None:
            return 'This conversation has no sandbox.'
        if sandbox.unused:
            return await self._local()
        await sandbox.provider.delete(sandbox.ref)
        sandbox.deleted = True
        return f'Deleted {sandbox.provider.title} sandbox {sandbox.ref.id}. Run /new to start another conversation.'

    async def on_turn_start(self, event: TurnStart) -> None:
        if (sandbox := self.current()) is not None:
            sandbox.unused = False

    async def on_conversation_changed(self, event: ConversationChanged) -> None:
        await self._delete_unused(keep=event.conversation_id)

    async def on_session_end(self, event: SessionEnd) -> None:
        await self._delete_unused(keep=None)

    async def _delete_unused(self, *, keep: str | None) -> None:
        """Delete the sandboxes `/sandbox` made for conversations other than `keep` that never sent a prompt."""
        for key, sandbox in list(self._sandboxes.items()):
            if key != keep and sandbox is not None and sandbox.unused:
                del self._sandboxes[key]
                with anyio.CancelScope(shield=True):
                    await sandbox.provider.delete(sandbox.ref)


def _github() -> GitHubCredential | None:
    """`gh`'s working login on github.com; `None` without `gh`, without a login, or when GitHub refuses it."""
    try:
        token = gh_token('github.com')
        login = gh_user('github.com') if token is not None else None
    except UserError:
        return None
    return None if token is None or login is None else GitHubCredential(login=login, token=token)


def _started(
    provider: SandboxProvider, ref: WorkspaceRef, checkout: Snapshot, *, github: GitHubCredential | None
) -> str:
    """What `/sandbox PROVIDER` says once the sandbox holds the checkout."""
    if checkout.head is None and checkout.work is None:
        holds = 'an empty folder: this folder is not in a Git repository with files'
    else:
        at = f'{checkout.branch} at {checkout.head[:7]}' if checkout.branch and checkout.head else 'this checkout'
        holds = f'{at}{", with your uncommitted changes" if checkout.changed else ""}'
    lines = [f'This conversation now runs in {provider.title} sandbox {ref.id}. {provider.working_dir} holds {holds}.']
    if github is None:
        lines.append(
            'gh has no GitHub login here, so the sandbox cannot push or reach private repositories. Run gh auth login, then /new.'
        )
    elif checkout.clone_url is None and checkout.head is not None:
        lines.append('origin is not on GitHub, so the sandbox has no credentials to push it.')
    else:
        lines.append(
            f'git and gh there act as @{github.login}, on every repository that login reaches. '
            f"{provider.title}'s proxy, outside the sandbox, holds the token until the sandbox is deleted."
        )
    lines.append('Your files here stay as they are: ask the agent to push, then pull here. ! commands still run here.')
    return '\n'.join(lines)
