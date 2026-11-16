"""`/sandbox boxd`: boxd's proxy swaps a stand-in in the machine's `GH_TOKEN` for the user's GitHub token.

boxd's proxy reads only secrets, so each machine gets one of its own, made just before it and deleted with
it: the user's (`private-unattached`, which no other org member can see or use), bound to `GITHUB_HOSTS`.
The machine has a variable of the secret's name holding the stand-in, and `create` copies it into `GH_TOKEN`
in the machine's spec, so `gh` and git's credential helper send it after a reopen too.
"""

import uuid
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING

import anyio

from pydantic_ai.workspaces import WorkspaceRef
from pydantic_clai2.builtin_plugins.sandbox.providers.base import (
    GITHUB_HOSTS,
    SANDBOX_ENV,
    GitHubCredential,
    saved_key,
)
from pydantic_clai2.builtin_plugins.sandbox.seed import SandboxBackend

if TYPE_CHECKING:
    import boxd

    from pydantic_ai_harness.boxd_sandbox import BoxdSandboxBackend

KEY_NAME = 'BOXD_API_KEY'

SECRET_PREFIX = 'CLAI_GITHUB_'
"""How `delete` tells this machine's GitHub secret from others its spec names."""

_SCOPE = 'private-unattached'


@dataclass(frozen=True)
class BoxdProvider:
    """[boxd](https://boxd.sh) machines, with the key from `/keys` or the `BOXD_API_KEY` environment variable."""

    name: str = 'boxd'
    title: str = 'boxd'
    working_dir: str = '/home/boxd/workspace'

    def missing(self) -> str | None:
        if find_spec('boxd') is None:
            return "boxd support is not installed. Install CLAI with it: uv tool install 'pydantic-clai2[boxd]'."
        if saved_key(KEY_NAME) is None:
            return f'Add your boxd API key in /keys as {KEY_NAME}. Make one with boxd api-keys create.'
        return None

    def _client(self) -> 'boxd.AsyncBoxd':
        import boxd

        return boxd.AsyncBoxd(api_key=saved_key(KEY_NAME))

    async def create(self, *, github: GitHubCredential | None) -> tuple[WorkspaceRef, SandboxBackend]:
        client = self._client()
        if github is None:
            backend = self._backend(client)
            machine = await backend.get_sandbox()
            return WorkspaceRef(provider=self.name, id=machine.id), backend
        # boxd refuses a spec naming a secret that does not exist yet, so the secret comes first.
        secret = f'{SECRET_PREFIX}{uuid.uuid4().hex.upper()}'
        await client.secrets.set(secret, github.token, scope=_SCOPE, domains=list(GITHUB_HOSTS))
        backend = self._backend(client, secrets=[secret])
        try:
            machine = await backend.get_sandbox()
            stand_in = (await backend.run(['printenv', secret])).stdout.strip()
            await client.machines.update_spec(machine.id, {'env': {'GH_TOKEN': stand_in}})
        except BaseException:
            with anyio.CancelScope(shield=True):
                if backend.ref is not None:
                    await self.delete(backend.ref)
                else:
                    await client.secrets.delete(secret, scope=_SCOPE)
            raise
        return WorkspaceRef(provider=self.name, id=machine.id), backend

    def _backend(self, client: 'boxd.AsyncBoxd', *, secrets: list[str] | None = None) -> 'BoxdSandboxBackend':
        from pydantic_ai_harness.boxd_sandbox import BoxdSandboxBackend

        # The backend sets `env` on each command it runs, not on the machine, so each reopen passes it again.
        return BoxdSandboxBackend(client=client, working_dir=self.working_dir, env=SANDBOX_ENV, secrets=secrets)

    def reopen(self, ref: WorkspaceRef) -> SandboxBackend:
        from pydantic_ai_harness.boxd_sandbox import BoxdSandboxBackend

        return BoxdSandboxBackend(client=self._client(), ref=ref, working_dir=self.working_dir, env=SANDBOX_ENV)

    async def delete(self, ref: WorkspaceRef) -> None:
        import boxd

        from pydantic_ai_harness.boxd_sandbox import BoxdSandbox

        client = self._client()
        try:
            try:
                secrets = ((await client.machines.get(ref.id)).spec or {}).get('secrets', [])
            except boxd.NotFoundError:
                secrets = []
            await BoxdSandbox[None](client=client).destroy(ref)
            for secret in secrets:
                if secret.startswith(SECRET_PREFIX):
                    await client.secrets.delete(secret, scope=_SCOPE)
        finally:
            await client.close()
