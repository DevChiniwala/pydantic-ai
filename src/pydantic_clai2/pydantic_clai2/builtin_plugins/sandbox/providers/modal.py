"""`/sandbox modal`: Modal's outbound policy sets GitHub's `Authorization` header on the sandbox's requests.

The token is in the sandbox's policy as written, so it ends with the sandbox. A Modal secret would outlive
it, and anyone in the Modal workspace could use it.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.util import find_spec

from pydantic_ai.workspaces import WorkspaceRef
from pydantic_clai2.builtin_plugins.sandbox.providers.base import GITHUB_HOSTS, SANDBOX_ENV, GitHubCredential
from pydantic_clai2.builtin_plugins.sandbox.seed import SandboxBackend

APP_NAME = 'clai-sandboxes'
"""The Modal app every `/sandbox modal` sandbox belongs to; Modal creates it on first use."""


@dataclass(frozen=True)
class ModalProvider:
    """[Modal](https://modal.com) sandboxes, signed in with `modal token new` or `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`."""

    name: str = 'modal'
    title: str = 'Modal'
    working_dir: str = '/home/user/workspace'

    def missing(self) -> str | None:
        if find_spec('modal') is None:
            return "Modal support is not installed. Install CLAI with it: uv tool install 'pydantic-clai2[modal]'."
        from modal.config import config

        if not (config.get('token_id') and config.get('token_secret')):
            return 'Sign in to Modal: run uvx modal token new, or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.'
        return None

    async def create(self, *, github: GitHubCredential | None) -> tuple[WorkspaceRef, SandboxBackend]:
        import modal
        from modal.experimental import OutboundPolicy

        from pydantic_ai_harness.modal_sandbox import ModalSandboxBackend

        env: Mapping[str, str] = SANDBOX_ENV
        policy: OutboundPolicy | None = None
        if github is not None:
            policy = OutboundPolicy()
            for host in GITHUB_HOSTS:
                policy = policy.with_header_replacement(
                    domain=host, headers={'Authorization': github.authorization(host)}
                )
            env = {**env, 'GH_TOKEN': 'placeholder'}  # `gh` sends a header only with a token set; the proxy replaces it
        backend = ModalSandboxBackend(
            # Modal's default image has neither git nor gh.
            image=modal.Image.debian_slim(python_version='3.12').apt_install('git', 'gh', 'ripgrep'),
            app_name=APP_NAME,
            working_dir=self.working_dir,
            env=env,  # Modal keeps it on the sandbox, so commands after a reopen get it too
            outbound_policy=policy,
        )
        await backend.get_sandbox()
        assert backend.ref is not None
        return backend.ref, backend

    def reopen(self, ref: WorkspaceRef) -> SandboxBackend:
        from pydantic_ai_harness.modal_sandbox import ModalSandboxBackend

        return ModalSandboxBackend(ref=ref, working_dir=self.working_dir)

    async def delete(self, ref: WorkspaceRef) -> None:
        from pydantic_ai_harness.modal_sandbox import ModalSandbox

        await ModalSandbox[None](warn_if_no_tools=False).destroy(ref)
