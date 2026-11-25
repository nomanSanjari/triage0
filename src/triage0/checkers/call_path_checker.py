"""Confirm witnessed direct C call paths; never infer unreachability."""

from __future__ import annotations

from triage0.checkers.tools.ast_tools import BadPathError, resolve_target, sandbox_target
from triage0.checkers.tools.callgraph import find_path, is_identifier, seal_graph
from triage0.core.evidence import EvidenceResult, EvidenceStatus
from triage0.core.interfaces import SandboxSession
from triage0.core.models import AnyClaim, CallPathClaim, ClaimKind
from triage0.sandbox.providers.docker_local import SANDBOX_SRC, SandboxError


class CallPathChecker:
    """``BaseClaimChecker`` for direct paths in a named C translation unit."""

    claim_kind = ClaimKind.CALL_PATH

    def supports(self, claim: AnyClaim) -> bool:
        return isinstance(claim, CallPathClaim)

    def check(self, claim: AnyClaim, sandbox: SandboxSession) -> EvidenceResult:
        if not isinstance(claim, CallPathClaim):
            raise TypeError(f"CallPathChecker cannot check a {claim.kind} claim")
        if not claim.expected_reachable:
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                summary="this checker only confirms positive static call paths",
            )
        # The endpoints are only compared against graph names, never executed,
        # but a name that cannot appear in the graph would waste a Clang run.
        if not (is_identifier(claim.entrypoint) and is_identifier(claim.target)):
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                summary="entrypoint and target must be C identifiers",
            )
        if not claim.file_hint:
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                summary="call-path analysis requires a C source file hint",
            )
        try:
            path = sandbox_target(claim.file_hint)
            if resolve_target(sandbox, path) != path:
                raise BadPathError("call-path analysis does not follow source symlinks")
        except BadPathError as exc:
            return EvidenceResult(status=EvidenceStatus.CANT_CHECK, summary=str(exc))
        except SandboxError:
            return EvidenceResult(status=EvidenceStatus.CANT_CHECK, summary="sandbox path check failed")
        if not path.endswith(".c"):
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                summary="call-path analysis requires a .c translation unit",
            )

        try:
            proof = seal_graph(sandbox, path)
        except SandboxError:
            return EvidenceResult(status=EvidenceStatus.CANT_CHECK, summary="sandbox analysis failed")
        if proof.timed_out or proof.truncated:
            detail = "timed out" if proof.timed_out else "hit the output limit"
        elif proof.exit_code != 0:
            detail = f"Clang call-graph extraction failed (exit {proof.exit_code})"
        else:
            hops = find_path(proof.stdout, claim.entrypoint, claim.target)
            if hops:
                relative = path.removeprefix(SANDBOX_SRC + "/")
                citations = "; ".join(
                    f"{hop.caller} -> {hop.callee} ({relative}:{hop.line})" for hop in hops
                )
                return EvidenceResult(
                    status=EvidenceStatus.CONFIRMED,
                    proof=proof,
                    summary=f"Direct C call path under fallback Clang configuration: {citations}",
                )
            detail = "no direct call path established within the analysis limits"
        return EvidenceResult(status=EvidenceStatus.CANT_CHECK, proof=proof, summary=detail)
