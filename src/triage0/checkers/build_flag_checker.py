"""Check literal build-option defaults without configuring the project."""

from __future__ import annotations

from triage0.checkers.tools.build_tools import READ_TIMEOUT_S, parse_default, read_argv
from triage0.core.evidence import EvidenceResult, EvidenceStatus
from triage0.core.interfaces import SandboxSession
from triage0.core.models import AnyClaim, BuildFlagClaim, BuildSystem, ClaimKind
from triage0.sandbox.providers.docker_local import SandboxError


class BuildFlagChecker:
    claim_kind = ClaimKind.BUILD_FLAG

    def supports(self, claim: AnyClaim) -> bool:
        return isinstance(claim, BuildFlagClaim)

    def check(self, claim: AnyClaim, sandbox: SandboxSession) -> EvidenceResult:
        if not isinstance(claim, BuildFlagClaim):
            raise TypeError(f"BuildFlagChecker cannot check a {claim.kind} claim")
        if claim.build_system not in (BuildSystem.CMAKE, BuildSystem.AUTOCONF):
            return EvidenceResult(status=EvidenceStatus.CANT_CHECK, summary="unsupported build system")
        if claim.config is not None or claim.target is not None:
            return EvidenceResult(status=EvidenceStatus.CANT_CHECK, summary="target-specific and configuration-specific defaults are not analyzed")

        try:
            proof = sandbox.seal(read_argv(claim.build_system), timeout_s=READ_TIMEOUT_S)
        except SandboxError:
            return EvidenceResult(status=EvidenceStatus.CANT_CHECK, summary="sandbox read failed")
        if proof.exit_code != 0 or proof.timed_out or proof.truncated:
            return EvidenceResult(status=EvidenceStatus.CANT_CHECK, proof=proof, summary="build-file read failed or was incomplete")
        default = parse_default(claim.build_system, proof.stdout, claim.flag)
        if default is None:
            return EvidenceResult(
                status=EvidenceStatus.CANT_CHECK,
                proof=proof,
                summary="no unique, unconditional literal default could be established; it may be missing, conditional, or unsupported",
            )
        filename = "CMakeLists.txt" if claim.build_system is BuildSystem.CMAKE else "configure.ac"
        status = EvidenceStatus.CONFIRMED if default.enabled == claim.expected_enabled else EvidenceStatus.CONTRADICTED
        return EvidenceResult(
            status=status,
            proof=proof,
            summary=f"{filename}:{default.line} declares the option default {'enabled' if default.enabled else 'disabled'}",
        )
