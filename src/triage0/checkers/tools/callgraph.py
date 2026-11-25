"""Bounded direct-call analysis of a single C translation unit.

This is a source-level graph under Clang's fallback configuration, not runtime
reachability. Cross-file definitions, function pointers, callbacks, macro-origin
calls, and code that needs generated sources or build flags are not resolved.
Cited lines are the physical ones, so a ``#line`` directive cannot change them.

Both inputs and outputs are capped: an AST over 16 MiB or a graph over 100k
edges fails the run, because a partial graph could still contain a path and
look complete.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass

from triage0.core.evidence import DeterministicProof
from triage0.core.interfaces import SandboxSession

_TIMEOUT_S = 15

# Longest witnessed chain the checker confirms. The target check runs before
# the queue guard in find_path, so a path of exactly _MAX_DEPTH edges is
# reported and a longer one is not.
_MAX_DEPTH = 8

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")

# The image already provides Perl and JSON::PP. Projecting inside the sandbox
# keeps the sealed output legible and avoids the 1 MiB capture limit on raw ASTs.
# No source text, report symbols, or project flags are executed by this adapter.
_PROJECT_AST = r"""
use strict;
use warnings;
use JSON::PP;

my ($path) = @ARGV;
die "not a regular file\n" unless -f $path;
open my $clang, '-|', '/usr/bin/clang', '--no-default-config',
    '-x', 'c', '-fsyntax-only', '-fno-color-diagnostics', '-ferror-limit=1',
    '-I/src/include', '-I/src/lib', '-Xclang', '-ast-dump=json', '--', $path
    or die "could not start clang\n";
my $limit = 16 * 1024 * 1024;
my $raw = '';
while (1) {
    my $chunk;
    my $got = read $clang, $chunk, 65536;
    die "could not read clang output\n" unless defined $got;
    last if $got == 0;
    $raw .= $chunk;
    die "AST input limit exceeded\n" if length $raw > $limit;
}
close $clang or die "clang did not finish successfully\n";

# Clang writes "file"/"line" only when they differ from the previous bare
# location, so replaying them in stream order reconstructs its elision chain.
# They are the physical file and line; presumed (#line) numbering lives in
# separate keys and is ignored, because every citation must point at the file
# this run actually read. A macro-expanded call site has a range wrapper
# holding spellingLoc/expansionLoc instead of a bare location, so it carries
# no physicalLine and its edge is dropped rather than attributed to a guess.
my ($file, $line);
my $json = JSON::PP->new->utf8;
my $ast = $json->filter_json_object(sub {
    my ($obj) = @_;
    if (exists $obj->{offset} && exists $obj->{tokLen} && exists $obj->{col}) {
        $file = $obj->{file} if exists $obj->{file};
        $line = $obj->{line} if exists $obj->{line};
        $obj->{physicalLine} = $line if defined $file && $file eq $path
            && !exists $obj->{includedFrom};
    }
    return $obj;
})->decode($raw);
die "unexpected clang AST\n" unless $ast->{kind} eq 'TranslationUnitDecl';

my %prev;
my %body;
my %definition;
for my $decl (@{$ast->{inner}}) {
    next unless ($decl->{kind} // '') eq 'FunctionDecl';
    my $id = $decl->{id};
    $prev{$id} = $decl->{previousDecl} if exists $decl->{previousDecl};
    my $name = $decl->{name} // '';
    next unless $name =~ /\A[A-Za-z_][A-Za-z_0-9]*\z/
        && $name eq ($decl->{mangledName} // '') && $decl->{loc}{physicalLine};
    my ($compound) = grep { ($_->{kind} // '') eq 'CompoundStmt' } @{$decl->{inner} // []};
    next unless $compound && $compound->{range}{begin}{physicalLine}
        && $compound->{range}{end}{physicalLine};
    $body{$name} = $compound;
    while (defined $id && !exists $definition{$id}) {
        $definition{$id} = $name;
        $id = $prev{$id};
    }
}

my @edges;
# One bounded list, so a pathological translation unit cannot turn the sealed
# output into a graph that merely looks complete.
my $edge_limit = 100_000;
for my $caller (sort keys %body) {
    my @nodes = ($body{$caller});
    while (my $node = pop @nodes) {
        my $kind = $node->{kind} // '';
        # These subtrees can contain syntactically present but unevaluated calls.
        next if $kind =~ /\A(?:UnaryExprOrTypeTraitExpr|GenericSelectionExpr|ChooseExpr|TypeTraitExpr|OffsetOfExpr|ConstantExpr|BlockDecl|FunctionDecl)\z/;
        push @nodes, reverse @{$node->{inner} // []};
        next unless $kind eq 'CallExpr';
        my $site = $node->{range}{begin}{physicalLine};
        next unless $site && $node->{range}{end}{physicalLine};
        my $callee = $node->{inner}[0];
        while (($callee->{kind} // '') eq 'ParenExpr'
            || (($callee->{kind} // '') eq 'ImplicitCastExpr'
                && ($callee->{castKind} // '') eq 'FunctionToPointerDecay')) {
            $callee = $callee->{inner}[0];
        }
        next unless ($callee->{kind} // '') eq 'DeclRefExpr'
            && !$callee->{nonOdrUseReason} && $callee->{range}{begin}{physicalLine};
        my $ref = $callee->{referencedDecl};
        next unless ($ref->{kind} // '') eq 'FunctionDecl';
        my $name = $definition{$ref->{id}};
        next unless defined $name && $name eq $ref->{name};
        die "call graph edge limit exceeded\n" if @edges >= $edge_limit;
        push @edges, [$caller, $name, $site];
    }
}
print JSON::PP->new->ascii->encode(\@edges), "\n";
"""


@dataclass(frozen=True)
class CallHop:
    caller: str
    callee: str
    line: int


def seal_graph(sandbox: SandboxSession, path: str) -> DeterministicProof:
    """Seal the direct edges of a previously validated, absolute sandbox path."""
    return sandbox.seal(
        ["/usr/bin/perl", "-e", _PROJECT_AST, "--", path], timeout_s=_TIMEOUT_S
    )


def is_identifier(name: str) -> bool:
    """Whether ``name`` can appear as a function in the projected graph."""
    return _IDENTIFIER.fullmatch(name) is not None


def find_path(stdout: str, entrypoint: str, target: str) -> tuple[CallHop, ...]:
    """Find the shortest witnessed path, requiring at least one call edge."""
    try:
        edges = json.loads(stdout)
    except (ValueError, RecursionError):
        return ()
    if not isinstance(edges, list):
        return ()

    outgoing: dict[str, list[CallHop]] = defaultdict(list)
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 3:
            return ()
        caller, callee, line = edge
        if (
            not isinstance(caller, str) or not _IDENTIFIER.fullmatch(caller)
            or not isinstance(callee, str) or not _IDENTIFIER.fullmatch(callee)
            or type(line) is not int or line < 1
        ):
            return ()
        outgoing[caller].append(CallHop(caller, callee, line))

    queue: deque[tuple[str, tuple[CallHop, ...]]] = deque([(entrypoint, ())])
    seen = {entrypoint}
    while queue:
        caller, path = queue.popleft()
        for hop in outgoing.get(caller, []):
            if hop.callee == target:
                return (*path, hop)
            if len(path) + 1 < _MAX_DEPTH and hop.callee not in seen:
                queue.append((hop.callee, (*path, hop)))
                seen.add(hop.callee)
    return ()
