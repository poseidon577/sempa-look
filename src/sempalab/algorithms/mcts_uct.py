"""
Generic MCTS baseline for the surrogate problem P2 (Table II of the
TCOM paper).

The paper describes it in one sentence (Sec. V-A): "a generic MCTS
baseline that sequentially constructs G using UCT and Monte-Carlo
rollouts with reward (1/N) sum_i psi(C_i, W)", compared with SemPA-Look
"under the same number of encoding step budget". Everything else is
our choice (docs/assumptions.md, A16-A20):

- State: the subpackets committed so far. Action: the next subpacket,
  an M-subset of the unassigned tokens. The last subpacket is forced.
- Decisions: N - 1 depth-wise commits, as in SemPA-Look. Before each
  commit MCTS runs from the current state with an equal share of the
  remaining encoder budget; the tree below the committed child is kept.
- Selection: UCT, mean + c * sqrt(ln n_parent / n_child), c = 1/sqrt(2)
  (Kocsis and Szepesvari's constant for rewards in [0, 1]).
- Expansion: the action space has C(|remaining|, M) elements (10,626 at
  the root for K = 24, M = 4), so a node keeps a list of sampled
  actions and widens progressively: it may hold ceil(n_visits ** 0.5)
  children, each a uniformly random new M-subset of its tokens. A node
  whose children are leaves (2M tokens left) is widened fully, since
  revisiting a leaf only replays a known reward.
- Rollout: complete the partition with uniformly random disjoint
  subpackets; reward (1/N) sum over all N subpackets of
  psi(C, W) = phi(W \\ C, W) (RSS, eq. 18).
- Budget: encoder evaluations of distinct psi values. psi values are
  memoized, so a subpacket seen twice costs once, and a rollout whose
  new psi values would overrun the budget is not started. Commit rule:
  most-visited child (ties: higher mean reward).

Logical calls in the result are the psi evaluations actually made, so
they are directly comparable with SemPA-Look's.
"""

from __future__ import annotations

import math
import time
from itertools import combinations
from math import comb

import numpy as np

from sempalab.metrics import batch_semantic_similarity
from sempalab.types import SearchResult


class _Node:
    __slots__ = ("remaining", "children", "visits", "value", "tried")

    def __init__(self, remaining: tuple[int, ...]):
        self.remaining = remaining
        self.children: dict[tuple[int, ...], _Node] = {}
        self.visits = 0
        self.value = 0.0
        self.tried: set[tuple[int, ...]] = set()

    def mean(self) -> float:
        return self.value / self.visits if self.visits else 0.0


def _sample_action(remaining, M, rng, exclude, total_actions, max_tries=64):
    """
    A uniformly random M-subset of `remaining` not in `exclude`, or None
    when every action has been tried.
    """

    if len(exclude) >= total_actions:
        return None

    if total_actions <= 4 * max_tries:
        untried = [c for c in combinations(remaining, M) if c not in exclude]
        return untried[int(rng.integers(len(untried)))] if untried else None

    for _ in range(max_tries):
        action = tuple(sorted(int(i) for i in rng.choice(remaining, size=M, replace=False)))
        if action not in exclude:
            return action

    return None


def _random_completion(remaining, M, rng):
    shuffled = [int(i) for i in rng.permutation(remaining)]
    return [tuple(sorted(shuffled[s:s + M])) for s in range(0, len(shuffled), M)]


def mcts_uct(
    message,
    encoder,
    M,
    budget=190,
    c=1 / math.sqrt(2),
    widening=0.5,
    seed=None,
):
    """
    Parameters
    ----------
    budget:
        Maximum number of distinct psi evaluations (encoder calls). The
        default 190 is SemPA-Look's measured count at K=24, M=4, P=10,
        k=4 (disjoint lookaheads); Table III's closed form gives 250.
    c:
        UCT exploration constant.
    widening:
        Progressive-widening exponent: a node holds at most
        ceil(visits ** widening) children (at least one).
    """

    K = len(message.tokens)

    if K == 0:
        raise ValueError("Message must contain at least one token.")
    if M <= 0 or K % M != 0:
        raise ValueError("M must be positive and divide K exactly.")
    if budget < 0:
        raise ValueError("budget must be non-negative.")
    if c < 0 or not 0 < widening <= 1:
        raise ValueError("c must be >= 0 and widening in (0, 1].")

    N = K // M
    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()

    initial = (encoder.logical_calls, encoder.miss_calls, encoder.cache_hits, encoder.encoder_batches)

    all_indices = tuple(range(K))
    psi_memo: dict[tuple[int, ...], float] = {}
    spent = 0

    def psi_many(groups) -> list[float] | None:
        """psi for each group, or None if the new ones exceed the budget."""
        nonlocal spent

        new = list(dict.fromkeys(g for g in groups if g not in psi_memo))

        if spent + len(new) > budget:
            return None

        if new:
            survivors = [tuple(i for i in all_indices if i not in set(g)) for g in new]
            scores = batch_semantic_similarity(encoder, message, survivors)
            for g, s in zip(new, scores):
                psi_memo[g] = float(s)
            spent += len(new)

        return [psi_memo[g] for g in groups]

    def simulate(root: _Node, committed: list[tuple[int, ...]]) -> bool:
        """One MCTS iteration. Returns False if the budget stopped it."""

        node, path, groups = root, [root], list(committed)

        # Selection / expansion.
        while len(node.remaining) > M:
            total = comb(len(node.remaining), M)
            limit = max(1, math.ceil(node.visits ** widening))

            if len(node.remaining) == 2 * M:
                # Children are leaves: revisiting one replays a known
                # reward, so expand before reselecting.
                limit = total

            action = None
            if len(node.children) < limit:
                action = _sample_action(node.remaining, M, rng, node.tried, total)

            if action is not None:
                node.tried.add(action)
                rest = tuple(i for i in node.remaining if i not in set(action))
                child = _Node(rest)
                node.children[action] = child
                groups.append(action)
                path.append(child)
                node = child
                break

            log_n = math.log(max(node.visits, 1))
            action, node = max(
                node.children.items(),
                key=lambda item: item[1].mean() + c * math.sqrt(log_n / item[1].visits)
                if item[1].visits else math.inf,
            )
            groups.append(action)
            path.append(node)

        # Rollout.
        groups.extend(_random_completion(node.remaining, M, rng))
        assert len(groups) == N

        scores = psi_many(groups)
        if scores is None:
            # Undo an expansion that will never be visited.
            if len(path) > 1 and path[-1].visits == 0:
                parent = path[-2]
                for action, child in list(parent.children.items()):
                    if child is path[-1]:
                        del parent.children[action]
                        parent.tried.discard(action)
            return False

        reward = float(np.mean(scores))
        for visited in path:
            visited.visits += 1
            visited.value += reward

        return True

    committed: list[tuple[int, ...]] = []
    root = _Node(all_indices)
    best_score = float("nan")

    while len(root.remaining) > M:
        decisions_left = len(root.remaining) // M - 1
        share = (budget - spent) // decisions_left
        target = spent + share

        # Run until this decision's share is spent, or the budget runs out.
        # Rollouts whose psi values are all memoized are free; cap them
        # so a fully explored small subtree cannot loop forever.
        stalls = 0
        while spent < target and stalls < 50:
            before = spent
            if not simulate(root, committed):
                break
            stalls = stalls + 1 if spent == before else 0

        if root.children:
            action, child = max(root.children.items(), key=lambda item: (item[1].visits, item[1].mean()))
            best_score = child.mean()
        else:
            # No budget left: commit a random subpacket.
            action = _random_completion(root.remaining, M, rng)[0]
            child = _Node(tuple(i for i in root.remaining if i not in set(action)))

        committed.append(action)
        root = child

    committed.append(tuple(sorted(root.remaining)))

    return SearchResult(
        best_partition=tuple(committed),
        best_score=float(best_score),
        objective="rss-mcts",
        partitions_evaluated=1,
        logical_calls=encoder.logical_calls - initial[0],
        miss_calls=encoder.miss_calls - initial[1],
        cache_hits=encoder.cache_hits - initial[2],
        encoder_batches=encoder.encoder_batches - initial[3],
        runtime_seconds=time.perf_counter() - start_time,
    )
