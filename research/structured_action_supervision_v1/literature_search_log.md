# Scoped Literature Search Log

Search date: `2026-07-29`  
Purpose: direct-substitute and nearest-neighbor audit, not a systematic review
or meta-analysis

## Search questions

1. Has agentic RL already applied token-level emphasis to action/tool tokens?
2. Has multi-turn agent RL already assigned credit at turns or transitions?
3. Has prior work separated function-name and argument components inside tool
   calls?
4. Are trajectory filters or selective memories close enough to invalidate a
   broad novelty claim?

## Search strings

- `"agentic reinforcement learning" action token credit assignment`
- `"multi-turn" tool use reinforcement learning turn-level advantage`
- `"tool call" function name argument process reward model`
- `"agent RL" transition-level credit assignment`
- `"trajectory filtering" multi-turn agent reinforcement learning`

Sources searched: arXiv and ACL Anthology. Only primary paper pages or PDFs were
used for the decision.

## Included direct and adjacent work

| Work | Relation to this study | Decision impact |
|---|---|---|
| [ActFocus: Resolving Action Bottleneck](https://arxiv.org/abs/2605.14558) | Direct: upweights action tokens and downweights reasoning tokens in agentic RL | Kills the broad claim that static action-focused token selection is an open gap |
| [TRACE](https://arxiv.org/abs/2607.13988) | Direct neighbor: dense turn-level credit at tool-call boundaries | Narrows any turn-level credit novelty claim |
| [MT-GRPO](https://arxiv.org/abs/2505.11821) | Direct neighbor: turn-level advantage for multi-turn tool use | Establishes prior turn-granular RL optimization |
| [Agent Lightning](https://arxiv.org/abs/2508.03680) | Adjacent: transition decomposition and hierarchical credit assignment | Establishes general agent-transition credit framing |
| [ECHO](https://arxiv.org/abs/2606.31650) | Adjacent: selective turn memory and positive credit routing | Further narrows broad selective-turn claims |
| [RAGEN](https://arxiv.org/abs/2504.20073) | Adjacent: trajectory filtering and stability in agent RL | Requires separation of token masking from trajectory filtering |
| [SimpleTIR](https://arxiv.org/abs/2509.02479) | Adjacent: filters trajectories containing void turns | Shows selection/filtering is already used for tool-integrated reasoning |
| [ToolPRM](https://aclanthology.org/2026.acl-long.855/) | Adjacent intra-call supervision: separates function and argument quality | Kills “tool-call argument masking has no precedent” wording |

## Synthesis

The literature supports a conservative distinction:

- **Occupied:** action-focused token weighting, turn-level credit, transition
  credit, trajectory filtering, selective turn memory, and function/argument
  decomposition.
- **Potentially useful but not yet novel:** a controlled 2×2 ablation of temporal
  breadth and intra-call content under an identical AReno policy-training path.
- **Defensible contribution now:** reproducible diagnostic evidence about when
  static span selection helps, hurts, or produces no update in a deterministic
  structured-action environment.

## Search limitations

- This was a targeted substitute search, not an exhaustive database review.
- Very recent 2026 papers may change status or receive revisions.
- No citation-count, benchmark-reproduction, or independent result verification
  was performed.
- A paper claim must therefore use “to our scoped search” and avoid
  first-ever/unique language.

## Literature gate

`KILL_METHOD_NOVELTY_AS_PRIMARY_CONTRIBUTION`

`PASS_DIAGNOSTIC_REPLICATION_POSITIONING`

