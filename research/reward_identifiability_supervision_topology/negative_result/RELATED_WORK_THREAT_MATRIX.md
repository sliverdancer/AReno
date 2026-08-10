# Related-work threat matrix

This matrix fixes the novelty claim before manuscript writing. The target claim
is not a new credit-assignment algorithm; it is a pre-training estimability
diagnostic for group-relative tool-use RL.

| Work | What it solves | Threat to our claim | Differentiation |
| --- | --- | --- | --- |
| TRACE: Turn-level Reward Assignment via Credit Estimation | Dense turn-level rewards for long-horizon agents using reference-model value changes at tool-call boundaries. | High. It directly targets turn-level credit assignment for agentic RL. | TRACE assumes a mechanism for generating dense turn-level signal. RIST studies the prior failure mode where sparse group-relative terminal rewards have no usable contrast, so supervision-topology effects are not estimable without additional shaping. |
| PORTool: Importance-Aware Policy Optimization with Rewarded Tree | Step importance estimation via rewarded rollout trees and local branch comparisons. | High. It addresses tool-use credit ambiguity and intermediate decision importance. | PORTool creates local alternatives in a tree to resolve step credit. Our diagnostic asks whether a fixed evaluation/training pool has enough reward resolution before such credit machinery is meaningful. |
| TSPO: Turn-level Stage-aware Policy Optimization | Addresses process and intra-group homogenization in multi-turn search policy optimization. | Very high. It explicitly names intra-group homogenization in GRPO-like sampling. | TSPO proposes a reward allocation method to increase reward variance. RIST contributes an evaluation-validity gate and terminal case studies showing when the original group-relative setup should be stopped before training. |
| Tree-structured credit assignment for RL with LLMs | Uses shared prefixes / tree structure for credit assignment under delayed reward. | Medium. Similar motivation around sparse delayed rewards. | Tree methods change rollout structure. RIST focuses on diagnosing reward-resolution collapse in existing task pools and reporting when no topology experiment is scientifically admissible. |
| Token/turn reward shaping and teacher-derived rewards | Densifies sparse rewards through teacher scores, process labels, or token-level calibration. | Medium. Could be positioned as a fix for collapse. | These are mitigation methods. Our paper is a precondition check: without reward-resolution evidence, null or positive topology results are not interpretable. |
| BFCL / ToolSandbox / tau-bench style tool benchmarks | Evaluate function calling or tool-agent-user interaction. | Medium. They may already expose tool-use failure modes. | Existing benchmarks usually report task success. RIST asks for group-level reward-resolution statistics required by group-relative training objectives. |
| GRPO/RLVR sparse-reward analyses | Discuss outcome-reward learning and group-relative normalization. | Medium. The core math is adjacent. | RIST ties the math to multi-turn tool-use supervision topology and provides evidence-integrity case studies with manifest/receipt-bound collection. |

## Novelty boundary

The manuscript may claim:

- reward-resolution diagnostics are a necessary validity layer before
  group-relative supervision-topology experiments;
- all-fail, all-pass, and non-transportable mixedness are distinct collapse
  modes;
- RIST v3.1/v4.0 are controlled terminal case studies showing that valid
  infrastructure does not imply estimable supervision effects.

The manuscript may not claim:

- a new state-of-the-art credit-assignment method;
- that TRACE, PORTool, or TSPO are unnecessary;
- that action-span supervision is ineffective in settings with real reward
  contrast;
- that RIST is a production benchmark.

## Required reviewer-defense language

Use this wording in the introduction and related-work section:

> Credit-assignment methods answer how to distribute a learning signal across
> steps. Reward-resolution diagnostics answer whether the sampled group contains
> a learning signal to distribute at all.

This distinction is the paper's main defense against direct-substitute reviews.
