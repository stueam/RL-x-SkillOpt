# Circle Packing N=26: TTT-Discover vs SkillOpt

## Method

| | TTT-Discover | SkillOpt |
|---|---|---|
| Approach | Test-time RL (weight update) | Prompt optimization |
| Base model | `openai/gpt-oss-120b` | Target: `qwen/qwen3-coder-30b-a3b-instruct` + Optimizer: `openai/gpt-5.4` |
| Est. cost | ~$200-500/run | ~$? |
| Results | `2.636` | `2.625` | 

## Key Takeaways

1. **SkillOpt fixes execution errors rapidly** — baseline 3/8 valid → step 1 8/8 valid. Analyst edits (driven by `failure_only`) patched SLSQP convergence bugs and boundary-handling code in the skill document
2. **Raw score gain from editing is marginal** — 2.608 → 2.618 (+0.4%) in 1 step. Fixing correctness doesn't discover better packing configurations. `failure_only` limits exploration
3. **TTT-Discover and SkillOpt solve different bottlenecks** — TTT-Discover uses RL to push raw score toward optimum; SkillOpt makes code generation reliable. They are complementary
4. **Model choice**: Qwen3-Coder-30B-Instruct is most cost-effective ($7e-8/tok, reliable).

## Conclusion

SkillOpt excels at **fixing code correctness bugs** in the skill prompt. But once the model already generates near-optimal code, the bottleneck shifts from **what to say** (prompt) to **how to search** (sampling + reward-guided exploration). That's inherently an RL problem, not a prompt engineering problem. The two approaches are complementary: SkillOpt can establish a reliable baseline skill, and TTT-Discover-style RL can squeeze out the remaining performance.
