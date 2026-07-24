You are an expert cap set construction analyst.

The target model produced successful cap set constructions. Extract the working strategy as reusable guidance for the skill document.

## Rules
- Focus on patterns that appear across multiple trajectories.
- Be concise. Patterns must generalize beyond specific n or seed.
- Prefer reinforcing existing sections over adding new top-level sections.

Respond ONLY with a valid JSON object:
{
  "batch_size": <number of trajectories>,
  "success_patterns": ["<pattern 1>", "<pattern 2>"],
  "patch": {
    "reasoning": "<why these patterns are worth encoding>",
    "edits": [
      {"op": "append",       "content": "<markdown>"},
      {"op": "insert_after", "target": "<heading/text>", "content": "<markdown>"},
      {"op": "replace",      "target": "<old text>",     "content": "<new text>"},
      {"op": "delete",       "target": "<exact text to remove>"}
    ]
  }
}

Produce at most L=3 edits. You MUST produce at least one edit — always extract and encode what worked.
