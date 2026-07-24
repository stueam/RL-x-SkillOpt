You are an expert regex engineer.

The target model produced a correct regex. Extract the working synthesis strategy as reusable guidance for the skill document.

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

Produce at most L=3 edits. You MUST produce at least one edit.
