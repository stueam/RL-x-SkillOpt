You are an expert regex engineer.

The target model produced a `solve(positives, negatives)` that generated an invalid or incorrect regex. Identify the root cause and propose concrete edits to fix the skill.

Respond ONLY with a valid JSON object:
{
  "batch_size": 1,
  "patch": {
    "reasoning": "<root cause analysis>",
    "edits": [
      {"op": "append",       "content": "<markdown>"},
      {"op": "insert_after", "target": "<heading/text>", "content": "<markdown>"},
      {"op": "replace",      "target": "<old text>",     "content": "<new text>"},
      {"op": "delete",       "target": "<exact text to remove>"}
    ]
  }
}

Produce at most L=3 edits. Focus on the most critical fix.
