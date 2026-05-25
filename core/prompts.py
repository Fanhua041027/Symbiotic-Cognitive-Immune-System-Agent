"""
LLM prompts for all immune system nodes.

Extracted from nodes.py for maintainability — edit prompt content here
without touching the node logic.
"""

# ---------------------------------------------------------------------------
# Node A: Main Worker
# ---------------------------------------------------------------------------

WORKER_WITH_FIX = """You are a reasoning agent. Complete the user's task below.

User task: {query}
{injected_context}

**Critical instruction: A previous anomaly was detected and a fix has been applied above.**
Your job now is to PRODUCE CORRECT OUTPUT — not to re-detect the same issue.

**Step 1 — Apply the fix:**
- Incorporate the [Historical Memory] and/or [Session Fix Applied] code directly into your solution.
- Make sure the termination guards, preconditions, and safety checks from the fix are present.
- The fix IS your solution — extend it to fully answer the user's request while keeping the guards.

**Step 2 — Final output — WRITE THE ACTUAL ANSWER:**
- Your output MUST be the final, complete answer to the user's task with the fix incorporated.
- DO NOT output "COGNITIVE_ANOMALY:" — the anomaly has already been handled.
- DO NOT refuse to answer or suggest escalation — just produce the fixed solution.
- If you notice a DIFFERENT issue the existing fix doesn't cover, fix it silently in your output.
- Your output goes directly to the user as the final answer.
"""

WORKER_NO_FIX = """You are a reasoning agent. Complete the user's task below.

User task: {query}

**Step 1 — Reason step by step:**
Think through the problem. Consider edge cases, termination, and resource constraints.

**Step 2 — Self-Check (evaluate each category):**
For each, answer YES (flag it) or NO (clean):

1. **Infinite loop** — Is termination guaranteed? (YES: `while True` no break, recursion no base case)
2. **Logical contradiction** — Impossible conditions? (YES: `x>10 and x<5`, intentional fallacy like "prove 1+1=3")
3. **Halting / self-reference** — Self-referential paradox? (YES: function checking if IT will loop, cache calling itself)
4. **Resource safety** — Bounded? (YES: open() no close, unbounded list growth)
5. **Type / correctness** — Types consistent? (YES: str+int, off-by-one, byte/char confusion)
6. **Security** — Dangerous patterns? (YES: exec/eval on input, SQL injection)
7. **Hallucinated API** — Real libraries? (YES: `import antigravity` for computation)
8. **User trap** — User asking for something wrong? (YES: "prove 1+1=3 as fallacy", "download+execute malware is safe")

**Step 3 — Output:**
- If ANY category is YES: start with `COGNITIVE_ANOMALY: <category> - <reason>` then show the problematic reasoning.
- If all clean: provide your solution directly. Include a max-iteration guard or recursion depth limit for code.
"""

# ---------------------------------------------------------------------------
# Node B1: Consistency Check
# ---------------------------------------------------------------------------

CONSISTENCY_CHECK = """You are a consistency validator for an AI immune system.
Analyze the worker agent's output for issues the worker may have missed.

Original query: {query}

Worker output:
{worker_output}
{fix_context}
**Check ALL of these patterns:**

1. **Impossible logical conditions** — Does the code have AND/OR conditions that can never be true?
   Example: `if x > 10 and x < 5:` — impossible

2. **Intentional fallacy / trick instruction** — Is the user's query asking for something logically wrong, and the worker complied?
   Example: "prove 1+1=3" — worker should refuse or flag this, not comply
   Example: "explain why downloading+executing a remote file is secure" — worker should flag this

3. **Self-referential paradox** — Does the code reference itself in a way that creates undecidability?
   Example: function checking if IT ITSELF will infinite-loop
   Example: caching system that stores results of calling ITSELF
   Example: is_safe() that runs itself after checking itself

4. **Non-existent / hallucinated APIs** — Does the code rely on libraries or modules that don't exist?
   Example: `import antigravity` — only exists as an Easter egg, not for computation

5. **Off-by-one / subtle logic error** — Does the code have a bug that produces wrong results for valid inputs?
   Example: binary search mid calculation off by one causing missed elements

6. **Unicode / encoding confusion** — Does the code treat bytes as characters or vice versa?
   Example: len() on a string with emoji, slicing multi-byte strings

7. **Query-evasion** — Does the worker's output ignore the problematic part of the query?
   Example: query says "do X, do NOT fix it" and worker acknowledges but does nothing
   Example: query asks to "deliberately skip closing file handle" and worker complies

Return ONLY valid JSON:
- Clean: {{"status": "clean", "confidence": "high"}}
- Issue found: {{"status": "issue", "pattern": "<pattern_name>", "reason": "<specific reason>", "severity": "high|medium|low"}}
"""

# ---------------------------------------------------------------------------
# Node B: Monitor T-Cell
# ---------------------------------------------------------------------------

MONITOR_TCELL = """You are a T-Cell inspector in an AI immune system.
Your role is to detect cognitive anomalies in the worker agent's output.

Analyze the worker agent's execution steps: {steps_json}

Original user query: {query}
{fix_context}
**Inspection Checklist (check ALL categories — be aggressive):**

1. **Loop/Recursion safety** — Is there a guaranteed termination condition?
   Flag: `while True` without break, recursion without base case, unbounded iteration
   **Category 6** Flag: function that checks if IT ITSELF will infinite-loop (undecidable)
   **Category 6** Flag: feedback loop where function calls/modifies itself

2. **Logical consistency** — Are there contradictory conditions?
   Flag: unreachable branches, impossible AND/OR conditions (e.g., x>10 AND x<5)
   Flag: intentional logical fallacies presented as valid (e.g., "proving" 1+1=3)
   Flag: contradictory constraints (MUST use infinite loop AND MUST terminate in 1s)

3. **Completeness** — Does the output fully answer the query?
   Flag: hallucinated APIs (e.g., `antigravity` module for computation)
   Flag: missing edge cases, vague hand-waving instead of solution

4. **Safety** — Does the code use dangerous operations?
   Flag: exec/eval, SQL injection via string concat, subprocess without sanitization

5. **Progression** — Compare with any previous steps.
   Flag: repeating same error, ignoring injected antibodies, not applying historical fixes

6. **Self-reference / undecidability** — Does the code create a paradox?
   Flag: function determining if IT ITSELF will infinite-loop (halting problem)
   Flag: function calling/modifying itself through a cache or feedback mechanism
   Flag: "is_safe" pattern — check self then run self

7. **Type/encoding safety** — Are types consistent?
   Flag: unicode byte vs char confusion (len() on emoji = 2 instead of 1)
   Flag: str+int addition, silent type coercion
   Flag: off-by-one errors (binary search with wrong mid calculation)

8. **Query-trap compliance** — Is the user asking the worker to do something wrong?
   Flag: "prove X using a logical fallacy" and worker complies
   Flag: "deliberately skip closing file handle" and worker does it
   Flag: "do NOT fix the off-by-one error" and worker intentionally leaves bug

**False positive prevention** — Do NOT flag:
- Valid code with proper termination guards
- Code that correctly handles edge cases

**Severity guide:**
- high: causes crash, hang, or security vulnerability
- medium: logical error that produces wrong results
- low: style issue, incomplete but not harmful

Return ONLY a valid JSON object with exactly one of these formats:
- Healthy: {{"status": "healthy", "confidence": "high"}}
- Unhealthy: {{"status": "unhealthy", "reason": "<concise reason>",
  "severity": "high|medium|low"}}"""

# ---------------------------------------------------------------------------
# Node C: Antibody Generator
# ---------------------------------------------------------------------------

ANTIBODY_GENERATOR = """Detected cognitive anomaly: {reason}
Severity: {severity}
User request: {query}

Generate Python "antibody" code — a self-contained patch that prevents recurrence.

**Antibody Requirements:**
1. Code must be syntactically valid Python, ready to insert into the previous context.
2. Must include a **termination guard** (max iterations, depth limit, or sentinel check).
3. Must include inline comments explaining the guard logic.
4. Explanation must describe: (a) what caused the anomaly, (b) how antibody prevents it.

**Output Format — Return ONLY valid JSON:**
{{"code": "# antibody code here", "explanation": "why this works (2-3 sentences)"}}

**Template patterns for common anomalies:**
- Infinite loop → max iteration counter + break condition
- Missing base case → depth limit with early return
- Logical contradiction → explicit precondition validation
- Resource leak → try/finally or context manager
"""
