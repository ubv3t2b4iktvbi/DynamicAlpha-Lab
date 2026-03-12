# Essence Extraction Schema

Ask these questions:
1. What information must flow from input to output?
2. What information can be compressed into a macro-state?
3. Which scales must interact?
4. Which components carry memory?
5. Where is control or conditioning injected?
6. Which constraints are task-defining versus implementation-specific?
7. Which observed gains are robust across regimes?

Strong essence statements are short and architecture-independent.
Example pattern:
- "The model needs a coarse macro-state with slow dynamics, plus scale-local corrective injections that restore fine detail under fast disturbances."
