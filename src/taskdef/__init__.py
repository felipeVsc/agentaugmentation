"""Task definition component (TASKDEFINITION_SPEC.md).

Builds a fixed, versioned task set ``(n, H, G)`` from a property graph:
snapshot -> centrality -> context -> popularity -> sampling (Algorithm 2)
-> difficulty -> tasks.csv / ground_truth.csv / manifest.json.
"""

__version__ = "0.1.0"
