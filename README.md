# spectral-graph-agent-c

**Your team's org chart says one thing. The spectral graph says another. The Fiedler value tells you who's actually connected.**

Spectral graph theory for multi-agent network analysis. When agents communicate, depend on, and trust each other, they form a graph. The *spectral properties* of that graph — eigenvalues of the Laplacian, Cheeger constants, mixing times — determine everything about how the system behaves: convergence speed, robustness to failures, information flow, and hidden community structure.

## What It Computes

| Analysis | What It Tells You |
|---|---|
| **Algebraic Connectivity** (Fiedler value λ₂) | How well-connected is your agent network? λ₂ = 0 means disconnected. Higher = more cohesive. |
| **Cheeger Constant** (sweep cut) | Where's the bottleneck? Finds the cut that most severely partitions your agents. |
| **Mixing Time** | How fast does information spread through the agent network? Derived from the spectral gap. |
| **Spectral Clustering** | Which agents naturally form communities? Uses the Fiedler vector for k-way partitioning. |
| **Eigenvector Centrality** | Which agents are structurally most important? Not just most connected — most *influentially* connected. |
| **Robustness Score** | How many edges/nodes can fail before the network fragments? Composite metric with articulation point analysis. |
| **Expander Quality** | How close is your agent network to a Ramanujan graph? Better expanders = faster, more resilient communication. |

## Building

```bash
make            # Build static library
make test       # Build and run 60+ tests
make clean      # Clean build artifacts
```

## Quick Start

```c
#include "spectral_graph.h"

// Build an agent communication graph
sg_graph *g = sg_graph_create(10, false);
sg_graph_add_edge(g, 0, 1, 1.0);  // agent 0 talks to agent 1
sg_graph_add_edge(g, 1, 2, 1.0);  // agent 1 talks to agent 2
// ... add all edges ...
_ensure_finalized(g);

// How connected is the network?
sg_fiedler_result *fiedler = sg_compute_fiedler(g);
printf("Algebraic connectivity: %.4f\n", fiedler->fiedler_value);
sg_fiedler_destroy(fiedler);

// Where's the bottleneck?
sg_cheeger_result *cheeger = sg_compute_cheeger(g);
printf("Cheeger constant: %.4f\n", cheeger->cheeger_constant);
sg_cheeger_destroy(cheeger);

// How fast does information spread?
sg_mixing_result *mix = sg_compute_mixing_time(g);
printf("Mixing time: %.2f steps\n", mix->mixing_time);
sg_mixing_destroy(mix);

// Which agents are most important?
sg_centrality_result *cent = sg_compute_centrality(g);
printf("Most important agent: %u (centrality=%.3f)\n",
       cent->dominant_agent, cent->max_centrality);
sg_centrality_destroy(cent);

sg_graph_destroy(g);
```

## Graph Construction Helpers

```c
// Standard topologies (great for benchmarking)
sg_graph_build_complete(g, 1.0);   // All-to-all (fully connected)
sg_graph_build_path(g, 1.0);      // Linear chain (worst case)
sg_graph_build_cycle(g, 1.0);     // Ring
sg_graph_build_star(g, 1.0);      // Hub-and-spoke
sg_graph_build_grid2d(g, r, c, 1.0);  // 2D grid
sg_graph_build_random(g, 0.3, seed);   // Erdős–Rényi random
```

## Ecosystem Integration

### evolving-sheaf-c (Sheaf Theory)

If your agent network *is* a sheaf (local data with consistency conditions), spectral gap degradation predicts communication failures:

```c
// Compare current spectral gap to healthy baseline
sg_sheaf_prediction pred = sg_sheaf_predict_failures(g, baseline_gap);
if (pred.communication_risk) {
    printf("WARNING: %.1f%% spectral gap degradation\n", pred.degradation_pct);
    printf("Communication failure probability: %.2f\n", pred.failure_probability);
}
```

### ergodic-transport-c (Ergodic Theory)

Is the agent network's mixing time related to convergence speed from ergodic theory?

```c
sg_ergodic_correlation corr = sg_ergodic_correlate(g);
printf("Mixing ↔ Ergodic convergence correlation: %.3f\n", corr.correlation);
```

## The Math (Briefly)

- **Laplacian**: L = D - A (degree matrix minus adjacency)
- **Fiedler value**: λ₂, the second-smallest Laplacian eigenvalue. Zero iff disconnected.
- **Cheeger inequality**: λ₂/2 ≤ h(G) ≤ √(2λ₂) — connects algebraic to geometric connectivity
- **Mixing time**: τ ≈ (1/λ₂) · log(n/ε) for normalized Laplacian
- **Ramanujan bound**: |λ₂| ≤ 2√(d-1) for d-regular graphs — optimal spectral expanders

## Test Coverage

60+ tests covering:
- Graph construction (path, cycle, complete, star, grid, random)
- Matrix operations (adjacency, Laplacian, normalized Laplacian, trace)
- Eigendecomposition (power iteration, QR-based full decomposition)
- All spectral analyses with known analytical values
- Cheeger inequality verification (λ₂/2 ≤ h ≤ √(2λ₂))
- Monotonicity properties (adding edges increases connectivity)
- NULL safety and edge cases
- Sheaf and ergodic integration

## License

MIT
