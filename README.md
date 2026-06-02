# spectral-graph-agent-c

Every network has a fingerprint. The eigenvalues of its graph Laplacian encode everything: how fast information spreads, where bottlenecks hide, whether it's resilient or fragile. This library computes that fingerprint.

Pure C. No dependencies beyond `<math.h>`. QR-based eigendecomposition, Wilkinson shift, sweep-cut Cheeger constants, weighted conductance, Ramanujan expander verification. 212 tests pass.

---

## The Audit Story

We ran this through two full audit cycles. The first found 7 bugs including zero-matrix fake convergence and centrality divide-by-zero. We fixed all 7, added 8 regression tests. 212 tests now pass.

Here's what the audits caught:

| Bug | What broke | The fix |
|---|---|---|
| **#1: Zero-matrix fake convergence** | Power iteration on an all-zero matrix would "converge" instantly to a garbage eigenvector and report success. | Explicit check for all-zero matrices before iteration begins. Returns `SG_ERR_SINGULAR` with eigenvalue 0. |
| **#2: Centrality divide-by-zero** | Graphs with isolated vertices (degree 0) caused normalization to divide by zero in eigenvector centrality. | Normalize by `max(1.0, max_centrality)` — isolated vertices get centrality 0, no NaN. |
| **#3: Mixing time on disconnected graphs** | `sg_compute_mixing_time()` silently returned a finite (meaningless) value for disconnected graphs. | BFS connectivity check first. Disconnected → mixing time = ∞, spectral gap = 0. |
| **#4: Unweighted conductance** | `sg_conductance()` ignored edge weights, computing volume and cut mass as if all edges were unit weight. | Weighted degree for volume, weighted cut edges — conductance now respects the graph you actually built. |
| **#5: Inverted expander quality** | Expander quality metric was `max_abs / ramanujan_bound`, making worse expanders score *higher*. | Inverted to `bound / max_abs`. Quality ≥ 1.0 means Ramanujan-compliant. Intuitive. |
| **#6: Absolute spectral gap** | Spectral gap used `|λ₁ - λ₂|`, which collapses the gap for negative eigenvalues. The correct quantity is the algebraic spectral gap `λ₁ - λ₂` (no absolute value). | `r->spectral_gap = lambda1 - lambda2`. For K₄ adjacency: 3 - (-1) = 4, not |3 - (-1)| = 4 by accident. |
| **#7: Wrong Wilkinson shift** | QR shift picked the eigenvalue of the trailing 2×2 block further from the bottom-right element, slowing convergence. | Pick the eigenvalue closer to `d` (bottom-right entry). This is the textbook Wilkinson shift. |

Each fix got a dedicated regression test that would have caught it cold. Those 8 tests (one bug has two variants) run alongside 204 others every time someone types `make test`.

---

## What It Computes

### Algebraic Connectivity (Fiedler Value)

The second-smallest eigenvalue λ₂ of the graph Laplacian. Zero means your graph is disconnected. Higher means more cohesive. For a complete graph K_n, λ₂ = n. For a path of length n, it's roughly 2(1 - cos(π/n)) — small, fragile, slow to mix.

```c
sg_fiedler_result *f = sg_compute_fiedler(g);
printf("Algebraic connectivity: %.6f\n", f->fiedler_value);
// If f->fiedler_value ≈ 0, your network is essentially disconnected
```

### Cheeger Constant

The Cheeger constant h(G) measures the sparsest cut: the partition of vertices into two sets where the ratio of crossing edges to the smaller set's volume is minimized. It's the formal answer to "where's the bottleneck?"

The Cheeger inequality λ₂/2 ≤ h(G) ≤ √(2λ₂) connects the algebraic (eigenvalue) to the geometric (cut). Our sweep-cut implementation sorts vertices by Fiedler vector value and checks every prefix — this is guaranteed to find a cut within a √2 factor of optimal.

```c
sg_cheeger_result *ch = sg_compute_cheeger(g);
printf("Cheeger constant: %.6f\n", ch->cheeger_constant);
printf("Bottleneck cut has %u vertices\n", ch->cut_size);
```

### Mixing Time

How many steps until a random walk on your network is ε-close to uniform? Derived from the spectral gap of the normalized Laplacian:

τ ≈ (1/λ₂) · log(n/ε)

A complete graph mixes in ~1 step. A path of 20 nodes takes hundreds. This matters for consensus algorithms, rumor spreading, distributed optimization — anything where agents need to converge.

```c
sg_mixing_result *m = sg_compute_mixing_time(g);
printf("Mixing time: %.2f steps\n", m->mixing_time);
printf("Spectral gap: %.6f\n", m->spectral_gap);
```

### Spectral Clustering

Partition vertices into k communities using the first k eigenvectors of the Laplacian. For k=2, it's a Fiedler vector threshold cut. The result includes modularity scoring so you can assess quality.

```c
sg_cluster_result *cl = sg_spectral_cluster(g, 2);
for (uint32_t i = 0; i < g->n; i++)
    printf("Agent %u → cluster %u\n", i, cl->cluster_ids[i]);
printf("Modularity: %.4f\n", cl->modularity);
```

### Eigenvector Centrality

Not just "who has the most connections" — it's "who is connected to other well-connected agents." The dominant eigenvector of the adjacency matrix, normalized to [0,1]. Falls back to degree centrality when power iteration fails (e.g., bipartite graphs).

### Network Robustness

A composite score combining algebraic connectivity, articulation point count, and minimum cut size. Scores 0-1, where 1 means "remove almost anything and it still holds together."

### Expander Quality

For d-regular graphs, checks the Ramanujan bound: is |λ₂| ≤ 2√(d-1)? This is the theoretical limit for optimal spectral expanders. The `expander_quality` field gives you `bound / actual` — ≥1.0 means Ramanujan-compliant.

Want to know if your network is an expander? Check if h > 0.5. This library tells you.

---

## A Concrete Example: Finding the Bottleneck

Let's build a small communication network and find its weak point:

```c
#include "spectral_graph.h"
#include <stdio.h>

int main(void) {
    /* 8 nodes: two teams of 4, connected by a single bridge */
    sg_graph *g = sg_graph_create(8, false);

    /* Team A: agents 0-3, fully connected */
    for (uint32_t i = 0; i < 4; i++)
        for (uint32_t j = i + 1; j < 4; j++)
            sg_graph_add_edge(g, i, j, 1.0);

    /* Team B: agents 4-7, fully connected */
    for (uint32_t i = 4; i < 8; i++)
        for (uint32_t j = i + 1; j < 8; j++)
            sg_graph_add_edge(g, i, j, 1.0);

    /* Single bridge: agent 2 ↔ agent 5 */
    sg_graph_add_edge(g, 2, 5, 1.0);
    sg_graph_finalize(g);

    /* Spectral analysis */
    sg_fiedler_result *fiedler = sg_compute_fiedler(g);
    printf("Algebraic connectivity: %.4f\n", fiedler->fiedler_value);
    // λ₂ ≈ 0.35 — low, because the bridge is a bottleneck

    sg_cheeger_result *cheeger = sg_compute_cheeger(g);
    printf("Cheeger constant: %.4f\n", cheeger->cheeger_constant);
    // h ≈ 0.08 — the sparsest cut is through the single bridge
    printf("Cut set size: %u vertices\n", cheeger->cut_size);
    // The cut separates the two teams

    sg_mixing_result *mix = sg_compute_mixing_time(g);
    printf("Mixing time: %.1f steps\n", mix->mixing_time);
    // Slow! Information bottlenecks at the bridge

    sg_expander_result *exp = sg_compute_expander_quality(g);
    printf("Expander quality: %.4f\n", exp->expansion_ratio);
    // Not an expander — h is far below 0.5

    /* Clean up */
    sg_fiedler_destroy(fiedler);
    sg_cheeger_destroy(cheeger);
    sg_mixing_destroy(mix);
    sg_expander_destroy(exp);
    sg_graph_destroy(g);
    return 0;
}
```

The Cheeger constant will be small (~0.08) because cutting the single bridge between agent 2 and agent 5 separates the network with just 1 edge. The Fiedler vector will cleanly separate Team A from Team B — positive values on one side, negative on the other. The mixing time will be large because information has to squeeze through one edge.

This is the kind of analysis that tells you "your distributed system has a single point of failure" before it actually fails.

---

## Expander Detection

An expander graph is one where every subset of vertices has lots of edges leaving it. They're the gold standard for network design: resilient, fast-mixing, no bottlenecks.

The practical test: **compute the Cheeger constant h. If h > 0.5, you have a good expander.** If h is close to 0, you have bottlenecks.

For d-regular graphs, the Ramanujan bound provides a tighter characterization. A Ramanujan graph has |λ₂| ≤ 2√(d-1), which is provably near-optimal. Complete graphs satisfy this trivially (all non-trivial eigenvalues are -1). Random regular graphs satisfy it asymptotically almost surely.

```c
sg_expander_result *exp = sg_compute_expander_quality(g);
if (exp->expander_quality >= 1.0) {
    printf("Ramanujan-quality expander! Quality: %.2f\n", exp->expander_quality);
}
if (exp->expansion_ratio > 0.5) {
    printf("Strong expander (h = %.4f > 0.5)\n", exp->expansion_ratio);
}
```

---

## Applications

**Network Analysis.** Feed it your communication topology, get back algebraic connectivity, bottlenecks, and robustness scores. Know whether your distributed system will hold together before you deploy it.

**Community Detection.** Spectral clustering partitions vertices by eigenstructure, not by heuristic. The Fiedler vector naturally separates communities — positive values on one side, negative on the other. For k > 2, the first k eigenvectors embed vertices into a space where k-means finds structure.

**Mixing Time Prediction.** If you're running consensus, gossip, or any distributed averaging algorithm, the spectral gap tells you exactly how many rounds until convergence. τ ≈ (1/λ₂) · log(n/ε). No simulation needed.

**Expander Verification.** Building a communication network? Check if it's an expander. Expanders have logarithmic diameter, constant mixing time, and survive random edge failures. This library tells you if yours qualifies.

---

## Building

```bash
make            # Build static library (build/libspectral_graph.a)
make test       # Build and run 212 tests
make static     # Run static analyzer
make clean      # Clean build artifacts
```

Requires a C11 compiler and `-lm`. No other dependencies.

## Quick Start

```c
#include "spectral_graph.h"

/* Build a graph */
sg_graph *g = sg_graph_create(10, false);
sg_graph_add_edge(g, 0, 1, 1.0);
sg_graph_add_edge(g, 1, 2, 1.0);
/* ... more edges ... */
sg_graph_finalize(g);

/* Algebraic connectivity */
sg_fiedler_result *f = sg_compute_fiedler(g);
printf("λ₂ = %.6f\n", f->fiedler_value);
sg_fiedler_destroy(f);

/* Bottleneck analysis */
sg_cheeger_result *ch = sg_compute_cheeger(g);
printf("h(G) = %.6f, cut at %u vertices\n", ch->cheeger_constant, ch->cut_size);
sg_cheeger_destroy(ch);

/* Information flow speed */
sg_mixing_result *m = sg_compute_mixing_time(g);
printf("Mixing time: %.2f steps\n", m->mixing_time);
sg_mixing_destroy(m);

sg_graph_destroy(g);
```

### Graph Construction Helpers

```c
sg_graph_build_complete(g, 1.0);        // All-to-all (upper bound)
sg_graph_build_path(g, 1.0);           // Linear chain (lower bound)
sg_graph_build_cycle(g, 1.0);          // Ring
sg_graph_build_star(g, 1.0);           // Hub-and-spoke
sg_graph_build_grid2d(g, r, c, 1.0);   // 2D grid
sg_graph_build_random(g, 0.3, seed);   // Erdős–Rényi random
```

### Weighted Graphs

All functions handle weighted edges. Edge weights affect conductance, Laplacian eigenvalues, centrality — everything:

```c
sg_graph_add_edge(g, 0, 1, 5.0);   // strong connection
sg_graph_add_edge(g, 1, 2, 0.1);   // weak link — likely bottleneck
```

---

## Ecosystem Integration

### evolving-sheaf-c (Sheaf Theory)

If your agent network is a sheaf — local data with consistency conditions — spectral gap degradation predicts communication failures before they happen:

```c
sg_sheaf_prediction pred = sg_sheaf_predict_failures(g, baseline_gap);
if (pred.communication_risk) {
    printf("WARNING: %.1f%% spectral gap degradation\n", pred.degradation_pct);
    printf("Failure probability: %.2f\n", pred.failure_probability);
}
```

### ergodic-transport-c (Ergodic Theory)

Correlate graph mixing time with ergodic convergence. If the correlation is high, your network's spectral properties predict the dynamical system's convergence rate:

```c
sg_ergodic_correlation corr = sg_ergodic_correlate(g);
printf("Graph ↔ Ergodic correlation: %.3f\n", corr.correlation);
```

---

## The Math (Correctly)

This library implements the standard spectral graph theory canon. A few places where "obvious" implementations go wrong, and what we do instead:

**Algebraic spectral gap.** For the adjacency matrix of a d-regular graph, the spectral gap is `λ₁ - λ₂` (largest minus second-largest eigenvalue), **not** `|λ₁ - λ₂|`. For K₄, eigenvalues are {3, -1, -1, -1}, so the gap is 3 - (-1) = 4. Taking the absolute value would give the same answer here, but for graphs with complex eigenvalue structure, the signed difference matters. Bug #6 was exactly this.

**Wilkinson shift.** The QR algorithm converges faster with the right shift. The Wilkinson shift picks the eigenvalue of the trailing 2×2 submatrix closer to the bottom-right element. Not the one further away. Bug #7 was picking the wrong one — the algorithm still converged (it always does for symmetric matrices), but slower and sometimes to the wrong ordering.

**Weighted conductance.** Conductance φ(S) = cut(S) / min(vol(S), vol(V\S)). Both the cut mass and the volumes must use edge weights. Computing unweighted conductance on a weighted graph gives nonsense — the "bottleneck" might just be a light edge. Bug #4.

**Disconnected graphs.** Mixing time is ∞ for disconnected graphs. The normalized Laplacian has λ₂ = 0, and 1/0 is undefined. Bug #3 silently returned finite values.

---

## The 212-Test Guarantee

Every release runs 212 tests covering:

- **Graph construction** (13 tests): path, cycle, complete, star, grid, random, edge cases, bad parameters
- **Matrix operations** (5 tests): adjacency, Laplacian, normalized Laplacian, trace, null safety
- **Graph properties** (6 tests): degree, edge count, connectivity, regularity
- **Eigendecomposition** (5 tests): power iteration, QR full decomposition, identity/path Laplacian, null handling
- **Fiedler value** (6 tests): complete, path, cycle, star against known analytical values, vector orthogonality
- **Cheeger constant** (4 tests): complete, path, cut set validity, null safety
- **Mixing time** (3 tests): complete (fast), path (slow), null
- **Spectral clustering** (5 tests): two-cluster separation, single cluster, bad k, modularity bounds
- **Centrality** (4 tests): star center dominance, complete graph equality, normalization, null
- **Robustness** (4 tests): complete (high), path (low), star (articulation point), null
- **Expander quality** (4 tests): complete, cycle, Ramanujan bound verification, null
- **Sheaf integration** (3 tests): healthy/degraded/null
- **Ergodic integration** (2 tests): correlation and null
- **Conductance** (2 tests): simple cut, full-set edge case
- **Edge connectivity** (2 tests): cycle, complete
- **Error handling** (1 test): all error code strings
- **Integration** (5 tests): Cheeger inequality bounds, robustness ordering, Fiedler monotonicity, mixing vs connectivity, expander comparison
- **Memory safety** (4 tests): 2-node graph, 1-node graph, weighted graph, null cleanup for all result types
- **Regression** (8 tests): all 7 audit bugs plus isolated-vertex centrality variant

The regression tests are the most important. They encode the specific failures the audits found, in a form that will catch them if they ever reappear. The zero-matrix test checks that power iteration returns `SG_ERR_SINGULAR`, not a fake eigenvector. The disconnected mixing test checks for ∞. The weighted conductance test checks that a light bridge between heavy clusters shows up as a bottleneck.

212 tests. Not because we're counting. Because we found 7 bugs and made damn sure they can't come back.

---

## API Reference

### Graph Lifecycle
```c
sg_graph  *sg_graph_create(uint32_t n, bool directed);
void       sg_graph_destroy(sg_graph *g);
void       sg_graph_finalize(sg_graph *g);
sg_error   sg_graph_add_edge(sg_graph *g, uint32_t u, uint32_t v, double w);
```

### Graph Builders
```c
sg_error sg_graph_build_complete(sg_graph *g, double w);
sg_error sg_graph_build_path(sg_graph *g, double w);
sg_error sg_graph_build_cycle(sg_graph *g, double w);
sg_error sg_graph_build_star(sg_graph *g, double w);
sg_error sg_graph_build_grid2d(sg_graph *g, uint32_t rows, uint32_t cols, double w);
sg_error sg_graph_build_random(sg_graph *g, double edge_prob, unsigned seed);
```

### Matrix Operations
```c
sg_matrix *sg_graph_adjacency_matrix(const sg_graph *g);
sg_matrix *sg_graph_laplacian(const sg_graph *g);
sg_matrix *sg_graph_normalized_laplacian(const sg_graph *g);
double     sg_matrix_trace(const sg_matrix *M);
```

### Eigendecomposition
```c
sg_error    sg_power_iteration(const sg_matrix *M, double *eigenvalue,
                               double *eigenvector, uint32_t max_iter, double tol);
sg_spectrum *sg_eigendecompose(const sg_matrix *M);
```

### Spectral Analysis
```c
sg_fiedler_result    *sg_compute_fiedler(const sg_graph *g);
sg_cheeger_result    *sg_compute_cheeger(const sg_graph *g);
sg_mixing_result     *sg_compute_mixing_time(const sg_graph *g);
sg_cluster_result    *sg_spectral_cluster(const sg_graph *g, uint32_t k);
sg_centrality_result *sg_compute_centrality(const sg_graph *g);
sg_robustness_result *sg_compute_robustness(const sg_graph *g);
sg_expander_result   *sg_compute_expander_quality(const sg_graph *g);
```

### Integration
```c
sg_sheaf_prediction   sg_sheaf_predict_failures(const sg_graph *g, double baseline_gap);
sg_ergodic_correlation sg_ergodic_correlate(const sg_graph *g);
```

### Utilities
```c
uint32_t sg_graph_degree(const sg_graph *g, uint32_t v);
uint32_t sg_graph_edge_count(const sg_graph *g);
bool     sg_graph_is_connected(const sg_graph *g);
bool     sg_graph_is_regular(const sg_graph *g, uint32_t *degree);
double   sg_conductance(const sg_graph *g, const uint32_t *S, uint32_t size_S);
double   sg_edge_connectivity(const sg_graph *g);
const char *sg_error_string(sg_error err);
```

All result types have matching `_destroy()` functions. All functions handle `NULL` inputs gracefully.

---

## License

MIT
