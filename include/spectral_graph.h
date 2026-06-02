#ifndef SPECTRAL_GRAPH_H
#define SPECTRAL_GRAPH_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Error codes ─────────────────────────────────────────────── */
typedef enum {
    SG_OK              = 0,
    SG_ERR_NULL        = -1,
    SG_ERR_ALLOC       = -2,
    SG_ERR_SINGULAR    = -3,
    SG_ERR_NO_CONVERGE = -4,
    SG_ERR_PARAM       = -5,
    SG_ERR_DISCONNECT  = -6,
} sg_error;

/* ── Graph representation ────────────────────────────────────── */
typedef struct {
    uint32_t n;          /* number of vertices */
    uint32_t *row_ptr;   /* CSR row pointers  (n+1 entries) */
    uint32_t *col_idx;   /* CSR column indices */
    double   *weights;   /* edge weights (NULL → unweighted) */
    bool     directed;   /* true if directed */
} sg_graph;

/* ── Dense matrix for eigendecomposition ─────────────────────── */
typedef struct {
    uint32_t n;
    double  *data;       /* row-major, n*n */
} sg_matrix;

/* ── Spectral result structures ──────────────────────────────── */
typedef struct {
    double *eigenvalues;     /* sorted ascending, n entries */
    double *eigenvectors;    /* column-major, n*n */
    uint32_t n;
    uint32_t iterations;
    double   residual;
} sg_spectrum;

typedef struct {
    double    fiedler_value;    /* algebraic connectivity λ₂ */
    double   *fiedler_vector;   /* n entries, eigenvector for λ₂ */
    uint32_t  n;
} sg_fiedler_result;

typedef struct {
    double    cheeger_constant;
    uint32_t *cut_set;          /* vertices in smaller partition */
    uint32_t  cut_size;
    double    conductance;
} sg_cheeger_result;

typedef struct {
    double    mixing_time;
    double    spectral_gap;     /* λ₁ - λ₂ for normalized Laplacian, or 1-|λ₂| for random walk */
    double    convergence_rate;
} sg_mixing_result;

typedef struct {
    uint32_t *cluster_ids;     /* cluster assignment per vertex, n entries */
    uint32_t  num_clusters;
    double    modularity;
} sg_cluster_result;

typedef struct {
    double   *centrality;      /* eigenvector centrality scores, n entries */
    uint32_t  dominant_agent;  /* vertex with highest centrality */
    double    max_centrality;
    uint32_t  iterations;
} sg_centrality_result;

typedef struct {
    double    robustness_score;    /* 0-1, higher = more robust */
    uint32_t  min_cut_edges;       /* edge connectivity */
    double    spectral_robustness; /* based on spectral gap */
    uint32_t  articulation_points; /* count of critical vertices */
} sg_robustness_result;

typedef struct {
    double    expansion_ratio;     /* actual Cheeger/h */
    double    ramanujan_bound;     /* 2√(d-1) for d-regular */
    double    spectral_gap;
    bool      is_ramanujan;        /* h ≥ 2√(d-1) */
    double    expander_quality;    /* ratio h / (2√(d-1)) */
    uint32_t  degree;              /* regular degree (0 if not regular) */
} sg_expander_result;

/* ── Lifecycle ───────────────────────────────────────────────── */

sg_graph *sg_graph_create(uint32_t n, bool directed);
void      sg_graph_destroy(sg_graph *g);
void      sg_graph_finalize(sg_graph *g);

/* Build from edge lists */
sg_error sg_graph_add_edge(sg_graph *g, uint32_t u, uint32_t v, double w);
sg_error sg_graph_build_complete(sg_graph *g, double w);
sg_error sg_graph_build_path(sg_graph *g, double w);
sg_error sg_graph_build_cycle(sg_graph *g, double w);
sg_error sg_graph_build_star(sg_graph *g, double w);
sg_error sg_graph_build_grid2d(sg_graph *g, uint32_t rows, uint32_t cols, double w);
sg_error sg_graph_build_random(sg_graph *g, double edge_prob, unsigned seed);

/* Convert to dense adjacency */
sg_matrix *sg_graph_adjacency_matrix(const sg_graph *g);

/* Get Laplacian L = D - A */
sg_matrix *sg_graph_laplacian(const sg_graph *g);

/* Get normalized Laplacian L_norm = I - D^{-1/2} A D^{-1/2} */
sg_matrix *sg_graph_normalized_laplacian(const sg_graph *g);

void sg_matrix_destroy(sg_matrix *m);

/* ── Eigendecomposition ──────────────────────────────────────── */

/* Power iteration for dominant eigenvector */
sg_error sg_power_iteration(const sg_matrix *M, double *eigenvalue,
                             double *eigenvector, uint32_t max_iter, double tol);

/* Full QR-based eigendecomposition (for small matrices) */
sg_spectrum *sg_eigendecompose(const sg_matrix *M);

void sg_spectrum_destroy(sg_spectrum *s);

/* ── Core spectral analyses ──────────────────────────────────── */

sg_fiedler_result   *sg_compute_fiedler(const sg_graph *g);
sg_cheeger_result   *sg_compute_cheeger(const sg_graph *g);
sg_mixing_result    *sg_compute_mixing_time(const sg_graph *g);
sg_cluster_result   *sg_spectral_cluster(const sg_graph *g, uint32_t k);
sg_centrality_result *sg_compute_centrality(const sg_graph *g);
sg_robustness_result *sg_compute_robustness(const sg_graph *g);
sg_expander_result  *sg_compute_expander_quality(const sg_graph *g);

void sg_fiedler_destroy(sg_fiedler_result *r);
void sg_cheeger_destroy(sg_cheeger_result *r);
void sg_mixing_destroy(sg_mixing_result *r);
void sg_cluster_destroy(sg_cluster_result *r);
void sg_centrality_destroy(sg_centrality_result *r);
void sg_robustness_destroy(sg_robustness_result *r);
void sg_expander_destroy(sg_expander_result *r);

/* ── Sheaf integration (from evolving-sheaf-c) ───────────────── */
typedef struct {
    double spectral_gap_healthy;     /* baseline spectral gap */
    double spectral_gap_current;     /* current observed gap */
    double degradation_pct;          /* % degradation */
    bool   communication_risk;       /* true if gap degraded > threshold */
    double failure_probability;      /* estimated comm failure prob */
} sg_sheaf_prediction;

sg_sheaf_prediction sg_sheaf_predict_failures(const sg_graph *g,
                                               double baseline_gap);

/* ── Ergodic/transport integration ───────────────────────────── */
typedef struct {
    double mixing_time_graph;
    double convergence_time_ergodic;  /* from ergodic theory */
    double correlation;               /* correlation between graph mixing and ergodic convergence */
} sg_ergodic_correlation;

sg_ergodic_correlation sg_ergodic_correlate(const sg_graph *g);

/* ── Utility ─────────────────────────────────────────────────── */
double sg_matrix_trace(const sg_matrix *M);
uint32_t sg_graph_degree(const sg_graph *g, uint32_t v);
uint32_t sg_graph_edge_count(const sg_graph *g);
bool     sg_graph_is_connected(const sg_graph *g);
bool     sg_graph_is_regular(const sg_graph *g, uint32_t *degree);
double   sg_conductance(const sg_graph *g, const uint32_t *S, uint32_t size_S);
double   sg_edge_connectivity(const sg_graph *g);

const char *sg_error_string(sg_error err);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_GRAPH_H */
