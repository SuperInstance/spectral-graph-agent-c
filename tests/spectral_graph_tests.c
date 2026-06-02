/*
 * spectral_graph_tests.c — 60+ tests for spectral graph theory on agent networks
 *
 * Compile: gcc -O2 -lm -o test_spectral tests/spectral_graph_tests.c src/spectral_graph.c -Iinclude
 * Run:     ./test_spectral
 */

#include "spectral_graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int g_pass = 0, g_fail = 0, g_total = 0;

#define TEST(name) static void test_##name(void)
#define RUN(name) do { printf("  %-50s", #name); test_##name(); } while(0)

#define ASSERT_TRUE(cond) do { \
    g_total++; \
    if (cond) { g_pass++; printf("PASS\n"); } \
    else { g_fail++; printf("FAIL  [%s:%d]\n", __FILE__, __LINE__); } \
} while(0)

#define ASSERT_EQ_DBL(a, b, eps) ASSERT_TRUE(fabs((a)-(b)) < (eps))
#define ASSERT_EQ_INT(a, b)      ASSERT_TRUE((a) == (b))
#define ASSERT_NEQ_DBL(a, b, eps) ASSERT_TRUE(fabs((a)-(b)) >= (eps))
#define ASSERT_NOT_NULL(p)       ASSERT_TRUE((p) != NULL)
#define ASSERT_NULL(p)           ASSERT_TRUE((p) == NULL)

#define EPS 1e-6
#define EPS_LOOSE 1e-3

/* ═══════════════════════════════════════════════════════════════
   1. GRAPH CONSTRUCTION TESTS
   ═══════════════════════════════════════════════════════════════ */

TEST(graph_create_destroy) {
    sg_graph *g = sg_graph_create(5, false);
    ASSERT_NOT_NULL(g);
    ASSERT_EQ_INT(g->n, 5);
    sg_graph_destroy(g);
}

TEST(graph_create_zero) {
    sg_graph *g = sg_graph_create(0, false);
    ASSERT_NOT_NULL(g);
    sg_graph_destroy(g);
}

TEST(graph_null_destroy) {
    sg_graph_destroy(NULL);  /* should not crash */
    ASSERT_TRUE(1);
}

TEST(graph_add_edge_basic) {
    sg_graph *g = sg_graph_create(4, false);
    sg_error err = sg_graph_add_edge(g, 0, 1, 1.0);
    ASSERT_EQ_INT(err, SG_OK);
    sg_graph_build_path(g, 0); /* finalize */
    sg_graph_destroy(g);
}

TEST(graph_add_edge_out_of_bounds) {
    sg_graph *g = sg_graph_create(3, false);
    sg_error err = sg_graph_add_edge(g, 0, 5, 1.0);
    ASSERT_EQ_INT(err, SG_ERR_PARAM);
    sg_graph_destroy(g);
}

TEST(graph_null_add_edge) {
    sg_error err = sg_graph_add_edge(NULL, 0, 1, 1.0);
    ASSERT_EQ_INT(err, SG_ERR_NULL);
}

TEST(graph_build_path) {
    sg_graph *g = sg_graph_create(5, false);
    sg_error err = sg_graph_build_path(g, 1.0);
    ASSERT_EQ_INT(err, SG_OK);
    ASSERT_EQ_INT(sg_graph_edge_count(g), 4);
    ASSERT_TRUE(sg_graph_is_connected(g));
    sg_graph_destroy(g);
}

TEST(graph_build_cycle) {
    sg_graph *g = sg_graph_create(6, false);
    sg_error err = sg_graph_build_cycle(g, 1.0);
    ASSERT_EQ_INT(err, SG_OK);
    ASSERT_EQ_INT(sg_graph_edge_count(g), 6);
    ASSERT_TRUE(sg_graph_is_connected(g));
    sg_graph_destroy(g);
}

TEST(graph_build_complete) {
    sg_graph *g = sg_graph_create(4, false);
    sg_error err = sg_graph_build_complete(g, 1.0);
    ASSERT_EQ_INT(err, SG_OK);
    ASSERT_EQ_INT(sg_graph_edge_count(g), 6); /* n(n-1)/2 */
    ASSERT_TRUE(sg_graph_is_connected(g));
    sg_graph_destroy(g);
}

TEST(graph_build_star) {
    sg_graph *g = sg_graph_create(5, false);
    sg_error err = sg_graph_build_star(g, 1.0);
    ASSERT_EQ_INT(err, SG_OK);
    ASSERT_EQ_INT(sg_graph_edge_count(g), 4);
    ASSERT_TRUE(sg_graph_is_connected(g));
    ASSERT_EQ_INT(sg_graph_degree(g, 0), 4); /* center */
    ASSERT_EQ_INT(sg_graph_degree(g, 1), 1); /* leaf */
    sg_graph_destroy(g);
}

TEST(graph_build_grid2d) {
    sg_graph *g = sg_graph_create(6, false); /* 2x3 */
    sg_error err = sg_graph_build_grid2d(g, 2, 3, 1.0);
    ASSERT_EQ_INT(err, SG_OK);
    ASSERT_TRUE(sg_graph_is_connected(g));
    /* 2x3 grid: (2*(3-1) + 3*(2-1)) = 4+3 = 7 edges */
    ASSERT_EQ_INT(sg_graph_edge_count(g), 7);
    sg_graph_destroy(g);
}

TEST(graph_build_grid2d_bad_size) {
    sg_graph *g = sg_graph_create(7, false);
    sg_error err = sg_graph_build_grid2d(g, 2, 3, 1.0); /* 6 != 7 */
    ASSERT_EQ_INT(err, SG_ERR_PARAM);
    sg_graph_destroy(g);
}

TEST(graph_build_random) {
    sg_graph *g = sg_graph_create(10, false);
    sg_error err = sg_graph_build_random(g, 0.5, 42);
    ASSERT_EQ_INT(err, SG_OK);
    uint32_t ec = sg_graph_edge_count(g);
    ASSERT_TRUE(ec > 0 && ec <= 45); /* max 10*9/2 */
    sg_graph_destroy(g);
}

/* ═══════════════════════════════════════════════════════════════
   2. MATRIX OPERATION TESTS
   ═══════════════════════════════════════════════════════════════ */

TEST(adjacency_matrix_complete) {
    sg_graph *g = sg_graph_create(3, false);
    sg_graph_build_complete(g, 1.0);
    sg_matrix *A = sg_graph_adjacency_matrix(g);
    ASSERT_NOT_NULL(A);
    /* Complete graph: all off-diagonal = 1 */
    ASSERT_EQ_DBL(A->data[0*3+1], 1.0, EPS);
    ASSERT_EQ_DBL(A->data[1*3+0], 1.0, EPS);
    ASSERT_EQ_DBL(A->data[0*3+0], 0.0, EPS); /* diagonal */
    sg_matrix_destroy(A);
    sg_graph_destroy(g);
}

TEST(laplacian_complete) {
    sg_graph *g = sg_graph_create(3, false);
    sg_graph_build_complete(g, 1.0);
    sg_matrix *L = sg_graph_laplacian(g);
    ASSERT_NOT_NULL(L);
    /* Diagonal = degree = n-1 = 2 */
    ASSERT_EQ_DBL(L->data[0*3+0], 2.0, EPS);
    /* Off-diagonal = -1 */
    ASSERT_EQ_DBL(L->data[0*3+1], -1.0, EPS);
    /* Row sums = 0 */
    for (uint32_t i = 0; i < 3; i++) {
        double rs = 0;
        for (uint32_t j = 0; j < 3; j++) rs += L->data[i*3+j];
        ASSERT_EQ_DBL(rs, 0.0, EPS);
    }
    sg_matrix_destroy(L);
    sg_graph_destroy(g);
}

TEST(normalized_laplacian_complete) {
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_build_complete(g, 1.0);
    sg_matrix *Ln = sg_graph_normalized_laplacian(g);
    ASSERT_NOT_NULL(Ln);
    /* For K_n: L_norm has diagonal 1, off-diagonal -1/(n-1) */
    ASSERT_EQ_DBL(Ln->data[0*4+0], 1.0, EPS);
    ASSERT_EQ_DBL(Ln->data[0*4+1], -1.0/3.0, EPS_LOOSE);
    sg_matrix_destroy(Ln);
    sg_graph_destroy(g);
}

TEST(matrix_trace) {
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_build_complete(g, 1.0);
    sg_matrix *L = sg_graph_laplacian(g);
    /* Trace of Laplacian = sum of degrees = 2|E| */
    ASSERT_EQ_DBL(sg_matrix_trace(L), 12.0, EPS); /* 4*3 */
    sg_matrix_destroy(L);
    sg_graph_destroy(g);
}

TEST(null_matrix_ops) {
    ASSERT_NULL(sg_graph_adjacency_matrix(NULL));
    ASSERT_NULL(sg_graph_laplacian(NULL));
    ASSERT_NULL(sg_graph_normalized_laplacian(NULL));
    ASSERT_TRUE(sg_matrix_trace(NULL) == 0);
}

/* ═══════════════════════════════════════════════════════════════
   3. GRAPH PROPERTY TESTS
   ═══════════════════════════════════════════════════════════════ */

TEST(graph_degree) {
    sg_graph *g = sg_graph_create(5, false);
    sg_graph_build_star(g, 1.0);
    ASSERT_EQ_INT(sg_graph_degree(g, 0), 4); /* center */
    ASSERT_EQ_INT(sg_graph_degree(g, 4), 1); /* leaf */
    ASSERT_EQ_INT(sg_graph_degree(g, 99), 0); /* out of range */
    sg_graph_destroy(g);
}

TEST(graph_edge_count) {
    sg_graph *g = sg_graph_create(5, false);
    sg_graph_build_cycle(g, 1.0);
    ASSERT_EQ_INT(sg_graph_edge_count(g), 5);
    sg_graph_destroy(g);
}

TEST(graph_is_connected_path) {
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_build_path(g, 1.0);
    ASSERT_TRUE(sg_graph_is_connected(g));
    sg_graph_destroy(g);
}

TEST(graph_is_connected_disconnected) {
    /* Build a graph with no edges */
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_finalize(g);
    ASSERT_TRUE(!sg_graph_is_connected(g));
    sg_graph_destroy(g);
}

TEST(graph_is_regular_cycle) {
    sg_graph *g = sg_graph_create(6, false);
    sg_graph_build_cycle(g, 1.0);
    uint32_t deg = 0;
    bool reg = sg_graph_is_regular(g, &deg);
    ASSERT_TRUE(reg);
    ASSERT_EQ_INT(deg, 2);
    sg_graph_destroy(g);
}

TEST(graph_is_regular_star_not) {
    sg_graph *g = sg_graph_create(5, false);
    sg_graph_build_star(g, 1.0);
    uint32_t deg = 0;
    bool reg = sg_graph_is_regular(g, &deg);
    ASSERT_TRUE(!reg);
    sg_graph_destroy(g);
}

/* Expose internal helpers for testing */
extern sg_matrix *sg_matrix_create(uint32_t n);

/* ═══════════════════════════════════════════════════════════════
   4. EIGENDECOMPOSITION TESTS
   ═══════════════════════════════════════════════════════════════ */

TEST(power_iteration_complete) {
    sg_graph *g = sg_graph_create(3, false);
    sg_graph_build_complete(g, 1.0);
    sg_matrix *A = sg_graph_adjacency_matrix(g);
    double ev = 0;
    double vec[3] = {0};
    sg_error err = sg_power_iteration(A, &ev, vec, 100, 1e-10);
    ASSERT_EQ_INT(err, SG_OK);
    /* Dominant eigenvalue of K_3 adjacency = 2 */
    ASSERT_EQ_DBL(ev, 2.0, EPS_LOOSE);
    sg_matrix_destroy(A);
    sg_graph_destroy(g);
}

TEST(eigendecompose_identity) {
    sg_matrix *M = sg_matrix_create(3);
    for (uint32_t i = 0; i < 3; i++) M->data[i*3+i] = 1.0;
    sg_spectrum *s = sg_eigendecompose(M);
    ASSERT_NOT_NULL(s);
    /* All eigenvalues should be 1 */
    for (uint32_t i = 0; i < 3; i++)
        ASSERT_EQ_DBL(s->eigenvalues[i], 1.0, EPS_LOOSE);
    sg_spectrum_destroy(s);
    sg_matrix_destroy(M);
}

TEST(eigendecompose_laplacian_path) {
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_build_path(g, 1.0);
    sg_matrix *L = sg_graph_laplacian(g);
    sg_spectrum *s = sg_eigendecompose(L);
    ASSERT_NOT_NULL(s);
    /* Smallest eigenvalue of Laplacian = 0 */
    ASSERT_EQ_DBL(s->eigenvalues[0], 0.0, EPS_LOOSE);
    /* Second eigenvalue > 0 */
    ASSERT_TRUE(s->eigenvalues[1] > EPS);
    sg_spectrum_destroy(s);
    sg_matrix_destroy(L);
    sg_graph_destroy(g);
}

TEST(eigendecompose_null) {
    ASSERT_NULL(sg_eigendecompose(NULL));
}

TEST(power_iteration_null) {
    double ev = 0, vec[3] = {0};
    ASSERT_EQ_INT(sg_power_iteration(NULL, &ev, vec, 100, 1e-10), SG_ERR_NULL);
}

/* ═══════════════════════════════════════════════════════════════
   5. FIEDLER VALUE (ALGEBRAIC CONNECTIVITY) TESTS
   ═══════════════════════════════════════════════════════════════ */

TEST(fiedler_complete_graph) {
    sg_graph *g = sg_graph_create(5, false);
    sg_graph_build_complete(g, 1.0);
    sg_fiedler_result *r = sg_compute_fiedler(g);
    ASSERT_NOT_NULL(r);
    /* For K_n: λ₂ = n */
    ASSERT_EQ_DBL(r->fiedler_value, 5.0, EPS_LOOSE);
    sg_fiedler_destroy(r);
    sg_graph_destroy(g);
}

TEST(fiedler_path_graph) {
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_build_path(g, 1.0);
    sg_fiedler_result *r = sg_compute_fiedler(g);
    ASSERT_NOT_NULL(r);
    /* Path P4: λ₂ ≈ 0.5858 = 2 - 2cos(π/n) for path */
    ASSERT_TRUE(r->fiedler_value > 0);
    ASSERT_TRUE(r->fiedler_value < 1.0);
    sg_fiedler_destroy(r);
    sg_graph_destroy(g);
}

TEST(fiedler_cycle_graph) {
    sg_graph *g = sg_graph_create(6, false);
    sg_graph_build_cycle(g, 1.0);
    sg_fiedler_result *r = sg_compute_fiedler(g);
    ASSERT_NOT_NULL(r);
    /* Cycle C_n: λ₂ = 2 - 2cos(2π/n) */
    double expected = 2.0 - 2.0 * cos(2.0 * M_PI / 6.0);
    ASSERT_EQ_DBL(r->fiedler_value, expected, EPS_LOOSE);
    sg_fiedler_destroy(r);
    sg_graph_destroy(g);
}

TEST(fiedler_star_graph) {
    sg_graph *g = sg_graph_create(5, false);
    sg_graph_build_star(g, 1.0);
    sg_fiedler_result *r = sg_compute_fiedler(g);
    ASSERT_NOT_NULL(r);
    /* Star: λ₂ = 1 */
    ASSERT_EQ_DBL(r->fiedler_value, 1.0, EPS_LOOSE);
    sg_fiedler_destroy(r);
    sg_graph_destroy(g);
}

TEST(fiedler_null) {
    ASSERT_NULL(sg_compute_fiedler(NULL));
}

TEST(fiedler_vector_properties) {
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_build_cycle(g, 1.0);
    sg_fiedler_result *r = sg_compute_fiedler(g);
    ASSERT_NOT_NULL(r);
    /* Fiedler vector should be orthogonal to all-ones */
    double sum = 0;
    for (uint32_t i = 0; i < 4; i++) sum += r->fiedler_vector[i];
    ASSERT_EQ_DBL(sum, 0.0, EPS_LOOSE);
    sg_fiedler_destroy(r);
    sg_graph_destroy(g);
}

/* ═══════════════════════════════════════════════════════════════
   6. CHEEGER CONSTANT TESTS
   ═══════════════════════════════════════════════════════════════ */

TEST(cheeger_complete) {
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_build_complete(g, 1.0);
    sg_cheeger_result *r = sg_compute_cheeger(g);
    ASSERT_NOT_NULL(r);
    /* K_4: h(G) ≥ (4-1)/4 * ... approx high, close to 3/4 */
    ASSERT_TRUE(r->cheeger_constant > 0);
    sg_cheeger_destroy(r);
    sg_graph_destroy(g);
}

TEST(cheeger_path) {
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_build_path(g, 1.0);
    sg_cheeger_result *r = sg_compute_cheeger(g);
    ASSERT_NOT_NULL(r);
    /* Path has small Cheeger constant */
    ASSERT_TRUE(r->cheeger_constant > 0);
    ASSERT_TRUE(r->cheeger_constant < 1.0);
    sg_cheeger_destroy(r);
    sg_graph_destroy(g);
}

TEST(cheeger_null) {
    ASSERT_NULL(sg_compute_cheeger(NULL));
}

TEST(cheeger_cut_set_valid) {
    sg_graph *g = sg_graph_create(6, false);
    sg_graph_build_cycle(g, 1.0);
    sg_cheeger_result *r = sg_compute_cheeger(g);
    ASSERT_NOT_NULL(r);
    ASSERT_TRUE(r->cut_size > 0 && r->cut_size < 6);
    ASSERT_NOT_NULL(r->cut_set);
    sg_cheeger_destroy(r);
    sg_graph_destroy(g);
}

/* ═══════════════════════════════════════════════════════════════
   7. MIXING TIME TESTS
   ═══════════════════════════════════════════════════════════════ */

TEST(mixing_complete) {
    sg_graph *g = sg_graph_create(5, false);
    sg_graph_build_complete(g, 1.0);
    sg_mixing_result *r = sg_compute_mixing_time(g);
    ASSERT_NOT_NULL(r);
    /* Complete graph should mix fast */
    ASSERT_TRUE(r->mixing_time < 10);
    ASSERT_TRUE(r->spectral_gap > 0);
    sg_mixing_destroy(r);
    sg_graph_destroy(g);
}

TEST(mixing_path_slow) {
    sg_graph *g = sg_graph_create(20, false);
    sg_graph_build_path(g, 1.0);
    sg_mixing_result *r = sg_compute_mixing_time(g);
    ASSERT_NOT_NULL(r);
    /* Path should have larger mixing time than complete */
    ASSERT_TRUE(r->mixing_time > 0);
    sg_mixing_destroy(r);
    sg_graph_destroy(g);
}

TEST(mixing_null) {
    ASSERT_NULL(sg_compute_mixing_time(NULL));
}

/* ═══════════════════════════════════════════════════════════════
   8. SPECTRAL CLUSTERING TESTS
   ═══════════════════════════════════════════════════════════════ */

TEST(cluster_two_clusters) {
    /* Build a graph with two natural clusters */
    sg_graph *g = sg_graph_create(6, false);
    /* Cluster 1: 0-1-2 fully connected */
    sg_graph_add_edge(g, 0, 1, 1.0);
    sg_graph_add_edge(g, 0, 2, 1.0);
    sg_graph_add_edge(g, 1, 2, 1.0);
    /* Cluster 2: 3-4-5 fully connected */
    sg_graph_add_edge(g, 3, 4, 1.0);
    sg_graph_add_edge(g, 3, 5, 1.0);
    sg_graph_add_edge(g, 4, 5, 1.0);
    /* Bridge: single edge */
    sg_graph_add_edge(g, 2, 3, 1.0);
    sg_graph_finalize(g);

    sg_cluster_result *r = sg_spectral_cluster(g, 2);
    ASSERT_NOT_NULL(r);
    ASSERT_EQ_INT(r->num_clusters, 2);
    /* Vertices in same cluster should share cluster ID */
    ASSERT_EQ_INT(r->cluster_ids[0], r->cluster_ids[1]);
    ASSERT_EQ_INT(r->cluster_ids[3], r->cluster_ids[4]);
    /* Vertices in different clusters should differ */
    ASSERT_TRUE(r->cluster_ids[0] != r->cluster_ids[3]);
    sg_cluster_destroy(r);
    sg_graph_destroy(g);
}

TEST(cluster_one_cluster) {
    sg_graph *g = sg_graph_create(3, false);
    sg_graph_build_complete(g, 1.0);
    sg_cluster_result *r = sg_spectral_cluster(g, 1);
    ASSERT_NOT_NULL(r);
    ASSERT_EQ_INT(r->num_clusters, 1);
    for (uint32_t i = 0; i < 3; i++)
        ASSERT_EQ_INT(r->cluster_ids[i], 0);
    sg_cluster_destroy(r);
    sg_graph_destroy(g);
}

TEST(cluster_bad_k) {
    sg_graph *g = sg_graph_create(3, false);
    ASSERT_NULL(sg_spectral_cluster(g, 0));
    ASSERT_NULL(sg_spectral_cluster(g, 5));
    sg_graph_destroy(g);
}

TEST(cluster_null) {
    ASSERT_NULL(sg_spectral_cluster(NULL, 2));
}

TEST(cluster_modularity) {
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_build_cycle(g, 1.0);
    sg_cluster_result *r = sg_spectral_cluster(g, 2);
    ASSERT_NOT_NULL(r);
    /* Modularity should be defined */
    ASSERT_TRUE(r->modularity >= -1.0 && r->modularity <= 1.0);
    sg_cluster_destroy(r);
    sg_graph_destroy(g);
}

/* ═══════════════════════════════════════════════════════════════
   9. EIGENVECTOR CENTRALITY TESTS
   ═══════════════════════════════════════════════════════════════ */

TEST(centrality_star_center_highest) {
    sg_graph *g = sg_graph_create(5, false);
    sg_graph_build_star(g, 1.0);
    sg_centrality_result *r = sg_compute_centrality(g);
    ASSERT_NOT_NULL(r);
    /* Center of star should have highest centrality */
    ASSERT_EQ_INT(r->dominant_agent, 0);
    ASSERT_EQ_DBL(r->max_centrality, 1.0, EPS);
    ASSERT_TRUE(r->centrality[0] >= r->centrality[1]);
    sg_centrality_destroy(r);
    sg_graph_destroy(g);
}

TEST(centrality_complete_equal) {
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_build_complete(g, 1.0);
    sg_centrality_result *r = sg_compute_centrality(g);
    ASSERT_NOT_NULL(r);
    /* Complete graph: all centrality values equal */
    for (uint32_t i = 1; i < 4; i++)
        ASSERT_EQ_DBL(r->centrality[i], r->centrality[0], EPS_LOOSE);
    sg_centrality_destroy(r);
    sg_graph_destroy(g);
}

TEST(centrality_null) {
    ASSERT_NULL(sg_compute_centrality(NULL));
}

TEST(centrality_normalized) {
    sg_graph *g = sg_graph_create(6, false);
    sg_graph_build_cycle(g, 1.0);
    sg_centrality_result *r = sg_compute_centrality(g);
    ASSERT_NOT_NULL(r);
    /* All values should be between 0 and 1 */
    for (uint32_t i = 0; i < 6; i++) {
        ASSERT_TRUE(r->centrality[i] >= 0.0);
        ASSERT_TRUE(r->centrality[i] <= 1.0 + EPS);
    }
    sg_centrality_destroy(r);
    sg_graph_destroy(g);
}

/* ═══════════════════════════════════════════════════════════════
   10. ROBUSTNESS TESTS
   ═══════════════════════════════════════════════════════════════ */

TEST(robustness_complete) {
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_build_complete(g, 1.0);
    sg_robustness_result *r = sg_compute_robustness(g);
    ASSERT_NOT_NULL(r);
    ASSERT_TRUE(r->robustness_score > 0);
    ASSERT_EQ_INT(r->articulation_points, 0); /* K_n has no articulation points */
    sg_robustness_destroy(r);
    sg_graph_destroy(g);
}

TEST(robustness_path) {
    sg_graph *g = sg_graph_create(5, false);
    sg_graph_build_path(g, 1.0);
    sg_robustness_result *r = sg_compute_robustness(g);
    ASSERT_NOT_NULL(r);
    /* Path has many articulation points */
    ASSERT_TRUE(r->articulation_points > 0);
    sg_robustness_destroy(r);
    sg_graph_destroy(g);
}

TEST(robustness_star_low) {
    sg_graph *g = sg_graph_create(5, false);
    sg_graph_build_star(g, 1.0);
    sg_robustness_result *r = sg_compute_robustness(g);
    ASSERT_NOT_NULL(r);
    /* Star: center is articulation point */
    ASSERT_TRUE(r->articulation_points >= 1);
    sg_robustness_destroy(r);
    sg_graph_destroy(g);
}

TEST(robustness_null) {
    ASSERT_NULL(sg_compute_robustness(NULL));
}

/* ═══════════════════════════════════════════════════════════════
   11. EXPANDER QUALITY TESTS
   ═══════════════════════════════════════════════════════════════ */

TEST(expander_complete) {
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_build_complete(g, 1.0);
    sg_expander_result *r = sg_compute_expander_quality(g);
    ASSERT_NOT_NULL(r);
    ASSERT_TRUE(r->expansion_ratio > 0);
    ASSERT_TRUE(r->degree == 3); /* K_4 is 3-regular */
    sg_expander_destroy(r);
    sg_graph_destroy(g);
}

TEST(expander_cycle) {
    sg_graph *g = sg_graph_create(6, false);
    sg_graph_build_cycle(g, 1.0);
    sg_expander_result *r = sg_compute_expander_quality(g);
    ASSERT_NOT_NULL(r);
    ASSERT_TRUE(r->degree == 2); /* cycle is 2-regular */
    ASSERT_TRUE(r->ramanujan_bound > 0);
    sg_expander_destroy(r);
    sg_graph_destroy(g);
}

TEST(expander_null) {
    ASSERT_NULL(sg_compute_expander_quality(NULL));
}

TEST(expander_ramanujan_bound) {
    /* For 3-regular: Ramanujan bound = 2√(3-1) = 2√2 ≈ 2.828 */
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_build_complete(g, 1.0);
    sg_expander_result *r = sg_compute_expander_quality(g);
    ASSERT_NOT_NULL(r);
    ASSERT_EQ_DBL(r->ramanujan_bound, 2.0 * sqrt(2.0), EPS_LOOSE);
    sg_expander_destroy(r);
    sg_graph_destroy(g);
}

/* ═══════════════════════════════════════════════════════════════
   12. SHEAF INTEGRATION TESTS
   ═══════════════════════════════════════════════════════════════ */

TEST(sheaf_healthy_network) {
    sg_graph *g = sg_graph_create(5, false);
    sg_graph_build_complete(g, 1.0);
    sg_sheaf_prediction p = sg_sheaf_predict_failures(g, 1.0);
    ASSERT_TRUE(p.spectral_gap_current > 0);
    ASSERT_TRUE(p.degradation_pct >= 0);
    /* For healthy complete graph, degradation should be reasonable */
    sg_graph_destroy(g);
}

TEST(sheaf_degraded_network) {
    sg_graph *g = sg_graph_create(10, false);
    sg_graph_build_path(g, 1.0);
    /* Use a high baseline as if this were once a complete graph */
    sg_sheaf_prediction p = sg_sheaf_predict_failures(g, 5.0);
    ASSERT_TRUE(p.degradation_pct > 30.0); /* significant degradation */
    ASSERT_TRUE(p.communication_risk);
    sg_graph_destroy(g);
}

TEST(sheaf_null_graph) {
    sg_sheaf_prediction p = sg_sheaf_predict_failures(NULL, 1.0);
    ASSERT_EQ_DBL(p.spectral_gap_current, 0.0, EPS);
}

/* ═══════════════════════════════════════════════════════════════
   13. ERGODIC INTEGRATION TESTS
   ═══════════════════════════════════════════════════════════════ */

TEST(ergodic_correlate_complete) {
    sg_graph *g = sg_graph_create(5, false);
    sg_graph_build_complete(g, 1.0);
    sg_ergodic_correlation c = sg_ergodic_correlate(g);
    ASSERT_TRUE(c.mixing_time_graph > 0);
    ASSERT_TRUE(c.convergence_time_ergodic > 0);
    ASSERT_TRUE(c.correlation > 0);
    sg_graph_destroy(g);
}

TEST(ergodic_correlate_null) {
    sg_ergodic_correlation c = sg_ergodic_correlate(NULL);
    ASSERT_EQ_DBL(c.mixing_time_graph, 0.0, EPS);
}

/* ═══════════════════════════════════════════════════════════════
   14. CONDUCTANCE UTILITY TESTS
   ═══════════════════════════════════════════════════════════════ */

TEST(conductance_simple) {
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_build_cycle(g, 1.0);
    /* S = {0,1}, edges across: (1,2), (0,3) = 2, vol(S)=4, vol(comp)=4 */
    uint32_t S[] = {0, 1};
    double cond = sg_conductance(g, S, 2);
    ASSERT_TRUE(cond > 0);
    ASSERT_TRUE(cond <= 1.0);
    sg_graph_destroy(g);
}

TEST(conductance_full_set) {
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_build_cycle(g, 1.0);
    uint32_t S[] = {0, 1, 2, 3};
    double cond = sg_conductance(g, S, 4);
    ASSERT_EQ_DBL(cond, 1.0, EPS); /* Full set → 1.0 */
    sg_graph_destroy(g);
}

/* ═══════════════════════════════════════════════════════════════
   15. EDGE CONNECTIVITY TESTS
   ═══════════════════════════════════════════════════════════════ */

TEST(edge_connectivity_cycle) {
    sg_graph *g = sg_graph_create(6, false);
    sg_graph_build_cycle(g, 1.0);
    double ec = sg_edge_connectivity(g);
    ASSERT_EQ_DBL(ec, 2.0, EPS); /* cycle: λ = 2 */
    sg_graph_destroy(g);
}

TEST(edge_connectivity_complete) {
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_build_complete(g, 1.0);
    double ec = sg_edge_connectivity(g);
    ASSERT_EQ_DBL(ec, 3.0, EPS); /* K_4: min degree = 3 */
    sg_graph_destroy(g);
}

/* ═══════════════════════════════════════════════════════════════
   16. ERROR STRING TESTS
   ═══════════════════════════════════════════════════════════════ */

TEST(error_strings) {
    ASSERT_TRUE(strcmp(sg_error_string(SG_OK), "OK") == 0);
    ASSERT_TRUE(strcmp(sg_error_string(SG_ERR_NULL), "NULL pointer") == 0);
    ASSERT_TRUE(strcmp(sg_error_string(SG_ERR_ALLOC), "allocation failure") == 0);
    ASSERT_TRUE(strcmp(sg_error_string(SG_ERR_NO_CONVERGE), "no convergence") == 0);
    ASSERT_TRUE(strcmp(sg_error_string(SG_ERR_PARAM), "invalid parameter") == 0);
}

/* ═══════════════════════════════════════════════════════════════
   17. INTEGRATION / COMPARISON TESTS
   ═══════════════════════════════════════════════════════════════ */

TEST(fiedler_vs_cheeger_bound) {
    /* Cheeger inequality: λ₂/2 ≤ h(G) ≤ √(2λ₂)
       Note: our conductance is volume-based (matching normalized Laplacian),
       so we use a relaxed bound that accounts for the degree factor. */
    sg_graph *g = sg_graph_create(8, false);
    sg_graph_build_cycle(g, 1.0);
    sg_fiedler_result *f = sg_compute_fiedler(g);
    sg_cheeger_result *c = sg_compute_cheeger(g);
    ASSERT_NOT_NULL(f);
    ASSERT_NOT_NULL(c);
    /* Relaxed bounds: h should be between 0 and 1, and positive */
    ASSERT_TRUE(c->cheeger_constant > 0);
    ASSERT_TRUE(c->cheeger_constant <= 1.0);
    /* Upper Cheeger bound should hold */
    double upper = sqrt(2.0 * f->fiedler_value) + EPS_LOOSE;
    ASSERT_TRUE(c->cheeger_constant <= upper);
    sg_fiedler_destroy(f);
    sg_cheeger_destroy(c);
    sg_graph_destroy(g);
}

TEST(more_connected_more_robust) {
    /* Complete graph should be more robust than path */
    sg_graph *g1 = sg_graph_create(5, false);
    sg_graph_build_complete(g1, 1.0);
    sg_robustness_result *r1 = sg_compute_robustness(g1);

    sg_graph *g2 = sg_graph_create(5, false);
    sg_graph_build_path(g2, 1.0);
    sg_robustness_result *r2 = sg_compute_robustness(g2);

    ASSERT_TRUE(r1->robustness_score > r2->robustness_score);
    sg_robustness_destroy(r1); sg_robustness_destroy(r2);
    sg_graph_destroy(g1); sg_graph_destroy(g2);
}

TEST(fiedler_monotone_with_edges) {
    /* Adding edges should not decrease algebraic connectivity */
    sg_graph *g1 = sg_graph_create(4, false);
    sg_graph_build_path(g1, 1.0);
    sg_fiedler_result *f1 = sg_compute_fiedler(g1);

    sg_graph *g2 = sg_graph_create(4, false);
    sg_graph_build_cycle(g2, 1.0);
    sg_fiedler_result *f2 = sg_compute_fiedler(g2);

    ASSERT_TRUE(f2->fiedler_value >= f1->fiedler_value - EPS_LOOSE);
    sg_fiedler_destroy(f1); sg_fiedler_destroy(f2);
    sg_graph_destroy(g1); sg_graph_destroy(g2);
}

TEST(mixing_time_decreases_with_connectivity) {
    sg_graph *g1 = sg_graph_create(6, false);
    sg_graph_build_path(g1, 1.0);
    sg_mixing_result *m1 = sg_compute_mixing_time(g1);

    sg_graph *g2 = sg_graph_create(6, false);
    sg_graph_build_complete(g2, 1.0);
    sg_mixing_result *m2 = sg_compute_mixing_time(g2);

    /* Complete should have shorter mixing time */
    ASSERT_TRUE(m2->mixing_time < m1->mixing_time);
    sg_mixing_destroy(m1); sg_mixing_destroy(m2);
    sg_graph_destroy(g1); sg_graph_destroy(g2);
}

TEST(expander_cycle_vs_complete) {
    /* Complete graph should be better expander than cycle */
    sg_graph *gc = sg_graph_create(6, false);
    sg_graph_build_cycle(gc, 1.0);
    sg_expander_result *ec = sg_compute_expander_quality(gc);

    sg_graph *gk = sg_graph_create(6, false);
    sg_graph_build_complete(gk, 1.0);
    sg_expander_result *ek = sg_compute_expander_quality(gk);

    ASSERT_TRUE(ek->expansion_ratio > ec->expansion_ratio);
    sg_expander_destroy(ec); sg_expander_destroy(ek);
    sg_graph_destroy(gc); sg_graph_destroy(gk);
}

/* ═══════════════════════════════════════════════════════════════
   18. MEMORY SAFETY / EDGE CASE TESTS
   ═══════════════════════════════════════════════════════════════ */

TEST(small_graph_two_nodes) {
    sg_graph *g = sg_graph_create(2, false);
    sg_graph_add_edge(g, 0, 1, 1.0);
    sg_graph_finalize(g);
    sg_fiedler_result *f = sg_compute_fiedler(g);
    ASSERT_NOT_NULL(f);
    ASSERT_EQ_DBL(f->fiedler_value, 2.0, EPS_LOOSE); /* 2-node path: λ₂ = 2 */
    sg_fiedler_destroy(f);
    sg_graph_destroy(g);
}

TEST(single_node_graph) {
    sg_graph *g = sg_graph_create(1, false);
    sg_graph_finalize(g);
    ASSERT_TRUE(sg_graph_is_connected(g));
    ASSERT_EQ_INT(sg_graph_edge_count(g), 0);
    ASSERT_EQ_INT(sg_graph_degree(g, 0), 0);
    sg_graph_destroy(g);
}

TEST(weighted_graph) {
    sg_graph *g = sg_graph_create(3, false);
    sg_graph_add_edge(g, 0, 1, 2.0);
    sg_graph_add_edge(g, 1, 2, 3.0);
    sg_graph_add_edge(g, 0, 2, 1.0);
    sg_graph_finalize(g);

    sg_matrix *A = sg_graph_adjacency_matrix(g);
    ASSERT_EQ_DBL(A->data[0*3+1], 2.0, EPS);
    ASSERT_EQ_DBL(A->data[1*3+2], 3.0, EPS);
    sg_matrix_destroy(A);
    sg_graph_destroy(g);
}

TEST(destroy_null_results) {
    sg_fiedler_destroy(NULL);
    sg_cheeger_destroy(NULL);
    sg_mixing_destroy(NULL);
    sg_cluster_destroy(NULL);
    sg_centrality_destroy(NULL);
    sg_robustness_destroy(NULL);
    sg_expander_destroy(NULL);
    sg_spectrum_destroy(NULL);
    sg_matrix_destroy(NULL);
    ASSERT_TRUE(1); /* no crash = pass */
}

/* ═══════════════════════════════════════════════════════════════
   19. REGRESSION TESTS FOR AUDIT BUGS
   ═══════════════════════════════════════════════════════════════ */

/* Bug #1: Power iteration on zero matrix should return error, not fake convergence */
TEST(regression_power_iter_zero_matrix) {
    sg_matrix *M = sg_matrix_create(3); /* all zeros */
    double ev = 999.0;
    double vec[3] = {0};
    sg_error err = sg_power_iteration(M, &ev, vec, 100, 1e-10);
    ASSERT_EQ_INT(err, SG_ERR_SINGULAR);
    ASSERT_EQ_DBL(ev, 0.0, EPS);  /* eigenvalue must be 0 */
    sg_matrix_destroy(M);
}

/* Bug #2: Centrality on graph with isolated vertices should not crash */
TEST(regression_centrality_isolated_vertices) {
    /* 4-node graph: 0-1 connected, 2 and 3 isolated */
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_add_edge(g, 0, 1, 1.0);
    sg_graph_finalize(g);
    sg_centrality_result *r = sg_compute_centrality(g);
    ASSERT_NOT_NULL(r);
    /* All values should be finite and in [0, 1] */
    for (uint32_t i = 0; i < 4; i++) {
        ASSERT_TRUE(r->centrality[i] >= 0.0);
        ASSERT_TRUE(r->centrality[i] <= 1.0 + EPS);
    }
    sg_centrality_destroy(r);
    sg_graph_destroy(g);
}

/* Bug #2: Centrality on completely isolated graph (all degree 0) */
TEST(regression_centrality_all_isolated) {
    sg_graph *g = sg_graph_create(3, false);
    sg_graph_finalize(g);
    sg_centrality_result *r = sg_compute_centrality(g);
    ASSERT_NOT_NULL(r);
    /* Should not crash, all centrality = 0 */
    for (uint32_t i = 0; i < 3; i++)
        ASSERT_EQ_DBL(r->centrality[i], 0.0, EPS);
    sg_centrality_destroy(r);
    sg_graph_destroy(g);
}

/* Bug #3: Mixing time for disconnected graph should be INFINITY */
TEST(regression_mixing_disconnected) {
    /* Two disconnected components */
    sg_graph *g = sg_graph_create(6, false);
    sg_graph_add_edge(g, 0, 1, 1.0);
    sg_graph_add_edge(g, 1, 2, 1.0);
    sg_graph_add_edge(g, 3, 4, 1.0);
    sg_graph_add_edge(g, 4, 5, 1.0);
    sg_graph_finalize(g);
    ASSERT_TRUE(!sg_graph_is_connected(g));
    sg_mixing_result *r = sg_compute_mixing_time(g);
    ASSERT_NOT_NULL(r);
    ASSERT_TRUE(r->mixing_time == INFINITY || r->mixing_time < 0);
    ASSERT_EQ_DBL(r->spectral_gap, 0.0, EPS);
    sg_mixing_destroy(r);
    sg_graph_destroy(g);
}

/* Bug #4: Conductance should use edge weights */
TEST(regression_conductance_weighted) {
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_add_edge(g, 0, 1, 5.0);  /* heavy edge */
    sg_graph_add_edge(g, 1, 2, 1.0);  /* light bridge */
    sg_graph_add_edge(g, 2, 3, 5.0);  /* heavy edge */
    sg_graph_add_edge(g, 3, 0, 1.0);  /* light bridge */
    sg_graph_finalize(g);

    /* S = {0,1}: cut edges are (1,2) and (0,3) with weights 1.0 + 1.0 = 2.0
       vol(S) = 6+6=12, vol(comp) = 6+6=12, cond = 2/12 = 1/6 */
    uint32_t S[] = {0, 1};
    double cond = sg_conductance(g, S, 2);
    ASSERT_TRUE(cond > 0);
    ASSERT_TRUE(fabs(cond - 1.0/6.0) < EPS_LOOSE);
    sg_graph_destroy(g);
}

/* Bug #5: Expander quality should be higher for better expanders */
TEST(regression_expander_quality_semantics) {
    /* Complete graph K_6 should have expander_quality >= 1.0 for Ramanujan */
    sg_graph *gk = sg_graph_create(6, false);
    sg_graph_build_complete(gk, 1.0);
    sg_expander_result *ek = sg_compute_expander_quality(gk);
    ASSERT_NOT_NULL(ek);
    /* For K_6 (5-regular): Ramanujan bound = 2√4 ≈ 4.0
       All eigenvalues of K_n besides n-1 are -1, so max_abs = 1.
       quality = bound/max_abs = 4.0/1.0 = 4.0 >= 1.0 */
    ASSERT_TRUE(ek->expander_quality >= 1.0 - EPS_LOOSE);
    sg_expander_destroy(ek);
    sg_graph_destroy(gk);
}

/* Bug #6: Spectral gap should use algebraic diff (λ₁ - λ₂), not absolute values */
TEST(regression_expander_spectral_gap) {
    sg_graph *g = sg_graph_create(4, false);
    sg_graph_build_complete(g, 1.0);
    sg_expander_result *r = sg_compute_expander_quality(g);
    ASSERT_NOT_NULL(r);
    /* K_4: eigenvalues of adjacency = [3, -1, -1, -1]
       spectral_gap = 3 - (-1) = 4 (algebraic diff) */
    ASSERT_TRUE(r->spectral_gap > 0);
    ASSERT_EQ_DBL(r->spectral_gap, 4.0, EPS_LOOSE);
    sg_expander_destroy(r);
    sg_graph_destroy(g);
}

/* Bug #7: Wilkinson shift should pick closer eigenvalue */
TEST(regression_eigendecompose_consistency) {
    /* Build a matrix with known eigenvalues and verify QR converges correctly */
    sg_matrix *M = sg_matrix_create(3);
    /* Diagonal matrix with eigenvalues 1, 5, 10 */
    M->data[0*3+0] = 1.0; M->data[0*3+1] = 0.5; M->data[0*3+2] = 0.0;
    M->data[1*3+0] = 0.5; M->data[1*3+1] = 5.0; M->data[1*3+2] = 0.3;
    M->data[2*3+0] = 0.0; M->data[2*3+1] = 0.3; M->data[2*3+2] = 10.0;
    sg_spectrum *s = sg_eigendecompose(M);
    ASSERT_NOT_NULL(s);
    /* Eigenvalues should be close to 1, 5, 10 (allowing some QR iteration error) */
    ASSERT_TRUE(fabs(s->eigenvalues[0] - 1.0) < 0.5);
    ASSERT_TRUE(fabs(s->eigenvalues[1] - 5.0) < 0.5);
    ASSERT_TRUE(fabs(s->eigenvalues[2] - 10.0) < 0.5);
    sg_spectrum_destroy(s);
    sg_matrix_destroy(M);
}

/* ═══════════════════════════════════════════════════════════════
   MAIN
   ═══════════════════════════════════════════════════════════════ */

int main(void) {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  spectral-graph-agent-c test suite\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    printf("── Graph Construction ───────────────────────────────────\n");
    RUN(graph_create_destroy);
    RUN(graph_create_zero);
    RUN(graph_null_destroy);
    RUN(graph_add_edge_basic);
    RUN(graph_add_edge_out_of_bounds);
    RUN(graph_null_add_edge);
    RUN(graph_build_path);
    RUN(graph_build_cycle);
    RUN(graph_build_complete);
    RUN(graph_build_star);
    RUN(graph_build_grid2d);
    RUN(graph_build_grid2d_bad_size);
    RUN(graph_build_random);

    printf("\n── Matrix Operations ──────────────────────────────────\n");
    RUN(adjacency_matrix_complete);
    RUN(laplacian_complete);
    RUN(normalized_laplacian_complete);
    RUN(matrix_trace);
    RUN(null_matrix_ops);

    printf("\n── Graph Properties ──────────────────────────────────\n");
    RUN(graph_degree);
    RUN(graph_edge_count);
    RUN(graph_is_connected_path);
    RUN(graph_is_connected_disconnected);
    RUN(graph_is_regular_cycle);
    RUN(graph_is_regular_star_not);

    printf("\n── Eigendecomposition ────────────────────────────────\n");
    RUN(power_iteration_complete);
    RUN(eigendecompose_identity);
    RUN(eigendecompose_laplacian_path);
    RUN(eigendecompose_null);
    RUN(power_iteration_null);

    printf("\n── Fiedler Value ─────────────────────────────────────\n");
    RUN(fiedler_complete_graph);
    RUN(fiedler_path_graph);
    RUN(fiedler_cycle_graph);
    RUN(fiedler_star_graph);
    RUN(fiedler_null);
    RUN(fiedler_vector_properties);

    printf("\n── Cheeger Constant ─────────────────────────────────\n");
    RUN(cheeger_complete);
    RUN(cheeger_path);
    RUN(cheeger_null);
    RUN(cheeger_cut_set_valid);

    printf("\n── Mixing Time ──────────────────────────────────────\n");
    RUN(mixing_complete);
    RUN(mixing_path_slow);
    RUN(mixing_null);

    printf("\n── Spectral Clustering ──────────────────────────────\n");
    RUN(cluster_two_clusters);
    RUN(cluster_one_cluster);
    RUN(cluster_bad_k);
    RUN(cluster_null);
    RUN(cluster_modularity);

    printf("\n── Eigenvector Centrality ───────────────────────────\n");
    RUN(centrality_star_center_highest);
    RUN(centrality_complete_equal);
    RUN(centrality_null);
    RUN(centrality_normalized);

    printf("\n── Robustness ───────────────────────────────────────\n");
    RUN(robustness_complete);
    RUN(robustness_path);
    RUN(robustness_star_low);
    RUN(robustness_null);

    printf("\n── Expander Quality ─────────────────────────────────\n");
    RUN(expander_complete);
    RUN(expander_cycle);
    RUN(expander_null);
    RUN(expander_ramanujan_bound);

    printf("\n── Sheaf Integration ────────────────────────────────\n");
    RUN(sheaf_healthy_network);
    RUN(sheaf_degraded_network);
    RUN(sheaf_null_graph);

    printf("\n── Ergodic Integration ──────────────────────────────\n");
    RUN(ergodic_correlate_complete);
    RUN(ergodic_correlate_null);

    printf("\n── Conductance ──────────────────────────────────────\n");
    RUN(conductance_simple);
    RUN(conductance_full_set);

    printf("\n── Edge Connectivity ────────────────────────────────\n");
    RUN(edge_connectivity_cycle);
    RUN(edge_connectivity_complete);

    printf("\n── Error Handling ───────────────────────────────────\n");
    RUN(error_strings);

    printf("\n── Integration / Comparison ─────────────────────────\n");
    RUN(fiedler_vs_cheeger_bound);
    RUN(more_connected_more_robust);
    RUN(fiedler_monotone_with_edges);
    RUN(mixing_time_decreases_with_connectivity);
    RUN(expander_cycle_vs_complete);

    printf("\n── Memory Safety / Edge Cases ───────────────────────\n");
    RUN(small_graph_two_nodes);
    RUN(single_node_graph);
    RUN(weighted_graph);
    RUN(destroy_null_results);

    printf("\n── Regression Tests (Audit Bugs) ─────────────────────\n");
    RUN(regression_power_iter_zero_matrix);
    RUN(regression_centrality_isolated_vertices);
    RUN(regression_centrality_all_isolated);
    RUN(regression_mixing_disconnected);
    RUN(regression_conductance_weighted);
    RUN(regression_expander_quality_semantics);
    RUN(regression_expander_spectral_gap);
    RUN(regression_eigendecompose_consistency);

    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("  Results: %d/%d passed, %d failed\n", g_pass, g_total, g_fail);
    printf("═══════════════════════════════════════════════════════════\n");

    return g_fail > 0 ? 1 : 0;
}
