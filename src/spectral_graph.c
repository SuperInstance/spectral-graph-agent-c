#include "spectral_graph.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

/* ══════════════════════════════════════════════════════════════ */
/*  Utility helpers                                            */
/* ══════════════════════════════════════════════════════════════ */

static double *vec_alloc(uint32_t n) { return (double *)calloc(n, sizeof(double)); }
static void    vec_zero(double *v, uint32_t n) { memset(v, 0, n * sizeof(double)); }
static double  vec_dot(const double *a, const double *b, uint32_t n) {
    double s = 0; for (uint32_t i = 0; i < n; i++) s += a[i]*b[i]; return s;
}
static double  vec_norm(const double *v, uint32_t n) { return sqrt(vec_dot(v, v, n)); }
static void    vec_normalize(double *v, uint32_t n) {
    double nm = vec_norm(v, n);
    if (nm > 1e-15) for (uint32_t i = 0; i < n; i++) v[i] /= nm;
}
static void    vec_scale(double *v, uint32_t n, double s) {
    for (uint32_t i = 0; i < n; i++) v[i] *= s;
}

/* Mat-vec: y = M*x, row-major */
static void mat_vec(const sg_matrix *M, const double *x, double *y) {
    for (uint32_t i = 0; i < M->n; i++) {
        double s = 0;
        for (uint32_t j = 0; j < M->n; j++)
            s += M->data[i * M->n + j] * x[j];
        y[i] = s;
    }
}

/* Dense matrix multiply C = A*B */
static void mat_mul(const sg_matrix *A, const sg_matrix *B, sg_matrix *C) {
    uint32_t n = A->n;
    for (uint32_t i = 0; i < n; i++)
        for (uint32_t j = 0; j < n; j++) {
            double s = 0;
            for (uint32_t k = 0; k < n; k++)
                s += A->data[i*n+k] * B->data[k*n+j];
            C->data[i*n+j] = s;
        }
}

const char *sg_error_string(sg_error err) {
    switch (err) {
        case SG_OK:              return "OK";
        case SG_ERR_NULL:        return "NULL pointer";
        case SG_ERR_ALLOC:       return "allocation failure";
        case SG_ERR_SINGULAR:    return "singular matrix";
        case SG_ERR_NO_CONVERGE: return "no convergence";
        case SG_ERR_PARAM:       return "invalid parameter";
        case SG_ERR_DISCONNECT:  return "disconnected graph";
        default:                 return "unknown error";
    }
}

/* ══════════════════════════════════════════════════════════════ */
/*  Graph lifecycle                                            */
/* ══════════════════════════════════════════════════════════════ */

/* Internal edge list for building */
typedef struct { uint32_t u, v; double w; } _edge;

struct _sg_build {
    _edge   *edges;
    uint32_t edge_cap;
    uint32_t edge_count;
    uint32_t n;
    bool     directed;
};

sg_graph *sg_graph_create(uint32_t n, bool directed) {
    sg_graph *g = (sg_graph *)calloc(1, sizeof(sg_graph));
    if (!g) return NULL;
    g->n = n;
    g->directed = directed;
    /* Allocate internal build state */
    struct _sg_build *b = (struct _sg_build *)calloc(1, sizeof(struct _sg_build));
    b->n = n; b->directed = directed;
    b->edge_cap = 64;
    b->edges = (_edge *)malloc(b->edge_cap * sizeof(_edge));
    b->edge_count = 0;
    /* Store in row_ptr temporarily */
    g->row_ptr = (uint32_t *)b;
    g->col_idx = NULL;
    g->weights = NULL;
    return g;
}

static void _finalize_csr(sg_graph *g) {
    struct _sg_build *b = (struct _sg_build *)g->row_ptr;
    uint32_t n = g->n;
    uint32_t ne = b->edge_count;

    /* Count degrees */
    uint32_t *deg = (uint32_t *)calloc(n, sizeof(uint32_t));
    for (uint32_t i = 0; i < ne; i++) deg[b->edges[i].u]++;

    /* Build CSR */
    uint32_t *rp = (uint32_t *)calloc(n + 1, sizeof(uint32_t));
    uint32_t *ci = (uint32_t *)calloc(ne, sizeof(uint32_t));
    double   *wt = (double *)calloc(ne, sizeof(double));

    for (uint32_t i = 0; i < n; i++) rp[i+1] = rp[i] + deg[i];
    uint32_t *pos = (uint32_t *)calloc(n, sizeof(uint32_t));
    for (uint32_t i = 0; i < ne; i++) {
        uint32_t u = b->edges[i].u;
        uint32_t idx = rp[u] + pos[u]++;
        ci[idx] = b->edges[i].v;
        wt[idx] = b->edges[i].w;
    }

    free(b->edges);
    free(b);
    free(deg);
    free(pos);

    g->row_ptr = rp;
    g->col_idx = ci;
    g->weights = wt;
}

void sg_graph_destroy(sg_graph *g) {
    if (!g) return;
    /* Check if still in build mode */
    if (!g->col_idx && g->row_ptr) {
        struct _sg_build *b = (struct _sg_build *)g->row_ptr;
        free(b->edges); free(b);
    } else {
        free(g->row_ptr);
        free(g->col_idx);
        free(g->weights);
    }
    free(g);
}

/* Internal: finalize if needed */
void sg_graph_finalize(sg_graph *g) {
    if (!g->col_idx && g->row_ptr) _finalize_csr(g);
}

sg_error sg_graph_add_edge(sg_graph *g, uint32_t u, uint32_t v, double w) {
    if (!g) return SG_ERR_NULL;
    if (u >= g->n || v >= g->n) return SG_ERR_PARAM;

    struct _sg_build *b = (struct _sg_build *)g->row_ptr;
    if (g->col_idx) return SG_ERR_PARAM; /* already finalized */

    if (b->edge_count >= b->edge_cap) {
        b->edge_cap *= 2;
        b->edges = (_edge *)realloc(b->edges, b->edge_cap * sizeof(_edge));
    }
    b->edges[b->edge_count++] = (_edge){u, v, w};
    if (!g->directed && u != v) {
        if (b->edge_count >= b->edge_cap) {
            b->edge_cap *= 2;
            b->edges = (_edge *)realloc(b->edges, b->edge_cap * sizeof(_edge));
        }
        b->edges[b->edge_count++] = (_edge){v, u, w};
    }
    return SG_OK;
}

/* Helper: build common graphs using add_edge */
static sg_graph *_build_simple(uint32_t n, bool directed) {
    return sg_graph_create(n, directed);
}

sg_error sg_graph_build_complete(sg_graph *g, double w) {
    if (!g) return SG_ERR_NULL;
    for (uint32_t i = 0; i < g->n; i++)
        for (uint32_t j = i+1; j < g->n; j++)
            sg_graph_add_edge(g, i, j, w);
    sg_graph_finalize(g);
    return SG_OK;
}

sg_error sg_graph_build_path(sg_graph *g, double w) {
    if (!g) return SG_ERR_NULL;
    for (uint32_t i = 0; i < g->n - 1; i++)
        sg_graph_add_edge(g, i, i+1, w);
    sg_graph_finalize(g);
    return SG_OK;
}

sg_error sg_graph_build_cycle(sg_graph *g, double w) {
    if (!g) return SG_ERR_NULL;
    for (uint32_t i = 0; i < g->n; i++)
        sg_graph_add_edge(g, i, (i+1) % g->n, w);
    sg_graph_finalize(g);
    return SG_OK;
}

sg_error sg_graph_build_star(sg_graph *g, double w) {
    if (!g) return SG_ERR_NULL;
    for (uint32_t i = 1; i < g->n; i++)
        sg_graph_add_edge(g, 0, i, w);
    sg_graph_finalize(g);
    return SG_OK;
}

sg_error sg_graph_build_grid2d(sg_graph *g, uint32_t rows, uint32_t cols, double w) {
    if (!g) return SG_ERR_NULL;
    if (rows * cols != g->n) return SG_ERR_PARAM;
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < cols; c++) {
            uint32_t v = r * cols + c;
            if (c + 1 < cols) sg_graph_add_edge(g, v, v + 1, w);
            if (r + 1 < rows) sg_graph_add_edge(g, v, v + cols, w);
        }
    sg_graph_finalize(g);
    return SG_OK;
}

/* Simple LCG random */
static uint32_t _lcg_rand(unsigned *seed) {
    *seed = *seed * 1103515245u + 12345u;
    return (*seed >> 16) & 0x7fff;
}

sg_error sg_graph_build_random(sg_graph *g, double edge_prob, unsigned seed) {
    if (!g) return SG_ERR_NULL;
    for (uint32_t i = 0; i < g->n; i++)
        for (uint32_t j = i+1; j < g->n; j++) {
            double r = (double)_lcg_rand(&seed) / 32768.0;
            if (r < edge_prob)
                sg_graph_add_edge(g, i, j, 1.0);
        }
    sg_graph_finalize(g);
    return SG_OK;
}

/* ══════════════════════════════════════════════════════════════ */
/*  Matrix operations                                          */
/* ══════════════════════════════════════════════════════════════ */

sg_matrix *sg_matrix_create(uint32_t n) {
    sg_matrix *m = (sg_matrix *)calloc(1, sizeof(sg_matrix));
    if (!m) return NULL;
    m->n = n;
    m->data = (double *)calloc((size_t)n * n, sizeof(double));
    if (!m->data) { free(m); return NULL; }
    return m;
}

void sg_matrix_destroy(sg_matrix *m) {
    if (!m) return;
    free(m->data);
    free(m);
}

sg_matrix *sg_graph_adjacency_matrix(const sg_graph *g) {
    if (!g) return NULL;
    sg_matrix *A = sg_matrix_create(g->n);
    if (!A) return NULL;
    for (uint32_t i = 0; i < g->n; i++)
        for (uint32_t j = g->row_ptr[i]; j < g->row_ptr[i+1]; j++) {
            double w = g->weights ? g->weights[j] : 1.0;
            A->data[i * g->n + g->col_idx[j]] = w;
        }
    return A;
}

sg_matrix *sg_graph_laplacian(const sg_graph *g) {
    if (!g) return NULL;
    sg_matrix *L = sg_matrix_create(g->n);
    if (!L) return NULL;
    for (uint32_t i = 0; i < g->n; i++) {
        double deg = 0;
        for (uint32_t j = g->row_ptr[i]; j < g->row_ptr[i+1]; j++) {
            double w = g->weights ? g->weights[j] : 1.0;
            L->data[i * g->n + g->col_idx[j]] = -w;
            deg += w;
        }
        L->data[i * g->n + i] = deg;
    }
    return L;
}

sg_matrix *sg_graph_normalized_laplacian(const sg_graph *g) {
    if (!g) return NULL;
    uint32_t n = g->n;
    /* Compute degrees */
    double *deg = vec_alloc(n);
    for (uint32_t i = 0; i < n; i++)
        for (uint32_t j = g->row_ptr[i]; j < g->row_ptr[i+1]; j++) {
            double w = g->weights ? g->weights[j] : 1.0;
            deg[i] += w;
        }
    /* D^{-1/2} */
    double *d_inv_sqrt = vec_alloc(n);
    for (uint32_t i = 0; i < n; i++)
        d_inv_sqrt[i] = (deg[i] > 1e-15) ? 1.0 / sqrt(deg[i]) : 0.0;

    sg_matrix *L = sg_matrix_create(n);
    if (!L) { free(deg); free(d_inv_sqrt); return NULL; }
    /* L_norm = I - D^{-1/2} A D^{-1/2} */
    for (uint32_t i = 0; i < n; i++) {
        L->data[i * n + i] = 1.0; /* identity */
        for (uint32_t j = g->row_ptr[i]; j < g->row_ptr[i+1]; j++) {
            uint32_t k = g->col_idx[j];
            double w = g->weights ? g->weights[j] : 1.0;
            L->data[i * n + k] -= d_inv_sqrt[i] * w * d_inv_sqrt[k];
        }
    }
    free(deg); free(d_inv_sqrt);
    return L;
}

double sg_matrix_trace(const sg_matrix *M) {
    if (!M) return 0;
    double t = 0;
    for (uint32_t i = 0; i < M->n; i++) t += M->data[i * M->n + i];
    return t;
}

/* ══════════════════════════════════════════════════════════════ */
/*  Eigendecomposition                                         */
/* ══════════════════════════════════════════════════════════════ */

/* In-place QR decomposition using Householder reflections */
static void _qr_decompose(double *A, double *Q, uint32_t n) {
    /* Copy A to Q initially, work in-place on A for R */
    memcpy(Q, A, (size_t)n * n * sizeof(double));
    /* Q starts as identity */
    memset(Q, 0, (size_t)n * n * sizeof(double));
    for (uint32_t i = 0; i < n; i++) Q[i * n + i] = 1.0;

    double *v = (double *)malloc(n * sizeof(double));

    for (uint32_t k = 0; k < n && k < n - 1; k++) {
        /* Extract column */
        double norm_x = 0;
        for (uint32_t i = k; i < n; i++) norm_x += A[i * n + k] * A[i * n + k];
        norm_x = sqrt(norm_x);
        if (norm_x < 1e-15) continue;

        double sign = (A[k * n + k] >= 0) ? 1.0 : -1.0;
        for (uint32_t i = 0; i < n; i++) v[i] = 0;
        for (uint32_t i = k; i < n; i++) v[i] = A[i * n + k];
        v[k] += sign * norm_x;

        double norm_v = sqrt(vec_dot(v, v, n));
        if (norm_v < 1e-15) continue;
        vec_scale(v, n, 1.0 / norm_v);

        /* Apply H = I - 2vv^T to A: A = A - 2v(v^T A) */
        for (uint32_t j = k; j < n; j++) {
            double dot = 0;
            for (uint32_t i = k; i < n; i++) dot += v[i] * A[i * n + j];
            for (uint32_t i = k; i < n; i++) A[i * n + j] -= 2.0 * v[i] * dot;
        }

        /* Update Q: Q = Q * H = Q - 2(Qv)v^T */
        for (uint32_t i = 0; i < n; i++) {
            double dot = 0;
            for (uint32_t j = k; j < n; j++) dot += Q[i * n + j] * v[j];
            for (uint32_t j = k; j < n; j++) Q[i * n + j] -= 2.0 * dot * v[j];
        }
    }
    free(v);
}

sg_spectrum *sg_eigendecompose(const sg_matrix *M) {
    if (!M) return NULL;
    uint32_t n = M->n;
    if (n == 0) return NULL;

    sg_spectrum *s = (sg_spectrum *)calloc(1, sizeof(sg_spectrum));
    s->n = n;
    s->eigenvalues = vec_alloc(n);
    s->eigenvectors = (double *)calloc((size_t)n * n, sizeof(double));

    /* Copy matrix for QR iteration */
    double *A = (double *)malloc((size_t)n * n * sizeof(double));
    memcpy(A, M->data, (size_t)n * n * sizeof(double));

    double *Q_total = (double *)calloc((size_t)n * n, sizeof(double));
    for (uint32_t i = 0; i < n; i++) Q_total[i * n + i] = 1.0;

    double *Q_step = (double *)malloc((size_t)n * n * sizeof(double));
    double *R = (double *)malloc((size_t)n * n * sizeof(double));

    uint32_t max_iter = 200 + n * 10;
    for (uint32_t iter = 0; iter < max_iter; iter++) {
        /* Wilkinson shift */
        double mu = 0;
        if (n >= 2) {
            double a = A[(n-2)*(n)+(n-2)];
            double b = A[(n-2)*(n)+(n-1)];
            double c = A[(n-1)*(n)+(n-2)];
            double d = A[(n-1)*(n)+(n-1)];
            double tr = a + d;
            double det = a*d - b*c;
            double disc = sqrt(fabs(tr*tr/4.0 - det));
            mu = tr/2.0 + disc;  /* pick closer eigenvalue */
            /* Shift */
            for (uint32_t i = 0; i < n; i++) A[i*n+i] -= mu;
        }

        memcpy(R, A, (size_t)n * n * sizeof(double));
        _qr_decompose(R, Q_step, n);

        /* A_new = R * Q_step + mu * I */
        {
            double *_tmp = (double *)malloc((size_t)n * n * sizeof(double));
            for (uint32_t i = 0; i < n; i++)
                for (uint32_t j = 0; j < n; j++) {
                    double sum = 0;
                    for (uint32_t k = 0; k < n; k++)
                        sum += R[i*n+k] * Q_step[k*n+j];
                    _tmp[i*n+j] = sum;
                }
            memcpy(A, _tmp, (size_t)n * n * sizeof(double));
            free(_tmp);
        }

        /* Re-add shift */
        for (uint32_t i = 0; i < n; i++) A[i*n+i] += mu;

        /* Accumulate Q_total = Q_total * Q_step */
        {
            double *tmp = (double *)malloc((size_t)n * n * sizeof(double));
            for (uint32_t i = 0; i < n; i++)
                for (uint32_t j = 0; j < n; j++) {
                    double sum = 0;
                    for (uint32_t k = 0; k < n; k++)
                        sum += Q_total[i*n+k] * Q_step[k*n+j];
                    tmp[i*n+j] = sum;
                }
            memcpy(Q_total, tmp, (size_t)n * n * sizeof(double));
            free(tmp);
        }

        s->iterations = iter + 1;

        /* Check convergence */
        double off = 0;
        for (uint32_t i = 0; i < n; i++)
            for (uint32_t j = 0; j < n; j++)
                if (i != j) off += A[i*n+j] * A[i*n+j];
        s->residual = sqrt(off);
        if (s->residual < 1e-12) break;
    }

    /* Extract eigenvalues from diagonal */
    for (uint32_t i = 0; i < n; i++)
        s->eigenvalues[i] = A[i * n + i];

    /* Sort eigenvalues (ascending) and reorder eigenvectors */
    for (uint32_t i = 0; i < n; i++) {
        uint32_t min_idx = i;
        for (uint32_t j = i + 1; j < n; j++)
            if (s->eigenvalues[j] < s->eigenvalues[min_idx]) min_idx = j;
        if (min_idx != i) {
            double tmp_ev = s->eigenvalues[i];
            s->eigenvalues[i] = s->eigenvalues[min_idx];
            s->eigenvalues[min_idx] = tmp_ev;
            /* Swap columns in Q_total */
            for (uint32_t r = 0; r < n; r++) {
                double tmp = Q_total[r * n + i];
                Q_total[r * n + i] = Q_total[r * n + min_idx];
                Q_total[r * n + min_idx] = tmp;
            }
        }
    }

    /* Copy eigenvectors (columns of Q_total) */
    memcpy(s->eigenvectors, Q_total, (size_t)n * n * sizeof(double));

    free(A); free(Q_total); free(Q_step); free(R);
    return s;
}

void sg_spectrum_destroy(sg_spectrum *s) {
    if (!s) return;
    free(s->eigenvalues);
    free(s->eigenvectors);
    free(s);
}

/* Power iteration */
sg_error sg_power_iteration(const sg_matrix *M, double *eigenvalue,
                             double *eigenvector, uint32_t max_iter, double tol) {
    if (!M || !eigenvalue || !eigenvector) return SG_ERR_NULL;
    uint32_t n = M->n;

    /* Random initial vector */
    srand(42);
    for (uint32_t i = 0; i < n; i++) eigenvector[i] = (double)rand() / RAND_MAX;
    vec_normalize(eigenvector, n);

    double *y = vec_alloc(n);
    for (uint32_t iter = 0; iter < max_iter; iter++) {
        mat_vec(M, eigenvector, y);
        double ev = vec_dot(eigenvector, y, n);
        vec_normalize(y, n);
        /* Check convergence */
        double diff = 0;
        for (uint32_t i = 0; i < n; i++)
            diff += (eigenvector[i] - y[i]) * (eigenvector[i] - y[i]);
        memcpy(eigenvector, y, n * sizeof(double));
        *eigenvalue = ev;
        if (sqrt(diff) < tol) { free(y); return SG_OK; }
    }
    free(y);
    return SG_ERR_NO_CONVERGE;
}

/* ══════════════════════════════════════════════════════════════ */
/*  Graph utilities                                            */
/* ══════════════════════════════════════════════════════════════ */

uint32_t sg_graph_degree(const sg_graph *g, uint32_t v) {
    if (!g || v >= g->n) return 0;
    return g->row_ptr[v+1] - g->row_ptr[v];
}

uint32_t sg_graph_edge_count(const sg_graph *g) {
    if (!g) return 0;
    uint32_t cnt = 0;
    for (uint32_t i = 0; i < g->n; i++) cnt += sg_graph_degree(g, i);
    return g->directed ? cnt : cnt / 2;
}

/* BFS connectivity check */
bool sg_graph_is_connected(const sg_graph *g) {
    if (!g || g->n <= 1) return true;
    uint32_t n = g->n;
    bool *visited = (bool *)calloc(n, sizeof(bool));
    uint32_t *queue = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t head = 0, tail = 0;
    queue[tail++] = 0; visited[0] = true;
    while (head < tail) {
        uint32_t v = queue[head++];
        for (uint32_t j = g->row_ptr[v]; j < g->row_ptr[v+1]; j++) {
            uint32_t u = g->col_idx[j];
            if (!visited[u]) { visited[u] = true; queue[tail++] = u; }
        }
    }
    bool connected = (tail == n);
    free(visited); free(queue);
    return connected;
}

bool sg_graph_is_regular(const sg_graph *g, uint32_t *degree) {
    if (!g || g->n == 0) { if (degree) *degree = 0; return true; }
    uint32_t d = sg_graph_degree(g, 0);
    for (uint32_t i = 1; i < g->n; i++)
        if (sg_graph_degree(g, i) != d) { if (degree) *degree = 0; return false; }
    if (degree) *degree = d;
    return true;
}

/* ══════════════════════════════════════════════════════════════ */
/*  Fiedler value (algebraic connectivity)                     */
/* ══════════════════════════════════════════════════════════════ */

sg_fiedler_result *sg_compute_fiedler(const sg_graph *g) {
    if (!g) return NULL;
    sg_matrix *L = sg_graph_laplacian(g);
    if (!L) return NULL;

    sg_spectrum *spec = sg_eigendecompose(L);
    sg_matrix_destroy(L);
    if (!spec || spec->n < 2) { sg_spectrum_destroy(spec); return NULL; }

    sg_fiedler_result *r = (sg_fiedler_result *)calloc(1, sizeof(sg_fiedler_result));
    r->n = g->n;
    r->fiedler_value = spec->eigenvalues[1]; /* Second smallest */
    r->fiedler_vector = vec_alloc(g->n);
    /* Extract second column of eigenvector matrix */
    for (uint32_t i = 0; i < g->n; i++)
        r->fiedler_vector[i] = spec->eigenvectors[i * g->n + 1];

    sg_spectrum_destroy(spec);
    return r;
}

void sg_fiedler_destroy(sg_fiedler_result *r) {
    if (!r) return;
    free(r->fiedler_vector);
    free(r);
}

/* ══════════════════════════════════════════════════════════════ */
/*  Cheeger constant (sweep cut)                               */
/* ══════════════════════════════════════════════════════════════ */

double sg_conductance(const sg_graph *g, const uint32_t *S, uint32_t size_S) {
    if (!g || !S || size_S == 0 || size_S >= g->n) return 1.0;
    /* Compute vol(S) = sum of degrees in S, and cut edges */
    double vol_S = 0, vol_comp = 0, cut = 0;
    bool *in_S = (bool *)calloc(g->n, sizeof(bool));
    for (uint32_t i = 0; i < size_S; i++) in_S[S[i]] = true;

    for (uint32_t v = 0; v < g->n; v++) {
        double deg = sg_graph_degree(g, v);
        if (in_S[v]) vol_S += deg;
        else vol_comp += deg;
        if (in_S[v]) {
            for (uint32_t j = g->row_ptr[v]; j < g->row_ptr[v+1]; j++) {
                if (!in_S[g->col_idx[j]]) cut += 1.0;
            }
        }
    }
    free(in_S);
    double vol_min = (vol_S < vol_comp) ? vol_S : vol_comp;
    return (vol_min > 0) ? cut / vol_min : 1.0;
}

sg_cheeger_result *sg_compute_cheeger(const sg_graph *g) {
    if (!g) return NULL;
    sg_fiedler_result *fiedler = sg_compute_fiedler(g);
    if (!fiedler) return NULL;

    uint32_t n = g->n;
    /* Create index array sorted by Fiedler vector value */
    uint32_t *idx = (uint32_t *)malloc(n * sizeof(uint32_t));
    for (uint32_t i = 0; i < n; i++) idx[i] = i;
    /* Simple insertion sort by Fiedler value */
    for (uint32_t i = 1; i < n; i++) {
        uint32_t key = idx[i];
        double kv = fiedler->fiedler_vector[key];
        int32_t j = (int32_t)i - 1;
        while (j >= 0 && fiedler->fiedler_vector[idx[j]] > kv) {
            idx[j+1] = idx[j];
            j--;
        }
        idx[j+1] = key;
    }

    /* Sweep cut: try all prefix sets */
    double best_cond = 1.0;
    uint32_t best_k = 1;
    for (uint32_t k = 1; k < n; k++) {
        double cond = sg_conductance(g, idx, k);
        if (cond < best_cond) { best_cond = cond; best_k = k; }
    }

    sg_cheeger_result *r = (sg_cheeger_result *)calloc(1, sizeof(sg_cheeger_result));
    r->cheeger_constant = best_cond;
    r->conductance = best_cond;
    r->cut_size = best_k;
    r->cut_set = (uint32_t *)malloc(best_k * sizeof(uint32_t));
    memcpy(r->cut_set, idx, best_k * sizeof(uint32_t));

    free(idx);
    sg_fiedler_destroy(fiedler);
    return r;
}

void sg_cheeger_destroy(sg_cheeger_result *r) {
    if (!r) return;
    free(r->cut_set);
    free(r);
}

/* ══════════════════════════════════════════════════════════════ */
/*  Mixing time                                                */
/* ══════════════════════════════════════════════════════════════ */

sg_mixing_result *sg_compute_mixing_time(const sg_graph *g) {
    if (!g) return NULL;
    sg_matrix *Lnorm = sg_graph_normalized_laplacian(g);
    if (!Lnorm) return NULL;

    sg_spectrum *spec = sg_eigendecompose(Lnorm);
    sg_matrix_destroy(Lnorm);
    if (!spec) return NULL;

    sg_mixing_result *r = (sg_mixing_result *)calloc(1, sizeof(sg_mixing_result));
    uint32_t n = g->n;
    if (n < 2) {
        r->mixing_time = 0;
        r->spectral_gap = 0;
        r->convergence_rate = 0;
        sg_spectrum_destroy(spec);
        return r;
    }

    /* Spectral gap = smallest nonzero eigenvalue of normalized Laplacian
       = λ₁ for connected graph (since λ₀ = 0) */
    double lambda2 = spec->eigenvalues[1];
    r->spectral_gap = lambda2;

    /* Mixing time ≈ (1/λ₂) * log(n/ε), using ε=0.01 */
    if (lambda2 > 1e-15)
        r->mixing_time = (1.0 / lambda2) * log((double)n / 0.01);
    else
        r->mixing_time = INFINITY;

    r->convergence_rate = lambda2;

    sg_spectrum_destroy(spec);
    return r;
}

void sg_mixing_destroy(sg_mixing_result *r) { free(r); }

/* ══════════════════════════════════════════════════════════════ */
/*  Spectral clustering                                        */
/* ══════════════════════════════════════════════════════════════ */

sg_cluster_result *sg_spectral_cluster(const sg_graph *g, uint32_t k) {
    if (!g || k == 0 || k > g->n) return NULL;
    sg_matrix *L = sg_graph_laplacian(g);
    if (!L) return NULL;

    sg_spectrum *spec = sg_eigendecompose(L);
    sg_matrix_destroy(L);
    if (!spec) return NULL;

    uint32_t n = g->n;
    sg_cluster_result *r = (sg_cluster_result *)calloc(1, sizeof(sg_cluster_result));
    r->cluster_ids = (uint32_t *)calloc(n, sizeof(uint32_t));
    r->num_clusters = k;

    if (k == 1) {
        /* All in one cluster */
        sg_spectrum_destroy(spec);
        return r;
    }

    /* Use first k eigenvectors for embedding, then k-means */
    /* Simplified: use Fiedler vector for 2-cluster, extend for k */
    uint32_t dim = (k < n) ? k : n;

    /* Simple threshold-based clustering using Fiedler vector */
    if (k == 2) {
        for (uint32_t i = 0; i < n; i++)
            r->cluster_ids[i] = (spec->eigenvectors[i * n + 1] >= 0) ? 0 : 1;
    } else {
        /* Multi-way: use k eigenvectors, assign by sign patterns / simple quantization */
        /* For simplicity, use the first k non-trivial eigenvectors and quantize */
        for (uint32_t i = 0; i < n; i++) {
            double min_dist = DBL_MAX;
            uint32_t best_c = 0;
            /* Cluster by rounding the first eigenvector values */
            for (uint32_t c = 0; c < k; c++) {
                double center = -1.0 + 2.0 * c / (k - 1);
                double val = spec->eigenvectors[i * n + 1];
                double dist = fabs(val - center);
                if (dist < min_dist) { min_dist = dist; best_c = c; }
            }
            r->cluster_ids[i] = best_c;
        }
    }

    /* Compute modularity */
    double m = (double)sg_graph_edge_count(g);
    double mod = 0;
    for (uint32_t i = 0; i < n; i++)
        for (uint32_t j = g->row_ptr[i]; j < g->row_ptr[i+1]; j++) {
            uint32_t v = g->col_idx[j];
            if (r->cluster_ids[i] == r->cluster_ids[v]) {
                double deg_i = sg_graph_degree(g, i);
                double deg_v = sg_graph_degree(g, v);
                mod += 1.0 - deg_i * deg_v / (2.0 * m);
            }
        }
    r->modularity = mod / (2.0 * m);

    sg_spectrum_destroy(spec);
    return r;
}

void sg_cluster_destroy(sg_cluster_result *r) {
    if (!r) return;
    free(r->cluster_ids);
    free(r);
}

/* ══════════════════════════════════════════════════════════════ */
/*  Eigenvector centrality                                     */
/* ══════════════════════════════════════════════════════════════ */

sg_centrality_result *sg_compute_centrality(const sg_graph *g) {
    if (!g) return NULL;
    sg_matrix *A = sg_graph_adjacency_matrix(g);
    if (!A) return NULL;

    sg_centrality_result *r = (sg_centrality_result *)calloc(1, sizeof(sg_centrality_result));
    r->centrality = vec_alloc(g->n);

    double ev = 0;
    sg_error err = sg_power_iteration(A, &ev, r->centrality, 1000, 1e-10);
    sg_matrix_destroy(A);

    if (err != SG_OK) {
        /* Fallback: degree centrality */
        for (uint32_t i = 0; i < g->n; i++)
            r->centrality[i] = (double)sg_graph_degree(g, i);
    }

    /* Normalize so max = 1.0 */
    r->max_centrality = 0;
    r->dominant_agent = 0;
    for (uint32_t i = 0; i < g->n; i++) {
        if (r->centrality[i] < 0) r->centrality[i] = -r->centrality[i];
        if (r->centrality[i] > r->max_centrality) {
            r->max_centrality = r->centrality[i];
            r->dominant_agent = i;
        }
    }
    if (r->max_centrality > 1e-15)
        for (uint32_t i = 0; i < g->n; i++)
            r->centrality[i] /= r->max_centrality;
    r->max_centrality = 1.0;

    return r;
}

void sg_centrality_destroy(sg_centrality_result *r) {
    if (!r) return;
    free(r->centrality);
    free(r);
}

/* ══════════════════════════════════════════════════════════════ */
/*  Network robustness                                         */
/* ══════════════════════════════════════════════════════════════ */

/* Count articulation points using DFS */
static uint32_t _count_articulation_points(const sg_graph *g) {
    uint32_t n = g->n;
    if (n <= 2) return 0;

    bool *visited = (bool *)calloc(n, sizeof(bool));
    int *disc = (int *)malloc(n * sizeof(int));
    int *low = (int *)malloc(n * sizeof(int));
    bool *ap = (bool *)calloc(n, sizeof(bool));
    int *parent = (int *)malloc(n * sizeof(int));
    for (uint32_t i = 0; i < n; i++) parent[i] = -1;

    int time = 0;

    /* Iterative DFS for articulation points */
    typedef struct { uint32_t v; uint32_t adj_idx; bool processed; } _dfs_frame;
    _dfs_frame *stack = (_dfs_frame *)malloc(n * sizeof(_dfs_frame) * 2);

    for (uint32_t start = 0; start < n; start++) {
        if (visited[start]) continue;
        int sp = 0;
        stack[sp++] = (_dfs_frame){start, g->row_ptr[start], false};

        while (sp > 0) {
            _dfs_frame *frame = &stack[sp - 1];
            uint32_t v = frame->v;

            if (!visited[v]) {
                visited[v] = true;
                disc[v] = low[v] = time++;
            }

            if (frame->adj_idx < g->row_ptr[v+1]) {
                uint32_t u = g->col_idx[frame->adj_idx++];
                if (!visited[u]) {
                    parent[u] = (int)v;
                    stack[sp++] = (_dfs_frame){u, g->row_ptr[u], false};
                } else if ((int)u != parent[v]) {
                    low[v] = (low[v] < disc[u]) ? low[v] : disc[u];
                }
            } else {
                /* All neighbors processed */
                if (parent[v] != -1) {
                    uint32_t p = (uint32_t)parent[v];
                    low[p] = (low[p] < low[v]) ? low[p] : low[v];
                    /* Only check low >= disc for non-root parents */
                    if (parent[p] != -1 && low[v] >= disc[p]) ap[p] = true;
                } else {
                    /* Root: AP if has 2+ children */
                    uint32_t children = 0;
                    for (uint32_t j = g->row_ptr[v]; j < g->row_ptr[v+1]; j++)
                        if (parent[g->col_idx[j]] == (int)v) children++;
                    if (children > 1) ap[v] = true;
                }
                sp--;
            }
        }
    }

    uint32_t count = 0;
    for (uint32_t i = 0; i < n; i++) if (ap[i]) count++;

    free(visited); free(disc); free(low); free(ap); free(parent); free(stack);
    return count;
}

sg_robustness_result *sg_compute_robustness(const sg_graph *g) {
    if (!g) return NULL;
    sg_robustness_result *r = (sg_robustness_result *)calloc(1, sizeof(sg_robustness_result));

    uint32_t n = g->n;
    if (n <= 1) { r->robustness_score = 1.0; return r; }

    /* Spectral robustness from algebraic connectivity */
    sg_fiedler_result *fiedler = sg_compute_fiedler(g);
    if (fiedler) {
        r->spectral_robustness = fiedler->fiedler_value;
        sg_fiedler_destroy(fiedler);
    }

    /* Articulation points */
    r->articulation_points = _count_articulation_points(g);

    /* Min cut estimate from edge connectivity (approximate via Cheeger) */
    sg_cheeger_result *cheeger = sg_compute_cheeger(g);
    if (cheeger) {
        r->min_cut_edges = (uint32_t)(cheeger->cheeger_constant * sg_graph_edge_count(g) + 0.5);
        if (r->min_cut_edges < 1) r->min_cut_edges = 1;
        sg_cheeger_destroy(cheeger);
    }

    /* Overall robustness: composite score */
    bool connected = sg_graph_is_connected(g);
    if (!connected) {
        r->robustness_score = 0;
    } else {
        double conn_score = (r->spectral_robustness > 2.0) ? 1.0 : r->spectral_robustness / 2.0;
        double ap_penalty = (double)r->articulation_points / n;
        r->robustness_score = conn_score * (1.0 - ap_penalty);
        if (r->robustness_score < 0) r->robustness_score = 0;
        if (r->robustness_score > 1) r->robustness_score = 1;
    }

    return r;
}

void sg_robustness_destroy(sg_robustness_result *r) { free(r); }

/* ══════════════════════════════════════════════════════════════ */
/*  Expander graph quality                                     */
/* ══════════════════════════════════════════════════════════════ */

sg_expander_result *sg_compute_expander_quality(const sg_graph *g) {
    if (!g) return NULL;
    sg_expander_result *r = (sg_expander_result *)calloc(1, sizeof(sg_expander_result));

    uint32_t d = 0;
    bool regular = sg_graph_is_regular(g, &d);
    r->degree = regular ? d : 0;

    /* Cheeger constant */
    sg_cheeger_result *cheeger = sg_compute_cheeger(g);
    if (cheeger) {
        r->expansion_ratio = cheeger->cheeger_constant;
        sg_cheeger_destroy(cheeger);
    }

    /* Spectral gap from adjacency matrix */
    sg_matrix *A = sg_graph_adjacency_matrix(g);
    if (A) {
        sg_spectrum *spec = sg_eigendecompose(A);
        if (spec && spec->n >= 2) {
            /* Largest and second-largest eigenvalues of adjacency */
            double lambda1 = spec->eigenvalues[spec->n - 1];
            double lambda2 = spec->eigenvalues[spec->n - 2];
            r->spectral_gap = fabs(lambda1) - fabs(lambda2);
        }
        sg_spectrum_destroy(spec);
        sg_matrix_destroy(A);
    }

    /* Ramanujan bound: for d-regular graph, is |λ₂| ≤ 2√(d-1)? */
    if (regular && d >= 2) {
        r->ramanujan_bound = 2.0 * sqrt((double)(d - 1));
        /* Check if second eigenvalue satisfies Ramanujan bound */
        sg_matrix *Adj = sg_graph_adjacency_matrix(g);
        if (Adj) {
            sg_spectrum *sp = sg_eigendecompose(Adj);
            if (sp && sp->n >= 2) {
                /* Second largest absolute eigenvalue */
                double max_abs = 0;
                for (uint32_t i = 0; i < sp->n - 1; i++) {
                    double a = fabs(sp->eigenvalues[i]);
                    if (a > max_abs) max_abs = a;
                }
                r->is_ramanujan = (max_abs <= r->ramanujan_bound + 1e-10);
                r->expander_quality = (r->ramanujan_bound > 1e-15) ?
                    max_abs / r->ramanujan_bound : 0;
            }
            sg_spectrum_destroy(sp);
            sg_matrix_destroy(Adj);
        }
    } else {
        r->ramanujan_bound = 0;
        r->is_ramanujan = false;
        r->expander_quality = 0;
    }

    return r;
}

void sg_expander_destroy(sg_expander_result *r) { free(r); }

/* ══════════════════════════════════════════════════════════════ */
/*  Sheaf integration                                          */
/* ══════════════════════════════════════════════════════════════ */

sg_sheaf_prediction sg_sheaf_predict_failures(const sg_graph *g, double baseline_gap) {
    sg_sheaf_prediction p = {0};
    if (!g) return p;

    sg_mixing_result *mix = sg_compute_mixing_time(g);
    if (!mix) return p;

    p.spectral_gap_healthy = baseline_gap;
    p.spectral_gap_current = mix->spectral_gap;
    p.degradation_pct = (baseline_gap > 1e-15) ?
        (1.0 - mix->spectral_gap / baseline_gap) * 100.0 : 100.0;
    if (p.degradation_pct < 0) p.degradation_pct = 0;

    p.communication_risk = (p.degradation_pct > 30.0);
    p.failure_probability = (p.degradation_pct > 100.0) ? 1.0 :
        p.degradation_pct / 100.0;

    sg_mixing_destroy(mix);
    return p;
}

/* ══════════════════════════════════════════════════════════════ */
/*  Ergodic integration                                        */
/* ══════════════════════════════════════════════════════════════ */

sg_ergodic_correlation sg_ergodic_correlate(const sg_graph *g) {
    sg_ergodic_correlation c = {0};
    if (!g) return c;

    sg_mixing_result *mix = sg_compute_mixing_time(g);
    if (!mix) return c;

    c.mixing_time_graph = mix->mixing_time;

    /* Ergodic convergence time: for a reversible Markov chain,
       τ_ergodic ≈ (1/γ) * log(1/(ε*π_min))
       where γ is spectral gap, π_min is minimum stationary probability */
    uint32_t n = g->n;
    double pi_min = 1.0 / n;
    if (mix->spectral_gap > 1e-15) {
        c.convergence_time_ergodic = (1.0 / mix->spectral_gap) *
            log(1.0 / (0.01 * pi_min));
    } else {
        c.convergence_time_ergodic = INFINITY;
    }

    /* Correlation: high correlation means graph mixing predicts ergodic convergence */
    if (c.mixing_time_graph > 0 && c.convergence_time_ergodic > 0 &&
        c.mixing_time_graph < INFINITY && c.convergence_time_ergodic < INFINITY) {
        c.correlation = 1.0 - fabs(c.mixing_time_graph - c.convergence_time_ergodic) /
            (c.mixing_time_graph + c.convergence_time_ergodic);
    } else {
        c.correlation = 0;
    }

    sg_mixing_destroy(mix);
    return c;
}

/* ══════════════════════════════════════════════════════════════ */
/*  Edge connectivity (approximate via max-flow min-cut)       */
/* ══════════════════════════════════════════════════════════════ */

double sg_edge_connectivity(const sg_graph *g) {
    if (!g) return 0;
    /* Approximate: min degree for simple undirected graph */
    uint32_t min_deg = sg_graph_degree(g, 0);
    for (uint32_t i = 1; i < g->n; i++) {
        uint32_t d = sg_graph_degree(g, i);
        if (d < min_deg) min_deg = d;
    }
    return (double)min_deg;
}
