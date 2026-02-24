/**
 * AlgebraicMaxCliques - C++ Implementation
 * Based on the AlgebraicMaxCliques Canonical + BatchedReplicatorSparseAdaptive algorithm
 *
 * Compile: g++ -O3 -std=c++17 -o algebraic_cliques algebraic_cliques.cpp
 * Usage:   ./algebraic_cliques <graph.mtx> [K] [max_iter] [tol] [stable_thresh] [rel_thresh]
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include <chrono>
#include <string>
#include <tuple>
#include <functional>
#include <iomanip>

// ─── Types ───────────────────────────────────────────────────────────────────

using NodeId  = int;
using Clique  = std::vector<NodeId>;

// CSR sparse matrix (float)
struct CSRMatrix {
    int rows, cols;
    std::vector<int>   row_ptr;   // size rows+1
    std::vector<int>   col_idx;
    std::vector<float> values;
};

// Dense column-major matrix  (rows x cols)
struct DenseMatrix {
    int rows, cols;
    std::vector<float> data; // data[i + j*rows]
    float& at(int i, int j)       { return data[i + j*rows]; }
    float  at(int i, int j) const { return data[i + j*rows]; }
    void zero() { std::fill(data.begin(), data.end(), 0.f); }
};

// ─── Graph ───────────────────────────────────────────────────────────────────

struct Graph {
    int n = 0;
    std::vector<std::vector<NodeId>> adj;

    void resize(int n_) {
        n = n_;
        adj.assign(n, {});
    }
    void add_edge(NodeId u, NodeId v) {
        if (u == v) return;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    int degree(NodeId v) const { return (int)adj[v].size(); }
};

// ─── Shared adjacency dedup ───────────────────────────────────────────────────

void dedup_adj(Graph& G) {
    for (int i = 0; i < G.n; i++) {
        auto& a = G.adj[i];
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
    }
}

// ─── MTX loader ──────────────────────────────────────────────────────────────

Graph load_mtx(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; exit(1); }

    std::string line;
    int rows = 0, cols = 0, nnz = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream ss(line);
        ss >> rows >> cols >> nnz;
        break;
    }
    int n = std::max(rows, cols);
    Graph G;
    G.resize(n);
    int u, v;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream ss(line);
        if (!(ss >> u >> v)) continue;
        u--; v--; // 1-indexed → 0-indexed
        G.add_edge(u, v);
    }
    dedup_adj(G);
    return G;
}

// ─── .edges loader ────────────────────────────────────────────────────────────
//
// Supports the common SNAP / NetworkX edge-list format:
//   • Lines starting with '#' or '%' are comments (skipped)
//   • Each data line: <u> <v> [optional weight or extra columns — ignored]
//   • Node IDs may be 0-indexed or 1-indexed; we auto-detect by checking
//     whether any ID equals 0. If the minimum ID is 1 we shift to 0-indexed.
//   • Isolated nodes not mentioned in the edge list will not appear, which
//     matches the behaviour of typical edge-list datasets.

Graph load_edges(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; exit(1); }

    std::vector<std::pair<int,int>> raw_edges;
    int min_id = std::numeric_limits<int>::max(), max_id = 0;

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#' || line[0] == '%') continue;
        std::istringstream ss(line);
        int u, v;
        if (!(ss >> u >> v)) continue;
        raw_edges.emplace_back(u, v);
        min_id = std::min(min_id, std::min(u, v));
        max_id = std::max(max_id, std::max(u, v));
    }

    // Auto-detect indexing: shift to 0-based if smallest id is 1
    int shift = (min_id == 1) ? 1 : 0;
    int n = max_id - shift + 1;

    Graph G;
    G.resize(n);
    for (auto [u, v] : raw_edges)
        G.add_edge(u - shift, v - shift);

    dedup_adj(G);
    return G;
}

// ─── Unified loader (dispatch by extension) ───────────────────────────────────

Graph load_graph(const std::string& path) {
    // extract extension (lower-cased)
    std::string ext;
    auto dot = path.rfind('.');
    if (dot != std::string::npos) {
        ext = path.substr(dot + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    if (ext == "mtx") {
        std::cout << "[loader] detected MatrixMarket (.mtx)\n";
        return load_mtx(path);
    } else if (ext == "edges" || ext == "edgelist" || ext == "el" || ext == "txt") {
        std::cout << "[loader] detected edge-list (." << ext << ")\n";
        return load_edges(path);
    } else {
        // Peek at the first non-comment line to guess format
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line))
            if (!line.empty() && line[0] != '#' && line[0] != '%') break;

        if (!line.empty() && line[0] == '%') {
            // MatrixMarket banner
            std::cout << "[loader] guessed MatrixMarket format\n";
            return load_mtx(path);
        } else {
            std::cout << "[loader] unknown extension '" << ext
                      << "', treating as edge-list\n";
            return load_edges(path);
        }
    }
}

// ─── Core numbers (Batagelj-Zaversnik) ───────────────────────────────────────

std::vector<int> core_number(const Graph& G) {
    int n = G.n;
    std::vector<int> deg(n), core(n, 0);
    int max_deg = 0;
    for (int v = 0; v < n; v++) { deg[v] = G.degree(v); max_deg = std::max(max_deg, deg[v]); }

    std::vector<int> bin(max_deg + 1, 0);
    for (int v = 0; v < n; v++) bin[deg[v]]++;
    std::vector<int> start(max_deg + 1, 0);
    for (int d = 1; d <= max_deg; d++) start[d] = start[d-1] + bin[d-1];

    std::vector<int> order(n), pos(n);
    std::vector<int> cnt(max_deg + 1, 0);
    for (int v = 0; v < n; v++) { pos[v] = start[deg[v]] + cnt[deg[v]]++; order[pos[v]] = v; }

    for (int i = 0; i < n; i++) {
        int v = order[i];
        core[v] = deg[v];
        for (int u : G.adj[v]) {
            if (core[u] > core[v]) {
                int du = deg[u], pu = pos[u];
                int pw = start[du];
                int w  = order[pw];
                if (w != u) { pos[u] = pw; pos[w] = pu; order[pu] = w; order[pw] = u; }
                start[du]++;
                deg[u]--;
            }
        }
    }
    return core;
}

// ─── Degeneracy order positions ───────────────────────────────────────────────

std::vector<int> degeneracy_order_positions(const Graph& G, const std::vector<int>& core) {
    // Simple degeneracy ordering: repeatedly remove min-degree vertex
    // We use bucket-based approach; position = rank in removal order
    int n = G.n;
    std::vector<int> deg(n);
    for (int v = 0; v < n; v++) deg[v] = G.degree(v);

    std::vector<bool> removed(n, false);
    std::vector<int> pos(n, 0);
    // Use the core order: sort by core number then degree
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){
        if (core[a] != core[b]) return core[a] < core[b];
        return deg[a] < deg[b];
    });
    for (int i = 0; i < n; i++) pos[order[i]] = i;
    return pos;
}

// ─── CSR from induced subgraph ────────────────────────────────────────────────

CSRMatrix build_csr(const Graph& G, const std::vector<NodeId>& nodes) {
    int s = (int)nodes.size();
    std::unordered_map<NodeId, int> idx;
    idx.reserve(s);
    for (int i = 0; i < s; i++) idx[nodes[i]] = i;

    CSRMatrix M;
    M.rows = M.cols = s;
    M.row_ptr.resize(s + 1, 0);

    // count
    for (int i = 0; i < s; i++) {
        for (NodeId nb : G.adj[nodes[i]]) {
            auto it = idx.find(nb);
            if (it != idx.end()) M.row_ptr[i + 1]++;
        }
    }
    for (int i = 0; i < s; i++) M.row_ptr[i+1] += M.row_ptr[i];
    int nnz = M.row_ptr[s];
    M.col_idx.resize(nnz);
    M.values.assign(nnz, 1.f);
    std::vector<int> cur(M.row_ptr.begin(), M.row_ptr.begin() + s);
    for (int i = 0; i < s; i++) {
        for (NodeId nb : G.adj[nodes[i]]) {
            auto it = idx.find(nb);
            if (it != idx.end()) M.col_idx[cur[i]++] = it->second;
        }
    }
    return M;
}

// ─── Sparse-dense matmul: Y = A * X ──────────────────────────────────────────

void spmm(const CSRMatrix& A, const DenseMatrix& X, DenseMatrix& Y) {
    // Y: A.rows x X.cols
    int s = A.rows, p = X.cols;
    Y.rows = s; Y.cols = p;
    Y.data.assign((size_t)s * p, 0.f);
    for (int i = 0; i < s; i++) {
        for (int jj = A.row_ptr[i]; jj < A.row_ptr[i+1]; jj++) {
            int k = A.col_idx[jj];
            float av = A.values[jj];
            for (int j = 0; j < p; j++) Y.at(i,j) += av * X.at(k,j);
        }
    }
}

// ─── BatchedReplicatorSparseAdaptive ─────────────────────────────────────────

DenseMatrix batched_replicator(const CSRMatrix& Acsr, DenseMatrix X,
                               int max_iter, float tol,
                               int stable_iter_threshold,
                               int& out_iters) {
    int s = X.rows, p = X.cols;
    const float eps = 1e-9f;
    const float support_eps = 1e-6f;

    DenseMatrix AX;
    AX.rows = s; AX.cols = p;
    AX.data.resize((size_t)s * p);

    std::vector<std::vector<int>> prev_supports(p);
    std::vector<int> stable_counts(p, 0);
    bool has_prev = false;
    std::vector<float> prev_data;

    for (int t = 1; t <= max_iter; t++) {
        // AX = Acsr * X
        spmm(Acsr, X, AX);

        // X = X ⊙ AX, then normalize columns
        for (int j = 0; j < p; j++) {
            float col_sum = 0.f;
            for (int i = 0; i < s; i++) {
                float v = X.at(i,j) * AX.at(i,j);
                X.at(i,j) = v;
                col_sum += v;
            }
            if (col_sum > eps) {
                float inv = 1.f / col_sum;
                for (int i = 0; i < s; i++) X.at(i,j) *= inv;
            } else {
                for (int i = 0; i < s; i++) X.at(i,j) = 0.f;
            }
        }

        // Convergence: L1 norm check
        if (has_prev) {
            bool conv = true;
            for (int j = 0; j < p && conv; j++) {
                float diff = 0.f;
                for (int i = 0; i < s; i++) diff += std::fabs(X.at(i,j) - prev_data[i + j*s]);
                if (diff >= tol) conv = false;
            }
            if (conv) { out_iters = t; return X; }
        }

        // Support stability check
        for (int j = 0; j < p; j++) {
            float mx = 0.f;
            for (int i = 0; i < s; i++) mx = std::max(mx, X.at(i,j));
            float thresh = support_eps * mx;
            std::vector<int> sup;
            for (int i = 0; i < s; i++) if (X.at(i,j) >= thresh) sup.push_back(i);
            if (sup == prev_supports[j]) stable_counts[j]++;
            else { stable_counts[j] = 0; prev_supports[j] = sup; }
        }
        bool all_stable = true;
        for (int j = 0; j < p; j++) if (stable_counts[j] < stable_iter_threshold) { all_stable = false; break; }
        if (all_stable) { out_iters = t; return X; }

        prev_data = X.data;
        has_prev = true;
    }
    out_iters = max_iter;
    return X;
}

// ─── Clique check ────────────────────────────────────────────────────────────

bool is_clique(const Graph& G, const std::vector<NodeId>& nodes) {
    std::unordered_set<NodeId> s(nodes.begin(), nodes.end());
    for (NodeId u : nodes)
        for (NodeId v : nodes)
            if (u != v) {
                bool found = false;
                for (NodeId nb : G.adj[u]) if (nb == v) { found = true; break; }
                if (!found) return false;
            }
    return true;
}

// faster adjacency check using sorted adj
bool adjacent(const Graph& G, NodeId u, NodeId v) {
    // binary search (adj lists are sorted)
    return std::binary_search(G.adj[u].begin(), G.adj[u].end(), v);
}

bool is_clique_fast(const Graph& G, const std::vector<NodeId>& nodes) {
    for (int i = 0; i < (int)nodes.size(); i++)
        for (int j = i+1; j < (int)nodes.size(); j++)
            if (!adjacent(G, nodes[i], nodes[j])) return false;
    return true;
}

// ─── Expand to maximal clique ─────────────────────────────────────────────────

Clique expand_to_maximal(const Graph& G, const std::vector<NodeId>& seed) {
    std::unordered_set<NodeId> clique_set(seed.begin(), seed.end());
    Clique clique(seed);

    // candidates: neighbors of all clique members
    std::unordered_map<NodeId, int> cnt;
    for (NodeId u : clique)
        for (NodeId nb : G.adj[u])
            if (!clique_set.count(nb)) cnt[nb]++;

    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& [v, c] : cnt) {
            if (c == (int)clique.size()) {
                // v is adjacent to all clique members
                clique.push_back(v);
                clique_set.insert(v);
                // update cnt
                for (NodeId nb : G.adj[v])
                    if (!clique_set.count(nb)) cnt[nb]++;
                cnt.erase(v);
                changed = true;
                break;
            }
        }
    }
    std::sort(clique.begin(), clique.end());
    return clique;
}

// ─── LRU cache (simple map-based, unbounded for simplicity) ──────────────────

struct SubgraphCache {
    std::map<std::vector<NodeId>, std::pair<CSRMatrix, std::vector<NodeId>>> cache;
    size_t max_size;
    SubgraphCache(size_t ms = 500) : max_size(ms) {}

    bool contains(const std::vector<NodeId>& key) { return cache.count(key) > 0; }
    std::pair<CSRMatrix, std::vector<NodeId>>& get(const std::vector<NodeId>& key) { return cache.at(key); }
    void put(const std::vector<NodeId>& key, CSRMatrix&& M, std::vector<NodeId>&& nodes) {
        if (cache.size() >= max_size) cache.erase(cache.begin());
        cache[key] = {std::move(M), std::move(nodes)};
    }
};

// ─── Main algorithm ───────────────────────────────────────────────────────────

std::set<Clique> algebraic_max_cliques(const Graph& G,
                                        int K = 10,
                                        int max_iter = 200,
                                        float tol = 1e-5f,
                                        int stable_iter_threshold = 5,
                                        float rel_thresh = 0.5f) {
    auto core = core_number(G);
    auto order_pos = degeneracy_order_positions(G, core);

    SubgraphCache cache(1000);
    std::set<Clique> F;

    for (NodeId v = 0; v < G.n; v++) {
        const auto& nbrs_raw = G.adj[v];
        if (nbrs_raw.empty()) continue;

        // top K neighbors by core number
        std::vector<NodeId> neighs(nbrs_raw.begin(), nbrs_raw.end());
        if ((int)neighs.size() > K) {
            std::partial_sort(neighs.begin(), neighs.begin() + K, neighs.end(),
                [&](NodeId a, NodeId b){ return core[a] > core[b]; });
            neighs.resize(K);
        }

        // canonical seeds: neighbors with higher order position
        std::vector<NodeId> seeds;
        for (NodeId u : neighs)
            if (order_pos[u] >= order_pos[v]) seeds.push_back(u);
        bool none_seed = seeds.empty();
        if (none_seed) seeds.push_back(-1); // sentinel for "None"

        // induced subgraph nodes
        std::vector<NodeId> S_nodes;
        S_nodes.push_back(v);
        for (NodeId u : neighs) S_nodes.push_back(u);
        std::sort(S_nodes.begin(), S_nodes.end());
        S_nodes.erase(std::unique(S_nodes.begin(), S_nodes.end()), S_nodes.end());

        // cache lookup
        const CSRMatrix* Acsr_ptr;
        const std::vector<NodeId>* nodes_list_ptr;
        CSRMatrix tmp_csr; std::vector<NodeId> tmp_nodes;

        if (cache.contains(S_nodes)) {
            auto& [c, nl] = cache.get(S_nodes);
            Acsr_ptr = &c; nodes_list_ptr = &nl;
        } else {
            tmp_csr = build_csr(G, S_nodes);
            tmp_nodes = S_nodes;
            cache.put(S_nodes, std::move(tmp_csr), std::move(tmp_nodes));
            auto& [c, nl] = cache.get(S_nodes);
            Acsr_ptr = &c; nodes_list_ptr = &nl;
        }
        const CSRMatrix& Acsr = *Acsr_ptr;
        const std::vector<NodeId>& nodes_list = *nodes_list_ptr;

        int ns = (int)nodes_list.size();
        // local index of v
        int v_local = (int)(std::find(nodes_list.begin(), nodes_list.end(), v) - nodes_list.begin());

        int p = (int)seeds.size();

        // Build X0
        DenseMatrix X0;
        X0.rows = ns; X0.cols = p;
        X0.data.assign((size_t)ns * p, 0.f);

        for (int j = 0; j < p; j++) {
            NodeId seed = seeds[j];
            if (seed == -1) {
                X0.at(v_local, j) = 1.f;
            } else {
                X0.at(v_local, j) = 0.5f;
                int u_local = (int)(std::find(nodes_list.begin(), nodes_list.end(), seed) - nodes_list.begin());
                if (u_local < ns) X0.at(u_local, j) = 0.5f;
            }
            // normalize column to sum 1
            float s = 0.f;
            for (int i = 0; i < ns; i++) s += X0.at(i,j);
            if (s > 1e-9f) for (int i = 0; i < ns; i++) X0.at(i,j) /= s;
        }

        int iters = 0;
        DenseMatrix Xfinal = batched_replicator(Acsr, std::move(X0), max_iter, tol, stable_iter_threshold, iters);

        for (int j = 0; j < p; j++) {
            float mx = 0.f;
            for (int i = 0; i < ns; i++) mx = std::max(mx, Xfinal.at(i,j));
            if (mx == 0.f) continue;

            float thresh = rel_thresh * mx;
            std::vector<NodeId> candidate;
            for (int i = 0; i < ns; i++)
                if (Xfinal.at(i,j) >= thresh) candidate.push_back(nodes_list[i]);

            if (!is_clique_fast(G, candidate)) continue;

            Clique clique = expand_to_maximal(G, candidate);

            // canonical: min-order-pos node must be v
            NodeId min_node = *std::min_element(clique.begin(), clique.end(),
                [&](NodeId a, NodeId b){ return order_pos[a] < order_pos[b]; });
            if (min_node != v) continue;

            F.insert(clique);
        }
    }
    return F;
}

// ─── Bron-Kerbosch with degeneracy ordering (Eppstein et al. 2010) ────────────
//
// This is the state-of-the-art exact maximal clique enumeration algorithm.
// It processes vertices in degeneracy order and runs BK with pivoting on each
// induced subgraph, giving O(dn · 3^(d/3)) time where d is the degeneracy.

// Inner BK with pivot (Tomita pivot: choose pivot maximising |P ∩ N(u)|)
static void bk_pivot(const Graph& G,
                     std::vector<NodeId>  R,   // current clique (by value – small)
                     std::vector<NodeId>& P,   // candidates
                     std::vector<NodeId>& X,   // already processed
                     std::set<Clique>&    out) {
    if (P.empty() && X.empty()) {
        Clique c = R;
        std::sort(c.begin(), c.end());
        out.insert(c);
        return;
    }
    if (P.empty()) return;

    // Choose pivot u from P ∪ X that maximises |P ∩ N(u)|
    NodeId pivot = -1;
    int    best  = -1;
    for (NodeId u : P) {
        int cnt = 0;
        for (NodeId w : P) if (adjacent(G, u, w)) cnt++;
        if (cnt > best) { best = cnt; pivot = u; }
    }
    for (NodeId u : X) {
        int cnt = 0;
        for (NodeId w : P) if (adjacent(G, u, w)) cnt++;
        if (cnt > best) { best = cnt; pivot = u; }
    }

    // Iterate over P \ N(pivot)
    std::vector<NodeId> candidates;
    for (NodeId v : P)
        if (!adjacent(G, pivot, v)) candidates.push_back(v);

    for (NodeId v : candidates) {
        // new_P = P ∩ N(v),  new_X = X ∩ N(v)
        std::vector<NodeId> newP, newX;
        for (NodeId w : P) if (adjacent(G, v, w)) newP.push_back(w);
        for (NodeId w : X) if (adjacent(G, v, w)) newX.push_back(w);

        R.push_back(v);
        bk_pivot(G, R, newP, newX, out);
        R.pop_back();

        // move v from P to X
        P.erase(std::find(P.begin(), P.end(), v));
        X.push_back(v);
    }
}

std::set<Clique> bron_kerbosch(const Graph& G) {
    // Compute degeneracy ordering
    auto core     = core_number(G);
    auto order_pos = degeneracy_order_positions(G, core);

    // Sort vertices by degeneracy order position
    std::vector<NodeId> order(G.n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](NodeId a, NodeId b){ return order_pos[a] < order_pos[b]; });

    std::set<Clique> out;
    std::vector<bool> processed(G.n, false);

    for (NodeId v : order) {
        // P = later neighbours of v (not yet processed), X = earlier neighbours
        std::vector<NodeId> P, X;
        for (NodeId nb : G.adj[v]) {
            if (!processed[nb]) P.push_back(nb);
            else                X.push_back(nb);
        }
        bk_pivot(G, {v}, P, X, out);
        processed[v] = true;
    }
    return out;
}

// ─── Stats helper ─────────────────────────────────────────────────────────────

struct CliqueStats {
    size_t count      = 0;
    size_t max_size   = 0;
    double avg_size   = 0.0;
    std::map<int,int> histogram;
    long long time_ms = 0;
};

CliqueStats compute_stats(const std::set<Clique>& cliques, long long ms) {
    CliqueStats s;
    s.count    = cliques.size();
    s.time_ms  = ms;
    size_t total = 0;
    for (auto& c : cliques) {
        s.max_size = std::max(s.max_size, c.size());
        total += c.size();
        s.histogram[(int)c.size()]++;
    }
    s.avg_size = s.count ? (double)total / s.count : 0.0;
    return s;
}

// ─── Parameter tuner ─────────────────────────────────────────────────────────
//
// Grid-searches K, rel_thresh, max_iter, stable_thresh against BK ground truth.
// Ranks every configuration by recall (cliques found / BK total), breaking ties
// by speed.  Prints a ranked table and highlights the best config.

struct TuneResult {
    int   K, max_iter, stable_thresh;
    float rel_thresh;
    size_t found, bk_total, correct;
    long long time_ms;
    double recall()    const { return bk_total ? (double)correct / bk_total : 0.0; }
    double precision() const { return found    ? (double)correct / found    : 0.0; }
};

void run_tuner(const Graph& G, const std::set<Clique>& bk_cliques) {
    size_t bk_total = bk_cliques.size();

    // Search grids — deliberately wide so the user can see the full picture
    std::vector<int>   K_vals          = {5, 10, 15, 20, 30, 50};
    std::vector<float> rel_thresh_vals = {0.1f, 0.2f, 0.3f, 0.5f, 0.7f};
    std::vector<int>   max_iter_vals   = {50, 100, 200, 500};
    std::vector<int>   stable_vals     = {3, 5, 10};

    int total_configs = (int)(K_vals.size() * rel_thresh_vals.size()
                              * max_iter_vals.size() * stable_vals.size());

    std::cout << "\n== Parameter Tuner (" << total_configs
              << " configs, BK ground truth = " << bk_total << " cliques) ==\n";
    std::cout << "Running grid search";
    std::cout.flush();

    std::vector<TuneResult> results;
    results.reserve(total_configs);

    int done = 0;
    for (int K : K_vals) {
        for (float rt : rel_thresh_vals) {
            for (int mi : max_iter_vals) {
                for (int st : stable_vals) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    auto cliques = algebraic_max_cliques(G, K, mi, 1e-5f, st, rt);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();

                    size_t correct = 0;
                    for (auto& c : cliques) if (bk_cliques.count(c)) correct++;

                    results.push_back({K, mi, st, rt,
                                       cliques.size(), bk_total, correct, ms});

                    if (++done % 20 == 0) { std::cout << '.'; std::cout.flush(); }
                }
            }
        }
    }
    std::cout << " done.\n\n";

    // Sort: primary = recall (desc), secondary = time (asc)
    std::sort(results.begin(), results.end(), [](const TuneResult& a, const TuneResult& b){
        if (std::fabs(a.recall() - b.recall()) > 1e-9) return a.recall() > b.recall();
        return a.time_ms < b.time_ms;
    });

    // ── Full ranked table ───────────────────────────────────────────────────
    // Header
    std::cout << std::left
              << std::setw(5)  << "Rank"
              << std::setw(5)  << "K"
              << std::setw(10) << "rel_thr"
              << std::setw(10) << "max_iter"
              << std::setw(8)  << "stable"
              << std::setw(10) << "Found"
              << std::setw(10) << "Correct"
              << std::setw(10) << "Recall%"
              << std::setw(10) << "Prec%"
              << std::setw(10) << "ms"
              << "\n";
    std::cout << std::string(96, '-') << "\n";

    int rank = 1;
    for (auto& r : results) {
        bool is_best = (rank == 1);
        if (is_best) std::cout << ">>> "; else std::cout << "    ";
        std::cout << std::left
                  << std::setw(5)  << rank++
                  << std::setw(5)  << r.K
                  << std::setw(10) << r.rel_thresh
                  << std::setw(10) << r.max_iter
                  << std::setw(8)  << r.stable_thresh
                  << std::setw(10) << r.found
                  << std::setw(10) << r.correct
                  << std::setw(10) << std::fixed << std::setprecision(1) << (r.recall()*100)
                  << std::setw(10) << std::fixed << std::setprecision(1) << (r.precision()*100)
                  << std::setw(10) << r.time_ms
                  << "\n";
    }
    std::cout << std::string(96, '-') << "\n";

    // ── Best config summary ─────────────────────────────────────────────────
    const TuneResult& best = results[0];
    std::cout << "\nBest configuration:\n";
    std::cout << "  K             = " << best.K             << "\n";
    std::cout << "  rel_thresh    = " << best.rel_thresh    << "\n";
    std::cout << "  max_iter      = " << best.max_iter      << "\n";
    std::cout << "  stable_thresh = " << best.stable_thresh << "\n";
    std::cout << "  => Found " << best.found << " cliques, "
              << best.correct << "/" << bk_total << " verified ("
              << std::fixed << std::setprecision(1) << (best.recall()*100) << "% recall), "
              << best.time_ms << " ms\n";
    std::cout << "\nTo run with best params:\n";
    std::cout << "  ./algebraic_cliques <graph> "
              << best.K << " "
              << best.max_iter << " 1e-5 "
              << best.stable_thresh << " "
              << best.rel_thresh << "\n";

    // ── Parameter sensitivity analysis ──────────────────────────────────────
    std::cout << "\n-- Sensitivity: Recall by K (best rel_thresh/iter/stable per K) --\n";
    std::map<int, double> best_recall_by_K;
    std::map<int, long long> best_time_by_K;
    for (auto& r : results) {
        if (!best_recall_by_K.count(r.K) || r.recall() > best_recall_by_K[r.K]) {
            best_recall_by_K[r.K] = r.recall();
            best_time_by_K[r.K]   = r.time_ms;
        }
    }
    for (auto& [k, rec] : best_recall_by_K)
        std::cout << "  K=" << std::setw(3) << k
                  << "  recall=" << std::fixed << std::setprecision(1) << (rec*100) << "%"
                  << "  time=" << best_time_by_K[k] << "ms\n";

    std::cout << "\n-- Sensitivity: Recall by rel_thresh (best K/iter/stable per thresh) --\n";
    std::map<float, double> best_recall_by_rt;
    for (auto& r : results) {
        if (!best_recall_by_rt.count(r.rel_thresh) || r.recall() > best_recall_by_rt[r.rel_thresh])
            best_recall_by_rt[r.rel_thresh] = r.recall();
    }
    for (auto& [rt, rec] : best_recall_by_rt)
        std::cout << "  rel_thresh=" << std::setw(4) << rt
                  << "  recall=" << std::fixed << std::setprecision(1) << (rec*100) << "%\n";

    std::cout << "\n-- Sensitivity: Recall by max_iter --\n";
    std::map<int, double> best_recall_by_mi;
    for (auto& r : results) {
        if (!best_recall_by_mi.count(r.max_iter) || r.recall() > best_recall_by_mi[r.max_iter])
            best_recall_by_mi[r.max_iter] = r.recall();
    }
    for (auto& [mi, rec] : best_recall_by_mi)
        std::cout << "  max_iter=" << std::setw(4) << mi
                  << "  recall=" << std::fixed << std::setprecision(1) << (rec*100) << "%\n";
}

// ─── Main / comparison + tuner harness ───────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  Normal mode:  " << argv[0]
                  << " <graph> [K=10] [max_iter=200] [tol=1e-5] [stable=5] [rel_thresh=0.5]\n"
                  << "  Tuner mode:   " << argv[0] << " <graph> --tune\n"
                  << "  Supported formats: .mtx, .edges/.edgelist/.el/.txt\n";
        return 1;
    }

    std::string path = argv[1];

    // Check for --tune flag (can be anywhere after the graph path)
    bool tune_mode = false;
    for (int i = 2; i < argc; i++)
        if (std::string(argv[i]) == "--tune") { tune_mode = true; break; }

    // ── Load graph ──────────────────────────────────────────────────────────
    std::cout << "Loading graph from: " << path << "\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    Graph G = load_graph(path);
    auto t1 = std::chrono::high_resolution_clock::now();

    long long edges = 0;
    for (int v = 0; v < G.n; v++) edges += G.degree(v);
    std::cout << "Nodes: " << G.n << "  Edges: " << edges / 2 << "\n";
    std::cout << "Load time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()
              << " ms\n";

    // ── Always run BK first (ground truth) ──────────────────────────────────
    std::cout << "\n-- Bron-Kerbosch (ground truth) --\n";
    auto tb0 = std::chrono::high_resolution_clock::now();
    auto bk_cliques = bron_kerbosch(G);
    auto tb1 = std::chrono::high_resolution_clock::now();
    long long bk_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tb1-tb0).count();
    CliqueStats bk_stats = compute_stats(bk_cliques, bk_ms);
    std::cout << "Found " << bk_stats.count << " maximal cliques in " << bk_ms << " ms\n";

    if (tune_mode) {
        // ── Tuner mode ───────────────────────────────────────────────────────
        run_tuner(G, bk_cliques);
        return 0;
    }

    // ── Normal comparison mode ───────────────────────────────────────────────
    int   K                     = (argc > 2) ? std::stoi(argv[2])   : 10;
    int   max_iter              = (argc > 3) ? std::stoi(argv[3])   : 200;
    float tol                   = (argc > 4) ? std::stof(argv[4])   : 1e-5f;
    int   stable_iter_threshold = (argc > 5) ? std::stoi(argv[5])   : 5;
    float rel_thresh            = (argc > 6) ? std::stof(argv[6])   : 0.5f;

    std::cout << "\n-- AlgebraicMaxCliques --\n";
    std::cout << "Parameters: K=" << K
              << "  max_iter=" << max_iter
              << "  tol=" << tol
              << "  stable_thresh=" << stable_iter_threshold
              << "  rel_thresh=" << rel_thresh << "\n";

    auto ta0 = std::chrono::high_resolution_clock::now();
    auto alg_cliques = algebraic_max_cliques(G, K, max_iter, tol, stable_iter_threshold, rel_thresh);
    auto ta1 = std::chrono::high_resolution_clock::now();
    long long alg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ta1-ta0).count();
    CliqueStats alg_stats = compute_stats(alg_cliques, alg_ms);

    // ── Side-by-side comparison ─────────────────────────────────────────────
    const int W = 22;
    auto sep = [&]{ std::cout << std::string(52, '-') << "\n"; };

    std::cout << "\n";
    sep();
    std::cout << std::left
              << std::setw(W) << "Metric"
              << std::setw(15) << "Algebraic"
              << std::setw(15) << "Bron-Kerbosch" << "\n";
    sep();

    auto row = [&](const std::string& label, auto a, auto b) {
        std::cout << std::left  << std::setw(W) << label
                  << std::setw(15) << a
                  << std::setw(15) << b << "\n";
    };

    row("Runtime (ms)",    alg_stats.time_ms,  bk_stats.time_ms);
    row("Maximal cliques", alg_stats.count,     bk_stats.count);
    row("Max clique size", alg_stats.max_size,  bk_stats.max_size);

    std::ostringstream avg_a, avg_b;
    avg_a << std::fixed << std::setprecision(2) << alg_stats.avg_size;
    avg_b << std::fixed << std::setprecision(2) << bk_stats.avg_size;
    row("Avg clique size", avg_a.str(), avg_b.str());

    size_t alg_correct = 0;
    for (auto& c : alg_cliques) if (bk_cliques.count(c)) alg_correct++;
    double recall    = bk_stats.count ? (double)alg_correct / bk_stats.count * 100.0 : 0.0;
    double precision = alg_stats.count ? (double)alg_correct / alg_stats.count * 100.0 : 0.0;

    std::ostringstream cov_a, cov_b, rec_a, prec_a;
    cov_a  << alg_correct << "/" << alg_stats.count;
    cov_b  << bk_stats.count << "/" << bk_stats.count;
    rec_a  << std::fixed << std::setprecision(1) << recall    << "%";
    prec_a << std::fixed << std::setprecision(1) << precision << "%";
    row("Verified in BK",  cov_a.str(),  cov_b.str());
    row("Recall",          rec_a.str(),  "100.0%");
    row("Precision",       prec_a.str(), "100.0%");
    sep();

    // ── Per-algorithm histograms ─────────────────────────────────────────────
    auto print_detail = [&](const std::string& name,
                             const CliqueStats& s,
                             const std::set<Clique>& cliques) {
        std::cout << "\n" << name << " -- clique size histogram:\n";
        for (auto& [sz, cnt] : s.histogram)
            std::cout << "  size " << std::setw(3) << sz << ": " << cnt << "\n";
        std::cout << "\n" << name << " -- first 10 cliques (1-indexed):\n";
        int printed = 0;
        for (auto& c : cliques) {
            if (printed++ >= 10) break;
            std::cout << "  [";
            for (int i = 0; i < (int)c.size(); i++) { if (i) std::cout << ", "; std::cout << c[i]+1; }
            std::cout << "]\n";
        }
    };

    print_detail("AlgebraicMaxCliques", alg_stats, alg_cliques);
    print_detail("Bron-Kerbosch",       bk_stats,  bk_cliques);

    return 0;
}
