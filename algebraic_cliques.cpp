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

// ─── Main / test harness ──────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <graph.mtx|graph.edges> [K=10] [max_iter=200] [tol=1e-5]"
                     " [stable_thresh=5] [rel_thresh=0.5]\n";
        return 1;
    }

    std::string path = argv[1];
    int   K                    = (argc > 2) ? std::stoi(argv[2])   : 10;
    int   max_iter             = (argc > 3) ? std::stoi(argv[3])   : 200;
    float tol                  = (argc > 4) ? std::stof(argv[4])   : 1e-5f;
    int   stable_iter_threshold= (argc > 5) ? std::stoi(argv[5])   : 5;
    float rel_thresh           = (argc > 6) ? std::stof(argv[6])   : 0.5f;

    std::cout << "Loading graph from: " << path << "\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    Graph G = load_graph(path);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Nodes: " << G.n << "  Edges (approx): ";
    long long edges = 0;
    for (int v = 0; v < G.n; v++) edges += G.degree(v);
    std::cout << edges / 2 << "\n";
    std::cout << "Load time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()
              << " ms\n\n";

    std::cout << "Parameters: K=" << K
              << " max_iter=" << max_iter
              << " tol=" << tol
              << " stable_thresh=" << stable_iter_threshold
              << " rel_thresh=" << rel_thresh << "\n\n";

    auto t2 = std::chrono::high_resolution_clock::now();
    auto cliques = algebraic_max_cliques(G, K, max_iter, tol, stable_iter_threshold, rel_thresh);
    auto t3 = std::chrono::high_resolution_clock::now();

    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count();
    std::cout << "Found " << cliques.size() << " maximal cliques\n";
    std::cout << "Algorithm time: " << ms << " ms\n\n";

    // Stats
    size_t max_size = 0, total = 0;
    std::map<int,int> size_hist;
    for (auto& c : cliques) {
        max_size = std::max(max_size, c.size());
        total += c.size();
        size_hist[(int)c.size()]++;
    }
    std::cout << "Max clique size found: " << max_size << "\n";
    if (!cliques.empty())
        std::cout << "Avg clique size:       " << std::fixed << std::setprecision(2)
                  << (double)total / cliques.size() << "\n\n";

    std::cout << "Clique size histogram:\n";
    for (auto& [sz, cnt] : size_hist)
        std::cout << "  size " << std::setw(3) << sz << ": " << cnt << "\n";

    // Print first 10 cliques
    std::cout << "\nFirst 10 cliques (1-indexed):\n";
    int printed = 0;
    for (auto& c : cliques) {
        if (printed++ >= 10) break;
        std::cout << "  [";
        for (int i = 0; i < (int)c.size(); i++) {
            if (i) std::cout << ", ";
            std::cout << c[i] + 1;
        }
        std::cout << "]\n";
    }

    return 0;
}
