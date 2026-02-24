# Algebraic Maximal Clique Enumeration via Replicator Dynamics

## Overview

This repository contains an implementation of an algebraic maximal clique enumeration algorithm based on replicator dynamics and the Motzkin–Straus formulation of the maximum clique problem.  

The project compares this continuous optimization–based approach against the classical Bron–Kerbosch backtracking algorithm.

The goal is to study:

- The effectiveness of algebraic methods for clique detection
- Parameter sensitivity in replicator-based enumeration
- Empirical performance comparison with exact combinatorial methods
- Practical behavior on sparse and medium-scale graphs

---

## Mathematical Background

The maximum clique problem admits a continuous reformulation via the Motzkin–Straus theorem:

Maximizing  
xᵀ A x  
over the simplex Δ corresponds to detecting maximum cliques in a graph with adjacency matrix A.

Replicator dynamics provides an iterative method:

xᵢ ← xᵢ (A x)ᵢ  
followed by normalization.

Stationary supports of the dynamics correspond to cliques under suitable conditions.

This implementation:

- Uses batched replicator dynamics
- Restricts computation to induced local subgraphs
- Applies degeneracy ordering for canonical enumeration
- Uses core-number pruning to reduce search space
- Validates algebraic supports combinatorially

---

## Implemented Algorithms

### 1. Algebraic Clique Enumeration

- Local induced subgraph construction
- Batched sparse replicator dynamics
- Support extraction via relative thresholding
- Maximal expansion and canonical filtering
- CSR sparse matrix representation

### 2. Bron–Kerbosch (Baseline)

- Classical recursive backtracking algorithm
- Used for:
  - Exact enumeration
  - Verification of algebraic outputs
  - Runtime comparison

---
## Default parameters:
  K = 50
  max_iter = 500
  tol = 1e-6
  stable_iter_threshold = 10
  rel_thresh = 1e-2

---
## Experimental Results

<img width="418" height="177" alt="3" src="https://github.com/user-attachments/assets/d6cb8917-1fe3-48bc-a5b1-50b09275050e" />
<img width="413" height="175" alt="1" src="https://github.com/user-attachments/assets/0336d51f-01ee-49d6-b13e-b7b672e96e9f" />
<img width="414" height="170" alt="2" src="https://github.com/user-attachments/assets/5757cbb2-8e40-4874-b18a-cff2318a3d4d" />


---

## Observations 

 Observations:

   Bron–Kerbosch is significantly faster for sparse graphs.

   Algebraic enumeration depends strongly on parameter selection.

   Differences in clique counts arise from:

   Local neighborhood truncation (parameter K)
  
   Numerical thresholding
 
   Convergence tolerance

   All algebraically detected cliques are verified combinatorially.


---
## Limitations
   Not complete by default: the algorithm is not guaranteed to enumerate all maximal cliques. A clique C may be missed if its members fall outside the top-K neighborhood of its minimum degeneracy-order vertex v.

   Parameter sensitivity: results depend strongly on K, tol, and rel_thresh. Poor parameter choices can cause both false negatives (missed cliques) and wasted computation.

   Dense graphs: performance degrades on dense graphs where local induced subgraphs G[S] become large (up to K² edges).

   Algebraically detected cliques are always verified combinatorially before output, so there are no false positives (soundness holds unconditionally).


---
## Documentation
Formal write-ups are in the docs/ folder:

src/Algebraic-Algorithm.pdf — full pseudocode specification

docs/complexity_analysis_algebraic_max_cliques.md — time and space complexity analysis

docs/correctness_proof_algebraic_max_cliques.md — soundness and conditional completeness proof






---
## Parameter Sensitivity

  Key parameters affecting performance:

  K: number of neighbors considered per vertex

  max_iter: replicator iteration cap

  tol: convergence tolerance

  stable_iter_threshold: support stability requirement

  rel_thresh: support extraction threshold

  Smaller tolerances improve precision but increase runtime.
  Larger K increases coverage but may cause exponential growth in local subgraphs.
---
## How to Run

### Requirements
- C++17 or higher
- CMake ≥ 3.15 (or `g++` directly)

### Compile
```bash
g++ -O2 -std=c++17 src/algebraic_cliques.cpp -o algebraic_cliques


### Run
 ./algebraic_cliques.exe <graph_file.edges>

### Input Format
Plain edge list, one edge per line (0-indexed node IDs):
  0 1
  0 2
  1 2
  2 3
### Datasets Used
examples attached but you can use any other graphs with file formate .edges or .mtx




















---
## Author

Islam Ahmed
PhD student in Discrete Mathematics MIPT
Research interests: combinatorial geometry, discrete geometry, Borsuk problem, algebraic geometry, algebraic graph methods
