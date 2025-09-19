# StringZilla: GPU-accelerated sequence processing

| **Long story short** | **StringZilla** |
|---|---|
| StringZilla v4 ([the StringWa.rs release](https://ashvardanian.com/posts/stringwars-on-gpus/#traditional-string-similarity-measures)) was engineered to move classic dynamic-programming string- and sequence-comparison kernels onto GPUs. As a result, throughput for edit-distance and alignment tasks was increased by orders of magnitude versus single-core CPU runs, making proteome-scale pairwise scoring, library-scale deduplication and other previously HPC-bound workloads practical to run on cloud GPU infrastructure. | [StringZilla](https://github.com/ashvardanian/StringZilla) is a CUDA-capable, SIMD-first library for high-throughput string and biological sequence processing. The library re-implements Levenshtein, Needleman–Wunsch and Smith–Waterman (with Gotoh affine gaps) as GPU-friendly kernels, adds high-performance hashing/fingerprinting primitives and exposes fast, reproducible baselines for deduplication, similarity search and alignment tasks used in databases, Information Retrieval and bioinformatics. |

## Bioinformatics applications and impact

Massive sequence comparison workloads (read alignment, long-read error detection, protein similarity searches, deduplication and large-scale variant matching) were identified as direct beneficiaries of the library’s GPU kernels. For bioinformatics pipelines that compute a large number of pairwise dynamic programming (DP) scores or perform local alignments (Smith-Waterman) with affine-gap penalties, DP evaluation is the main computational bottleneck – moving these kernels to GPUs removes that bottleneck while preserving biologically meaningful scoring (substitution matrices and affine-gap penalties). This makes exploratory analyses (large-scale pairwise similarity, proteome-scale scans, and library deduplication) dramatically faster.

## GPU implementation and optimizations

Classic Levenshtein (Wagner–Fischer) fills a DP matrix row-by-row; even storing only two rows preserves a sequential dependency that blocks vectorization. For parallel/SIMD/GPU implementations the computation is re-ordered along anti-diagonals: by keeping the current diagonal and the two previous ones, each new diagonal can be computed from the prior two, which preserves correctness while enabling massive parallelism.

Affine-gap penalties are handled with the three canonical DP matrices (match/mismatch, gap-open, gap-extend), and protein alignments apply a 20x20 substitution matrix. StringZilla uses NVIDIA integer dot-product primitives (DP4A/DPX-style) to accelerate small-integer DP.

A new AES-based hash was designed for both short-input velocity and long-stream throughput; it supports streaming updates, seed influence on every output bit, dynamic dispatch for different instruction set architectures and identical outputs across platforms. AES rounds are combined with complementary SIMD shuffles and adds to exploit port-parallelism, larger states/blocks are used for inputs >64 B, and masked/predicated loads ensure uniform per-byte weighting. In benchmarks stringzilla::hash reached ~1.84 GiB/s on short lines and ~11.23 GiB/s on long lines.

MinHash signatures are computed using a trick based on double-precision arithmetic: double values (53 bits of precision – 52 stored bits + 1 implicit bit) are used to implement 52-bit integer-style modulo mixing and rolling hashes so the CPU and GPU kernels remain vector-friendly and numerically consistent. This yields large speedups (single-threaded Rust ~ 0.5 MiB/s vs H100 GPU ~ 392.37 MiB/s in the author’s tests) with competitive collision rates and entropy. 

Finally, string sorting is accelerated by pre-sorting integer prefixes, improving comparison throughput (e.g., `sz::argsort_permutation` ~182.9M comparisons/s vs `std::sort_unstable_by_key` ~54.4M).

## Reported performance highlights

Performance was measured in Cell Updates Per Second (CUPS) and reported using the author’s MCUPS notation for large-scale throughput numbers. Benchmarks were run on ~1,000-byte string comparisons (for generic edit-distance) and ~1,000-length amino-acid sequences (for protein alignment), and the following results were reported:

For Levenshtein / edit-distance (~1,000-byte strings):
* `rapidfuzz::levenshtein`: ~**14,316 MCUPS** when executed on a single Intel Sapphire Rapids core.
* `stringzillas::LevenshteinDistances`: ~**13,084 MCUPS** on that same single Sapphire Rapids core.
* `stringzillas::LevenshteinDistances`: ~**624,730 MCUPS** running on an NVIDIA H100 GPU, demonstrating an orders-of-magnitude throughput uplift when the kernel was mapped to a modern GPU accelerator.

For context, legacy Python tooling was far slower: NLTK was reported at roughly ~2 MCUPS. A RAPIDS-based comparison (`cudf` wrapping `nvtext::levenshtein`) was reported at ~1,950 MCUPS on the same H100 – a reminder that not all GPU-enabled stacks deliver the same kernel-level throughput (and that specialized implementations can substantially outperform general-purpose wrappers). A commercial Nvidia offering (Clara Parabricks) was noted as a likely competitor for optimized genomics workloads.

For protein alignment (Needleman-Wunsch / Gotoh with substitution costs and affine gaps, ~1,000-length amino-acid sequences):
* `biopython`: ~**303 MCUPS**.
* `stringzillas-cpus`: ~**276 MCUPS**, comparable to Biopython on a single core.
* `stringzillas-cuda`: ~**10,098 MCUPS** on an NVIDIA H100, again showing a multi-order throughput advantage for alignment workloads when the DP kernels were fully vectorized and executed on GPU.

## Infrastructure and reproducibility

The benchmarks were run on Nebius cloud GPU instances provisioned with NVIDIA H100 accelerators. Local CPU baselines were measured on Intel Sapphire Rapids cores and comparisons were made against AMD Zen4 in microbenchmarks where relevant. CUDA-enabled toolchains and platform-specific wheels were built and deployed via GitHub Actions to ensure consistent runtime environments across machines. RAPIDS (`cudf` / `nvtext`) was used as a representative GPU-enabled comparison stack and Biopython was used as a representative single-core CPU baseline for protein alignment workloads.

## Future roadmap and next steps

Planned work will focus on broadening platform support, scaling to multi-GPU and distributed deployments, optimizing memory and data layouts, and adding multiple-sequence alignment capabilities:

* Support for additional GPU backends will be expanded, including ROCm for AMD accelerators, so users on diverse cloud platforms can run GPU-accelerated kernels.

* Multi-GPU and distributed execution modes will be implemented to enable very large-scale pairwise scoring and batch alignment across nodes.

* Memory and data-layout optimizations will be applied (for example, moving substitution matrices out of constant memory into tiled/texture-backed layouts) to remove current bottlenecks.

* A multiple-sequence-alignment (MSA) algorithm will be developed and integrated to broaden the library’s applicability to more complex alignment tasks.

These items will be pursued while leveraging Nebius’ cloud GPU infrastructure to validate performance at scale and to ensure the work remains practical and reproducible for the broader bioinformatics community.