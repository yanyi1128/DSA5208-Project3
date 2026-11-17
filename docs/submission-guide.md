# Submission and team handoff

This guide separates the current deliverable from experiments that can still be added if Vanda GPU resources become available. It does not certify that unmeasured configurations work or that the submission will receive a particular grade.

## Current evidence boundary

- The CPU MPI contribution contains real Vanda strong- and weak-scaling measurements through 256 CPU ranks. The measured weak-scaling base is `N0 = 1024`; earlier preparation examples using `256` are not the final data configuration.
- The CUDA contribution has been executed on a local NVIDIA RTX 3070 Laptop GPU with one MPI rank. Small correctness cases passed against freshly generated serial fields for both tested block shapes. Local GPU measurements must retain their local platform label.
- Vanda GPU and multi-rank/multi-GPU CUDA runs remain unmeasured. A single-rank run does not validate inter-rank halo exchange, rank-to-GPU placement across nodes, or multi-GPU scaling.
- Formal timing records with `validation = not_run` remain so. A separate correctness suite supports the implementation but does not turn every large timing run into a validated field comparison.
- Serial timings and speedup baselines must identify their execution environment and timed region. Do not interpret a ratio between unlike machines as a controlled hardware speedup experiment.

The report should explicitly state these boundaries, explain the algorithm and design, analyse the available scaling evidence, describe correctness checks, and disclose AI assistance. The assignment asks for runtime figures from Atlas or Vanda; the Vanda CPU MPI results should be identified directly. Local GPU evidence must not be presented as Vanda GPU evidence. Whether the remaining GPU coverage is sufficient for marking is the instructor's decision.

## What belongs in the final delivery

Include:

- The final PDF report with all group members' names and student IDs checked by the group.
- Reproducible source: `src`, `include`, the relevant build files, scripts and tests, plus `SPEC.md` and `README.md`.
- Selected raw measurement CSVs, processed CSV summaries and report figures, with enough concise provenance to identify platform, source version, problem size, process/GPU count and repetitions. Original numerical CSV data may be retained; keep local results clearly labelled.
- The source-code access method required by the assignment. If choosing a public GitHub link, the final report must link to a repository that a marker can actually access. A private repository link alone is insufficient for that option.

Exclude from the final report/repository archive:

- Log files and terminal-output transcripts, including `.stdout`, `.stderr`, PBS output/error files, and screenshots of terminal output.
- Complete `results/runs` directories or raw audit JSON that embeds terminal output. Keep these privately as reproducibility evidence; extract a concise provenance description for the report instead.
- Build directories, executables, object files, caches, machine-specific temporary files, credentials and obsolete handoff ZIPs.
- Empty result/figure placeholders, duplicate report versions and claims that queued GPU jobs have completed.

The scripts are allowed to generate diagnostic logs when run. The exclusion applies to the delivered artefacts, not to retaining evidence privately during development.

## Check before packaging

1. Use the commands in the root README. Bare `make` and the legacy `scripts/run_serial.sh` do not build/run a complete experiment in this snapshot. `make -f Makefile.cuda serial` builds the serial solver without CUDA; CPU MPI uses the explicit `mpicxx` command.
2. Ensure the report's figures and tables match the selected CSV data, including warm-up exclusion, repetition count, medians and local/cluster labels. Retain unusual measurements rather than removing them without explanation.
3. Keep validation assertions within the tested scope. Report Vanda GPU and multi-GPU CUDA as limitations if no actual results arrive.
4. Check the final PDF for missing figures, broken references, placeholder author details and truncated text. Check names, IDs and AI-use disclosure with the group.
5. Inspect the archive contents for the exclusions above. Keep the full private evidence bundle separately from the submission archive.

## Integrate with the team's GitHub repositories

These working directories originated from ZIP downloads. They are not necessarily independent Git clones; a Git command may discover an unrelated parent workspace repository. Do not push from them until the repository root and remote have been checked.

Use a fresh clone of the team's current source repository, create a contribution branch such as `codex/cuda-contribution`, and copy/review only the intended source and documentation changes. Compare with the latest teammate work before merging. Do the same separately for the report repository. Commit selected result CSVs with their correct provenance; do not copy the private run directories or overwrite teammates' newer files with an older ZIP snapshot.

Publishing or pushing is a separate team action. No push or public-access change is implied by preparing this local submission. Before finalising a public-link submission, verify that the link resolves to the intended final version without relying on the author's authenticated session.

## If Vanda GPU results arrive

The default `scripts/vanda_cuda.pbs` job runs a single-GPU pilot only. Check its completion, environment and validation results first. A successful pilot is followed by a separate benchmark submission using `CUDA_PHASE=benchmark`; multi-GPU work needs an appropriate allocation and explicit rank/GPU validation.

Import actual measurements into the report only after checking the source version, configuration, raw repetitions and validation status. Add their Vanda hardware/software provenance and update the limitations accordingly. Keep the local results separately identified. A successful pilot alone is not a completed scaling study.
