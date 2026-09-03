# Conformer Search Baseline Design

## Goal

在 `xtb-ase` 中加入一个不强制依赖 PyTorch、Numba、iRMSD 或 `pamssw` 的
构象/团簇构型搜索 baseline：它能够生成可复现的高温 GFN-FF Langevin 轨迹，
按相对能量窗口分层并用 FPS 进行廉价下采样，使用精确 iRMSD（可选依赖）或
明确标注为近似的距离指纹完成去重，并为后续 MACE/TorchSim 重排和 g-xTB
热化学收口提供稳定的数据契约。

## Scope

本次实现包含：

1. ASE-based Langevin trajectory iterator/list helper，参数包含温度、步长、摩擦、
   步数、抽帧间隔和随机种子。
2. 以 ASE eV 为内部能量单位的相对能量窗口和能量分层 FPS；Numba 仅作为可选
   加速后端，NumPy 实现始终可用。
3. 由按元素分块排序的成对距离组成的刚体变换/原子重排不变预筛选指纹。
4. 可选 `irmsd` 后端；缺少可选依赖时对 `rmsd_backend="irmsd"` fail closed，
   不静默把近似距离指纹冒充精确 RMSD。
5. 统一的 `ConformerSearchResult`、候选记录、能量分层和聚类成员索引，保证
   选出的代表结构来自输入轨迹且按最低能量优先保留。
6. g-xTB `--hess` 输出中的 total free energy、enthalpy、zero-point energy、
   thermochemical correction 和温度解析为 ASE calculator properties，并保留
   `get_gibbs_free_energy()` 等明确单位的 Python API。
7. 可选 `pamssw` adapter：将 ASE `Atoms` 转为 PAM-SSW `State`，用现有
   `SSWConfig.target_uphill_energy` 配置小 uphill-height，对外返回与 baseline
   兼容的 minima/energy 摘要。该依赖不加入基础安装，也不参与 baseline 主路径。

## Non-goals

- 本次不实现 MACE 模型加载、TorchSim backend 或 latent feature 的具体模型绑定；
  pipeline 通过 callable descriptor/refiner 接口预留这些能力。
- 本次不把 iRMSD 重新实现为 Python/Numba 算法；精确置换、旋转和对称处理由
  `irmsd` optional dependency 负责。
- 本次不实现 PLUMED、PySAGES、SPONGE ITS/SITS 或 BO；它们作为后续同预算消融。
- 本次不把 g-xTB RRHO free energy 当作原始 MD 快照的自由能；热化学字段只在
  `--hess` 输出可用时解析，文档明确要求对优化后的候选进行 Hessian。
- 不修改 `pam-ssw` 仓库；adapter 只依赖其公开 Python API。

## Architecture

```text
sample_langevin(atoms, GFNFF)
        -> frames + source metadata
        -> energy extraction / relative energy window
        -> default pair-distance fingerprint or injected MACE descriptor
        -> energy-stratified FPS
        -> exact iRMSD dedup (or explicit approximate fallback)
        -> ConformerSearchResult
        -> optional injected batch refiner
        -> GXTB --hess thermochemistry API
```

### Geometry and descriptor contract

`pair_distance_fingerprint(atoms)` returns a one-dimensional float array. Distances
are grouped by unordered atomic-number pairs and sorted inside each group. Therefore
it is invariant to rigid translation/rotation and permutation of atoms with the same
element, but it is not a complete graph invariant and is not called an RMSD.

An injected descriptor callable has signature:

```python
descriptor = descriptor_fn(atoms)  # shape (n_features,)
```

All descriptors are validated to be finite, one-dimensional, and equal length before
FPS. Descriptor standardization is performed column-wise over the candidate set, with
zero-variance columns assigned unit scale.

### Selection contract

```python
config = SearchConfig(
    energy_window_kcal_mol=6.0,
    max_selected=32,
    energy_bins=4,
    fps_backend="auto",
    rmsd_backend="irmsd",
    rmsd_tolerance_angstrom=0.125,
)
result = ConformerSearch(config).select(frames, energies=energies)
```

Energy input is in eV and must be one scalar per frame. If omitted, the selector reads
`atoms.info["energy"]`, an attached calculator's cached `results["energy"]`, or calls
`atoms.get_potential_energy()` as a last resort. The relative window is converted from
kcal/mol using ASE's `units.kcal / units.mol`.

Selection is deterministic for fixed input, descriptor values, configuration, and
random seed. It first retains frames within the relative energy window, partitions
them into equal-width energy strata, allocates slots with a low-energy preference,
runs FPS within each non-empty stratum, and finally processes selected structures in
ascending energy order for deduplication. The lowest-energy eligible frame is always
included.

### Deduplication contract

`rmsd_backend="irmsd"` lazily imports `irmsd.get_irmsd_ase` and compares a new frame
with the lowest-energy representative of each existing group. Missing `irmsd` raises
an actionable `ImportError` instructing the user to install the search extra.

`rmsd_backend="distance_fingerprint"` compares the normalized RMS distance between
the pair-distance fingerprints and marks the result as approximate in diagnostics.
This mode is intended for dependency-light smoke tests and prefiltering, not final
scientific claims. `rmsd_backend="ordered"` is an explicit fixed-atom-order Kabsch
fallback for controlled tests only.

### Refinement and external methods

The result contains source indices and copies of `Atoms`, so a future MACE/TorchSim
batch refiner can consume `result.representatives` and return a sequence of refined
`Atoms` plus energies without changing selection. The first implementation does not
pretend to provide a TorchSim adapter where the dependency is absent.

`run_pamssw(initial_atoms, calculator, target_uphill_energy=0.05, ...)` lazily imports
`pamssw`, maps the input to its `State`, invokes `run_ssw`, and converts archive states
back to ASE. The default comparison value is deliberately small but remains an
explicit parameter in eV; benchmark scripts must record it rather than rely on a
hidden default.

### g-xTB thermochemistry

Thermochemistry property requests trigger the existing g-xTB Hessian operation. The
parser recognizes both the `::` and boxed `| TOTAL ... |` output forms. Returned
calculator result values are eV; `get_entropy()` is not added until a stable, tested
entropy unit/output contract is available. A request for free energy without a
Hessian-capable binary raises the same controlled execution/parse error as other
missing g-xTB outputs.

## Error handling

- Empty frame sequences, mismatched atom counts/compositions, non-finite positions,
  non-finite energies, invalid windows, invalid FPS counts, and inconsistent descriptor
  shapes raise `ValueError` before selection.
- Periodic frames are rejected by the non-periodic Langevin conformer helper unless
  all cell/PBC data are identical and the caller uses an injected descriptor/deduper;
  no unsupported periodic iRMSD claim is made.
- Optional dependency errors identify the missing package and the corresponding extra.
- A failed external refiner is represented by its caller; selection itself never
  suppresses calculator failures or silently inserts an invalid structure.
- All returned `Atoms` are copies and do not share a mutable calculator with the input
  trajectory.

## Testing and evidence boundaries

Unit tests must cover:

- Langevin configuration validation, deterministic frame count and detached frame
  calculators using ASE's Lennard-Jones calculator.
- Pair-distance fingerprint invariance under rigid transforms and same-element atom
  permutation, plus composition/shape validation.
- NumPy FPS exact behavior, backend selection, energy-window filtering, low-energy
  inclusion, strata allocation, deterministic tie breaking, and finite descriptors.
- Ordered Kabsch RMSD and explicit approximate diagnostics; an optional iRMSD test is
  skipped when the wheel is unavailable.
- Result groups retaining the lowest-energy representative.
- g-xTB thermochemistry parsing in `Eh` and calculator conversion to eV using the
  existing fake executable.
- PAM-SSW adapter validation through a monkeypatched module boundary, plus an optional
  integration test when the local `pamssw` checkout is installed.

The future benchmark must separate API/unit correctness, runtime/resource usage,
numeric convergence, and scientific conformer coverage. It will compare direct
GFNFF MD, CREST 3.1, the new baseline, and PAM-SSW with equal seeds and force/
CPU-hour budgets. No benchmark result is claimed by this implementation change.
