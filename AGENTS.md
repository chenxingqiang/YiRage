# AGENTS.md

## YiRage 无限优化闭环（Infinite Optimization Loop）— **RuntimeFusion × MetaX MACA**

> **当前主目标（2026-07-27）**：以 **RuntimeFusion（RF）** 将搜索期 **FusionPlan** 裁成可调度的 **FusionCapsule**，嵌入 vLLM / SGLang：引擎保留连续批处理、PagedAttention、RadixAttention 与总调度权；YiRage 在 serving **每一步**按引擎 meta **弹性选择/编排**融合块。这是「引擎原生协同」的工业路径，**不是**把整网烤进独占 MegaKernel 的编译霸权。MACA 真卡与 CUDA 执行能力仍是**支撑轨**。

> **当前主后端**：Loop **默认合并闸门**为 **CPU**（`YIRAGE_BACKEND=cpu` + real torch / `yirage.core`）验证 **逻辑与功能链路**（[G6](#设计目标价值评价锚点)）；CUDA / MetaX MACA / 其它真卡仅在 **目标环境** 做 parity 与 bench（见 [G6 验证阶梯](#g6-验证阶梯cpu-优先--目标环境真卡)）。CPU 闭环历史见文末 [历史归档](#历史归档cpu-无限优化闭环)。MACA↔CUDA 能力矩阵见 [CUDA 对标（支撑轨）](#cuda-对标优化目标与能力矩阵)。

> **概念切断（强制）**：对外叙事与新 Loop **禁止**以 Mirage / MPK / µGraph 作为产品身份。遗留符号（`mugraph`、`MuGraphStore`、`PersistentKernel` 类名等）按 [Legacy 对照表](#概念对照legacy--yirage-标准) 视为实现别名，符号大改名另开 Chore，不阻塞 S1。

> **历史北极星（2026-07-08，已降级为支撑）**：MACA 算子/搜索/融合逐项对标 CUDA。R0–R26 已闭合离线 Qwen3 PK 脚手架；旧 R27 offline latency 归档降级。S0 起主轨为 RuntimeFusion。

本仓库的持续改进**没有终止条件**。每一轮闭环的目标不是「做完就停」，而是：**感知现状 → 选定瓶颈 → 最小落地 → 用证据验证 → 把结论写回文档与契约 → 进入下一轮**。Cloud Agent 与人类协作者都应把 `AGENTS.md` 当作活文档，每轮验证通过后更新本节或下方 Gotchas。

**不要**为此闭环新增独立编排脚本（例如一键跑完全部阶段的 orchestrator），除非用户明确要求。闭环由 Agent 按层执行现有脚本与测试，并把经验沉淀进文档。

## Agent Development Protocol（Mandatory）

> Cloud Agent 与人类协作者均须遵守。与 [Serving 验证禁令](#serving-验证禁令严禁2026-07-27-起永久有效)、[Agent 行为性价比审查](#agent-行为性价比审查每步必做)、[Loop 真实价值判断](#loop-真实价值判断每轮必写) 冲突时，**本协议优先**。

### Core principles

| # | Principle | Requirement |
|---|-----------|-------------|
| 1 | **Test-Driven Development (TDD)** | Write tests first, then code. Requirements must be verifiable before implementation lands. |
| 2 | **Phased development** | Work strictly in **Design → Development → Testing → Verification**. Each phase has a clear objective; do not blur phases. |
| 3 | **Design-first precision** | The Design phase must be detailed and unambiguous. Do not arbitrarily change Design decisions in later phases; any change needs explicit justification and **human confirmation**. |
| 4 | **Minimal change** | Verification-phase edits must be minimal and targeted—fix what tests expose, not broad refactors. |
| 5 | **Production-targeted development** | See [Production standards](#production-standards) below. |
| 6 | **Real-environment validation** | Validate on the actual target system or a strictly identical environment. Cloud CPU green **does not** substitute GPU/MACA Serving gates. |
| 7 | **Minimize file creation** | **Modify, don't create.** New files are last resort; require clear architectural justification and **human approval** before adding. |
| 8 | **Lean documentation** | No superfluous summary reports. After feature completion, update **`README.md`** for user-facing changes; update **`AGENTS.md`** for agent protocol, gates, and loop notes. |
| 9 | **Pre-modification review** | Before any change, review existing code, tests, and docs. Changes must be minimal, necessary, and applied to existing artifacts where possible. |
| 10 | **Human approval gate** | Critical design choices, scope changes, new files, and merge-to-main require review and approval by the human programmer unless the user explicitly delegates. |
| 11 | **Collaborative iteration** | Iterate Design → Development → Testing → Verification → Refinement with the human; declare the current phase in PR notes or conversation when material. |

### Production standards

- **English-only artifacts**: Code, comments, commit messages, and documentation added or modified by Agent must be clear, professional **English** (existing non-English user docs may stay; do not expand Chinese in new agent-authored artifacts unless the user asks).
- **No mocking / simulation for core logic**: Do not use mocks, stubs, or simulated engines for Serving/RF **verification** paths. Real PyTorch (`TorchEngineModel`), real `vllm` when testing vLLM hooks, and MetaX VM for MACA are required. Legacy offline modules (e.g. `engine_stub.py`) must not be re-wired into cert or pytest.
- **Production-ready focus**: Every change must be concise, maintainable, and suitable for production integration—not throwaway demos, smoke scripts, or contract-only shortcuts.

### Phase checklist (Agent)

```
Design       → Requirements clear; test plan drafted; human approves design before coding.
Development  → Implement to approved design; unit/integration tests written per TDD.
Testing      → Run tests; report failures; no silent edits during test execution.
Verification → Minimal fixes from test evidence; human sign-off before merge.
Refinement   → Update README.md (user-facing) + AGENTS.md (protocol/loop notes); commit per phase.
```

### Explicit prohibitions (Serving / RF)

These reinforce item 5–6 and [Serving 验证禁令](#serving-验证禁令严禁2026-07-27-起永久有效):

- No `demo/serving/*smoke*.py`, `--contract-only`, or NumPy stub cert paths.
- No mock vLLM layers or torch fallbacks posing as RuntimeFusion parity.
- No new verification files when extending `tests/python/test_runtime_fusion_s*.py` suffices.

### 概念对照（Legacy → YiRage 标准）

| 维度 | 遗留概念（Legacy，勿作对外身份） | **YiRage 标准概念** | 工业界语义对齐 |
|------|----------------------------------|---------------------|----------------|
| **搜索产物** | µGraph / MuGraph / `MuGraphStore` | **FusionPlan** | 优化后的局部子图执行计划（Execution Plan） |
| **服务单元** | MegaKernel / MPK / 「一层烤死的 PK」 | **FusionCapsule** | 引擎侧可调度的动态融合粒度（Fused Block） |
| **运行时** | Mirage-style Persistent MegaKernel 霸占 | **RuntimeFusion（RF）** | 引擎内嵌的动态选择与编排（Dynamic Fusion Runtime） |
| **图层级（文档用语）** | KN Graph / TB Graph | **Device Graph / Tile Graph** | 设备级与 Tile 级算子抽象（代码符号可暂留 `KN*`/`TB*`） |
| **执行战术（非产品身份）** | `PersistentKernel` worker/scheduler | Capsule 的可选 launch backend | 持久化 worker 只是实现手段之一 |

**一句话关系**：

```text
引擎调度 (vLLM / SGLang)
    ↓ meta（block_table / radix hit / batch / SM 预算）
RuntimeFusion.step(...)
    ↓ 本步选择 0..N 个边界可变的
FusionCapsule+   ← 可降级回引擎原生算子
    ↑ 来自搜索期缓存的
FusionPlan 库    ← superoptimize / profile / shape-bucket cache
```

**与「编译霸权」的刻意对立**：FusionPlan 是搜索期方案，不是整网死刑判决；Capsule 边界可随 step 伸缩；RF 把调度权交还引擎，只拥有**融合身份**。

### 核心冲突与设计抉择（必读）

| 侧 | 假设 | 冲突点 |
|----|------|--------|
| **vLLM / SGLang** | 算子级流式调度 + 显存碎片化（`block_table` / Radix 前缀缓存） | 需要细粒度抢占、动态 KV、辅助算子（Sampler、NCCL、Vision）可随时 launch |
| **错误路径（已废弃）** | 整图/大图烤进独占持久化内核 | 与连续批处理、Radix、辅助流冲突；重复「编译霸权」 |
| **RuntimeFusion（正道）** | 弹性融合：按 meta 选 Capsule | 须定义 Capsule ABI、最小 `step` 钩子、meta 桥与 SM 预算 |

**结论**：禁止「替换单个 Attention 算子」冒充完成；采用 **Block/Layer-level Override + RF**：
1. 在 `vllm/model_executor/models/{qwen2,llama}.py`（或 SGLang 等价 model）覆盖可融合区间，而不是只改 `layers/` 内孤立 Linear。
2. 搜索/构图得到 **FusionPlan**；裁成 **FusionCapsule**（首刀：单层 MLP）。`forward`/`RF.step` 按本步 meta 决定是否启用该 Capsule。
3. KV / 调度元数据经 meta 缓冲实时注入；需要 Attention Capsule 时须支持 `block_table` / indptr **非连续寻址**（S4）。
4. Capsule launch 的 SM 占用可配置，给 Sampler / NCCL / 多模态留辅助流（S5）。
5. Radix：拓扑不必烤死；按 cache-hit meta **跳过或收缩** Capsule 工作集（S6）。

**建议切入点（硬约束）**：S1 交付 **第一个可被 RF 选中的 MLP FusionCapsule + 最小 `step` 钩子**（RMSNorm + Gated-Linear；**PagedAttention 仍留引擎**）。脚手架可复用实现细节：`demo/qwen3/demo_chat.py`、`demo/maca/qwen3_pk_*.py`、`qwen_kernel_utils.superoptimize_mlp_*`——但叙事与 API 外壳须挂在 FusionCapsule / RF 下，不得宣传为 MegaKernel/MPK。

### 核心原则

| 原则 | 含义 |
|------|------|
| **Serving / RF 优先于离线 demo** | 新 Loop 须闭合 [RuntimeFusion 能力矩阵](#runtimefusion-能力矩阵对标-vllmsglang) 的一行；纯 offline demo 美化 **降级** |
| **弹性融合先于全图独占** | 未打通「引擎调度 + MLP Capsule + `RF.step`」前，禁止全模型单一内核替换 |
| **融合身份 ≠ 调度权** | Capsule/RF 拥有融合；vLLM/SGLang 拥有批处理与 KV 池 |
| **先测后优** | 没有正确性与可复现 serving/RF **pytest（real torch）**，不改 Capsule ABI / meta / SM 配额 |
| **同后端评估** | Search/profile/execute 与 serving 插件须在同一 `backend` 完成 |
| **瓶颈驱动** | 优先：Capsule + `step` 钩子 → 层覆盖 → meta/`block_table` → SM → Radix |
| **最小改动** | 每轮 1～2 个瓶颈；复用现有搜索/执行实现作 legacy backend；禁止平行重写 runtime |
| **文档先于符号大挪移** | S0/S1 完成品牌与概念切断；`MuGraphStore` 等改名另开 Chore |
| **验证通过再沉淀** | RF/serving pytest 或 e2e demo 通过后再更新 `AGENTS.md` |
| **执行前价值闸门** | 每轮对照 [Serving 分阶](#serving-loop-分阶路线-s0sn) 与四轮自问 |
| **Loop 真实价值判断** | 每轮策略前按 [设计目标](#设计目标价值评价锚点) 判定 **做/不做/降级**；合并后写入 [轮次笔记](#当前轮次笔记serving--maca由-agent-持续追加) 的 **价值** 字段 |
| **行为性价比** | 禁止 stub 插件 / torch 回退 / **NumPy stub smoke** 冒充 RuntimeFusion |

### Agent 行为性价比审查（每步必做）

Cloud Agent 在**每一步**行动前须自问并写入 PR / 本轮笔记（可 1 句话）：

| 维度 | 问题 |
|------|------|
| **目标对齐** | 本步是否直接闭合当前 Serving 策略卡片（S{n}）的主瓶颈，且服务 [设计目标 G1–G7](#设计目标价值评价锚点)？ |
| **正式 vs 回退** | 是否在绕开根因（stub 插件、整层 torch 回退、用「烤死 MegaKernel」冒充 RF）？ |
| **功能闭合** | 是否推进 **G7** 一条 e2e 功能链（或明确段落/基础设施/拒绝零散堆砌）？ |
| **证据** | 是否有 **G6 CPU** RF pytest（real torch / `yirage.core`）或 **G7** 链路 demo？真卡/MetaX 仅作目标环境补充 |
| **机会成本** | 是否更应打通 Capsule/`step`/meta/SM，而非 offline 抛光或 legacy 大改名？ |

**结论为「性价比不足」或「属于回退」时**：停止落地，回到策略层重选 backlog；不得为凑 Loop 合并 PR。

### Loop 真实价值判断（每轮必写）

每一轮 Loop（Serving **S{n}** / MACA **R{n}**）在 **策略层写代码之前** 须完成一次 **真实价值判断**，并在 **PR 描述** 与 **合并后的轮次笔记** 中各写 **1～3 句话**。

**核心要求**：价值 **必须围绕 [设计目标](#设计目标价值评价锚点) 评价**——先回答「本轮推进了哪条设计目标、哪一行 RF/MACA 矩阵」，再给出四档结论；**禁止**仅用「pytest 绿了 / bench 跑了 / 文档写了」代替设计对齐。

#### 设计目标（价值评价锚点）

价值判断的**唯一参照系**是 YiRage 产品/design 目标，而非孤立性能数字或提交量。

| # | 设计目标 | 一句话检验 |
|---|----------|------------|
| **G1** | **引擎原生协同** | 是否让 vLLM/SGLang **保留**调度、PagedAttention、Radix、连续批处理，RF **只**编排 FusionCapsule？ |
| **G2** | **弹性融合（非编译霸权）** | 是否增强「按 step meta 选择/跳过/收缩 Capsule」，而非整网/MegaKernel/MPK 独占？ |
| **G3** | **FusionPlan → Capsule → RF.step 链路** | 是否闭合搜索期 Plan 裁成 Capsule、经 `RF.step` 在 serving 路径可验证地执行？ |
| **G4** | **Capsule 边界与 meta 桥** | 是否推进 MLP/decoder 片段 Capsule、KV/`block_table`、Radix hit、SM 预算等 **meta 打针**？ |
| **G5** | **正确性先于吞吐** | 是否有 Capsule/RF **numerical parity** 证据，再谈 latency/speedup？ |
| **G6** | **CPU 优先验证逻辑与功能链路** | 是否先在 **CPU**（`YIRAGE_BACKEND=cpu` + real torch / `yirage.core`）闭合逻辑、API、功能链路？CUDA/MACA/其它真卡留到 **目标环境**再验，不阻塞 CPU 轮合并 |
| **G7** | **完整功能闭合（非零散堆砌）** | 是否闭合一条 **端到端可跑通** 的功能链（明确输入→经过 RF/Plan/Capsule/引擎协同→可观测输出），而非仅 registry/命名/单点 helper/归档 JSON？ |

**反模式（默认低价值-拒绝）**：单 Attention 替换、offline demo 抛光、Mirage/MPK/µGraph 对外叙事、孤立 beat cuBLAS/mcPytorch、**registry/命名/FAST_PATH 闭合**、**无 e2e 链路的单测堆砌**、stub/torch 回退冒充 RF。

#### 何时写

| 时机 | 写什么 |
|------|--------|
| **策略前（闸门）** | 勾选 **G1–G7** 中本轮主服务项；判定 **做 / 不做 / 降级**；**G7**：本轮闭合哪条功能链，或明确仅为下一链路的阻塞项 |
| **PR 描述** | 「价值判断」：**设计目标对齐（Gx）** + 四档结论 + 1 条证据 |
| **合并后进化** | 轮次笔记必填 **价值** 字段（见下方模板） |

#### 价值结论（四档，必选其一）

四档结论 **必须** 能追溯到上表 **G1–G7**（主轨至少 **G1+G3 或 G4** 之一，且说明 **G7** 功能链闭合程度；**G6** 为默认验证阶梯）。

| 结论 | 含义（相对设计目标） | 合并门槛 |
|------|----------------------|----------|
| **高价值** | 直接推进 **G1–G5** 并闭合 RF 矩阵一行；且 **G7** 有一条可演示的 e2e 功能链（非单点/registry） | 矩阵 **CPU** 验证 + **G7 功能链** pytest/demo 全绿 |
| **中价值** | 为 **G3/G4/G5/G6/G7** 建立可回归基线或解除下一档主瓶颈；**G7** 仅闭合链路中的 **一段**（须在笔记写明「部分闭合」与下一段 backlog） | cert pytest 或 archive JSON + **功能链段落说明** |
| **低价值-拒绝** | 不服务 **G1–G7**；**G7 反模式**（零散 registry/命名/单测无链路）；或跳过 **G6** 用真卡冒充 | **不得合并** |
| **支撑轨** | 用户点名或编译/ABI **阻塞** Serving 的 **CUDA/MACA** 专项（**不等同于 G6**） | CPU 契约 pytest 绿 + **目标环境**（MetaX/NVIDIA）真卡 smoke；纯 CPU 不得宣称 GPU parity |

#### 策略卡片 + 轮次笔记模板（必填字段）

```text
价值：<高价值|中价值|低价值-拒绝|支撑轨> — <一句话结论>
设计目标：<G1–G7 编号 + 如何推进/不推进>
功能闭合：<G7 — 本轮闭合的 e2e 功能链名 + 输入→输出；或「部分闭合：…」/「仅基础设施，不宣称 G7」>
矩阵：<RuntimeFusion 能力矩阵 或 MACA 矩阵 哪一行>
理由：<为何服务设计目标；为何不是零散堆砌>
证据：<pytest / e2e demo / cert stage；真卡仅当已在目标环境跑过>
拒绝项：<未做且违背设计目标的工作；无则写「无」>
```

**示例（Serving S27，围绕设计目标）**：

```text
价值：中价值 — decode-step bench 验证 G3/G5/G6；G7 部分闭合。
设计目标：G3（Plan→Capsule 在 q_len=1 可测）、G5（parity 先于 speedup）、G6（CPU）。
功能闭合：部分闭合 — 「HF prefill → 单步 decode（native vs RF MLP）→ logits parity」；未闭合多步 generation / vLLM 插件全链（G1）。
矩阵：yirage_cpu MLP Capsule 执行（partial S14）。
证据：test_runtime_fusion_s27_qwen_decode_bench.py；serving_qwen_decode_bench.py --quick（parity_ok）。
拒绝项：MLP-only 微 bench — 若只做 timing 无 parity 链，违反 G7，留 S30。
```

**示例（Serving S29，围绕设计目标）**：

```text
价值：中价值 — tier archive 服务 G3：为 FusionPlan 搜索 tier（seed_verify vs full_tb_ray）提供选型证据。
设计目标：G3（Plan 搜索成本可见）；不直接推进 G1/G2 引擎嵌入。
功能闭合：仅基础设施 — archive/CI 选型 JSON；不宣称 G7 功能完成；链 E 段落闭合留 S30+。
矩阵：FusionPlan 搜索/缓存（partial）；nightly 量化 full_tb_ray 成本 ~500×。
理由：设计目标要求 Plan 可缓存、可按 tier 选型；非 latency 优化轮。
证据：make test-serving-search-tier-nightly-profile；compare_ok JSON。
拒绝项：decode nightly artifact — 属 G5/G7 延伸，留 S30。
```

#### G6 验证阶梯（CPU 优先 → 目标环境真卡）

```text
1. CPU（默认合并闸门）：逻辑 + 功能链路 — real torch、`yirage.core`（若触及）、RF pytest / e2e demo
2. 目标环境（后置、按需）：CUDA / MetaX MACA / 其它卡 — 同 seed 图 parity、融合 bench、mxcc 等
3. 禁止：无 CPU 逻辑/链路证据 → 用 Cloud CPU 绿或单次真卡 smoke 宣称 MACA/CUDA parity 完成
```

**示例（MACA 支撑轮，G6 + 支撑轨分工）**：

```text
价值：支撑轨 — mxcc 编译闭合，Serving 在 MetaX 可跑 Capsule。
设计目标：G6 已在 CPU 完成 API/搜索契约；本轮在目标环境补 MACA 执行证据（非替代 G6）。
矩阵：MACA 能力矩阵「构建/transpiler」行。
证据：pytest test_maca_config（CPU）+ MetaX VM demo_maca_superopt_test（目标环境）。
拒绝项：仅改文档不跑 MetaX — 不得合并 parity 项。
```

#### G7 功能闭合检验（完整功能 vs 零散堆砌）

**G7 问的是**：用户/引擎能否感知到一条 **跑通的功能**，而不是仓库里多了几个名字或单测。

| 闭合程度 | 含义 | 典型证据 | 能否宣称 G7 完成 |
|----------|------|----------|------------------|
| **全链闭合** | 输入→引擎 meta（若有）→ RF.step → Capsule 执行 → 输出可对照 | `demo/serving/qwen05b_cpu_e2e.py`、`vllm_mlp_e2e.py`、`test_runtime_fusion_s11_*` | 是 |
| **段落闭合** | 链路中 **一段** 端到端可跑（如单步 decode、单层 MLP RF） | S27 decode bench、S14 yirage_core e2e | **部分闭合**（笔记须写下一段） |
| **基础设施** | API/搜索/归档/契约 only | archive JSON、registry 键、config 单测 | **否**（不得写「功能完成」） |
| **零散堆砌（拒绝）** | 命名/FAST_PATH/矩阵 tier 无 runnable 链 | 仅 `kn_*_batch2_fast` parity | **低价值-拒绝** |

**功能链命名（策略卡片必填其一）**：

```text
链 A — MLP Capsule 最小链：tensor in → RF.step → MlpFusionCapsule → tensor out（test_runtime_fusion_s1）
链 B — HF Qwen decode 单步：prefill KV → next_token → RF MLP layers → logits parity（S27）
链 C — vLLM 插件全链：engine forward → layer override → RF.step → parity（S11）
链 D — SGLang ForwardBatch 全链：batch meta → hook → RF.step（S12）
链 E — Plan 搜索可执行：seed graph → superoptimize → execute vs reference（S14/S19）
```

**合并闸门（G7）**：

- **高价值**轮：须新增或加粗一条 **全链或段落闭合** 证据（e2e demo 或等价 pytest 含输入/输出断言）。
- **中价值**轮：须写明 **部分闭合** 哪一段、下一段 backlog 是什么；**禁止**用 archive/registry  alone 冒充 G7。
- **验证轮**（仅 registry/命名）：降级为 **低价值-拒绝**，除非用户显式要求 Chore。

#### 与行为性价比审查的关系

- **行为性价比**：逐步行动是否绕开根因（每步 1 句话）。
- **真实价值判断**：**整轮**是否服务 **G1–G7 设计目标**（策略前 + PR + 笔记）；**G7** 防止「绿单测但无功能链」。
- 二者任一为「不足/拒绝」→ 停止落地或降级 backlog。

### 执行前闸门：框架目标与优化价值（每轮必做）

**在勾选检查清单第 2 步「策略」、写代码之前**，Agent 必须先完成本闸门；若结论为「价值不足」，改选更高优先级 backlog。

#### YiRage 框架目标（RuntimeFusion 主轨 + MACA 支撑轨）

| 层级 | 工业参考 | YiRage 对应 | 本阶段重点 |
|------|----------|-------------|------------|
| **Serving 调度** | vLLM / SGLang | 引擎侧；RF 只消费 meta | S2+ 层覆盖 |
| **RuntimeFusion** | Dynamic fusion runtime | `RF.step(meta) → Capsule*` | S1 最小钩子 |
| **FusionCapsule** | Fused block | MLP Capsule（首刀）；后续 decoder 片段 | S1 |
| **FusionPlan** | Execution plan | `superoptimize` 产物 + cache（legacy: mugraph） | 搜索/复用 |
| **Device / Tile Graph** | 设备与 tile 抽象 | 代码暂留 KN/TB；文档用新名 | 实现层 |
| **原语** | cuBLAS / mcPytorch | `.cu` / `.maca`；不重造孤立 GEMM | 仅阻塞时修 |
| **验证** | generate 数值 + 吞吐 | Capsule vs eager；RF pytest | **G6：CPU 逻辑/链路优先**；真卡后置 |

**借鉴要点**：

- **调度在外、融合在内**：引擎拥有批处理与 KV；RF 只编排约定 Capsule。
- **Plan ≠ 死刑**：FusionPlan 可缓存、可按 shape bucket 命中；本步可不选或降级。
- **meta 打针**：`block_tables` / Radix hit / SM 预算 → RF.step。
- **SM 非全占**：Capsule launch 预留辅助流。
- **正确性优先**：先 Capsule/RF numerical parity，再 latency/throughput。

#### 四轮自问（策略卡片必填）

1. **层级**：Serving/RF（Capsule/`step`/meta/SM）还是支撑轨？若仅为 beat 孤立 GEMM、offline 抛光、或 Mirage 式整网烤核 → **拒绝**。
2. **RF 对标**：闭合 [RuntimeFusion 能力矩阵](#runtimefusion-能力矩阵对标-vllmsglang) 哪一行？是否仍在用 legacy MPK/µGraph 对外叙事？
3. **路径**：Capsule 边界是否清晰？`RF.step` 是否消费本轮所需 meta？SM 是否预留？
4. **收益**：Capsule 通路 / 层覆盖 / KV 桥 / SM / Radix / e2e？性能向须写基线（eager 引擎 vs RF）。
5. **验证入口**：**G6** — `make test-serving-cpu-cert-pytest` / RF e2e demo（CPU 逻辑+功能链路）；GPU/MACA 仅当本轮触及且在 **目标环境** 补跑；**禁止** stub/smoke 冒充完成。
6. **功能闭合**：**G7** — 本轮闭合哪条功能链（链 A–E 或自定义 input→output）？全链 / 段落 / 基础设施 / 零散堆砌？
7. **机会成本**：是否挤掉更高优先级的 Capsule/`step` backlog？
8. **真实价值**：本轮服务 [设计目标 G1–G7](#设计目标价值评价锚点) 哪几条？四档结论？若为 **低价值-拒绝**（不服务 G1–G7 或命中 G7 反模式）→ 改选 backlog，不得开 PR。

**性能向轮次**（按后端）：

```bash
# CUDA / RF hybrid（主轨）
export YIRAGE_BACKEND=cuda PYTHONPATH=.
# S1+：pytest tests/python/test_runtime_fusion_s*.py；e2e demo/serving/torch_e2e.py

# MACA 支撑（仅触及 MACA 时）
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:$LD_LIBRARY_PATH
export YIRAGE_BACKEND=maca
PYTHONPATH=. /opt/conda/bin/python3 benchmark/maca_vs_pytorch.py --quick
```

若 RF 混合通路未打通，**禁止**用 `maca_vs_pytorch` 冒充本轮主收益。

---

## RuntimeFusion 闭环（主轨，2026-07-27）

### Serving Loop 分阶路线（S0…Sn）

| 阶段 | 目标 | 复用脚手架（实现） | 验证门槛 | 状态 |
|------|------|--------------------|----------|------|
| **S0** | 北极星改为 FusionPlan / FusionCapsule / RF；Legacy 对照表；冻结反模式 | 本文档 | 文档评审合入 | **done（本 PR 含）** |
| **S1** | **第一个可被 RF 选中的 MLP FusionCapsule** + **最小 `RF.step` 钩子** | `python/yirage/serving/` | `pytest tests/python/test_runtime_fusion_s1.py` | **done** |
| **S2** | vLLM 形 MLP Override：Attention 留引擎；MLP 走 `RF.step` | `layer_override.py` | `test_runtime_fusion_s2_s3.py` | **done** |
| **S3** | 前 K 层 MLP Capsule 可配置混合 | `hybrid_model.py` | K∈{1,2,4}；`test_runtime_fusion_s2_s3.py` | **done** |
| **S4** | **PagedAttention meta 桥**（`block_table` → paged_kv_*） | `kv_meta.py`；`RF.step` auto-bridge | `test_runtime_fusion_s4_kv.py` | **done** |
| **S5** | **SM 预算**与 Sampler/NCCL 共驻 | `sm_budget.py`；`RF.step` 超预算 skip + engine fallback | `test_runtime_fusion_s5_sm.py`；`make test-serving-cpu-cert` | **done** |
| **S6** | SGLang **Radix**：hit meta → 跳过/收缩 Capsule | `radix_meta.py` | `test_runtime_fusion_s6_radix.py` | **done** |
| **S7** | 多 Capsule / 大段 Decoder Override | `capsule_orchestration.py`；`split_mlp_capsule.py`；`segment_override.py` | `test_runtime_fusion_s7_multi_capsule.py` | **done** |
| **S8** | vLLM Qwen2 MLP 插件契约 + segment torch 实测 bench 归档 | `torch_plugin.py`；`vllm_plugin.py`（须安装 vllm） | `test_runtime_fusion_s8_*`；`segment_torch_bench.py` | **done** |
| **S9** | **SGLang ForwardBatch meta 桥**（`extend_seq_lens` → Radix skip/shrink + KV） | `radix_meta.build_sglang_rf_step_meta` | `test_runtime_fusion_s9_sglang_meta.py` | **done** |
| **S10** | **SGLang model MLP hook**（ForwardBatch → RF meta + Qwen2 layer hook） | `sglang_plugin.rf_step_meta_from_forward_batch`；`SglangQwen2MlpRfHook` | `test_runtime_fusion_s10_sglang_plugin.py` | **done** |
| **S11** | **vLLM 全链路 MLP RF e2e**（单层 hook + 多层 hybrid + 可选真实 vllm） | `vllm_e2e.run_torch_vllm_*` / `run_vllm_qwen2_mlp_rf_e2e` | `test_runtime_fusion_s11_vllm_e2e.py`；`vllm_mlp_e2e.py` | **done** |
| **S12** | **SGLang ForwardBatch 全链路 e2e**（Radix partial/all-hit + KV meta + 可选 sglang） | `sglang_e2e.run_torch_sglang_*` / `run_sglang_qwen2_mlp_rf_e2e` | `test_runtime_fusion_s12_sglang_e2e.py`；`sglang_mlp_e2e.py` | **done** |
| **S13** | **vLLM PagedAttention 全层 MLP hook** | `vllm_paged_e2e.VllmPagedKvBatchSpec` + `run_torch_vllm_paged_full_layer_e2e` | `test_runtime_fusion_s13_vllm_paged_e2e.py`；`vllm_paged_e2e.py` | **done** |
| **S14** | **yirage.core MLP capsule 全层 hybrid e2e** | `yirage_core_e2e.run_yirage_core_full_layer_e2e` + `HybridModelOverride(mlp_backend=yirage_cpu)` | `test_runtime_fusion_s14_yirage_core_e2e.py`；`yirage_core_full_e2e.py` | **done** |
| **S15** | **MACA serving meta + vLLM-metax 插件 tier** | `maca_serving_meta.MacaServingRfSpec` + `vllm_metax_plugin` + `maca_serving_e2e` | `test_runtime_fusion_s15_maca_serving.py`；`maca_serving_e2e.py` | **done** |
| **S16** | **yirage_maca 全层 capsule + SGLang-metax 对称 tier** | `YirageMacaServingMlpRunner` + `yirage_maca_e2e` + `sglang_metax_plugin` + `sglang_metax_e2e` | `test_runtime_fusion_s16_metax_tiers.py`；`yirage_maca_full_e2e.py`（MetaX VM）；`sglang_metax_e2e.py` | **done** |
| **S17** | **yirage_maca generation latency 归档 + SGLang-metax 真 fork e2e** | `yirage_maca_generation` decode loop + bench archive + `run_sglang_metax_fork_e2e` | `test_runtime_fusion_s17_maca_generation.py`；`yirage_maca_generation_bench.py` | **done** |
| **S18** | **generation mcPytorch baseline JSON + SGLang-metax 多层 real fork** | `run_yirage_maca_generation_mcpytorch_baseline_archive` + `run_sglang_metax_multilayer_fork_e2e` | `test_runtime_fusion_s18_maca_baseline.py`；`yirage_maca_generation_bench.py --baseline-json` | **done** |
| **S19** | **Qwen2-0.5B YiRage CPU superoptimize + coordinator cert tier** | `hf_qwen_cpu_e2e` + `yirage_exec.superoptimize_down_matmul_via_coordinator` + `DistributedSearchCoordinator` | `test_runtime_fusion_s19_yirage_cpu_search.py`；`test_serving_ray_search.py`；`qwen05b_cpu_e2e.py --use-ray` | **done** |
| **S20** | **Ray cluster coordinator e2e for serving down matmul** | `superoptimize_down_matmul_via_coordinator(use_ray=True)` + `graph.superoptimize(use_ray=True)` blockdim partition | `test_serving_coordinator_ray_e2e.py`；`test_runtime_fusion_s20_coordinator_ray.py` | **done** |
| **S21** | **Tractable full TB down matmul search (no seed-verify)** | `YIRAGE_SERVING_FULL_TB_SEARCH=1` + capped TB matmul explore | `test_runtime_fusion_s21_full_tb_search.py`；`qwen05b_cpu_e2e.py --full-tb` | **done（本轮）** |

**明确不做什么（反模式）**：

- 对外宣称 YiRage 是 Mirage MPK / µGraph 复刻或对齐。
- 只换 Attention 单算子却不解决 RF/`step`/meta/SM。
- 把 S1 做成「不可选择的整层 MegaKernel 独占」，无 `step` 钩子。
- 未打通 S2 就宣称「已支持 vLLM」。
- 用整网 torch 回退冒充 RuntimeFusion。
- 在 S1 阶段做 `MuGraphStore` 等大规模符号重命名 Chore。

### Serving 验证禁令（**严禁**，2026-07-27 起永久有效）

> Cloud Agent 与人类协作者：**不得**再引入下列测试形态；已有项已删除，**禁止回滚或换名复活**。

| 严禁项 | 说明 |
|--------|------|
| **`demo/serving/*smoke*.py`** | 任何带 `smoke` 后缀的 Serving demo；能力须写入 `tests/python/test_runtime_fusion_s*.py` |
| **`--contract-only` / NumPy stub cert** | 无 torch 的「契约-only」cert 分支；`EngineModelStub` + `BACKEND_NUMPY_REF` **不得**作为 cert/pytest 主路径 |
| **Mock / duck-typed vLLM 层** | 如 `_MockVllmQwen2Layer`；vLLM 路径须真实 `pip install vllm` 或测负路径 `pytest.skip` |
| **重复验证脚本** | 与 pytest 同义的 one-off demo；仅保留 **`torch_e2e.py`**、**`segment_torch_bench.py`**、**`vllm_mlp_e2e.py`**、**`sglang_mlp_e2e.py`**、**`vllm_paged_e2e.py`**、**`maca_serving_e2e.py`**、**`sglang_metax_e2e.py`**、**`yirage_maca_generation_bench.py`**（实测 e2e/bench），以及可选 **`yirage_superopt_e2e.py`** / **`yirage_core_full_e2e.py`** / **`yirage_maca_full_e2e.py`**（yirage-core/MACA tier） |

**唯一认可的 Serving 验证栈**：

```bash
make test-serving-cpu-cert-pytest   # S1–S19 real torch pytest
make test-serving-cpu-cert          # 上式 + torch_e2e + segment_torch_bench
```

新能力：**先写/扩 pytest**（`TorchEngineModel`、`BACKEND_TORCH`、`tests/python/serving_test_utils.py`），必要时再加 **非 smoke 命名**的 e2e demo；**不得**为凑 stage 数新建 smoke 文件。

- 将旧 R27 offline latency 归档置于 S1–S2 之上。

### RuntimeFusion 能力矩阵（对标 vLLM/SGLang）

| 能力域 | 工业参照 | YiRage 落点（现状） | 真机验证入口（目标） | tier |
|--------|----------|--------------------|---------------------|------|
| **FusionPlan 搜索/缓存** | Execution plan + cache | `FusionPlan` API（S1）；存储仍 legacy mugraph | `test_runtime_fusion_s1` plan 契约；cache chore 另开 | partial |
| **MLP FusionCapsule** | Fused MLP block | `yirage.serving.MlpFusionCapsule`（默认 `backend=torch`；**S14** 全层 `yirage_cpu`） | `test_runtime_fusion_s1.py`；`test_runtime_fusion_s14_yirage_core_e2e.py` | **partial（S1/S14）** |
| **RF.step 钩子** | Dynamic fusion runtime | `RuntimeFusion.step`（select/skip） | 同上 | **partial（S1）** |
| **Model 层 Override** | `vllm/.../models/qwen2.py` | `RuntimeFusionMlpLayerOverride` + `TorchDecoderMlpRfHook` | `test_runtime_fusion_s2_s3.py` | **partial（S2）** |
| **前 K 层混合** | 可配置 fused layers | `HybridModelOverride(max_rf_mlp_layers=K)` | `test_runtime_fusion_s2_s3.py` | **partial（S3）** |
| **meta / KV 桥** | `block_tables` | `block_tables_to_paged_kv` + `StepMeta.with_paged_kv_bridge` + **S13 全层 hook** | `test_runtime_fusion_s4_kv.py`；`test_runtime_fusion_s13_vllm_paged_e2e.py` | **partial（S4/S13）** |
| **SM 配额共驻** | 引擎多流 | `resolve_sm_worker_quota` + `RF.step` 超预算 skip | `test_runtime_fusion_s5_sm.py` | **partial（S5）** |
| **Radix skip** | SGLang RadixAttention | `radix_meta` + `build_sglang_rf_step_meta` + `sglang_plugin` | `test_runtime_fusion_s6_radix.py`；`test_runtime_fusion_s9_sglang_meta.py`；`test_runtime_fusion_s10_sglang_plugin.py` | **partial（S6/S9/S10）** |
| **SGLang MLP 插件** | SGLang Qwen2 layer hook | `SglangQwen2MlpRfHook`（**须安装 sglang**）；实测 `SglangBatchTorchMlpRfHook` + `sglang_e2e` | `test_runtime_fusion_s10_*`；`test_runtime_fusion_s12_sglang_e2e.py`；`sglang_mlp_e2e.py` | **partial（S10/S12）** |
| **多 Capsule 编排** | 大段 fused blocks | `split_mlp_capsule` gate_up→down pipeline + `DecoderSegmentOverride` | `test_runtime_fusion_s7_multi_capsule.py` | **partial（S7）** |
| **vLLM MLP 插件** | `vllm/.../qwen2.py` hook | `VllmQwen2MlpRfHook`（**须安装 vllm**）；实测 `TorchDecoderMlpRfHook` + `vllm_e2e` | `test_runtime_fusion_s8_*`；`test_runtime_fusion_s11_vllm_e2e.py`；`vllm_mlp_e2e.py` | **partial（S8/S11）** |
| **Segment torch bench** | 实测 latency JSON | `run_segment_torch_bench_archive` | `segment_torch_bench.py`；cert | **partial（S8）** |
| **MLP capsule micro-bench** | Isolated fused MLP parity + timing | `run_mlp_capsule_bench`（G7 链 A 段落） | `benchmark/serving_mlp_capsule_bench.py`；`test_runtime_fusion_s30_*` | **partial（S30）** |
| **MACA / vLLM-metax** | MetaX vLLM fork | `maca_serving_meta` + `vllm_metax_plugin` + `maca_serving_e2e` | `test_runtime_fusion_s15_maca_serving.py`；MetaX VM `yirage_maca` tier | **partial（S15/S16）** |
| **SGLang-metax / yirage_maca capsule** | SGLang-metax fork | `sglang_metax_plugin` + `YirageMacaServingMlpRunner` + `yirage_maca_e2e` | `test_runtime_fusion_s16_metax_tiers.py`；MetaX VM `yirage_maca_full_e2e.py` | **partial（S16/S17）** |
| **yirage_maca generation archive** | Multi-step decode latency | `yirage_maca_generation` + baseline archive vs mcPytorch | `test_runtime_fusion_s17_maca_generation.py`；`test_runtime_fusion_s18_maca_baseline.py` | **partial（S17/S18）** |

### 现有脚手架 → RuntimeFusion 映射

| 脚手架（实现） | 路径 | RF 用途 |
|----------------|------|---------|
| 融合 MLP 构图 | `demo/qwen3/demo_chat.py` | S1 Capsule 形状参考 |
| 层追加 MLP 段 | `demo/maca/qwen3_pk_utils._append_maca_pk_decoder_layer` | 跨后端裁 Capsule 模板 |
| HF attach | `qwen3_pk_hf_utils.maca_pk_hf_weight_attach_map` | S2 权重名 |
| Plan 搜索 MLP | `qwen_kernel_utils.superoptimize_mlp_*` | FusionPlan 来源之一（过渡） |
| meta 缓冲 | `build_qwen3_pk_meta_tensors` 等 | S4 扩展，挂到 RF.step |
| 离线全图 demo | `demo/qwen3/demo.py` | **实现参考 only**；非 S1 产品目标 |
| C++ paged params | `pk_task_kernels.h` | S4 |
| Plan 持久化 | `MuGraphStore` / `~/.yirage/mugraphs/` | legacy FusionPlan 存储；改名 Chore 另开 |

### Serving 感知 / 策略 / 验证（主轨）

**感知**：读本表 + Legacy 对照；确认无 in-tree RF/`serving` 插件；列 S{n} gap。

**策略**：每轮闭合矩阵 **1 行**；优先 S1（Capsule+`step`）→ S2 → S4 → S5；Radix/多 Capsule 垫后。

**落地落点**：

- 新建薄 API 外壳：`python/yirage/serving/` 或等价（`FusionCapsule`、`RuntimeFusion.step`）；内部可委托 legacy 构图/launch
- 复用 `persistent_kernel/`、搜索缓存作 backend，**不**在叙事上称 MPK
- MACA：仅当 Capsule 在 C500 需 mxcc 时改支撑轨

**验证**：**`make test-serving-cpu-cert`** = S1–S15 **real torch pytest** + `torch_e2e` + `segment_torch_bench`；见 [Serving 验证禁令](#serving-验证禁令严禁2026-07-27-起永久有效)。可选 **`--yirage-core`** tier。

---

### CUDA 对标：优化目标与能力矩阵

> **支撑轨**：下列矩阵服务 MetaX MACA ↔ CUDA **能力/语义/融合 parity**，以及离线 PK 回归。主迭代缺口请优先查 [RuntimeFusion 能力矩阵](#runtimefusion-能力矩阵对标-vllmsglang)。仅当 Serving 被 MACA 编译/ABI 阻塞时，本轮才选本表 gap。

MACA 支撑闭环的目标：在 MetaX 真卡上对齐 CUDA 后端能力边界；性能基线为同机 mcPytorch；功能完备性以 CUDA 正式路径为金标准。

#### 三层对齐（优化目标）

| 对齐层 | 含义 | 通过标准 |
|--------|------|----------|
| **能力 parity** | CUDA 已支持的算子、TB epilogue、搜索 explore、transpiler 产物在 MACA 上可构图、可搜索、可 mxcc 编译、可执行 | 矩阵 tier ≠ `unsupported`；真卡 smoke exit 0 |
| **语义 parity** | 同 seed 图在 MACA/CUDA 上数值一致（允许 fp16/bf16 tol） | fingerprint 或 runtime vs reference 通过 |
| **融合 parity** | CUDA 上已验证的融合模式（rms+matmul、MLP、attention tile、forloop accum 等）MACA 可 superoptimize 且不慢于 mcPytorch 基线 | `maca_vs_pytorch` / e2e bench JSON `speedup ≥ 1.0`（同后端） |

**刻意不对齐（须文档化，不得当 bug 凑 parity）**：

| CUDA / NVIDIA 假设 | MACA / MetaX 现实 | 闭环处理 |
|--------------------|-------------------|----------|
| warp = 32 | **warp = 64** | block 为 64 倍数；shuffle 6 步；见 `MACA_WARP_SIZE` |
| smem 96KB（Volta 档） | **smem 64KB/block** | `maca::MAX_SMEM_SIZE`、搜索 smem 上限 |
| nvcc + PTX `mma.sync` | **mxcc + xcore1000** | 软件 MMA shadow（`YIRAGE_MACA_SOFTWARE_MMA`）等 |
| `torch.cuda` → cudart | mcPytorch → mcruntime | API 兼容，**评判仍用 YiRage backend=maca** |

#### 能力矩阵（对标 CUDA 路径）

维护原则：**CUDA 列**以 `YIRAGE_BACKEND=cuda` 正式 transpiler 路径为准；**MACA 列**以 mxcc 编译 + 真卡执行为准；**验证**列须在 MetaX VM 有对应命令（不可仅 CPU pytest）。

| 能力域 | CUDA 参照（金标准） | MACA 落点 | 真卡验证入口 |
|--------|---------------------|-----------|--------------|
| **构建 / transpiler** | `generate_cuda_program` + nvcc | `generate_cuda_program` + **mxcc**（`get_mxcc_cc_cmd`） | `demo/maca_superopt_test.py`；`tests/python/test_maca_config.py` |
| **搜索配置** | `get_cuda_search_config()` / CUDA `GeneratorConfig` | `get_maca_search_config()`、`maca_strategy.cc` | `demo/demo_maca_optimization.py`；`pytest -k maca` |
| **KN 原语 kernel** | `src/kernel/cuda/*.cu` | `src/kernel/maca/*.maca`（matmul、elementwise、reduction…） | `benchmark/maca_native_benchmark.py` |
| **TB customized / 融合** | customized CUDA kernel | `customized_kernel.maca` + TB interpreter 回退 | `maca_superopt_test.py`；`maca_vs_pytorch.py` |
| **RMSNorm + matmul 融合** | CUDA fused FusionPlan | 同构图 `backend=maca` superoptimize | `maca_vs_pytorch.py`（rms workload） |
| **Attention / softmax tile** | CUDA attention kernel 路径 | `attention_kernel.maca`、`softmax_kernel.maca` | `demo/maca/attention_smoke.py --inspect-only`（Cloud）；MetaX `--quick` superoptimize |
| **Persistent kernel** | CUDA PK backend | `maca_pk_backend.cc` + `demo/maca/qwen3_persistent_kernel_demo.py` | `--compile-plan`/`--compile-inspect`（Cloud）；MetaX `--compile-only` minimal embed + `--quick` |
| **Verifier** | CUDA fingerprint | MACA GPU fingerprint | `superoptimize(backend="maca")` 无 abort |
| **融合 vs 库基线** | fused vs `torch` CUDA | fused vs **mcPytorch** on MACA | `benchmark/maca_vs_pytorch.py --quick` |
| **Ray 分布式搜索** | `superoptimize(use_ray=True)` + `DistributedSearchCoordinator` | 同 API；MACA demo 暂 `use_ray=False` | `tests/integration/test_ray_maca_e2e.py`（MetaX VM）；CPU 镜像 `test_ray_cpu_e2e.py` |
| **RL 分层搜索** | `ConstrainedGraphEnv` + AccelForge prescreen + walkthrough | RL 编码含 `maca`；CPU walkthrough + **MACA walkthrough** | `tests/python/test_maca_rl_ray_capability.py`；**MetaX** `scripts/maca_capability_walkthrough.py` |
| **Qwen 全链路推理** | `demo/qwen2.5/demo.py`（HF load → superoptimize → decode） | `demo/maca/qwen_inference_demo.py`（合成权重 smoke）；`demo/maca/qwen_from_pretrained_demo.py`（HF 全权重） | **MetaX VM** `qwen_inference_demo.py --quick`；`qwen_from_pretrained_demo.py --max-layers 1 --quick` |

矩阵扩展时：在「当前轮次笔记」或 PR 中增一行 **CUDA 参照 PR/commit + MACA 验证命令**；禁止只改文档不补真卡测试。

#### RL / Ray 搜索优化：CUDA 对标（能力匹配）

搜索优化工具链须与 CUDA 路径 **同 API、同阶段、同 backend 语义**；MACA 上 Ray/RL 不得仅 CPU 单测绿而宣称对齐。

| 能力 | CUDA 参照 | MACA 现状 | 契约 / 真卡验证 |
|------|-----------|-----------|-----------------|
| **`superoptimize(..., use_ray=True)`** | 多 `griddims` 分区 + C++ search | 代码传 `backend=maca`；**烟雾默认 `use_ray=False`** | `tests/python/test_maca_rl_ray_capability.py`；MetaX Ray maca e2e（backlog） |
| **`DistributedSearchCoordinator.parallel_search`** | `backend=cuda` | 接受 `backend=maca` | `tests/integration/test_ray_maca_e2e.py` |
| **`RayDistributedEngine`** | GPU bundle via `torch.cuda` | 无 MACA 专用 placement；mcPytorch 或 CPU fallback | MetaX `gpus_per_worker` smoke（backlog） |
| **`scripts/bench_ray_search.py`** | CPU 公平 seq vs Ray | `--backend maca` + `--quick` | `scripts/bench_ray_search.py --backend maca --quick` |
| **`business_capability_walkthrough`** | YiRage×Ray×AccelForge×RL | CPU 版 `backend=cpu`；**MACA 版** `scripts/maca_capability_walkthrough.py` | MetaX VM `--quick` |
| **`ConstrainedGraphEnv` / FINISH** | `accelforge_metrics` + verify | `config_space` 含 maca warp=64；**无 maca env e2e** | `test_accelforge` + maca FINISH（backlog） |
| **`VerifierPool` / GPU verify** | `backend=cuda` | `VerifierPool(backend=maca)` + env default | `tests/python/test_rl/test_maca_verifier_pool.py`；MetaX Ray smoke |

**RL/Ray 对齐策略**：

1. **感知**：`PYTHONPATH=. pytest tests/python/test_maca_rl_ray_capability.py -v` + 对照上表 tier=`gap`/`partial` 项。
2. **策略**：每轮闭合 1 项（优先 Ray maca e2e 或 walkthrough maca，高于微基准）。
3. **落地**：复用 CUDA Ray 分区逻辑；MACA 差异仅 warp=64 / smem / mxcc；**禁止**永久 `use_ray=False` 而不记 backlog。
4. **验证**：契约 pytest 绿 + **MetaX VM** 上对应 Ray/RL smoke exit 0。

权威清单（与文档同步）：`tests/python/test_maca_rl_ray_capability.py` → `maca_rl_ray_capability_matrix()`。

#### Qwen 全链路推理：CUDA 对标

| | CUDA | MACA |
|---|------|------|
| **参照 demo** | `demo/qwen2.5/demo.py`（`Qwen/Qwen3-8B`、chat template、CUDA Graph decode） | `demo/maca/qwen_inference_demo.py`（合成）；`demo/maca/qwen_from_pretrained_demo.py`（HF 全权重） |
| **融合内核** | `modeling_qwen2.superoptimize_kernels()` → 默认 cuda transpiler | `modeling_qwen2_maca.superoptimize_kernels()` → `backend=maca` via `qwen_kernel_utils` |
| **Decode 路径** | `q_len==1` → YiRage MLP + attn kernels；CUDA Graph decode | 同语义；`qwen_decode_loop.py` 默认 CUDA Graph（`--no-cuda-graph` 回退 eager） |
| **e2e bench** | CI `tests/ci-tests/qwen2.5/demo.py` | `benchmark/end-to-end/maca/qwen_maca.py` |
| **Qwen3 PK** | `demo/qwen3/demo.py --use-yirage` | `demo/maca/qwen3_persistent_kernel_demo.py`（PK runtime smoke；`PersistentKernel.compile()` mxcc when `YIRAGE_BACKEND=maca`） |
| **真卡验证** | NVIDIA GPU | **MetaX VM** `--quick` smoke；`tests/integration/test_maca_qwen_inference_demo.py` |

**尚未对标（backlog）**：`demo/qwen3/demo.py --use-yirage` 全量多 layer task-graph e2e on MetaX（R17–R26 已闭合 scaffold/runtime/generation；**旧 R27 latency 归档降为支撑**）。**主轨改为** RuntimeFusion S1–S7（MLP FusionCapsule + RF.step → vLLM override → KV/SM/Radix → 多 Capsule）。


#### 对齐策略（每轮策略层）

1. **感知**：先查 [Serving 能力矩阵](#mpk-serving-能力矩阵对标-vllmsglang) gap；再查「CUDA supported ∧ MACA unsupported / 编译失败 / 无真卡测试」（仅当阻塞 Serving 或支撑回归时）。
2. **策略**：主轨每轮闭合 **1 个 Serving 阶段**（S1→S2→…）；支撑轨每轮至多 1 个 MACA parity 域。
3. **落地**：Serving 优先复用 PK API 与 meta；MACA 仅改执行层常量/编译；**禁止**用 `YIRAGE_MACA_TORCH_MATMUL` 等回退冒充 parity 或 Serving。
4. **验证**：Serving 真机入口或本域 MetaX 入口 **exit 0** 后方可合并宣称完成。
5. **进化**：更新 Serving 矩阵行 +（若触及）CUDA/MACA 矩阵 + 轮次笔记。

#### 效果检查手段（证据链）

| 检查类型 | 手段 | 适用场景 | 运行环境 |
|----------|------|----------|----------|
| **配置契约** | `pytest tests/python/test_maca_config.py`、`test_backends.py -k maca` | mxcc 命令、smem 上限、search quick | 可无卡；**合并前仍须真卡 smoke** |
| **编译 + 搜索 smoke** | `demo/maca_superopt_test.py` | transpiler→mxcc、FusionPlan 编译/profile | **MetaX 真卡 VM** |
| **设备 + search 感知** | `demo/demo_maca_optimization.py` | SDK、GPU 名、grid/block 约束 | **MetaX 真卡 VM** |
| **融合性能** | `benchmark/maca_vs_pytorch.py`（`--quick` 日常 / 全量 nightly） | 融合 vs mcPytorch；JSON speedup | **MetaX 真卡 VM** |
| **原语 kernel** | `benchmark/maca_native_benchmark.py` | `.maca` 孤立 kernel | **MetaX 真卡 VM** |
| **C500 专项 shape** | `benchmark/maca_c500_benchmark.py` | 104 SM / 64KB smem 边界 | **MetaX C500** |
| **LLM / LoRA e2e** | `benchmark/end-to-end/maca/*.py` | 图级 parity 回归 | **MetaX 真卡 VM** |
| **Qwen 全链路推理** | `demo/maca/qwen_inference_demo.py --quick` | 对齐 `demo/qwen2.5/demo.py` decode 路径 | **MetaX 真卡 VM** |
| **RL/Ray 契约** | `pytest tests/python/test_maca_rl_ray_capability.py` | API/矩阵与 CUDA 对齐检查 | 可无卡；Ray/RL 真卡 smoke 另需 MetaX |
| **CUDA 对照（可选）** | 同 seed 图 `backend=cuda` on NVIDIA 机 | 语义/reference 对照 | NVIDIA GPU（**非**合并硬门槛） |

**合并闸门（MACA + CUDA 对标）**：

- 触及 `.maca` / transpiler / search / fusion 的 PR：**MetaX VM 上表「真卡验证入口」相关项全绿**。
- 仅文档 / 纯 CPU pytest：**不得**宣称已完成 CUDA parity；须在笔记中标「待真卡 R{n}」。

### 五层结构

```mermaid
flowchart LR
  P[1 感知 Perceive] --> S[2 策略 Strategy]
  S --> I[3 落地 Implement]
  I --> V[4 验证 Verify]
  V --> M{MACA 验证通过?}
  M -->|否| S
  M -->|是| PR[开 PR + 合并 main]
  PR --> E[5 进化 Evolve]
  E --> N[扫描新瓶颈]
  N --> P
```

---

#### 第 1 层：感知（Perceive）— 我们在哪？

**目标**：弄清 **RuntimeFusion 嵌入缺口**（Capsule/`step`、层覆盖、meta/KV、SM、Radix）、FusionPlan 可复用面；以及（支撑）MACA↔CUDA parity。

**典型动作**：

- 读 [RuntimeFusion 闭环](#runtimefusion-闭环主轨2026-07-27) + [Legacy 对照](#概念对照legacy--yirage-标准) + [docs/HARDWARE_OPTIMIZATION.md](docs/HARDWARE_OPTIMIZATION.md)；对照 `demo_chat.py` / `qwen3_pk_*.py`（实现参考）
- 确认本轮后端与硬件（NVIDIA 或 MetaX）
- RF gap 扫描：是否已有 `python/yirage/serving` / `RF.step` / FusionCapsule API？
- （支撑）CUDA↔MACA kernel/search diff、`demo_maca_optimization`、`maca_vs_pytorch --quick`

**产出**：简短快照 — **Serving S{n} gap**、可复用实现列表、（可选）MACA 编译/parity 列表。

---

#### 第 2 层：策略（Strategy）— 下一步改什么？

**目标**：按 Serving 分阶排序，选定**单一**主攻（默认：S1 MLP FusionCapsule + RF.step → S2 vLLM override）。

**前置条件**：已完成上文 [执行前闸门](#执行前闸门框架目标与优化价值每轮必做) 的策略自问 + [Loop 真实价值判断](#loop-真实价值判断每轮必写)（四档结论）。

**决策参考**：

| 信号 | 优先策略 |
|------|----------|
| **无 MLP FusionCapsule / 无 RF.step** | **S1**：Capsule + 最小 step 钩子（委托现有构图实现） |
| **有 Capsule、无 vLLM 层覆盖** | **S2**：`models/qwen2.py`（或插件）调 RF.step；Attention 留引擎 |
| **混合通、KV 错/非连续失败** | **S4**：`block_table` → paged_kv_* adapter |
| **混合通、Sampler/NCCL hang** | **S5**：SM 预算 |
| **SGLang 前缀缓存语义错** | **S6**：Radix hit meta + Capsule 收缩/跳过 |
| **对外仍写 Mirage/MPK/µGraph 身份** | 改文档/API 外壳；符号 Chore 另开 |
| **CUDA 有、MACA 无**且阻塞 Serving/MetaX | 支撑轨：补 `.maca` / config / smoke |
| `mxcc` 编译失败阻塞 Capsule | 修 kernel / cmake / shims |
| 仅有 CPU pytest 绿、无 GPU RF smoke | **阻塞 Serving 完成宣称** |
| 想做 offline latency 归档（旧 R27）或 mugraph 大改名 | **降级**；不挡 S1 |

**产出**：本轮「策略卡片」— 1 句话目标（含 S{n}）、**价值四档结论**、触及文件、预期验证命令。

---

#### 第 3 层：落地（Implement）— 最小正确实现

**目标**：按策略做**最小**代码/配置改动，遵循仓库既有风格。

**常见落地点**（Serving 主轨）：

- PK API / 块构建：`python/yirage/persistent_kernel/`、MLP block helper
- Serving 适配：`python/yirage/serving/` 或 `demo/serving/`（vLLM hook、meta adapter）
- 权重名：复用 `qwen3_pk_hf_utils` attach map

**常见落地点**（MACA 支撑轨）：

- 执行：`src/kernel/maca/*.maca`、`src/backend/maca_backend.cc`
- 搜索：`python/yirage/backends/maca/config.py`、`maca_strategy.cc`
- Persistent kernel：`src/persistent_kernel/maca_pk_backend.cc`

**禁止**：新建独立全阶段 orchestrator；用现有 demo/benchmark + 薄适配层串联。

**产出**：可编译可 launch 的增量 patch；触及 Cython/native 则按后端 `pip install -e .`。

---

#### 第 4 层：验证（Verify）— 证据链

**目标**：正确性先于速度；**Serving 混合通路先于 offline speedup**。

| 层 | 机制 | 入口 |
|----|------|------|
| Serving | hybrid forward vs eager | S2+ vLLM/SGLang smoke（落地后） |
| PK 块 | MLP 块 vs torch MLP | S1 unit/demo smoke |
| Search | `ProbabilisticVerifier` 或 formal | `superoptimize`（支撑） |
| Runtime | optimized graph vs mcPytorch | `benchmark/maca_vs_pytorch.py`（支撑） |

**MACA 支撑最小验证集**（MetaX；仅本轮触及 MACA 时必跑）：

```bash
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:${LD_LIBRARY_PATH}
export YIRAGE_BACKEND=maca PYTHONPATH=.
pytest tests/python/test_backends.py -k maca -v
/opt/conda/bin/python3 demo/demo_maca_optimization.py
/opt/conda/bin/python3 demo/maca/qwen3_persistent_kernel_demo.py --inspect-only
```

**合并闸门**：Serving 改动须对应 S{n} 真机（或 S0 文档）入口通过；MACA kernel/search 须 MetaX 相关入口全绿；Cloud CPU 绿不能替代 GPU Serving/MACA 宣称。

---

### MACA Demo 与 Benchmark 烟雾层

Demo/benchmark 是 MACA 闭环的**用户可复现烟雾层**，证明 Agent 能在真卡上构图、搜索、执行；同时是 [CUDA 能力矩阵](#cuda-对标优化目标与能力矩阵) 的**真卡验证入口**。

| 闭环层 | 作用 | 典型命令 |
|--------|------|----------|
| **感知** | 确认 SDK、mcPytorch、mxcc、GPU 可见 | `mx-smi`；`demo/demo_maca_optimization.py`（device info） |
| **验证** | 改 kernel/search 后融合仍可跑 | `demo/maca_superopt_test.py`；`benchmark/maca_vs_pytorch.py` |
| **进化** | 新融合模式 e2e | `benchmark/end-to-end/maca/*.py` |

**推荐清单**：

| id | 脚本 | 层级 | 说明 |
|----|------|------|------|
| `demo_maca_optimization` | `demo/demo_maca_optimization.py` | 感知 | MACA 设备检测 + search config |
| `maca_superopt_test` | `demo/maca_superopt_test.py` | 验证 | superoptimize smoke |
| `maca_vs_pytorch` | `benchmark/maca_vs_pytorch.py` | 验证 | 融合 vs mcPytorch 基线 |
| `qwen_inference_maca` | `demo/maca/qwen_inference_demo.py` | 验证 | **Qwen 全链路**（对齐 `demo/qwen2.5/demo.py`） |
| `qwen_maca_bench` | `benchmark/end-to-end/maca/qwen_maca.py` | 验证 | Qwen e2e 入口 |
| `maca_native_benchmark` | `benchmark/maca_native_benchmark.py` | 验证 | 原生 kernel bench |
| `maca_c500_benchmark` | `benchmark/maca_c500_benchmark.py` | 进化 | C500 专项 shape |
| `llama_maca` | `benchmark/end-to-end/maca/llama_maca.py` | 进化 | LLM 片段 e2e |
| `lora_maca` | `benchmark/end-to-end/maca/lora_maca.py` | 进化 | LoRA blocked GEMM |

#### 推荐命令（MetaX GPU VM）

```bash
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:build/abstract_subexpr/release:build/formal_verifier/release:${LD_LIBRARY_PATH:-}
export YIRAGE_BACKEND=maca
export PYTHONPATH=.
PY=/opt/conda/bin/python3   # mcPytorch，勿用系统 /usr/bin/python3

$PY -m pip install z3-solver graphviz
YIRAGE_BACKEND=maca $PY -m pip install -e . --no-build-isolation

$PY demo/demo_maca_optimization.py
$PY demo/maca_superopt_test.py
$PY demo/maca/qwen_inference_demo.py --quick
$PY benchmark/maca_vs_pytorch.py
pytest tests/python/test_maca_rl_ray_capability.py -v
```

---

#### 第 5 层：进化（Evolve）— 写回知识，开启下一轮

**必须更新的位置**：

1. **`AGENTS.md`** — 「当前轮次笔记」（**含 `价值：` 字段**）、**[CUDA 能力矩阵](#cuda-对标优化目标与能力矩阵) 行状态** 或 **Gotchas**
2. **`docs/maca_quick_start.md` / `docs/maca_complete_guide.md`** — 环境/bench 基线、CUDA 差异表变更
3. **`docs/HARDWARE_OPTIMIZATION.md`** — MACA 评估标准、与 CUDA 分层对照变更
4. **集成测试 / benchmark** — 新融合或新 `.maca` 原语须补 **MetaX 真卡** smoke（可同步加 `test_maca_config` 契约）

---

### Cloud Agent 自主连续迭代协议（Serving 主轨）

用户未明确喊停时，Cloud Agent **默认连续跑多轮 Serving Loop（S{n}）**。每一轮：**感知 → 策略 → 落地 → 验证 → 自动合并 → 进化 → 扫描 backlog → 回到感知**。MACA 支撑轮仅在 Serving 被编译/ABI 阻塞或用户点名时插入。

#### MetaX GPU 开发机（SSH）

Agent 须在 **MetaX C500 VM** 上验证 **MACA 支撑**改动（Cloud 默认 Linux CPU VM **不能**替代）：

```bash
# 用户名格式为 root+<vm-id>，非裸 root
ssh -p 32222 'root+<vm-id>@<host>'

export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:$LD_LIBRARY_PATH
export YIRAGE_BACKEND=maca
cd /workspace   # 或 YiRage clone 路径
/opt/conda/bin/python3 -m pip install -e . --no-build-isolation
```

> **凭据**：SSH 主机/端口/密码由团队密钥管理，**不得**写入仓库。当前开发机：`140.207.205.81:32222`（MetaX C500，`mx-smi` / mcPytorch `2.8.0+metax*`）。

#### 验证通过后的自动合并

| 条件 | 要求 |
|------|------|
| 本地验证 | Serving：对应 S{n} 入口；MACA 支撑：MetaX 相关烟雾 exit 0；纯文档 S0：审阅即可 |
| 分支规范 | `cursor/<descriptive-name>-c4c0`（或仓库约定后缀），已 push，`base_branch=main` |
| PR 状态 | `mergeable`；若 draft，先 ready |
| 合并方式 | 使用仓库允许的 merge 流程（Cloud：`ManagePullRequest`；勿用只读 `gh` 写） |
| 合并后 | `git checkout main && git pull origin main` |

**不以远端 CI 绿作为 Serving/MACA 硬门槛**（CI 多在 CPU 上跑）；以 **真机烟雾** 为准。

#### 合并后感知扫描

```bash
# Serving：检查 S{n+1} gap 与 in-tree serving/ 插件是否存在
# MACA 支撑（若本轮触及）：
export MACA_PATH=/opt/maca YIRAGE_BACKEND=maca PYTHONPATH=.
PY=/opt/conda/bin/python3
$PY demo/demo_maca_optimization.py
$PY demo/maca/qwen3_persistent_kernel_demo.py --inspect-only
pytest tests/python/test_maca_config.py -v
```

**backlog 优先级**：见上文 Serving 优先序。

#### 连续迭代终止条件

- 用户明确停止
- 本轮所需 GPU（NVIDIA Serving 或 MetaX）不可用且无法修复
- 本轮纯文档且用户未要求继续（**S0 类文档轮除外**：用户已要求改目标时可合入）

---

### Cloud Agent 单轮检查清单（Serving 主轨）

```
[ ] 0. 闸门：策略卡片（含 Serving S{n} + 真机验证）+ [行为性价比审查](#agent-行为性价比审查每步必做) + [Loop 真实价值判断](#loop-真实价值判断每轮必写)（**G1–G7 设计目标对齐** + **G7 功能链闭合程度** + 四档结论 + 证据）
[ ] 1. 感知：RF 矩阵 gap、FusionPlan/Capsule 可复用面、Legacy 对照；（若触及 MACA）mx-smi / demo_maca_*
[ ] 2. 策略：只选 1 个 Serving 阶段（默认 S1 Capsule+step → S2→S4…）；拒绝 Mirage/MPK 叙事与单 Attention 替换
[ ] 3. 落地：最小 patch；优先 `serving/` RF 外壳 / Capsule / meta 桥；legacy 仅作委托；每步过行为性价比
[ ] 4. 验证：**G6 CPU 优先** + **G7 功能链** — `make test-serving-cpu-cert-pytest` 或对应 S{n} e2e pytest/demo；触及 MACA/CUDA 时在笔记标「待目标环境」；**禁止** stub cert / 零散单测冒充 G7
[ ] 5. 开 PR：push 分支 cursor/...-c4c0（或仓库约定后缀），base=main；PR 含 S{n} 与验证摘要
[ ] 6. 自动合并：验证闸门通过 → merge
[ ] 7. 同步 main：git checkout main && git pull
[ ] 8. 进化：更新 AGENTS.md Serving 矩阵行状态 + 轮次笔记（**含价值字段**）
[ ] 9. 扫描 backlog：S{n+1}、KV/SM/Radix、（支撑）MACA gap
[ ] 10. 下一轮：新分支继续 Serving Loop S{n+1}
```

### 现有工具索引

| 层 | 工具 / 路径 |
|----|-------------|
| 感知 | [RF 矩阵](#runtimefusion-能力矩阵对标-vllmsglang)、[Legacy 对照](#概念对照legacy--yirage-标准)、`demo_chat.py`、`qwen3_pk_*.py` |
| 策略 | Serving S0–S7（RF）、[CUDA 支撑矩阵](#cuda-对标优化目标与能力矩阵) |
| 落地 | 拟议 `python/yirage/serving/`（RF/Capsule API）、legacy `persistent_kernel/` / mugraph store 委托、MACA 支撑 |
| 验证 | `make test-serving-cpu-cert-pytest`；`demo/serving/torch_e2e.py`；`test_maca_*` |
| 进化 | **`AGENTS.md`（Serving + MACA 笔记）**, `docs/maca_*.md`, `docs/HARDWARE_OPTIMIZATION.md` |

### 当前轮次笔记（Serving + MACA，由 Agent 持续追加）

> **维护说明**：每合并一轮主轨（Serving）或支撑轨（MACA）PR，在此追加 3～5 行；**每行须含 `价值：` 与 `功能闭合：` 字段**（含 **G1–G7 设计目标编号** 与 G7 闭合程度，见 [Loop 真实价值判断](#loop-真实价值判断每轮必写)）。CPU 历史见 [归档](#历史归档cpu-无限优化闭环)。

- **Serving Loop S0（2026-07-27，RuntimeFusion 概念定稿）**：闸门：文档/协议层。北极星定为 **FusionPlan / FusionCapsule / RuntimeFusion（RF）**；写入 [Legacy 对照表](#概念对照legacy--yirage-标准)；明确反对 Mirage 式编译霸权与 MPK/µGraph 对外身份。MACA R0–R26 与旧 R27 为支撑。验证：文档审阅（PR #153）。下一轮：**S1** — 第一个可被 RF 选中的 **MLP FusionCapsule** + 最小 **`RF.step` 钩子**（PagedAttention 仍留引擎；不做符号大改名）。
- **Serving Loop S1（2026-07-27，MLP FusionCapsule + RF.step）**：闸门：Serving/RF API 外壳。新增 `python/yirage/serving/`（`FusionPlan`、`MlpFusionCapsule`、`RuntimeFusion.step`）；S1 执行器为 eager NumPy（Cloud 可测，不依赖 `yirage.core`）；`demo/serving/mlp_capsule_smoke.py`；契约 `tests/python/test_runtime_fusion_s1.py`。验证：pytest S1 + smoke select/skip。下一轮：**S2** — vLLM 模型层 Override 挂 `RF.step`（Attention/Paged 仍留引擎）。
- **Serving Loop S2（2026-07-27，MLP Layer Override）**：闸门：引擎协同。`RuntimeFusionMlpLayerOverride`：Attention 走 engine stub，MLP 走 `RF.step`，skip→`mlp_forward` fallback；`QWEN2_MLP_HF_ATTACH`；不 vendor vLLM。验证：`test_runtime_fusion_s2_s3` + `vllm_mlp_override_smoke` PASS。
- **Serving Loop S3（2026-07-27，first-K hybrid）**：闸门：多一层混合。`HybridModelOverride(max_rf_mlp_layers=K)` / `rf_mlp_layer_ids`；K∈{1,2,4} 与 engine 全路径数值对齐。验证：`hybrid_first_k_smoke` + pytest。下一轮：**S4** — `block_table` → paged_kv meta 桥。
- **Serving Loop S4（2026-07-27，KV meta bridge）**：闸门：引擎 meta。`kv_meta.block_tables_to_paged_kv`（indptr/indices/last_page_len）；`RuntimeFusion.step` 在存在 `block_tables`+`seq_lens` 时自动写入 Capsule extras。验证：`test_runtime_fusion_s4_kv` + `kv_meta_bridge_smoke`。下一轮：**S5** — SM 预算共驻契约。
- **Serving Loop S5（2026-07-27，SM 预算 + CPU cert）**：闸门：Serving/RF 执行契约。`sm_budget.resolve_sm_worker_quota` + `RF.step` 按 `sm_cost` 分配/超预算 skip；`RuntimeFusionMlpLayerOverride` SM skip→engine MLP fallback；CPU cert：`scripts/serving_cpu_cert.py` + `make test-serving-cpu-cert`。验证：26 pytest + cert 9 stage PASS。下一轮：**S6** — Radix hit meta。
- **Serving Loop S5b（2026-07-27，实测路径：PyTorch real）**：闸门：拒绝 mock 默认。新增 `torch_engine.TorchEngineModel` / `torch_exec.mlp_torch` / `bench_forward`；`MlpFusionCapsule` 默认 `backend=torch`；cert 仅 real torch（`torch_e2e` + latency ms）。验证：`make test-serving-cpu-cert` PASS。下一轮：**S5c** yirage.core tier。
- **Serving Loop S6（2026-07-27，Radix hit skip/shrink）**：闸门：SGLang Radix meta 驱动 Capsule 调度。`radix_meta.RadixHitMeta` + `RF.step` all-hit skip（`skipped_radix`）+ partial `apply_radix_shrink`；layer override all-hit → post-attn identity。验证：7 pytest + `radix_hit_smoke` + cert 8 stage PASS。下一轮：**S7** 多 Capsule 编排。
- **Serving Loop S7（2026-07-27，multi-Capsule segment override）**：闸门：图级编排层。每层 MLP 拆成 gate_up→down 两 Capsule；`StepMeta.pipeline` + `resolve_capsule_pipeline`；`DecoderSegmentOverride` / `SegmentHybridModelOverride` 大段 override（Attention/KV 仍引擎）。验证：7 pytest + `multi_capsule_segment_smoke` + cert。下一轮：**S8** — 真 vLLM 插件 / torch 实测 bench 归档。
- **Serving Loop S8（2026-07-27，vLLM plugin + segment torch bench）**：闸门：引擎插件 + 实测归档。`TorchDecoderMlpRfHook`（真实 torch 权重/forward，**禁止 mock**）；`VllmQwen2MlpRfHook` 仅当 `vllm` 已安装；`run_segment_torch_bench_archive` JSON ms。验证：cert + smokes 全 real torch。PR #161 待 main 合并。
- **Serving Loop S8b（2026-07-27，移除 contract-only / numpy stub cert）**：闸门：测试契约层。删除 `--contract-only` manifest；S1–S7 pytest 统一 real torch。验证：`make test-serving-cpu-cert-pytest` + `make test-serving-cpu-cert`。
- **Serving Loop S8c（2026-07-27，删除 serving smoke 脚本）**：闸门：工具层。删除 `demo/serving/*smoke*.py`；cert manifest 同步。
- **Serving Loop S8d（2026-07-27，验证禁令入 AGENTS）**：闸门：协议层。写入 [Serving 验证禁令](#serving-验证禁令严禁2026-07-27-起永久有效)；禁止再写 smoke/contract-only/stub cert。
- **Agent Protocol（2026-07-27，TDD + Production 开发协议）**：闸门：文档/协议层。新增 [Agent Development Protocol](#agent-development-protocolmandatory)（TDD、分阶段、设计冻结、最小改动、真实环境、少建文件、README/AGENTS 分工、人工审批门）。
- **Serving Loop S9（2026-07-27，SGLang meta bridge）**：闸门：图级/meta 层。`radix_hit_mask_from_sglang_extend_lens` + `build_sglang_rf_step_meta`（`extend_seq_lens`→Radix + 可选 KV 桥）；RF version **s9**。验证：`test_runtime_fusion_s9_sglang_meta.py` + cert。
- **Serving Loop S10（2026-07-27，SGLang model hook）**：闸门：图级/model 层。`rf_step_meta_from_forward_batch` + `SglangBatchTorchMlpRfHook` / `SglangQwen2MlpRfHook`（须安装 sglang）；RF version **s10**。验证：`test_runtime_fusion_s10_sglang_plugin.py` + cert。
- **Serving Loop S11（2026-07-27，vLLM full-path e2e）**：闸门：图级/e2e 层。`vllm_e2e.run_torch_vllm_mlp_rf_e2e` + `run_torch_vllm_hybrid_full_e2e` + 可选 `run_vllm_qwen2_mlp_rf_e2e`（须 vllm+transformers）；RF version **s11**。验证：`test_runtime_fusion_s11_vllm_e2e.py` + `vllm_mlp_e2e.py`。
- **Serving Loop S12（2026-07-27，SGLang ForwardBatch e2e）**：闸门：图级/e2e 层。`sglang_e2e.SglangForwardBatchSpec` + `run_torch_sglang_*` + 可选 `run_sglang_qwen2_mlp_rf_e2e`；RF version **s12**。验证：`test_runtime_fusion_s12_sglang_e2e.py` + `sglang_mlp_e2e.py`。
- **Serving Loop S13（2026-07-27，vLLM PagedAttention 全层 hook）**：闸门：图级/e2e 层。`VllmPagedKvBatchSpec` + 全层 `HybridModelOverride` + paged KV bridged RF steps；RF version **s13**。验证：`test_runtime_fusion_s13_vllm_paged_e2e.py` + `vllm_paged_e2e.py`。下一轮：**S14** — yirage-core MLP capsule 全层 e2e tier 或 MACA serving 支撑。
- **Serving Loop S14（2026-07-27，yirage.core 全层 hybrid e2e）**：闸门：图级/执行层。`HybridModelOverride(mlp_backend=yirage_cpu)` + `run_yirage_core_full_layer_e2e`（batch=1 decode；gate_up seed + superopt down）；RF version **s14**。验证：`test_runtime_fusion_s14_yirage_core_e2e.py` + `yirage_core_full_e2e.py`（skip 无 yirage.core）。下一轮：**S15** — MACA serving 支撑或 vLLM-metax 插件 tier。
- **Serving Loop S15（2026-07-27，MACA serving + vLLM-metax tier）**：闸门：图级/契约层。`MacaServingRfSpec`（64-warp/C500 SM meta 桥）+ `VllmMetaxQwen2MlpRfHook` + 全层 `maca_serving_e2e`；`BACKEND_YIRAGE_MACA` scaffold；RF version **s15**。验证：`test_runtime_fusion_s15_maca_serving.py` + `maca_serving_e2e.py`。下一轮：**S16** — MetaX VM `yirage_maca` 全层 capsule e2e 或 SGLang-metax 对称 tier。
- **Serving Loop S16（2026-07-27，yirage_maca 全层 + SGLang-metax tier）**：闸门：图级/执行层。`YirageMacaServingMlpRunner` + `MlpFusionCapsule(backend=yirage_maca)` + `yirage_maca_e2e`；`sglang_metax_plugin` + `sglang_metax_e2e`（ForwardBatch + MACA meta）；RF version **s16**。验证：`test_runtime_fusion_s16_metax_tiers.py` + `sglang_metax_e2e.py`（CPU torch）；MetaX VM `yirage_maca_full_e2e.py`。下一轮：**S17** — MetaX VM 全路径 yirage_maca generation latency 归档；SGLang-metax 真 fork e2e。
- **Serving Loop S17（2026-07-27，yirage_maca generation + SGLang-metax real fork）**：闸门：感知/图级层。`yirage_maca_generation` 多步 decode loop + `run_yirage_maca_generation_bench_archive`；`run_sglang_metax_fork_e2e` + `run_sglang_metax_hybrid_full_e2e_auto`；RF version **s17**。验证：`test_runtime_fusion_s17_maca_generation.py` + `yirage_maca_generation_bench.py`（CPU torch）；MetaX VM maca backend generation。下一轮：**S18** — MetaX VM generation latency JSON 归档 vs mcPytorch；SGLang-metax 多层 real fork。
- **Serving Loop S18（2026-07-27，mcPytorch baseline + SGLang-metax multi-layer fork）**：闸门：感知/基准层。`run_yirage_maca_generation_mcpytorch_baseline_archive`（`speedup_vs_mcpytorch` JSON）；`run_sglang_metax_multilayer_fork_e2e/auto`；RF version **s18**。验证：`test_runtime_fusion_s18_maca_baseline.py` + `yirage_maca_generation_bench.py --baseline-json`。下一轮：**S19** — Qwen2-0.5B YiRage CPU required superoptimize + coordinator cert tier。
- **Serving Loop S19（2026-07-29，Qwen2-0.5B YiRage CPU search + coordinator cert）**：闸门：图级/搜索/契约层。`superoptimize_down_matmul_via_coordinator` + `DistributedSearchCoordinator`（blockdim 分区 decode m=1）；`test_runtime_fusion_s19_yirage_cpu_search.py` + `test_serving_ray_search.py` 纳入 `--yirage-core` cert；RF version **s19**；禁止 seed fallback。验证：pytest S19 + ray contract + qwen05b e2e。下一轮：**S20** — Ray 真集群 coordinator e2e。
- **Serving Loop S20（2026-07-29，Ray cluster coordinator e2e）**：闸门：图级/搜索层。`tests/integration/test_serving_coordinator_ray_e2e.py`（Ray 可选 slow）；RF version **s20**。验证：pytest S20 + integration。下一轮：**S21** — 896×4864 tractable full TB search。
- **Serving Loop S21（2026-07-29，tractable full TB down matmul search）**：闸门：图级/搜索层。`YIRAGE_SERVING_FULL_TB_SEARCH=1` 禁用 seed-verify；C++ TB matmul-only explore + Python single-point search；`test_runtime_fusion_s21_full_tb_search.py`；RF version **s21**。验证：pytest S21（tiny slow + env 契约）。下一轮：**S22** — Qwen 896×4864 full TB + Ray coordinator 组合 / RL prescreen。
- **Serving Loop S22（2026-07-29，full TB + Ray coordinator combo）**：闸门：图级/搜索层。`serving_env` 经 coordinator config 传播至 Ray worker；full TB 单点 search + Qwen 896×4864 契约；可选 `YIRAGE_SERVING_ACCELFORGE_PRESCREEN=1`；`test_runtime_fusion_s22_full_tb_ray.py` + integration full TB+Ray；RF version **s22**；修复 `qwen05b_cpu_e2e.py --use-ray`。验证：pytest S22 + integration（Ray）。下一轮：**S23** — Qwen full TB+Ray e2e smoke / AccelForge prescreen 真路径 bench。
- **Serving Loop S23（2026-07-29，Qwen full TB+Ray e2e + AccelForge prescreen bench）**：闸门：图级/正确性 + 工具层。`resolve_serving_search_tier()` / `inspect_serving_search_tier()`；`bench_serving_accelforge_prescreen()` 真 coordinator payload；Qwen e2e report `serving_search_tier`；demo `--accelforge-prescreen`；`test_runtime_fusion_s23_full_tb_ray_e2e.py`；RF version **s23**。验证：pytest S23（slow Qwen full TB+Ray e2e + prescreen bench）。下一轮：**S24** — multi-layer full TB+Ray e2e / prescreen reject-path contract。
- **Serving Loop S24（2026-07-29，multi-layer full TB+Ray + prescreen reject）**：闸门：图级/正确性。`YIRAGE_SERVING_ACCELFORGE_LATENCY_BUDGET_MS` prescreen reject-path；Qwen 2-layer full TB+Ray e2e（`quick=False`）；`test_runtime_fusion_s24_multilayer_prescreen.py`；RF version **s24**。验证：pytest S24。下一轮：**S25** — full-model all-layer RF + search tier archive JSON。
- **Serving Loop S25（2026-07-29，all-layer RF + search tier archive）**：闸门：图级/工具层。`all_rf_layers` + `run_hf_qwen05b_search_tier_bench_archive()`；`ServingBenchArchive.search_tier` JSON；demo `--all-rf-layers --archive-json`；`test_runtime_fusion_s25_all_layer_archive.py`；RF version **s25**。验证：pytest S25（slow 24-layer archive）。下一轮：**S26** — archive CI artifact + multi-tier nightly compare。
- **Serving Loop S26（2026-07-29，multi-tier archive CI + nightly compare）**：闸门：感知/工具层。`search_tier_archive.py`（tier presets、`compare_serving_search_tier_archives`、`ServingMultiTierBenchArchive`）；`scripts/serving_search_tier_archive.py` + `validate_serving_search_tier_archive.py`；Makefile `test-serving-search-tier-archive-profile`；`.github/workflows/serving-search-tier-archive.yml`；`test_runtime_fusion_s26_multi_tier_archive.py`；RF version **s26**。验证：pytest S26 + seed_verify archive CI gate。下一轮：**S27** — decode fused vs torch bench 或 full_tb_ray nightly tier。
- **Serving Loop S27（2026-07-30，Qwen decode-step fused vs native bench）**：闸门：感知/性能层。`qwen_decode_bench.py` + `benchmark/serving_qwen_decode_bench.py`；RF **s27**。验证：pytest S27。**价值：中价值｜G3+G5+G6**。**功能闭合：G7 段落闭合（链 B）** — HF prefill → 单步 decode RF MLP → logits parity；未闭合 vLLM 全链（G1）。拒绝项：MLP-only 无 parity → S30。
- **Serving Loop S28（2026-07-30，multi-layer decode bench）**：闸门：感知/性能层。`run_qwen_multilayer_decode_bench()` + `--all-rf-layers`；RF **s28**。验证：pytest S28。**价值：中价值｜G3+G5+G6**。**功能闭合：G7 段落闭合（链 B 多层）** — 多层 per-layer superopt decode parity；未闭合 generation 多步。拒绝项：full_tb_ray nightly → S29。
- **Serving Loop S29（2026-07-30，full_tb_ray nightly multi-tier archive）**：闸门：感知/工具层。multi-tier archive + nightly workflow；RF **s29**。验证：pytest S29。**价值：中价值｜G3+G6**。**功能闭合：仅基础设施** — tier archive/CI；不宣称 G7 完成。拒绝项：decode nightly → S30。下一轮：**S30** — MLP-only micro-bench 须带 parity 链（G5+G6+G7 段落）或 vLLM/SGLang G1 回归。
- **Serving Loop G7（2026-07-31，完整功能闭合协议）**：闸门：文档/协议层。新增 **G7** 设计目标与 [G7 功能闭合检验](#g7-功能闭合检验完整功能-vs-零散堆砌)；价值四档、策略模板、行为审查、检查清单对齐 **G1–G7**；明确全链/段落/基础设施/零散堆砌四档。**价值：中价值｜G6+G7** — Loop 评价锚点补全，防止 registry/单测堆砌冒充功能完成。验证：AGENTS.md 审阅 + 模板示例 S27/S29。
- **Serving Loop S30（2026-07-31，MLP capsule micro-bench + G7 链 A）**：闸门：感知/正确性层。`mlp_capsule_bench.py` + `benchmark/serving_mlp_capsule_bench.py`；RF **s30**。验证：pytest S30。**价值：中价值｜G3+G5+G6+G7**。**功能闭合：G7 段落闭合（链 A）** — tensor in → RF.step → MlpFusionCapsule → parity vs eager + timing；非 HF/vLLM 全链。拒绝项：decode nightly artifact → S31；纯 timing 无 parity。下一轮：**S31** — decode bench nightly archive 或 vLLM/SGLang G1 回归。

- **MACA 后端基线（2026-07-07）**：主优化目标从 CPU 切换为 MetaX MACA；开发机 MetaX C500（`mx-smi` 2.2.12，mcPytorch `2.8.0+metax3.5.3.9`）；构建 `YIRAGE_BACKEND=maca pip install -e .`；文档锚点 `docs/maca_quick_start.md`。
- **Loop R0（2026-07-07，目标切换）**：闸门：文档/协议层。`AGENTS.md` 主闭环改为 MACA；Cloud Agent 须在 MetaX SSH VM 验证；CPU Loop R1–R137 迁入归档。验证：MetaX VM `mx-smi` + mcPytorch OK；下一轮：**R1 感知** — 跑 `demo_maca_optimization` + `maca_vs_pytorch`，建立 fusion vs mcPytorch 基线 JSON。
- **Loop R1（2026-07-07，demo/kernel 性能路径，PR #152）**：闸门：执行层 + 感知/工具层。根因：`yirage.maca_config` 缺失、`maca_call` 走 ascend 解释器、`compile()` 仅 nvcc、demo/bench 用 `backend=cpu` 与占位计时；**正式路径阻塞**：`USE_MACA` 链 `transpiler_stub` 致 `generate_cuda_program` segfault。修复：`maca_config.py` shim、`get_maca_search_config_quick`/`resolve_maca_search_config`、`maca_call→cuda_call`（mcPytorch）、`mxcc` JIT、`CMakeLists` MACA 启用 CUDA transpiler（`YIRAGE_BACKEND_CUDA_ENABLED`）、**撤销** `YIRAGE_MACA_SKIP_PROFILE`/`YIRAGE_MACA_TORCH_MATMUL` 回退、`demo/_maca_utils.py` + demo/bench 真实 `superoptimize(backend=maca)`；mxcc `maca_compat/` shims + resilient MACA profiling。验证：`pytest tests/python/test_maca_config.py`；MetaX VM：`demo_maca_optimization` + `maca_superopt_test` exit 0（search ~31–49s，部分 µGraph smem/mma 跳过）。
- **Loop R2（2026-07-08，smem 64KB 对齐，PR #152）**：闸门：执行/契约层（C500 65536 B/block vs Volta 98304 误判）。`maca::MAX_SMEM_SIZE`→64KB；`get_shared_memory_capacity()` MACA/mxcc 返回 `MACA_SHARED_MEM_PER_BLOCK`；`test_maca_config` smem 闸门。MetaX VM：`get_shared_memory_capacity(70)=65536`；`demo_maca_optimization`/`maca_superopt_test`/`test_maca_config` PASS；C++ `plan_stensor_memory` 仍报 98304 直至 VM 重编 `yirage.core`。下一轮：**R3** — mxcc 软件 MMA（`mma.sync` invalid on xcore1000）、VM 重编闭合 smem 警告、`maca_vs_pytorch`/`maca_native_benchmark` 全量 bench。
- **Loop R3（2026-07-08，mxcc 软件 MMA + bench quick，PR #152）**：闸门：执行/原语层（闭合 R2 mxcc `mma.sync` invalid on xcore1000）。`maca_compat/cute/arch/mma_sm70.hpp` warp-shuffle 软件 m8n8k4；`get_mxcc_cc_cmd()` 改 `YIRAGE_MACA_SOFTWARE_MMA=1`、移除 `CUTE_ARCH_MMA_SM70_ENABLED`/`LDSM_SM75`；`maca_vs_pytorch.py --quick`。MetaX VM：`maca_superopt_test` µGraph **14–15 编译+profile 成功**（原 mma 失败）；`maca_vs_pytorch --quick` **~154s PASS**；`demo_maca_optimization`/`test_maca_config` PASS。C++ `MAX_SMEM_SIZE(98304)` 警告仍在（VM 未重编 `yirage.core`）。下一轮：**R4** — VM 重编闭合 smem 警告、搜索 smem 超限 µGraph 收窄、`maca_native_benchmark --quick`。
- **Loop R4（2026-07-08，CUDA 对标协议文档，PR #152）**：闸门：文档/契约层（用户要求 MACA **对标 CUDA** 优化目标写入闭环）。`AGENTS.md` 增 [CUDA 对标：优化目标与能力矩阵](#cuda-对标优化目标与能力矩阵)（三层 parity、能力表、对齐策略、真卡效果检查、合并闸门）；核心原则/感知/策略/验证/检查清单/backlog 改为 **CUDA 路径为金标准 + MetaX 真卡必验**。下一轮：**R5** — VM 重编 `yirage.core` 闭合 smem；按能力矩阵补 attention/PK 真卡 smoke backlog。
- **Loop R5（2026-07-08，RL/Ray parity + Qwen 全链路 demo，PR #152）**：闸门：图级/正确性 + 工具契约层。`AGENTS.md` 增 RL/Ray CUDA 对标表 + Qwen 推理对标表；`demo/maca/qwen_inference_demo.py`（对齐 `demo/qwen2.5/demo.py`：maca superoptimize → prefill+decode）；`benchmark/end-to-end/maca/qwen_maca.py`；`tests/python/test_maca_rl_ray_capability.py`；`tests/integration/test_maca_qwen_inference_demo.py`。验证：pytest RL/Ray 契约（可无卡）；MetaX VM `qwen_inference_demo --quick`。下一轮：**R6** — `test_ray_maca_e2e`、`maca_capability_walkthrough`、HF Qwen `from_pretrained` MACA 路径。
- **Loop R6（2026-07-08，Ray MACA e2e + walkthrough，PR #152）**：闸门：图级/工具契约层（闭合 R5 Ray/walkthrough backlog）。`tests/integration/test_ray_maca_e2e.py`（镜像 CPU Ray e2e，`backend=maca`，block=256 warp 对齐）；`scripts/maca_capability_walkthrough.py`（YiRage×Ray×AccelForge×RL on MACA）；`test_maca_rl_ray_capability.py` 契约更新。验证：pytest 契约层（可无卡）；MetaX VM `test_ray_maca_e2e` + `maca_capability_walkthrough --quick`。下一轮：**R7** — VM 重编 `yirage.core` 闭合 smem 98304；`VerifierPool(backend=maca)`；HF Qwen `from_pretrained`。
- **Loop R7（2026-07-08，VerifierPool backend=maca，PR #152）**：闸门：RL/工具契约层（闭合 R6 `VerifierPool` gap）。`VerifierPool` 增 `backend` 参数（默认 `YIRAGE_BACKEND`）；Ray/local verifier 传递 `backend=maca`；`tests/python/test_rl/test_maca_verifier_pool.py`。验证：pytest 契约（可无卡）；MetaX VM Ray verifier smoke。下一轮：**R8** — VM 重编 `yirage.core` 闭合 smem 98304；`bench_ray_search.py backend=maca`；HF Qwen `from_pretrained`。
- **Loop R8（2026-07-08，bench_ray_search backend=maca，PR #152）**：闸门：感知/工具层（闭合 R7 `bench_ray_search` gap）。`scripts/bench_ray_search.py` 增 `--backend maca|cpu`、`--quick`、`--json`；MACA griddim 套件 + 64-warp blockdims；`tests/python/test_bench_ray_search_maca.py`。验证：pytest 契约（可无卡）；MetaX VM `bench_ray_search.py --backend maca --quick`。下一轮：**R9** — VM 重编 `yirage.core` 闭合 smem 98304；HF Qwen `from_pretrained`；`demo/_maca_utils` Ray opt-in。
- **Loop R9（2026-07-08，MACA Ray opt-in + smem 源码契约，PR #152）**：闸门：图级/工具契约层（闭合 R8 `use_ray` 硬编码 gap）。`demo/_maca_utils.py` 增 `resolve_maca_use_ray()` / `maca_superoptimize_ray_kwargs()`（`YIRAGE_MACA_USE_RAY=1` opt-in）；`maca_superopt_test`/`demo_maca_optimization`/`qwen_inference_demo` 统一调用；`test_maca_config` 增 `config.h` MACA smem 64KB 断言。验证：pytest `test_maca_config` + `test_maca_rl_ray_capability`（可无卡）；MetaX VM `YIRAGE_MACA_USE_RAY=1 maca_superopt_test.py`。下一轮：**R10** — VM 重编 `yirage.core` 闭合 smem 98304；HF Qwen `from_pretrained`；`RayDistributedEngine` MACA placement。
- **Loop R10（2026-07-08，RayDistributedEngine MACA placement，PR #152）**：闸门：图级/工具契约层（闭合 R9 `RayDistributedEngine` placement gap）。`maca/config.py` 增 `is_maca_torch_device_available` / `resolve_maca_gpus_per_worker` / `maca_ray_gpu_placement_kwargs`；`RayDistributedEngine._effective_gpus_per_worker` + `create_engine(backend=maca)` 接入；`test_ray_maca_e2e` 改用 placement helper。验证：pytest `test_maca_config` + `test_maca_rl_ray_capability`（可无卡）；MetaX VM `test_ray_maca_e2e::test_ray_distributed_engine_maca`。下一轮：**R11** — VM 重编 `yirage.core` 闭合 smem 98304；HF Qwen `from_pretrained`。
- **Loop R11（2026-07-08，HF Qwen config scaffold + smem rebuild script，PR #152）**：闸门：图级/工具契约层（闭合 R10 HF/smem backlog 的可 Cloud 落地部分）。`demo/maca/qwen_hf_utils.py`（Hub `config.json` 形状 + `--model`/`--config-only`）；`qwen_inference_demo.py` HF flags；`scripts/maca_rebuild_core.sh`（MetaX VM `YIRAGE_BACKEND=maca pip install -e .` + smem 65536 断言）；`test_maca_qwen_hf_contract.py`。验证：pytest 契约（可无卡）；MetaX VM `bash scripts/maca_rebuild_core.sh` + `qwen_inference_demo --model Qwen/Qwen3-8B --config-only --quick`。下一轮：**R12** — MetaX 全权重 `from_pretrained` + `modeling_qwen2.superoptimize_kernels` on MACA；PersistentKernel qwen3 路径。
- **Loop R12（2026-07-08，HF from_pretrained + modeling_qwen2_maca，PR #152）**：闸门：图级/正确性（闭合 R11 HF 全权重 backlog）。`demo/maca/qwen_kernel_utils.py`（共享 MACA superoptimize）；`demo/maca/models/modeling_qwen2_maca.py`（无 flashinfer，`superoptimize_kernels` → `backend=maca`）；`demo/maca/qwen_from_pretrained_demo.py`（对齐 `demo/qwen2.5/demo.py`）；`qwen_inference_demo` 复用 kernel utils；`test_maca_qwen_hf_contract` 扩展。验证：pytest 契约（可无卡）；MetaX VM `qwen_from_pretrained_demo.py --model Qwen/Qwen3-8B --max-layers 1 --quick`。下一轮：**R13** — CUDA Graph decode on MACA；`demo/qwen3` PersistentKernel 路径；VM `maca_rebuild_core.sh` smem 闭合。
- **Loop R13（2026-07-08，CUDA Graph decode on MACA，PR #152）**：闸门：图级/正确性（闭合 R12 CUDA Graph backlog）。`demo/maca/qwen_decode_loop.py`（prefill eager + decode CUDA Graph capture/replay，对齐 `demo/qwen2.5/demo.py`）；`qwen_from_pretrained_demo.py` 默认 CUDA Graph + `--no-cuda-graph`；`test_maca_qwen_from_pretrained_demo.py`。验证：pytest 契约（可无卡）；MetaX VM `qwen_from_pretrained_demo.py --max-layers 1 --quick --max-tokens 16`。下一轮：**R14** — `demo/qwen3` PersistentKernel MACA 路径；VM `maca_rebuild_core.sh` smem 闭合。
- **Loop R14（2026-07-08，Qwen3 PersistentKernel MACA scaffold，PR #152）**：闸门：图级/契约层（闭合 R13 qwen3 PK backlog 的可 Cloud 落地部分）。`demo/maca/qwen3_pk_utils.py` + `qwen3_persistent_kernel_demo.py`（对齐 `demo/qwen3/demo.py --use-yirage`；`--inspect-only` Cloud 契约 + MetaX `--quick` PKRuntime offline smoke）；`get_available_backends()` MetaX 检测；`BACKEND_CAPABILITIES` MACA smem 64KB；`test_maca_qwen3_persistent_kernel_demo.py`。验证：pytest 契约（可无卡）；MetaX VM `qwen3_persistent_kernel_demo.py --quick` + `maca_rebuild_core.sh`。下一轮：**R15** — 全量 `ypk.compile()` mxcc task-graph；VM smem 98304 警告闭合。
- **Loop R15（2026-07-08，PK smem 64KB + mxcc compile path，PR #152）**：闸门：执行/图级层（闭合 R14 mxcc compile + smem 98304 backlog）。`maca_pk_backend.cc` `get_shared_memory_size`→`maca::MAX_SMEM_SIZE`；`PersistentKernel.compile()` 增 `get_maca_pk_compile_command` + `_resolve_persistent_kernel_compiler`；`maca_rebuild_core.sh` PK smem 源码闸门；`test_maca_pk_smem_contract.py`。验证：pytest 契约（可无卡）；MetaX VM `bash scripts/maca_rebuild_core.sh` + qwen3 PK e2e smoke。下一轮：**R16** — attention smoke + qwen3 PK compile-inspect。
- **Loop R16（2026-07-08，attention smoke + PK compile-inspect，PR #152）**：闸门：图级/契约层（闭合 attention matrix backlog 与 R15 qwen3 PK e2e 前置）。`demo/maca/attention_utils.py` + `attention_smoke.py`（chameleon attention superoptimize；`--inspect-only` Cloud + MetaX `--quick`）；`inspect_maca_pk_compile_contract()` + `qwen3_persistent_kernel_demo.py --compile-inspect`；`test_maca_attention_contract.py` + `test_maca_attention_smoke.py`。验证：pytest 契约（可无卡）；MetaX VM `attention_smoke.py --quick` + `qwen3_persistent_kernel_demo.py --compile-inspect`。下一轮：**R17** — minimal qwen3 PK `ypk.compile()` e2e。
- **Loop R17（2026-07-08，minimal PK embed compile e2e，PR #152）**：闸门：执行/图级层（闭合 R16 全量 qwen3 e2e 前置）。`build_qwen3_pk_meta_tensors()` + `maca_pk_minimal_compile_smoke()`（embed-only task-graph + mxcc `ypk.compile()`）；`--compile-plan` Cloud + MetaX `--compile-only`；契约/集成测试扩展。验证：pytest 契约（可无卡）；MetaX VM `qwen3_persistent_kernel_demo.py --compile-only` + `maca_rebuild_core.sh`。下一轮：**R18** — 单层 Qwen3 PK task-graph（embed+attn+mlp）mxcc compile。
- **Loop R18（2026-07-08，one-layer Qwen3 PK compile e2e，PR #152）**：闸门：图级/执行层（闭合 R17 全量多 layer 前置）。`_build_maca_pk_one_layer_graph()`（对齐 `demo/qwen3/demo.py` layer[0]：embed+rmsnorm/linear QKV+paged_attention+o_proj residual+MLP silu_mul+down_proj residual）；`maca_pk_one_layer_compile_smoke()` + `inspect_maca_pk_compile_plan(variant=one_layer)`；`--compile-plan-variant one_layer` Cloud + MetaX `--compile-one-layer`。验证：pytest 契约（可无卡）；MetaX VM `qwen3_persistent_kernel_demo.py --compile-one-layer` + `maca_rebuild_core.sh`。下一轮：**R19** — N-layer stack + lm_head/argmax mxcc compile。
- **Loop R19（2026-07-08，stack PK compile + lm_head/argmax，PR #152）**：闸门：图级/执行层（闭合 R18 全量 `demo/qwen3/demo.py --use-yirage` 前置）。重构 `_append_maca_pk_decoder_layer` + `_append_maca_pk_lm_head_argmax`；`maca_pk_stack_compile_smoke()`（默认 2 layers + lm_head/argmax）；`--compile-plan-variant stack` Cloud + MetaX `--compile-stack`。验证：pytest 契约（可无卡）；MetaX VM `qwen3_persistent_kernel_demo.py --compile-stack --pk-compile-layers 2`。下一轮：**R20** — HF 全权重 multi-layer PK runtime e2e；attention native kernel bench。
- **Loop R20（2026-07-08，PK stack runtime + attention bench，PR #152）**：闸门：图级/执行层（闭合 R19 compile-only gap）。`prepare_maca_pk_runtime_meta` + `maca_pk_stack_runtime_smoke`（1 layer 默认 compile + `ypk()` launch）；`inspect_maca_pk_runtime_plan` / `inspect_maca_pk_hf_runtime_plan` Cloud 契约；`maca_attention_native_bench_quick`（mugraph vs mcPytorch reference）；demo `--runtime-stack` / `--bench`。验证：pytest 契约（可无卡）；MetaX VM `qwen3_persistent_kernel_demo.py --runtime-stack --pk-compile-layers 1` + `attention_smoke.py --bench --quick`。下一轮：**R21** — HF 权重注入 PK stack graph + 全 layer runtime e2e。
- **Loop R21（2026-07-08，HF 权重注入 PK stack runtime，PR #152）**：闸门：图级/正确性（闭合 R20 HF backlog）。`qwen3_pk_hf_utils.py`（`load_maca_pk_hf_weight_bundle` + attach_map）；`_build_maca_pk_stack_graph(hf_bundle=)` embed/layer/lm_head HF 路径；`maca_pk_hf_stack_runtime_smoke`；demo `--hf-weight-plan` / `--hf-runtime-stack`；修复 `_append_maca_pk_decoder_layer` 重复 attach_input。验证：pytest 契约（可无卡）；MetaX VM `qwen3_persistent_kernel_demo.py --hf-runtime-stack --pk-compile-layers 1`。下一轮：**R22** — 全 layer HF runtime、 padded lm_head (153600) argmax、对标 CUDA `demo/qwen3/demo.py` generation loop。
- **Loop R22（2026-07-08，padded lm_head + multi-layer + generation scaffold，PR #152）**：闸门：图级/正确性（闭合 R21 next_steps）。`use_padded_lm_head`（153600）贯穿 HF loader/stack runtime；`inspect_maca_pk_hf_padded_lm_head_plan` / `inspect_maca_pk_hf_generation_plan`；`maca_pk_hf_generation_smoke`（单步 launch + output_token）；demo `--hf-padded-plan` / `--hf-generation-plan` / `--hf-padded-lm-head`。验证：pytest 契约（可无卡）；MetaX VM `--hf-runtime-stack --pk-compile-layers 2` + `--hf-padded-lm-head`。下一轮：**R23** — multi-step decode generation loop + tokenizer vs CUDA demo。
- **Loop R23（2026-07-08，multi-step decode loop + tokenizer utils，PR #152）**：闸门：图级/正确性（闭合 R22 generation backlog）。`qwen3_pk_generation_utils.py`（`prepare_maca_pk_prompt_meta` / `advance_maca_pk_decode_step` / `run_maca_pk_decode_loop` / tokenizer encode-decode）；`_maca_pk_hf_init_compiled_stack` 共享 compile；`maca_pk_hf_generation_smoke(decode_steps, use_tokenizer)`；`multi_step_decode_ready=True`；demo `--hf-decode-step-plan` / `--hf-generation-smoke --decode-steps N`。验证：pytest 契约（可无卡）；MetaX VM `--hf-generation-smoke --decode-steps 4` + `--use-tokenizer`。下一轮：**R24** — tokenizer 全链路 generation e2e vs CUDA `demo/qwen3/demo.py --use-yirage`；multi-request batching。
- **Loop R24（2026-07-08，tokenizer full-path + multi-request batch，PR #152）**：闸门：图级/正确性（闭合 R23 backlog）。`compute_maca_pk_generation_latency`；`prepare_maca_pk_batched_prompt_meta`；`inspect_maca_pk_hf_tokenizer_generation_plan` / `inspect_maca_pk_multi_request_batch_plan`；`maca_pk_hf_tokenizer_generation_smoke`（tokenizer + latency）；demo `--hf-tokenizer-generation-plan` / `--hf-multi-request-batch-plan` / `--hf-tokenizer-generation-smoke`。验证：pytest 契约（可无卡）；MetaX VM `--hf-tokenizer-generation-smoke --decode-steps 4`。下一轮：**R25** — multi-request ypk() decode loop；MetaX generation e2e 对标 CUDA 全 layer。
- **Loop R25（2026-07-08，batched decode loop + full-layer e2e，PR #152）**：闸门：图级/正确性（闭合 R24 backlog）。`advance_maca_pk_batched_decode_step` / `run_maca_pk_batched_decode_loop`；`maca_pk_hf_batched_tokenizer_generation_smoke`；`maca_pk_hf_full_layer_tokenizer_generation_smoke`；`inspect_maca_pk_batched_decode_plan` / `inspect_maca_pk_hf_full_layer_generation_plan`；demo `--hf-batched-decode-plan` / `--hf-batched-generation-smoke` / `--hf-full-layer-generation-smoke`。验证：pytest 契约（可无卡）；MetaX VM `--hf-batched-generation-smoke --active-requests 2` + `--hf-full-layer-generation-smoke --decode-steps 4`。下一轮：**R26** — padded lm_head 全 layer batched e2e；per-request divergent prompts。
- **Loop R26（2026-07-08，divergent batch + full-layer batched padded，PR #152）**：闸门：图级/正确性（闭合 R25 backlog）。`prepare_maca_pk_batched_divergent_prompt_meta`；`maca_pk_hf_divergent_batched_tokenizer_generation_smoke`；`maca_pk_hf_full_layer_batched_padded_generation_smoke`；`inspect_maca_pk_divergent_batch_plan` / `inspect_maca_pk_hf_full_layer_batched_padded_generation_plan`；demo `--hf-divergent-batch-plan` / `--hf-divergent-generation-smoke` / `--hf-full-layer-batched-generation-smoke`。验证：pytest 契约（可无卡）；MetaX VM `--hf-full-layer-batched-generation-smoke --active-requests 2` + `--hf-divergent-generation-smoke --chat-prompts "Hello,What is AI?"`。下一轮（历史）：曾规划 **R27** offline latency 归档；**已由 Serving S0（2026-07-27）接管主轨**，R27 降为支撑 backlog。

- **协议（2026-07-07）**：MACA 改动须在 MetaX GPU VM 验证通过后合并；禁止用 CPU cert/loop-close 替代。
- **协议（2026-07-08，CUDA 对标）**：MACA 算子/融合/搜索须逐项对齐 CUDA 后端能力；每项能力须有 MetaX **真卡**验证入口；性能同后端对 mcPytorch，**功能**以 CUDA 正式路径为参照（**支撑轨**）。
- **协议（框架对齐）**：每轮策略前必过执行前闸门 — 融合上浮、原语对齐库，64-thread warp（MACA）。
- **协议（2026-07-27，RuntimeFusion）**：主轨为 **FusionPlan → FusionCapsule → RF.step** 嵌入 vLLM/SGLang；融合身份归 Capsule/RF，调度权归引擎；禁止 Mirage/MPK/µGraph 对外叙事与单 Attention 替换冒充完成；S1=可选择的 MLP Capsule+`step`，而非烤死 MegaKernel；完成宣称须真机 smoke。

---

## 历史归档：CPU 无限优化闭环

> 以下内容为 2026-06 至 2026-07 间 CPU 后端无限优化闭环笔记，**不再作为默认 Agent 主目标**。CPU 命令与 matrix 仍有效，见 `docs/cpu_support_matrix.yaml`、`make test-cpu-cert`。
- **CPU 后端基线（main）**：同后端 superoptimize 价值评估、`bench_fused_vs_mkl_baseline.py`、`eval_optimization_value.py`、P0 host BLAS + TB interpreter 路径已文档化于 `HARDWARE_OPTIMIZATION.md`。
- **Loop R1（2026-06-09，CPU 认证落地）**：合并 `cpu_support_matrix.yaml`、`make test-cpu-cert` / `test-cpu-value-verify`（47→49 项）、`get_cpu_search_config()` 搜索对齐、`search_explore_gaps=[]`。感知：`make test-cpu-cert` 26 passed，`bench_fused_vs_mkl_baseline --quick` rms_norm_matmul speedup≈1.01。
- **Loop R2（2026-06-09，forloop accum）**：瓶颈 `tb_forloop_accum_red_ld_sum_op` 解释器误用逐元素 `+=`（应为 last-dim reduce-sum）；补齐 value verify builders；`red_ld_sum` 从 experimental 升为 supported。验证：`make test-cpu-value-verify` **49 passed**，`make test-cpu-cert` 26 passed。
- **Loop R3（2026-06-09，`tb_forloop_accum_red_ld_mean_op`）**：CPU 解释器补齐 mean epilogue（与 CUDA reduce-sum + 末轮 `/n` 一致）；matrix 升为 supported 并加入 search explore；value verify **50 passed**。验证：`make test-cpu-value-verify`、`make test-cpu-cert`。
- **Loop R4（2026-06-09，redtox + max）**：补齐 `tb_forloop_accum_redtox_ld_sum_op`（`reduction_to_dimx` per tile 累加）与 `tb_forloop_accum_max_op`（逐元素 max）；value verify **52 passed**。验证：`make test-cpu-value-verify`、`make test-cpu-cert`。
- **Loop R5（2026-06-09，clamp + mul_scalar）**：`kn_clamp_op` / `tb_clamp_op` / `tb_mul_scalar_op` 升为 supported；C++ JSON 序列化 `min_val`/`max_val`/`scalar`；Cython `get_graph_structure` 暴露参数；`graph.py` CPU 解释器补齐。**根因**：`KNCustomizedOp` 复制 bgraph 时 `TB_CLAMP_OP` 误走 `elementunary(scalar)` 丢失边界，已改为 `elementunary_clamp`。验证：`make test-cpu-value-verify` **53 passed**（+1 tb_clamp），`make test-cpu-cert` 26 passed。
- **Loop R6（2026-06-09，`kn_mul_scalar_op`）**：补齐 KN `Graph::mul_scalar`、`KNElementUnaryOp::scalar` + JSON；Cython/Python API 与 CPU 解释器；matrix 升为 supported。验证：`make test-cpu-value-verify` **54 passed**，`make test-cpu-cert` **27 passed**。
- **Loop R7（2026-06-09，搜索空间对齐）**：`GeneratorConfig::get_cpu_search_config()` 与 `search_cpu_default_explore` 同步；纳入 clamp/mul_scalar、forloop mean/redtox/max；`op_utils` + abstract_expr 补齐；`test_cpu_search_explore_sync.py` 防漂移。验证：`make test-cpu-cert`、`make test-cpu-value-verify`。
- **Loop R8（2026-06-09，搜索扩展 unary/binary）**：CPU 搜索纳入 sigmoid/log/sub、KN reduction；补齐 abstract_expr（Sigmoid/Log/Sub）、symbolic_graph、irange、op_utils。TB reduction 暂缓（irange 未实现）。验证：`make test-cpu-cert`（29 passed）、`make test-cpu-value-verify`（54 passed）。
- **Loop R9（2026-06-09，TB reduction 搜索）**：`irange.cc` 补齐 `TB_REDUCTION_0/1/2` 与 `*_TO_DIMX_*` forward/backward（`extend_dim` + `truncate`，与 KN reduction 同型）；CPU 搜索 explore 纳入 `tb_reduction_0/1/2_op`；`test_cpu_search_explore_sync.py` 防漂移。验证：`make test-cpu-cert`（29 passed，superoptimize ~258s）、`make test-cpu-value-verify`（54 passed）。
- **Loop R10（2026-06-09，TB reduction_to_dimx）**：CPU 搜索 explore 纳入 `tb_reduction_*_to_dimx_op`；`op_utils::is_unary` 补齐；C++ `Graph::reduction_to_dimx(STensor*)` 指针重载 + Cython/Python API；`graph.py` 解释器 + value verify builders（**57 passed**）。验证：`make test-cpu-cert`（29 passed，superoptimize ~400s）、`make test-cpu-value-verify`（57 passed）。
- **Loop R11（2026-06-09，TB reduction_max CPU）**：`graph.py` 补齐 `tb_reduction_*_max_op` 双输出解释器（`ReductionMaxKernel` 语义：跨 forloop 迭代累积 max + diff）；matrix 三档 `supported`；value verify builders（**60 passed**）。搜索暂缓（双输出 + symbolic/irange 未接）。验证：`make test-cpu-cert`（29 passed，~598s）、`make test-cpu-value-verify`（60 passed）。
- **Loop R12（2026-06-09，TB reduction_max 搜索基础设施）**：闸门判定为图级/搜索层（非原语微优）；补齐 symbolic_graph、abstract_expr、irange、`op_utils`、`graph.cc` from_json；**刻意未**将 `tb_reduction_*_max_op` 纳入 explore（双输出 guid 仍会导致 superoptimize abort）。matrix 备注「search infra ready, explore deferred」。验证：`make test-cpu-value-verify`（60 passed）、`make test-cpu-cert`（29 passed，~270s）。
- **Loop R13（2026-06-09，TB reduction_max 纳入搜索 explore）**：闸门：图级搜索层，闭合 R12「infra ready / explore deferred」gap。根因：`irange.cc` 的 `forward_propagate`/`backward_propagate` 对双输出 MAX 仅返回 1 条 range（assert 失败）；`abstract_expr_eval` 未为 `output_tensors[1]` 注册 expr。修复后纳入 `search_cpu_default_explore` + `config.cc`。验证：`make test-cpu-value-verify`（60 passed）、`make test-cpu-cert`（29 passed，superoptimize ~810s）。
- **Loop R14（2026-06-09，TB concat CPU + 搜索输出上界，PR #66 → `9d7904b`）**：闸门：图级布局（TF/oneDNN `torch.cat`）。`graph.py` 补齐 `tb_concat_0/1/2`（含 `tb_concat_first_op_id` 枚举别名）；matrix 三档 `supported`；value verify **63 passed**；`get_cpu_search_config()` 设 `max_num_threadblock_graph_outputs=3`。concat search explore 暂缓（`abstract_expr` 无 Concat 节点）。验证：`make test-cpu-cert`（29 passed，~1233s）。
- **Loop R15（2026-06-09，TB concat search explore，PR #68 → `44b9634`）**：闸门：图级/搜索层（闭合 R14 explore deferred）。`abstract_expr` 新增 `Concat`（egg `(concat lhs rhs dim)`）；`abstract_expr_for_ops` / `op_utils` 补齐三档 `TB_CONCAT_*`；`tbop_to_explore` + matrix + sync test 纳入 explore。验证：`make test-cpu-value-verify`（63 passed）；`make test-cpu-cert`（29 passed，~1189s）。
- **Loop R1（TileKernel baseline，kn_sub_op，PR 待合并）**：闸门：执行层。``KN_SUB_OP`` + ``Graph.sub`` + CPU interpreter；verify **111**。
- **Loop R2（TileKernel baseline，kn_transpose_01_op，PR 待合并）**：闸门：layout 层（``torch.transpose(0,1)``）。``KN_TRANSPOSE_01_OP`` + ``Graph.transpose01`` + search explore；verify **112**。
- **Loop R3（TileKernel baseline，general ML 对齐，PR 待合并）**：闸门：图级/正确性（非孤立 GEMM 微优）。稳定 ``softmax``/``layer_norm``（TB reduction_max 路径）；``gemm_softmax``/``gemm_layernorm`` 对齐 ``F.softmax``/``F.layer_norm``；online softmax rescale accum 解释器；verify **118**（rescale search explore 见 YiRage Loop R6）。
- **Loop R4（YiRage main，runtime shape-aware MuGraph cache，PR 待合并）**：闸门：runtime/P0（搜索价值需动态 shape 命中）。``find_best_mugraph(..., input_shapes=...)`` 优先同 shape 条目；``YIRAGE_MUGraph_REQUIRE_SHAPE_MATCH=1`` 严格模式；``superoptimize`` persistent restore 传入 seed 图 input shapes。
- **Loop R5（YiRage main，MuGraph shape bucket，PR 待合并）**：闸门：runtime/P0（闭合 R4 相邻 bucket 动态 shape）。power-of-two ``bucket_dim``；exact → bucket → global 三级 restore；``YIRAGE_MUGraph_SHAPE_BUCKET=0`` 可关。
- **Loop R6（YiRage main，TB rescale search explore，PR 待合并）**：闸门：图级/搜索层（闭合 R3 matrix「search explore deferred」）。``TB_FORLOOP_ACCUM_*_RESCALE_OP`` 纳入 ``tbop_to_explore``；search 栈（``op_utils``、``abstract_expr``、``symbolic_graph``、``irange``、``formal_verifier``、``config.cc``）；``type.h`` JSON 名对齐 ``tb_forloop_accum_no_red_rescale_op``。验证：``make test-cpu-cert-quick`` **59 passed**；``make test-cpu-value-verify`` **118 passed**；``search_explore_gaps=[]``。
- **Loop R7（YiRage main，kn_conv2d CPU 契约，PR 待合并）**：闸门：执行层（CV 基础算子）。新增 ``KN_CONV2D_OP`` + ``Graph.conv2d``（NCHW/OIHW，stride/padding/dilation JSON）；Cython/Python + ``graph.py`` CPU 解释器（``F.conv2d``）；matrix ``kn_conv2d_op`` supported（search explore 见 R8）。验证：``make test-cpu-cert-quick`` **60 passed**；``make test-cpu-value-verify`` **119 passed**。
- **Loop R8（YiRage main，kn_conv2d search explore，PR 待合并）**：闸门：图级/搜索层（闭合 R7 matrix「search explore deferred」）。``KN_CONV2D_OP`` 纳入 ``knop_to_explore``；search 栈全链路。验证：``make test-cpu-cert-quick`` **60 passed**；``make test-cpu-value-verify`` **119 passed**；``search_explore_gaps=[]``。
- **Loop R9（YiRage main，stable self_attention CPU，PR 待合并）**：闸门：图级/正确性（闭合 COMET ``self_attention`` naive softmax）。``KNGraph.self_attention`` 改用 stable TB 路径；value-verify builder + general ML parity。验证：**120 passed**。
- **Loop R10（YiRage main，scaled self_attention，PR 待合并）**：闸门：图级/正确性（标准 ``1/sqrt(d)``）。``head_dim`` / ``scale`` + ``self_attention_scaled`` builder。验证：**121 passed**。
- **Loop R11（YiRage main，multi-head self_attention 3D，PR #9）**：闸门：图级/正确性（``[H,S,D]`` 多头契约）。``self_attention_multi_head``（chunk → per-head stable softmax → concat）；``_kn_customized_tb_softmax_last_dim`` 支持 ``[1,S,S]`` leading batch；``matmul_chain_shapes_from_cygraph`` 非 2D 安全回退。验证：**122 passed**。
- **Loop R12（YiRage main，batched self_attention 3D，PR #10）**：闸门：图级/正确性（``[B,S,D]`` 批量契约）。``self_attention_batched``（batch chunk → per-item stable softmax → concat）；``rms_matmul_shapes_from_cygraph`` 非 2D 安全回退。``[B,H,S,D]`` 多头批量留 R14+（需 reshape / 4D TB softmax）。验证：**123 passed**。
- **Loop R13（YiRage main，conv2d_bias 复合，PR #11）**：闸门：图级/正确性（CV bias 融合）。``conv2d_bias``（``kn_conv2d_op`` + broadcast ``kn_add_op``，bias ``[1,O,1,1]``）；builder + general ML parity。验证：**124 passed**。
- **Loop R14（YiRage main，kn_conv2d groups，PR #12）**：闸门：执行层（depthwise/grouped CV）。``KN_CONV2D_OP`` JSON + API 增 ``groups``；CPU 解释器 ``F.conv2d(groups=)``；``conv2d_groups`` builder + parity。验证：**125 passed**。
- **Loop R15（YiRage main，conv2d bias+groups 复合，PR #13）**：闸门：图级/正确性（闭合 R13/R14 CV 栈）。``conv2d_depthwise_bias``；``conv2d_bias_groups`` / ``conv2d_depthwise_bias`` builders + parity。验证：**127 passed**。
- **Loop R16（YiRage main，separable conv2d 复合，PR #14）**：闸门：图级/正确性（MobileNet depthwise+pointwise）。``conv2d_separable`` / ``conv2d_separable_bias``；builders + parity。验证：**129 passed**。
- **Loop R17（YiRage main，online rescale self_attention，PR #15）**：闸门：图级/正确性（FlashAttention-style TB forloop）。``self_attention_online``（``reduction_max`` + ``forloop_accum_rescale`` 按 key tile 流式）；builder + parity vs ``self_attention_scaled``。验证：**130 passed**。
- **Loop R18（YiRage main，conv2d_bias_relu 复合，PR #16）**：闸门：图级/正确性（CV 激活融合）。``conv2d_bias_relu``（``conv2d_bias`` + ``kn_relu_op``）；builder + parity vs ``F.relu(F.conv2d(...))``。3D online attention 扩展因 TB forloop 在 ``[1,S,D]`` chunk 切片上输出 shape 未闭合，留 R19+。验证：**131 passed**。
- **Loop R19（YiRage main，conv2d_bias_gelu 复合，PR #17）**：闸门：图级/正确性（CV GELU 融合）。``conv2d_bias_gelu``（``conv2d_bias`` + ``kn_gelu_op``）；builder + parity vs ``F.gelu(F.conv2d(...))``。验证：**132 passed**。
- **Loop R20（YiRage main，depthwise/separable ReLU 复合，PR #18）**：闸门：图级/正确性（MobileNet 激活融合）。``conv2d_depthwise_bias_relu``；``conv2d_separable_bias_relu``；builders + parity。验证：**134 passed**。
- **Loop R21（YiRage main，depthwise/separable GELU 复合，PR #19）**：闸门：图级/正确性（对称 R19/R20）。``conv2d_depthwise_bias_gelu``；``conv2d_separable_bias_gelu``；builders + parity。验证：**136 passed**。
- **Loop R22（YiRage main，gemm_gelu LLM 融合，PR #20）**：闸门：图级/正确性（FFN up-projection）。``gemm_gelu``（``matmul`` + ``kn_gelu_op``）；builder + parity vs ``F.gelu(A @ B)``。验证：**137 passed**。
- **Loop R23（YiRage main，gemm_silu gated MLP 融合，PR #21）**：闸门：图级/正确性（对称 R22）。``gemm_silu``（``matmul`` + ``kn_silu_op``）；builder + parity vs ``F.silu(A @ B)``。验证：**138 passed**。
- **Loop R24（YiRage main，CV SiLU 激活融合，PR #22）**：闸门：图级/正确性（闭合 ReLU/GELU/SiLU 矩阵）。``conv2d_bias_silu``；``conv2d_depthwise_bias_silu``；``conv2d_separable_bias_silu``；builders + parity。验证：**141 passed**。
- **Loop R25（YiRage main，gemm_relu + gated_mlp，PR #23）**：闸门：图级/正确性（GEMM 激活族 + LLM FFN）。``gemm_relu``；``gated_mlp`` builder + parity（``silu(gate)*up @ down``）。验证：**143 passed**。
- **Loop R26（YiRage main，rms_norm_linear LLM 融合，PR #24）**：闸门：图级/正确性（QKV 投影模式）。``rms_norm_linear`` builder + parity；``rms_norm_linear`` 改用 ``self.rms_norm``。验证：**144 passed**。
- **Loop R27（YiRage main，gated_mlp GELU 变体，PR #25）**：闸门：图级/正确性（LLM FFN GELU gate）。``gated_mlp_gelu`` builder + parity（``activation=\"gelu\"``）。3D ``[B,S,D]`` gated_mlp 留 R28+（KN matmul 仅 2D）。验证：**145 passed**。
- **Loop R28（YiRage main，gemm_bias 线性层融合，PR #26）**：闸门：图级/正确性（``matmul + broadcast bias``）。``gemm_bias``；builder + parity。3D gated_mlp 仍受 KN matmul batch 规则阻塞（``[B,S,D]@W`` 需 3D@3D 且 batch 对齐）。验证：**146 passed**。
- **Loop R29（YiRage main，gemm_bias 激活融合，PR #27）**：闸门：图级/正确性（对称 CV bias+act）。``gemm_bias_relu``；``gemm_bias_gelu``；builders + parity。验证：**148 passed**。
- **Loop R30（YiRage main，gemm_bias_silu 线性融合，PR #28）**：闸门：图级/正确性（闭合 GEMM+bias 激活矩阵）。``gemm_bias_silu``；builder + parity vs ``F.silu(A @ B + bias)``。验证：**149 passed**。
- **Loop R31（YiRage main，rms_norm_linear_gelu LLM 融合，PR #29）**：闸门：图级/正确性（QKV/FFN GELU epilogue）。``rms_norm_linear_gelu``；builder + parity vs ``F.gelu(RMSNorm(X) @ W)``。验证：**150 passed**。
- **Loop R32（YiRage main，rms_norm_linear ReLU/SiLU 融合，PR #30）**：闸门：图级/正确性（闭合 rms_norm_linear 激活矩阵）。``rms_norm_linear_relu``；``rms_norm_linear_silu``；builders + parity。验证：**152 passed**。
- **Loop R33（YiRage main，gemm_layernorm 激活融合，PR #31）**：闸门：图级/正确性（闭合 gemm_layernorm 激活矩阵）。``gemm_layernorm_gelu``；``gemm_layernorm_relu``；``gemm_layernorm_silu``；builders + parity。验证：**155 passed**。
- **Loop R34（YiRage main，gated_mlp_batched 3D FFN + cpu_call 3D 修复，PR #32）**：闸门：图级/正确性 + 工具层（``concat_matmul_shapes`` 对 4×3D 输入不再 abort ``cpu_call``）。``gated_mlp_batched``；``gated_mlp_batched_gelu``（``[B,S,D]`` chunk/concat + ``[1,D,D_ff]`` 权重）；builders + parity。验证：**157 passed**。
- **Loop R35（YiRage main，gemm_softmax_scaled 注意力分数融合，PR #33）**：闸门：图级/正确性（``softmax(scale * Q @ K)`` 分数段）。``gemm_softmax_scaled``；builder + parity vs ``softmax(matmul / sqrt(d))``。验证：**158 passed**。
- **Loop R36（YiRage main，gemm_softmax_scaled_batched 3D 分数，PR #34）**：闸门：图级/正确性（``[B,S,D]`` 批量注意力分数）。``gemm_softmax_scaled_batched``；builder + parity（``num_leading_dims=1`` stable softmax）。验证：**159 passed**。
- **Loop R37（YiRage main，KN 3D×2D matmul broadcast + gated_mlp_3d，PR #35）**：闸门：原语/P1 层（``[B,S,D] @ [D,F]`` 共享 2D 权重）。``create_matmul_op`` + ``_validate_kernel_matmul`` 支持 rank+1 broadcast；``kn_matmul_3d_2d_op``；``gated_mlp_3d`` builder + parity。验证：**161 passed**。
- **Loop R38（YiRage main，3D LLM FFN/QKV 契约扩展，PR #36）**：闸门：图级/正确性（R37 broadcast 落地）。``gated_mlp_3d_gelu``；``rms_norm_linear_3d``（``[B,S,D]`` + 2D 权重）；builders + parity。验证：**163 passed**。
- **Loop R39（YiRage main，rms_norm_linear 3D 激活融合，PR #37）**：闸门：图级/正确性（闭合 3D rms_norm_linear 激活矩阵）。``rms_norm_linear_3d_gelu``；``rms_norm_linear_3d_relu``；``rms_norm_linear_3d_silu``；builders + parity。验证：**166 passed**。
- **Loop R40（YiRage main，gemm_bias 3D 线性+bias 融合，PR #38）**：闸门：图级/正确性（R37 3D×2D broadcast 落地 linear+bias）。``gemm_bias_3d``；``gemm_bias_3d_relu``；``gemm_bias_3d_gelu``；``gemm_bias_3d_silu``（``[B,S,K] @ [K,N] + [1,1,N]``，复用现有 ``gemm_bias*``）；builders + parity。验证：**170 passed**。
- **Loop R41（YiRage main，gemm_layernorm 3D 激活融合，PR #39）**：闸门：图级/正确性（对称 R33，3D×2D GEMM + TB LayerNorm）。扩展 ``gemm_layernorm`` 支持 3D 输出（``num_leading_dims=1`` TB + batch chunk/concat）；``gemm_layernorm_3d``；``gemm_layernorm_3d_gelu/relu/silu``；builders + parity。验证：**174 passed**。
- **Loop R42（YiRage main，3D softmax/layernorm + gemm_gelu 3D，PR #40）**：闸门：图级/正确性（R37 3D×2D 注意力/归一化契约）。``_kn_layer_norm_batched_last_dim`` / ``_kn_softmax_batched_last_dim``；扩展 ``layer_norm``、``gemm_softmax``、``gemm_softmax_scaled``；``kn_layer_norm_3d``；``gemm_softmax_3d``；``gemm_softmax_scaled_3d``（``[B,S,D] @ [D,S]``）；``gemm_gelu_3d``（builder-only）。验证：**178 passed**。
- **Loop R43（YiRage main，3D GEMM 激活闭合 + self_attention_3d，PR #41）**：闸门：图级/正确性（共享 2D K/V 的 batched 注意力）。``self_attention_3d``（``[B,S,D]`` + ``[D,S]`` + ``[S,D]``）；``gemm_relu_3d``；``gemm_silu_3d``；``self_attention_scaled_3d``；builders + parity。验证：**182 passed**。
- **Loop R44（YiRage main，3D standalone softmax/RMSNorm，PR #42）**：闸门：图级/正确性（闭合 3D 归一化原语）。扩展 ``softmax`` 3D；``kn_softmax_3d``；``kn_softmax_3d_batch1``；``kn_rms_norm_3d``；``kn_rms_norm_3d_batch1``。验证：**186 passed**。
- **Loop R45（YiRage main，3D batch=1 快路径 builders，PR #43）**：闸门：图级/正确性（对称 R44 batch1 模式）。``kn_layer_norm_3d_batch1``；``gemm_layernorm_3d_batch1``；``gemm_softmax_3d_batch1``；``gemm_bias_3d_batch1``。验证：**190 passed**。
- **Loop R46（YiRage main，3D GEMM 激活 batch=1 闭合，PR #44）**：闸门：图级/正确性（闭合 3D GEMM+激活/bias 矩阵 batch1 快路径）。``gemm_gelu_3d_batch1``；``gemm_relu_3d_batch1``；``gemm_silu_3d_batch1``；``gemm_softmax_scaled_3d_batch1``；builders + parity。验证：**194 passed**。
- **Loop R47（YiRage main，3D LayerNorm/bias 激活 batch=1，PR #45）**：闸门：图级/正确性（闭合 3D layernorm/bias 激活 batch1 矩阵）。``gemm_layernorm_3d_gelu_batch1``；``gemm_layernorm_3d_relu_batch1``；``gemm_layernorm_3d_silu_batch1``；``gemm_bias_3d_relu_batch1``；builders + parity。验证：**198 passed**。
- **Loop R48（YiRage main，3D bias/RMSNorm/attention batch=1，PR #46）**：闸门：图级/正确性（闭合 3D bias 激活 + rms_norm_linear + scaled attention batch1 矩阵）。``gemm_bias_3d_gelu_batch1``；``gemm_bias_3d_silu_batch1``；``rms_norm_linear_3d_batch1``；``self_attention_scaled_3d_batch1``；builders + parity。验证：**202 passed**。
- **Loop R49（YiRage main，3D rms_norm_linear 激活 + attention batch=1，PR #47）**：闸门：图级/正确性（闭合 3D rms_norm_linear 激活 batch1 矩阵 + 非 scaled attention batch1）。``rms_norm_linear_3d_gelu_batch1``；``rms_norm_linear_3d_relu_batch1``；``rms_norm_linear_3d_silu_batch1``；``self_attention_3d_batch1``；builders + parity。验证：**206 passed**。
- **Loop R50（YiRage main，3D gated MLP + batched attention batch=1，PR #48）**：闸门：图级/正确性（3D FFN batch1 快路径 + per-batch K/V attention batch1）。``gated_mlp_3d_batch1``；``gated_mlp_3d_gelu_batch1``；``gated_mlp_batched_batch1``；``self_attention_batched_batch1``；builders + parity。验证：**210 passed**。
- **Loop R51（YiRage main，scaled batched scores + multi-head batch=1，PR #49）**：闸门：图级/正确性（闭合 batched 注意力分数/FFN GELU batch1 + KN 3D×2D matmul batch1）。``gemm_softmax_scaled_batched_batch1``；``gated_mlp_batched_gelu_batch1``；``self_attention_multi_head_batch1``；``kn_matmul_3d_2d_batch1_op``；builders + parity。验证：**214 passed**。
- **Loop R52（YiRage main，2D attention/FFN batch=1 推理契约，PR #50）**：闸门：图级/正确性（2D 单序列 batch=1 命名闭合）。``self_attention_scaled_batch1``；``self_attention_online_batch1``；``gated_mlp_batch1``；``gated_mlp_gelu_batch1``；builders + parity。验证：**218 passed**。
- **Loop R53（YiRage main，2D GEMM/RMSNorm/attention batch=1，PR #51）**：闸门：图级/正确性（2D LLM 线性/注意力 batch=1 命名闭合）。``self_attention_batch1``；``rms_norm_linear_batch1``；``gemm_gelu_batch1``；``gemm_bias_batch1``；builders + parity。验证：**222 passed**。
- **Loop R54（YiRage main，2D softmax/bias/rms GELU batch=1，PR #52）**：闸门：图级/正确性（2D 注意力分数 + bias 激活 + rms GELU batch=1 命名闭合）。``gemm_softmax_batch1``；``gemm_softmax_scaled_batch1``；``rms_norm_linear_gelu_batch1``；``gemm_bias_relu_batch1``；builders + parity。验证：**226 passed**。
- **Loop R55（YiRage main，2D GEMM 激活 + bias/rms ReLU batch=1，PR #54）**：闸门：图级/正确性（2D SiLU/ReLU/GELU + rms ReLU batch=1 命名闭合）。``gemm_silu_batch1``；``gemm_relu_batch1``；``gemm_bias_gelu_batch1``；``rms_norm_linear_relu_batch1``；builders + parity。验证：**230 passed**。
- **Loop R56（YiRage main，2D bias/rms SiLU + CV batch=2，PR #55）**：闸门：图级/正确性（闭合 2D bias/rms SiLU batch1 + CV N=2 推理契约）。``gemm_bias_silu_batch1``；``rms_norm_linear_silu_batch1``；``conv2d_bias_batch2``；``kn_conv2d_batch2_op``；builders + parity。验证：**234 passed**。
- **Loop R57（YiRage main，2D layernorm batch1 + CV act batch=2，PR #56）**：闸门：图级/正确性（闭合 2D GEMM+LayerNorm batch1 + CV N=2 激活推理契约）。``gemm_layernorm_batch1``；``gemm_layernorm_gelu_batch1``；``conv2d_bias_relu_batch2``；``conv2d_bias_gelu_batch2``；builders + parity。验证：**238 passed**。
- **Loop R58（YiRage main，2D layernorm act batch1 + CV SiLU/groups batch=2，PR #57）**：闸门：图级/正确性（闭合 2D LayerNorm ReLU/SiLU batch1 + CV N=2 SiLU/grouped 推理契约）。``gemm_layernorm_relu_batch1``；``gemm_layernorm_silu_batch1``；``conv2d_bias_silu_batch2``；``conv2d_bias_groups_batch2``；builders + parity。验证：**242 passed**。
- **Loop R59（YiRage main，CV groups/depthwise batch=2，PR #58）**：闸门：图级/正确性（闭合 grouped conv + depthwise CV N=2 推理契约）。``conv2d_groups_batch2``；``kn_conv2d_groups_batch2_op``；``conv2d_depthwise_bias_batch2``；``conv2d_depthwise_bias_relu_batch2``；builders + parity。验证：**246 passed**。
- **Loop R60（YiRage main，CV depthwise act + separable batch=2，PR #59）**：闸门：图级/正确性（闭合 depthwise GELU/SiLU + separable CV N=2 推理契约）。``conv2d_depthwise_bias_gelu_batch2``；``conv2d_depthwise_bias_silu_batch2``；``conv2d_separable_batch2``；``conv2d_separable_bias_batch2``；builders + parity。验证：**250 passed**。
- **Loop R61（YiRage main，separable act batch=2 + KN matmul batch=2，PR 待合并）**：闸门：图级/正确性（闭合 separable 激活 CV N=2 + KN 3D×2D matmul batch=2）。``conv2d_separable_bias_relu_batch2``；``conv2d_separable_bias_gelu_batch2``；``conv2d_separable_bias_silu_batch2``；``kn_matmul_3d_2d_batch2_op``；builders + parity。验证：**254 passed**。
- **Loop R62（YiRage main，3D GEMM/attention batch=2 命名闭合，PR #60）**：闸门：图级/正确性（闭合 3D softmax/scaled + rms GELU + self_attention batch=2 契约）。``gemm_softmax_3d_batch2``；``gemm_softmax_scaled_3d_batch2``；``rms_norm_linear_3d_gelu_batch2``；``self_attention_3d_batch2``；builders + parity。验证：**258 passed**。
- **Loop R63（YiRage main，LLM batch=2 命名闭合，PR #61）**：闸门：图级/正确性（闭合 3D GELU/gated MLP/rms ReLU + scaled attention batch=2）。``gemm_gelu_3d_batch2``；``gated_mlp_batched_batch2``；``rms_norm_linear_3d_relu_batch2``；``self_attention_scaled_3d_batch2``；builders + parity。验证：**262 passed**。
- **Loop R64（YiRage main，3D GEMM 激活 + rms/gated batch=2，PR 待合并）**：闸门：图级/正确性（闭合 3D ReLU/SiLU + rms SiLU + gated GELU batch=2）。``gemm_relu_3d_batch2``；``gemm_silu_3d_batch2``；``rms_norm_linear_3d_silu_batch2``；``gated_mlp_batched_gelu_batch2``；builders + parity。验证：**266 passed**。
- **Loop R65（YiRage main，KN/3D layernorm batch=2 命名闭合，PR 待合并）**：闸门：图级/正确性（闭合 KN softmax/rms/layer_norm + gemm_layernorm 3D batch=2）。``kn_softmax_3d_batch2``；``kn_rms_norm_3d_batch2``；``kn_layer_norm_3d_batch2``；``gemm_layernorm_3d_batch2``；builders + parity。验证：**270 passed**。
- **Loop R66（YiRage main，3D layernorm 激活 + bias batch=2，PR 待合并）**：闸门：图级/正确性（闭合 3D layernorm GELU/ReLU/SiLU + gemm_bias batch=2）。``gemm_layernorm_3d_gelu_batch2``；``gemm_layernorm_3d_relu_batch2``；``gemm_layernorm_3d_silu_batch2``；``gemm_bias_3d_batch2``；builders + parity。验证：**274 passed**。
- **Loop R67（YiRage main，3D gemm_bias 激活 + rms batch=2，PR #63）**：闸门：图级/正确性（闭合 3D gemm_bias ReLU/GELU/SiLU + rms_norm_linear batch=2）。``gemm_bias_3d_relu_batch2``；``gemm_bias_3d_gelu_batch2``；``gemm_bias_3d_silu_batch2``；``rms_norm_linear_3d_batch2``；builders + parity。验证：**278 passed**。
- **Loop R68（YiRage main，attention + gated MLP batch=2，PR #64）**：闸门：图级/正确性（闭合 batched/multi-head attention + gated MLP GELU + scaled batched softmax batch=2）。``self_attention_batched_batch2``；``self_attention_multi_head_batch2``；``gated_mlp_3d_gelu_batch2``；``gemm_softmax_scaled_batched_batch2``；builders + parity。验证：**282 passed**。
- **Loop R69（YiRage main，2D softmax + attention batch=2 命名，PR #65）**：闸门：图级/正确性（闭合 2D gemm_softmax/scaled + self_attention batch=2 推理契约）。``gemm_softmax_batch2``；``gemm_softmax_scaled_batch2``；``self_attention_batch2``；``self_attention_scaled_batch2``；builders + parity。验证：**286 passed**。
- **Loop R70（YiRage main，online + 3D MLP + 2D layernorm batch=2，PR #66）**：闸门：图级/正确性（闭合 online attention + gated MLP SiLU 3D + 2D layernorm/GELU batch=2）。``self_attention_online_batch2``；``gated_mlp_3d_batch2``；``gemm_layernorm_batch2``；``gemm_layernorm_gelu_batch2``；builders + parity。验证：**290 passed**。
- **Loop R71（YiRage main，2D layernorm act + gated MLP batch=2，PR #67）**：闸门：图级/正确性（闭合 2D layernorm ReLU/SiLU + 2D gated MLP SiLU/GELU batch=2）。``gemm_layernorm_relu_batch2``；``gemm_layernorm_silu_batch2``；``gated_mlp_batch2``；``gated_mlp_gelu_batch2``；builders + parity。验证：**294 passed**。
- **Loop R72（YiRage main，2D gemm_bias batch=2 命名闭合，PR #68）**：闸门：图级/正确性（闭合 2D GEMM+bias 及 ReLU/GELU/SiLU 激活 batch=2 推理契约）。``gemm_bias_batch2``；``gemm_bias_relu_batch2``；``gemm_bias_gelu_batch2``；``gemm_bias_silu_batch2``；builders + parity。验证：**298 passed**。
- **Loop R73（YiRage main，2D GEMM 激活 + rms batch=2，PR #69）**：闸门：图级/正确性（闭合 2D gemm GELU/ReLU/SiLU + rms_norm_linear batch=2 推理契约）。``gemm_gelu_batch2``；``gemm_relu_batch2``；``gemm_silu_batch2``；``rms_norm_linear_batch2``；builders + parity。验证：**302 passed**。
- **Loop R74（YiRage main，2D rms 激活 + KN softmax batch=2，PR #70）**：闸门：图级/正确性（闭合 2D rms_norm_linear GELU/ReLU/SiLU + kn_softmax batch=2 推理契约）。``rms_norm_linear_gelu_batch2``；``rms_norm_linear_relu_batch2``；``rms_norm_linear_silu_batch2``；``kn_softmax_batch2``；builders + parity。验证：**306 passed**。
- **Loop R75（YiRage main，2D KN softmax/layer_norm/rms batch 命名，PR #71）**：闸门：图级/正确性（闭合 2D KN softmax batch=1 + layer_norm batch=1/2 + rms_norm batch=1）。``kn_softmax_batch1``；``kn_layer_norm_batch1``；``kn_layer_norm_batch2``；``kn_rms_norm_batch1``；builders + parity。验证：**310 passed**。
- **Loop R76（YiRage main，KN matmul/conv/rms batch 命名，PR #72）**：闸门：图级/正确性（闭合 2D KN rms_norm batch=2 + matmul/conv2d/groups batch=1 推理契约）。``kn_rms_norm_batch2``；``kn_matmul_batch1``；``kn_conv2d_batch1``；``kn_conv2d_groups_batch1``；builders + parity。验证：**314 passed**。
- **Loop R77（YiRage main，CV batch=1 + KN matmul batch=2，PR #73）**：闸门：图级/正确性（闭合 CV conv bias/groups batch=1 命名 + 2D KN matmul batch=2）。``conv2d_bias_batch1``；``conv2d_groups_batch1``；``conv2d_bias_relu_batch1``；``kn_matmul_batch2``；builders + parity。验证：**318 passed**。
- **Loop R78（YiRage main，CV act/groups/depthwise batch=1，PR #74）**：闸门：图级/正确性（闭合 CV GELU/SiLU/groups/depthwise batch=1 推理契约）。``conv2d_bias_gelu_batch1``；``conv2d_bias_silu_batch1``；``conv2d_bias_groups_batch1``；``conv2d_depthwise_bias_batch1``；builders + parity。验证：**322 passed**。
- **Loop R79（YiRage main，CV depthwise act + separable batch=1，PR #75）**：闸门：图级/正确性（闭合 depthwise ReLU/GELU/SiLU + separable batch=1 推理契约）。``conv2d_depthwise_bias_relu_batch1``；``conv2d_depthwise_bias_gelu_batch1``；``conv2d_depthwise_bias_silu_batch1``；``conv2d_separable_batch1``；builders + parity。验证：**326 passed**。
- **Loop R80（YiRage main，CV separable bias act batch=1 命名，PR #76）**：闸门：图级/正确性（闭合 separable+bias 及 ReLU/GELU/SiLU batch=1 推理契约）。``conv2d_separable_bias_batch1``；``conv2d_separable_bias_relu_batch1``；``conv2d_separable_bias_gelu_batch1``；``conv2d_separable_bias_silu_batch1``；builders + parity。验证：**330 passed**。
- **Loop R81（YiRage main，KN conv2d batch 命名闭合，PR #77）**：闸门：图级/正确性（闭合 KN conv2d/groups batch=1/2 在 KN_OP 与 CUSTOMIZED 双 registry 命名）。``kn_conv2d_batch1_op``；``kn_conv2d_groups_batch1_op``；``kn_conv2d_batch2``；``kn_conv2d_groups_batch2``；registry + parity。验证：**334 passed**。
- **Loop R82（YiRage main，KN CUSTOMIZED base 命名闭合，PR #78）**：闸门：图级/正确性（补齐 CUSTOMIZED 中 matmul/rms_norm/conv2d/groups 无后缀 base 键，与 batch1/batch2 三档对齐）。``kn_matmul``；``kn_rms_norm``；``kn_conv2d``；``kn_conv2d_groups``；registry + parity。验证：**338 passed**。
- **Loop R83（YiRage main，KN matmul 3d×2d + transpose CUSTOMIZED，PR #79）**：闸门：图级/正确性（``[B,M,K]@[K,N]`` broadcast matmul 三档 + ``kn_transpose_01`` 迁入 CUSTOMIZED）。``kn_matmul_3d_2d``；``kn_matmul_3d_2d_batch1``；``kn_matmul_3d_2d_batch2``；``kn_transpose_01``；registry + parity。验证：**342 passed**。
- **Loop R84（YiRage main，rms_matmul + plain conv2d CV 命名，PR #80）**：闸门：图级/正确性（``kn_unfused_rms_matmul`` 迁入 CUSTOMIZED + 无 bias ``conv2d`` batch1/batch2 推理契约）。``kn_unfused_rms_matmul``；``conv2d``；``conv2d_batch1``；``conv2d_batch2``；registry + parity。验证：**346 passed**。
- **Loop R85（YiRage main，plain conv2d 激活 + relu batch1，PR #81）**：闸门：图级/正确性（无 bias ``conv2d`` + ReLU/GELU/SiLU 及 ``conv2d_relu_batch1`` 推理契约）。``conv2d_relu``；``conv2d_gelu``；``conv2d_silu``；``conv2d_relu_batch1``；builders + parity。验证：**350 passed**。
- **Loop R86（YiRage main，plain conv2d 激活 batch 命名，PR #82）**：闸门：图级/正确性（闭合无 bias conv2d 激活 batch1/batch2 推理契约）。``conv2d_relu_batch2``；``conv2d_gelu_batch1``；``conv2d_gelu_batch2``；``conv2d_silu_batch1``；builders + parity。验证：**354 passed**。
- **Loop R87（YiRage main，silu batch2 + rms_matmul batch 命名，PR #83）**：闸门：图级/正确性（``conv2d_silu_batch2`` + unfused rms_matmul batch1/batch2 及 ``[1,M,K]@[K,N]`` batched 契约）。``conv2d_silu_batch2``；``kn_unfused_rms_matmul_batch1``；``kn_unfused_rms_matmul_batch2``；``kn_unfused_rms_matmul_batched_batch1``；builders + parity。验证：**358 passed**。
- **Loop R88（YiRage main，grouped conv2d 无 bias 激活，PR #84）**：闸门：图级/正确性（groups=2 无 bias conv2d + ReLU/GELU/SiLU 及 ``conv2d_groups_relu_batch1``）。``conv2d_groups_relu``；``conv2d_groups_gelu``；``conv2d_groups_silu``；``conv2d_groups_relu_batch1``；builders + parity。验证：**362 passed**。
- **Loop R89（YiRage main，grouped conv2d 激活 batch 命名，PR #85）**：闸门：图级/正确性（闭合 groups=2 无 bias 激活 batch1/batch2 推理契约）。``conv2d_groups_relu_batch2``；``conv2d_groups_gelu_batch1``；``conv2d_groups_gelu_batch2``；``conv2d_groups_silu_batch1``；builders + parity。验证：**366 passed**。
- **Loop R90（YiRage main，groups silu batch2 + rms_matmul batched + depthwise，PR #86）**：闸门：图级/正确性（``conv2d_groups_silu_batch2``、plain ``conv2d_depthwise``、unfused rms_matmul batched 三档 ``[B,M,K]@[K,N]``）。``conv2d_groups_silu_batch2``；``conv2d_depthwise``；``kn_unfused_rms_matmul_batched``；``kn_unfused_rms_matmul_batched_batch2``；builders + parity。验证：**370 passed**。
- **Loop R91（YiRage main，plain depthwise batch + 激活，PR #87）**：闸门：图级/正确性（无 bias depthwise batch1/batch2 + ReLU/GELU 推理契约）。``conv2d_depthwise_batch1``；``conv2d_depthwise_batch2``；``conv2d_depthwise_relu``；``conv2d_depthwise_gelu``；builders + parity。验证：**374 passed**。
- **Loop R92（YiRage main，plain depthwise SiLU + 激活 batch1，PR #88）**：闸门：图级/正确性（闭合无 bias depthwise SiLU 及 ReLU/GELU/SiLU batch1 推理契约）。``conv2d_depthwise_silu``；``conv2d_depthwise_relu_batch1``；``conv2d_depthwise_gelu_batch1``；``conv2d_depthwise_silu_batch1``；builders + parity。验证：**378 passed**。
- **Loop R93（YiRage main，plain depthwise 激活 batch2 + separable relu，PR #89）**：闸门：图级/正确性（闭合 depthwise ReLU/GELU/SiLU batch2 + 无 bias separable ReLU 推理契约）。``conv2d_depthwise_relu_batch2``；``conv2d_depthwise_gelu_batch2``；``conv2d_depthwise_silu_batch2``；``conv2d_separable_relu``；builders + parity。验证：**382 passed**。
- **Loop R94（YiRage main，无 bias fused conv2d Graph API，PR #90）**：闸门：图级/正确性（对称 ``conv2d_bias_*``，新增 ``conv2d_relu/gelu/silu``、``conv2d_depthwise*``、``conv2d_separable_*`` fused API；builders 迁移 + separable gelu/silu batch1）。``conv2d_separable_gelu``；``conv2d_separable_silu``；``conv2d_separable_relu_batch1``；``conv2d_separable_gelu_batch1``；``graph.py`` + unit smoke。验证：**386 passed**。
- **Loop R95（YiRage main，separable 激活 batch 命名闭合，PR #91）**：闸门：图级/正确性（闭合无 bias separable SiLU batch1 + ReLU/GELU/SiLU batch2 推理契约）。``conv2d_separable_silu_batch1``；``conv2d_separable_relu_batch2``；``conv2d_separable_gelu_batch2``；``conv2d_separable_silu_batch2``；builders + parity。验证：**390 passed**。
- **Loop R96（YiRage main，grouped conv2d bias 激活，PR #92）**：闸门：图级/正确性（闭合 groups=2 bias+ReLU/GELU/SiLU 及 relu batch1 推理契约）。``conv2d_bias_groups_relu``；``conv2d_bias_groups_gelu``；``conv2d_bias_groups_silu``；``conv2d_bias_groups_relu_batch1``；builders + parity。验证：**394 passed**。
- **Loop R97（YiRage main，batched unfused rms_matmul 快路径，PR #93）**：闸门：原语/P0 层（``[B,M,K]@[K,N]`` unfused rms+matmul 识别 + ``cpu_rms_matmul`` 3D + FAST_PATH builders）。``kn_unfused_rms_matmul_batch1_fast``；``kn_unfused_rms_matmul_batched_fast``；``kn_unfused_rms_matmul_batched_batch1_fast``；``kn_unfused_rms_matmul_batched_batch2_fast``；``cpu_mlir_jit`` + ``cpu_native``。验证：**398 passed**。
- **Loop R98（YiRage main，grouped bias 激活 batch 命名，PR #94）**：闸门：图级/正确性（闭合 groups=2 bias 激活 batch1/batch2 余量）。``conv2d_bias_groups_gelu_batch1``；``conv2d_bias_groups_silu_batch1``；``conv2d_bias_groups_relu_batch2``；``conv2d_bias_groups_gelu_batch2``；builders + parity。验证：**402 passed**。
- **Loop R99（YiRage main，bias silu batch2 + KN/registry 闭合，PR #95）**：闸门：图级/正确性（``conv2d_bias_groups_silu_batch2``、``kn_conv2d_groups_op``、``kn_unfused_rms_matmul_batch2_fast``、``kn_matmul_batched_3d_2d``）。builders + parity。验证：**406 passed**。
- **Loop R100（YiRage main，grouped bias fused Graph API，PR #96）**：闸门：图级/正确性（``conv2d_bias_groups*`` fused API + FAST_PATH 四档 + builders 迁移）。``conv2d_bias_groups_relu_fast``；``conv2d_bias_groups_gelu_fast``；``conv2d_bias_groups_silu_fast``；``conv2d_bias_groups_relu_batch1_fast``；``graph.py`` + unit smoke。验证：**410 passed**。
- **Loop R101（YiRage main，grouped bias FAST_PATH batch 闭合，PR #97）**：闸门：图级/正确性（闭合 groups=2 bias 激活 FAST_PATH batch1/batch2 余量）。``conv2d_bias_groups_gelu_batch1_fast``；``conv2d_bias_groups_silu_batch1_fast``；``conv2d_bias_groups_relu_batch2_fast``；``conv2d_bias_groups_gelu_batch2_fast``；FAST_PATH + parity。验证：**414 passed**。
- **Loop R102（YiRage main，bias silu batch2 + depthwise FAST_PATH，PR #98）**：闸门：图级/正确性（``conv2d_bias_groups_silu_batch2_fast`` + plain depthwise ReLU/GELU/SiLU FAST_PATH）。builders + parity。验证：**418 passed**。
- **Loop R103（YiRage main，plain depthwise 迁移 + batch FAST_PATH，PR #99）**：闸门：图级/正确性（``build_conv2d_depthwise*`` 迁移 ``g.conv2d_depthwise()`` + FAST_PATH 四档 + unit smoke）。``conv2d_depthwise_fast``；``conv2d_depthwise_batch1_fast``；``conv2d_depthwise_batch2_fast``；``conv2d_depthwise_relu_batch1_fast``；builders + ``test_graph_conv2d_fused_api``。验证：**422 passed**。
- **Loop R104（YiRage main，depthwise 激活 FAST_PATH batch 闭合，PR #100）**：闸门：图级/正确性（plain depthwise ReLU/GELU/SiLU batch1/batch2 FAST_PATH 余量）。``conv2d_depthwise_gelu_batch1_fast``；``conv2d_depthwise_silu_batch1_fast``；``conv2d_depthwise_relu_batch2_fast``；``conv2d_depthwise_gelu_batch2_fast``；FAST_PATH + parity。验证：**426 passed**。
- **Loop R105（YiRage main，depthwise silu batch2 + separable FAST_PATH，PR #101）**：闸门：图级/正确性（``conv2d_depthwise_silu_batch2_fast`` + plain separable ReLU/GELU/SiLU FAST_PATH）。builders + parity。验证：**430 passed**。
- **Loop R106（YiRage main，separable depthwise 迁移 + batch FAST_PATH，PR #102）**：闸门：图级/正确性（``conv2d_separable`` 内层 ``conv2d_depthwise`` + FAST_PATH 四档 + unit smoke）。``conv2d_separable_fast``；``conv2d_separable_batch1_fast``；``conv2d_separable_batch2_fast``；``conv2d_separable_relu_batch1_fast``；``graph.py`` + ``test_graph_conv2d_fused_api``。验证：**434 passed**。
- **Loop R107（YiRage main，separable 激活 FAST_PATH batch 闭合，PR #103）**：闸门：图级/正确性（plain separable ReLU/GELU/SiLU batch1/batch2 FAST_PATH 余量）。``conv2d_separable_gelu_batch1_fast``；``conv2d_separable_silu_batch1_fast``；``conv2d_separable_relu_batch2_fast``；``conv2d_separable_gelu_batch2_fast``；FAST_PATH + parity。验证：**438 passed**。
- **Loop R108（YiRage main，separable silu batch2 + plain conv2d FAST_PATH，PR #104）**：闸门：图级/正确性（``conv2d_separable_silu_batch2_fast`` + 无 bias ``conv2d_relu/gelu/silu_fast``）。builders + parity。验证：**442 passed**。
- **Loop R109（YiRage main，conv2d_bias FAST_PATH + unit smoke，PR #105）**：闸门：图级/正确性（``conv2d_bias`` FAST_PATH 四档 + fused API smoke）。``conv2d_bias_fast``；``conv2d_bias_batch1_fast``；``conv2d_bias_batch2_fast``；``conv2d_bias_relu_fast``；``graph.py`` doc + ``test_graph_conv2d_fused_api``。验证：**446 passed**。
- **Loop R110（YiRage main，conv2d_bias 激活 FAST_PATH 余量，PR #106）**：闸门：图级/正确性（``conv2d_bias_gelu/silu_fast`` + ``conv2d_bias_relu_batch1/batch2_fast``）。FAST_PATH + parity。验证：**450 passed**。
- **Loop R111（YiRage main，plain conv2d batch FAST_PATH，PR #107）**：闸门：图级/正确性（无 bias ``conv2d_relu/gelu/silu_batch1_fast`` + ``conv2d_relu_batch2_fast``）。FAST_PATH + parity。验证：**454 passed**。
- **Loop R112（YiRage main，conv2d_bias_groups base FAST_PATH + unit smoke，PR #108）**：闸门：图级/正确性（``conv2d_bias_groups`` base/batch FAST_PATH + ``conv2d_bias_gelu_batch1_fast`` + fused API smoke）。``conv2d_bias_groups_fast``；``conv2d_bias_groups_batch1_fast``；``conv2d_bias_groups_batch2_fast``；``graph.py`` doc + ``test_graph_conv2d_fused_api``。验证：**458 passed**。
- **Loop R113（YiRage main，conv2d_bias 激活 batch FAST_PATH 余量，PR #109）**：闸门：图级/正确性（``conv2d_bias_gelu/silu_batch1/batch2_fast`` + ``conv2d_gelu_batch2_fast``）。FAST_PATH + parity。验证：**462 passed**。
- **Loop R114（YiRage main，plain silu batch2 + separable_bias FAST_PATH，PR #110）**：闸门：图级/正确性（``conv2d_silu_batch2_fast`` + separable_bias base ReLU/GELU FAST_PATH）。builders + parity。验证：**466 passed**。
- **Loop R115（YiRage main，separable_bias batch/silu FAST_PATH + unit smoke，PR #111）**：闸门：图级/正确性（separable_bias silu/batch/relu_batch1 FAST_PATH + fused API smoke）。``conv2d_separable_bias_silu_fast``；``conv2d_separable_bias_batch1_fast``；``conv2d_separable_bias_batch2_fast``；``conv2d_separable_bias_relu_batch1_fast``；``graph.py`` doc + ``test_graph_conv2d_fused_api``。验证：**470 passed**。
- **Loop R116（YiRage main，separable_bias 激活 batch FAST_PATH 余量，PR #112）**：闸门：图级/正确性（separable_bias ReLU/GELU/SiLU batch1/batch2 FAST_PATH 余量）。``conv2d_separable_bias_relu_batch2_fast``；``conv2d_separable_bias_gelu_batch1/batch2_fast``；``conv2d_separable_bias_silu_batch1_fast``；FAST_PATH + parity。验证：**474 passed**。
- **Loop R117（YiRage main，separable silu batch2 + depthwise_bias FAST_PATH，PR #113）**：闸门：图级/正确性（``conv2d_separable_bias_silu_batch2_fast`` + depthwise_bias base ReLU/GELU FAST_PATH）。builders + parity。验证：**478 passed**。
- **Loop R118（YiRage main，depthwise_bias batch/silu FAST_PATH + unit smoke，PR #114）**：闸门：图级/正确性（depthwise_bias batch/silu/relu_batch1 FAST_PATH + fused API smoke）。``conv2d_depthwise_bias_batch1_fast``；``conv2d_depthwise_bias_batch2_fast``；``conv2d_depthwise_bias_silu_fast``；``conv2d_depthwise_bias_relu_batch1_fast``；``graph.py`` doc + ``test_graph_conv2d_fused_api``。验证：**482 passed**。
- **Loop R119（YiRage main，depthwise_bias 激活 batch FAST_PATH 余量，PR #115）**：闸门：图级/正确性（depthwise_bias ReLU/GELU/SiLU batch1/batch2 FAST_PATH 余量）。``conv2d_depthwise_bias_relu_batch2_fast``；``conv2d_depthwise_bias_gelu_batch1/batch2_fast``；``conv2d_depthwise_bias_silu_batch1_fast``；FAST_PATH + parity。验证：**486 passed**。
- **Loop R120（YiRage main，depthwise silu batch2 + groups 激活 FAST_PATH，PR #117）**：闸门：图级/正确性（``conv2d_depthwise_bias_silu_batch2_fast`` + 无 bias groups=2 ReLU/GELU/SiLU FAST_PATH）。builders + parity。验证：**490 passed**。
- **Loop R121（YiRage main，conv2d_groups fused Graph API + FAST_PATH，PR #116）**：闸门：图级/正确性（``conv2d_groups*`` fused API + base/batch FAST_PATH + unit smoke）。``conv2d_groups_fast``；``conv2d_groups_batch1_fast``；``conv2d_groups_batch2_fast``；``conv2d_groups_relu_batch1_fast``；``graph.py`` + ``test_graph_conv2d_fused_api``。验证：**494 passed**。
- **Loop R122（YiRage main，groups 激活 batch FAST_PATH 余量，PR #118）**：闸门：图级/正确性（groups=2 ReLU/GELU/SiLU batch1/batch2 FAST_PATH 余量）。``conv2d_groups_relu_batch2_fast``；``conv2d_groups_gelu_batch1/batch2_fast``；``conv2d_groups_silu_batch1_fast``；FAST_PATH + parity。验证：**498 passed**。
- **Loop R123（YiRage main，groups silu batch2 + plain conv2d batch FAST_PATH，PR #119）**：闸门：图级/正确性（``conv2d_groups_silu_batch2_fast`` + plain ``conv2d/batch1/batch2_fast``）。builders + parity。验证：**502 passed**。
- **Loop R124（YiRage main，groups 激活 builders 迁移 + KN FAST_PATH，PR #120）**：闸门：图级/正确性（``build_conv2d_groups_{relu,gelu,silu}*`` 迁移 ``g.conv2d_groups_*()`` + KN FAST_PATH 四档 + unit smoke）。``kn_conv2d_fast``；``kn_conv2d_batch1_fast``；``kn_conv2d_batch2_fast``；``kn_conv2d_groups_batch2_fast``；``test_graph_conv2d_fused_api``。验证：**506 passed**。
- **Loop R125（YiRage main，KN groups FAST_PATH 闭合，PR #121）**：闸门：图级/正确性（``kn_conv2d_groups_{fast,batch1}_fast`` + ``kn_conv2d_groups_{relu,gelu}_fast``）。FAST_PATH + parity。验证：**510 passed**。
- **Loop R126（YiRage main，KN groups 激活 batch1 FAST_PATH，PR #122）**：闸门：图级/正确性（``kn_conv2d_groups_{silu,relu_batch1,gelu_batch1,silu_batch1}_fast``）。FAST_PATH + parity。验证：**514 passed**。
- **Loop R127（YiRage main，groups batch builder 对齐 + KN batch2 FAST_PATH，PR #123）**：闸门：图级/正确性（``build_conv2d_groups_batch*`` 委托 ``build_kn_conv2d_groups_batch*`` + KN batch2 激活 FAST_PATH + unit smoke）。``kn_conv2d_groups_{relu,gelu,silu}_batch2_fast``；``kn_conv2d_op_fast``；``graph.py`` doc + ``test_graph_conv2d_fused_api``。验证：**518 passed**。
- **Loop R128（YiRage main，plain conv2d KN 激活 + groups op FAST_PATH，PR #124）**：闸门：图级/正确性（``kn_conv2d_{relu,gelu,silu}_fast`` + ``kn_conv2d_groups_op_fast``）。FAST_PATH + parity。验证：**522 passed**。
- **Loop R129（YiRage main，KN conv2d batch1 激活/op FAST_PATH，PR #125）**：闸门：图级/正确性（``kn_conv2d_{relu,gelu,silu}_batch1_fast`` + ``kn_conv2d_batch1_op_fast``）。FAST_PATH + parity。验证：**526 passed**。
- **Loop R130（YiRage main，KN conv2d batch2 激活/op + graph doc，PR #126）**：闸门：图级/正确性（``kn_conv2d_{relu,gelu,silu}_batch2_fast`` + ``kn_conv2d_batch2_op_fast`` + ``graph.py`` See Also + unit smoke）。FAST_PATH + parity。验证：**530 passed**。
- **Loop R131（YiRage main，groups batch op + conv2d op FAST_PATH，PR #127）**：闸门：图级/正确性（``kn_conv2d_groups_batch{1,2}_op_fast`` + ``conv2d_{,groups_}op_fast``）。FAST_PATH + parity。验证：**534 passed**。
- **Loop R132（YiRage main，conv2d batch/groups op 别名 FAST_PATH，PR #128）**：闸门：图级/正确性（``conv2d_batch{1,2}_op_fast`` + ``conv2d_groups_batch{1,2}_op_fast``）。FAST_PATH + parity。验证：**538 passed**。
- **Loop R133（YiRage main，conv2d bias op FAST_PATH + graph doc，PR #129）**：闸门：图级/正确性（``conv2d_bias_{,batch1,batch2,groups_}op_fast`` + ``graph.py`` conv2d/bias See Also + unit smoke）。FAST_PATH + parity。验证：**542 passed**。
- **Loop R134（YiRage main，KN conv2d bias 激活 FAST_PATH，PR #131）**：闸门：图级/正确性（``kn_conv2d_bias_{,relu,gelu,silu}_fast``）。FAST_PATH + parity。验证：**546 passed**。
- **Loop R135（YiRage main，KN bias batch1 激活/op FAST_PATH，PR #130）**：闸门：图级/正确性（``kn_conv2d_bias_{relu,gelu,silu}_batch1_fast`` + ``kn_conv2d_bias_batch1_op_fast``）。FAST_PATH + parity。验证：**550 passed**。
- **Loop R136（YiRage main，KN bias batch2 激活/op + graph doc，PR #132）**：闸门：图级/正确性（``kn_conv2d_bias_{relu,gelu,silu}_batch2_fast`` + ``kn_conv2d_bias_batch2_op_fast`` + ``graph.py`` bias 激活 See Also + unit smoke）。FAST_PATH + parity。验证：**554 passed**。
- **Loop R137（YiRage main，KN bias groups 激活 FAST_PATH，PR #134）**：闸门：图级/正确性（``kn_conv2d_bias_groups_{,relu,gelu,silu}_fast``）。FAST_PATH + parity。验证：**558 passed**。
- **Loop R138（YiRage main，KN bias groups batch1 激活/op FAST_PATH，PR #133）**：闸门：图级/正确性（``kn_conv2d_bias_groups_{relu,gelu,silu}_batch1_fast`` + ``kn_conv2d_bias_groups_batch1_op_fast``）。FAST_PATH + parity。验证：**562 passed**。
- **Loop R139（YiRage main，KN bias groups batch2 激活/op + graph doc，PR #135）**：闸门：图级/正确性（``kn_conv2d_bias_groups_{relu,gelu,silu}_batch2_fast`` + ``kn_conv2d_bias_groups_batch2_op_fast`` + ``graph.py`` bias groups 激活 See Also + unit smoke）。FAST_PATH + parity。验证：**566 passed**。
- **Loop R140（YiRage main，bias groups op 别名 + KN depthwise base FAST_PATH，PR #136）**：闸门：图级/正确性（``conv2d_bias_groups_batch{1,2}_op_fast`` + ``kn_conv2d_depthwise_{,relu}_fast``）。FAST_PATH + parity。验证：**570 passed**。
- **Loop R141（YiRage main，KN depthwise 激活/batch FAST_PATH，PR #138）**：闸门：图级/正确性（``kn_conv2d_depthwise_{gelu,silu,batch1,batch2}_fast``）。FAST_PATH + parity。验证：**574 passed**。
- **Loop R142（YiRage main，depthwise batch1 KN 激活/op + graph doc，PR #137）**：闸门：图级/正确性（``kn_conv2d_depthwise_{relu,gelu,silu}_batch1_fast`` + ``conv2d_depthwise_op_fast`` + ``graph.py`` depthwise See Also + unit smoke）。FAST_PATH + parity。验证：**578 passed**。
- **Loop R143（YiRage main，depthwise batch2 KN + batch1 op FAST_PATH，PR #139）**：闸门：图级/正确性（``kn_conv2d_depthwise_{relu,gelu,silu}_batch2_fast`` + ``conv2d_depthwise_batch1_op_fast``）。FAST_PATH + parity。验证：**582 passed**。
- **Loop R144（YiRage main，KN depthwise_bias 激活 FAST_PATH，PR #140）**：闸门：图级/正确性（``kn_conv2d_depthwise_bias_{,relu,gelu,silu}_fast``）。FAST_PATH + parity。验证：**586 passed**。
- **Loop R145（YiRage main，depthwise_bias batch1 KN 激活/op + graph doc，PR #141）**：闸门：图级/正确性（``kn_conv2d_depthwise_bias_{relu,gelu,silu}_batch1_fast`` + ``conv2d_depthwise_bias_op_fast`` + ``graph.py`` depthwise_bias 激活 See Also + unit smoke）。FAST_PATH + parity。验证：**590 passed**。
- **Loop R146（YiRage main，KN depthwise_bias batch2 激活 FAST_PATH，PR #142）**：闸门：图级/正确性（``kn_conv2d_depthwise_bias_{,relu,gelu,silu}_batch2_fast``）。FAST_PATH + parity。验证：**594 passed**。
- **Loop R147（YiRage main，depthwise batch2 op + KN separable base FAST_PATH，PR #143）**：闸门：图级/正确性（``conv2d_depthwise_batch2_op_fast`` + ``kn_conv2d_separable_{,relu,gelu}_fast``）。FAST_PATH + parity。验证：**598 passed**。
- **Loop R148（YiRage main，KN separable silu/batch1 激活/op + graph doc，PR #144）**：闸门：图级/正确性（``kn_conv2d_separable_silu_fast`` + ``kn_conv2d_separable_{relu,gelu}_batch1_fast`` + ``conv2d_separable_op_fast`` + ``graph.py`` separable 激活 See Also + unit smoke）。FAST_PATH + parity。验证：**602 passed**。
- **Loop R149（YiRage main，KN separable silu batch1 + batch1 op + batch2 base FAST_PATH，PR #145）**：闸门：图级/正确性（``kn_conv2d_separable_silu_batch1_fast`` + ``conv2d_separable_batch1_op_fast`` + ``kn_conv2d_separable_{,relu}_batch2_fast``）。FAST_PATH + parity。验证：**606 passed**。
- **Loop R150（YiRage main，KN separable batch2 激活/op + bias base FAST_PATH，PR #146）**：闸门：图级/正确性（``kn_conv2d_separable_{gelu,silu}_batch2_fast`` + ``conv2d_separable_batch2_op_fast`` + ``kn_conv2d_separable_bias_fast``）。FAST_PATH + parity。验证：**610 passed**。
- **Loop R151（YiRage main，KN separable_bias 激活/op + graph doc，PR #147）**：闸门：图级/正确性（``kn_conv2d_separable_bias_{relu,gelu,silu}_fast`` + ``conv2d_separable_bias_op_fast`` + ``graph.py`` separable_bias 激活 See Also + unit smoke）。FAST_PATH + parity。验证：**614 passed**。
- **Loop R152（YiRage main，KN separable_bias batch1 激活/op FAST_PATH，PR #148）**：闸门：图级/正确性（``kn_conv2d_separable_bias_{relu,gelu,silu}_batch1_fast`` + ``conv2d_separable_bias_batch1_op_fast``）。FAST_PATH + parity。验证：**618 passed**。
- **Loop R153（YiRage main，KN separable_bias batch2 base/激活/op FAST_PATH，PR #149）**：闸门：图级/正确性（``kn_conv2d_separable_bias_{,relu,gelu}_batch2_fast`` + ``conv2d_separable_bias_batch2_op_fast``）。FAST_PATH + parity。验证：**622 passed**。
- **Loop R154（YiRage main，KN separable_bias silu batch2 + batch op/doc，PR #151）**：闸门：图级/正确性（``kn_conv2d_separable_bias_silu_batch2_fast`` + ``kn_conv2d_separable_bias_batch{1,2}_{,op}_fast`` + ``graph.py`` separable_bias batch See Also + unit smoke）。FAST_PATH + parity。验证：**626 passed**。
- **Loop 节奏（2026-06，用户确认：混合 C）**：**2 轮验证 + 1 轮实现** 交替，避免纯 registry 命名闭合凑 passed。
  - **验证轮**：registry + parity + inventory 闸门；闸门类型「图级/正确性 / 命名闭合」；**不改** `graph.py`/C++/search，除非发现静默错误。
  - **实现轮**：须动生产栈（`graph.py`、Cython/C++、matrix tier、search explore、fast path）；PR 描述必填「四轮自问」+ bench/cert 证据。
  - **下一批**：**R155–R156 验证**候选（CV/separable 命名余量闭合或转向 **gated_mlp f16 稳定性** / attention 图级 backlog）。
- **Loop R16（2026-06-09，CPU concat_matmul 搜索变换，PR #69 → `7484e59`）**：闸门：图级融合（LoRA 类 concat+matmul 模式）。`get_cpu_search_config()` 调用 `enable_concat_matmul_transformation()`；matrix 注明 `tb_concat_then_matmul_op` 为搜索宏。验证：value verify **63 passed**；cert **29 passed**（~927s）。
- **Loop R17（2026-06-09，KN chunk CPU，PR #70 → `07075eb`）**：闸门：layout 层（`torch.chunk`，对齐 TB concat）。Cython `chunk()` + `graph.py` 解释器；matrix 三档 `supported` + builders。验证：value verify **66 passed**；cert **32 passed**（~927s）。
- **Loop R18（2026-06-09，KN concat CPU，PR #71 → `66cbba9`）**：闸门：layout 层（`torch.cat`）。新增 `KNConcatOp` + Cython/解释器（含 `kn_concat_first_op_id` 别名）；matrix 三档 `supported`。验证：value verify **69 passed**；cert **35 passed**（~927s）。
- **Loop R19（2026-06-09，KN split CPU，PR #72 → `5fd7530`）**：闸门：layout 层（binary `torch.split`，concat 逆）。`KNSplitOp` + Cython/解释器（含 `kn_split_first_op_id`）；matrix 三档 `supported`。验证：value verify **72 passed**；cert **38 passed**（~912s）。
- **Loop R20（2026-06-09，TB split CPU，PR #73 → `c12cb54`）**：闸门：TB layout（binary split，concat 逆）。`TBSplitOp` + smem/customized 路径 + Cython/解释器；matrix 三档 `supported`。验证：value verify **75 passed**；cert **38 passed**（~919s）。
- **Loop R21（2026-06-09，TB split search explore，PR #74 → `5df8f27`）**：闸门：图级/搜索层（对称 R15 concat explore）。`abstract_expr` 新增 `Split`（egg `(split input part dim size)`）；symbolic_graph / irange / formal_verifier / `op_utils` 补齐三档 `TB_SPLIT_*`；`tbop_to_explore` + matrix + sync test。验证：value verify **75 passed**；cert **38 passed**（~933s）。
- **Loop R22（2026-06-09，KN layout search explore，PR #75）**：闸门：图级/搜索层（对称 R21 TB split）。`KN_CONCAT/SPLIT/CHUNK` 纳入 `knop_to_explore`；复用 `Concat`/`Split` abstract expr；`irange`/`symbolic_graph`/`formal_verifier`/`op_utils` 补齐；chunk explore 固定 `num_chunks=2`。验证：value verify **75 passed**；cert **38 passed**（~944s）。
- **Loop R23（2026-06-09，KN concat_matmul 搜索宏，PR #76）**：闸门：图级融合（对称 R16 TB concat_matmul）。新增 `KN_CONCAT_THEN_MATMUL_OP`（`concat+concat+matmul` 展开）；`get_knop_cand()` 在 `enable_concat_matmul_transformation()` 下注入；matrix 注明 `kn_concat_then_matmul_op` 为搜索宏。验证：value verify **75 passed**；cert **38 passed**（~1445s，valid mugraphs **139→143**）。`bench --full` 已启动（含 matmul_chain，搜索较慢）。
- **Loop R24（2026-06-09，bench 搜索可完成性，PR #77 → `996bf52`）**：闸门：感知/工具层（R22/R23 layout explore 后 bench 无法在合理时间完成）。`bench_fused_vs_mkl_baseline.py` 新增 `_apply_bench_search_tractability()`（`YIRAGE_CPU_MAX_*` + `YIRAGE_CPU_BENCH_MINIMAL_EXPLORE=1`）；`search_c.cc` bench 模式仅探索 GEMM/rms_matmul 融合 op。验证：`test_fused_vs_mkl_baseline` **~40s**；`bench --quick --json` plain_matmul **1.09×**、rms_norm_matmul **0.97×**（search **14.5s/21.1s**）。`--full` matmul_chain 仍较慢但可完成（较 R23 前 15M+ states 僵死显著改善）。
- **Loop R25（2026-06-09，LoRA concat_matmul bench + KN layout from_json，PR #78 → `af55020`）**：闸门：感知/契约（闭合 concat 图 JSON 缺口 + LoRA bench）。`graph.cc` 补齐 `KN_CONCAT/SPLIT/CHUNK` from_json；`apply_cpu_search_env` 不覆盖已设 bench env；bench 新增 `--workloads concat_matmul`（unfused interpreter vs MKL **0.18×**，fusion search 待 R26）。验证：`test_fused_vs_mkl_baseline` **2 passed**。
- **Loop R26（2026-06-09，TB concat_matmul abstract_expr，PR #79 → `eb3b1d9`）**：闸门：图级/搜索层（闭合 R25 assert 崩溃）。`abstract_expr_eval` 泛化 concat+matmul 三元组匹配；`YIRAGE_CPU_MAX_TB_GRAPH_INPUTS`；bench `concat_matmul` 重试 `config=lora` superoptimize（无 valid µGraph 则 interpreter 回退）。验证：`test_fused_vs_mkl_baseline` **2 passed**；superoptimize **无 abort**。
- **Loop R27（2026-06-11，KN concat_matmul abstract_expr 规范化，PR #80）**：闸门：图级/搜索层（闭合 R26「0 valid µGraph」）。根因：预处理目标 expr 为 `matmul(concat,concat)` 逐 op 组合，与搜索宏 `KN_CONCAT_THEN_MATMUL_OP` 的 `add(red(mul(...)))` 形式不等价。`abstract_expr_eval` 在 `KN_MATMUL_OP` 上检测双 KN concat（hi/lo dim）并规范化为 fused expr。验证：`bench --quick --workloads concat_matmul` **4 valid µGraphs**，`mugraph_source=fused_search`；`test_fused_vs_mkl_baseline` **2 passed**。
- **Loop R28（2026-06-11，concat_matmul cpu_call BLAS 快路径，PR #81）**：闸门：原语/P0 层（闭合 R27「搜索成功但执行仍慢」）。根因：`cpu_call` 无 concat_matmul 快路径，融合/非融合 µGraph 均走 Python TB 解释器。新增 `cpu_concat_matmul` + `is_production_concat_matmul_mugraph`（unfused 双 concat+matmul 与 fused `kn_customized`）；`YIRAGE_CPU_CONCAT_MATMUL_FAST`（默认开）。验证：`bench --quick --workloads concat_matmul` speedup **1.00×**；`test_cpu_native_gemm` + `test_fused_vs_mkl_baseline` 全绿。
- **Loop R29（2026-06-11，matmul_chain bench + BLAS 快路径，PR #82）**：闸门：感知/原语层（`USE_MLIR=0` 下优先 matmul_chain）。根因：bench `MAX_KN=2` 无法容纳 6-op 种子图；`cpu_call` 无链式 GEMM 快路径。修复：tractability `KN=8/TB=4`、搜索失败 interpreter 回退、`cpu_matmul_chain` + 混合 `kn_matmul+kn_customized` 检测；`test_mlir_jit_availability_smoke`。验证：unit/smoke；`bench --workloads matmul_chain` speedup **~1.04×**。
- **Loop R30（2026-06-11，matmul_chain quick bench 跳过搜索，PR #83）**：闸门：感知/工具层（P0 快路径已 ~1× MKL，90s 搜索无额外价值）。`--quick` 下 `matmul_chain` 若 `is_production_matmul_chain_mugraph` 则跳过 superoptimize（`YIRAGE_CPU_BENCH_SKIP_FUSION_SEARCH=1` 默认）；JSON 增 `fusion_search_skipped`。验证：bench **~4s**（原 ~90s）；`test_bench_matmul_chain_quick_skips_search`。**注**：`USE_MLIR=1` 构建在本 VM 触发 `llvm::DisableABIBreakingChecks` 链接错误，已回退 `USE_MLIR=0`。
- **Loop R31（2026-06-11，concat_matmul quick bench 跳过搜索，PR #84）**：闸门：感知/工具层（对称 R30）。`--quick` 下 `concat_matmul` 若 `is_production_concat_matmul_mugraph` 则跳过 superoptimize。验证：bench **~3s**（原 ~14s）；`test_fused_vs_mkl_baseline` **4 passed**。
- **Loop R32（2026-06-12，`USE_MLIR=1` yirage.core LLVM 链接，PR #84）**：闸门：实验/工具层（闭合 R30 MLIR 构建 import 失败）。根因：`setup.py` 以 `--whole-archive` 链 `libyirage_runtime.a` 时未传递 CMake PUBLIC 的 `libLLVM`/`libMLIR` dylib，加载 `yirage.core` 报 `llvm::DisableABIBreakingChecks` 未定义。修复：`tools/setup_backend_config.cython_mlir_extra_link_args()` 在 `USE_MLIR=ON` 时为 Cython 扩展追加 `-lLLVM-17 -lMLIR` + rpath；`setup.py` 接入；单测 `test_cython_mlir_link_args_*`。验证：`ldd yirage.core` 解析 `libLLVM-17.so`/`libMLIR.so`；`test_cpu_mlir_jit` **18 passed**；`test_fused_vs_mkl_baseline` **4 passed**。
- **Loop R33（2026-06-12，MLIR e2e tractability + bench JIT smoke，PR #85）**：闸门：感知/工具层（闭合 R32 慢 e2e）。根因：`test_mlir_jit_fused_customized_op_correctness` 未设 bench 级搜索上限，superoptimize 探索 15M+ states；bench `--mlir-jit` JSON 误读 `speedup_interp_over_jit` 键。修复：e2e 加 `_apply_rms_matmul_search_tractability()`、清 `_JIT_CACHE`、减 profile iters（**~13s**）；`bench_fused_vs_mkl_baseline.py` 修正 speedup 键；新增 `test_bench_rms_matmul_mlir_jit_quick_smoke`。验证：MLIR e2e + unit **28 passed**；`test_fused_vs_mkl_baseline` **5 passed**（~51s）。
- **Loop R34（2026-06-12，rms_norm_matmul quick bench 跳过搜索，PR #86）**：闸门：感知/工具层（对称 R30/R31）。`--quick` 下 `rms_norm_matmul` 若 `is_production_rms_matmul_mugraph` 则跳过 superoptimize；`test_bench_rms_matmul_quick_skips_search`。验证：bench **~3s**（原 ~12s）；`test_fused_vs_mkl_baseline` **6 passed**。
- **Loop R35（2026-06-12，plain_matmul quick bench 跳过搜索，PR #87）**：闸门：感知/工具层（闭合 R34 最后一项 quick workload）。`--quick` 下 `plain_matmul` 若 `_is_plain_matmul_mugraph` 则跳过 superoptimize；`test_bench_plain_matmul_quick_skips_search`；全量 quick smoke 兼容 `fusion_search_skipped`。验证：bench **~2.5s**（原 ~9s）；`test_fused_vs_mkl_baseline` **7 passed**。
- **Loop R36（2026-06-12，MLIR JIT 多级 emit 回退，PR #88）**：闸门：实验/原语层（闭合 R33 `YIRAGE_CPU_MLIR_JIT_BLAS=0` fused JIT 失败）。根因：fused bgrid tiled hand MLIR 无法被 `CPUJITKernel` 解析，且无 flat 回退。修复：`_rms_matmul_mlir_compile_candidates()` 依次尝试 tiled hand → dialect → lowered → tiled flat → **flat 无 tiling**；dialect 不再被 tiled hand 完全屏蔽。验证：fused e2e `YIRAGE_CPU_MLIR_JIT_BLAS=0` 通过；`test_rms_matmul_compile_candidates_end_with_flat_fallback`。
- **Loop R37（2026-06-12，quick bench P0 跳过搜索文档化，PR #89）**：闸门：感知/工具层（闭合 R30–R35 分散实现）。在 `bench_fused_vs_mkl_baseline.py` 模块 doc、`_bench_skip_fusion_search`、报告输出与 `docs/HARDWARE_OPTIMIZATION.md` 汇总四 workload 跳过谓词与 `YIRAGE_CPU_BENCH_SKIP_FUSION_SEARCH`；新增 `test_bench_fusion_search_skip.py`；quick smoke 断言默认 workload 均 `fusion_search_skipped`。验证：**8 passed**（`test_fused_vs_mkl_baseline` + unit）。
- **Loop R38（2026-06-12，bgrid tiled hand MLIR 解析 + JIT，PR #90）**：闸门：实验/原语层（闭合 R36 flat 回退根因）。根因：tiled emit 将 `yirage.*` 挂在 `%out` memref 参数上（parser `expected ')'`），且 M-tile 循环重复定义 `%c0`/`%cM`；动态 `memref.subview` 在 JIT 管线触发 `unrealized_conversion_cast`。修复：`_yirage_tiling_func_attrs()` 将元数据置于 `func.func` 级；M-grid matmul 用 `%m0 + %mi` 行偏移索引替代 subview；新增 `test_tiled_hand_mlir_compiles_in_jit_kernel`。验证：MLIR unit + e2e **25 passed**；`grid_m=2` hand emit **compile + invoke** 成功。
- **Loop R39（2026-06-12，`--full` bench P0 跳过搜索，PR #91）**：闸门：感知/工具层（闭合 R37「仅 quick 跳过」缺口）。`--full` 仍放大 shape，但 P0 种子图在 `YIRAGE_CPU_BENCH_SKIP_FUSION_SEARCH=1`（默认）下同样跳过 `superoptimize`；`_bench_skip_fusion_search` 不再依赖 `quick` 参数。验证：`test_bench_fusion_search_skip` + `test_bench_plain_matmul_full_skips_search`；`test_fused_vs_mkl_baseline` 全绿。
- **Loop R40（2026-06-12，hand tiled vs dialect 数值对齐，PR #92）**：闸门：实验/验证层（闭合 R38 tiled JIT 与 dialect 路径一致性缺口）。新增 `invoke_rms_matmul_mlir_text`、`compare_hand_tiled_vs_dialect_lowered_jit`（hand emit vs `yirage-cpu-jit-pipeline` lowered）；单测 M-grid synthetic tiling + fused e2e superoptimize 对齐断言。验证：MLIR unit + e2e **26 passed**。
- **Loop R41（2026-06-12，`--full` 强制搜索 tractability，PR #93）**：闸门：感知/工具层（闭合 R39「full 强制搜索 >5min」）。当 `fusion_search_skipped=False` 时，`--quick`/`--full` 均用 `_cap_bench_search_explore()` 单点 grid；`_apply_bench_search_tractability` 对 full 同步 TB cap。验证：`test_bench_rms_matmul_forced_search_tractable`（`--quick`+`SKIP=0`，search **<90s**）；`test_cap_bench_search_explore_single_grid_point`；`test_fused_vs_mkl_baseline` 全绿。注：`--full`+`SKIP=0` 因更大 shape 仍较慢，默认 `SKIP=1` 跳过搜索。
- **Loop R42（2026-06-12，bench `--mlir-jit` hand/dialect 速度 JSON，PR #94）**：闸门：实验/感知层（闭合 R40 数值对齐的速度侧）。新增 `bench_hand_vs_dialect_lowered_jit`、`_hand_and_dialect_lowered_mlir_texts`；bench JSON 增 `hand_mlir_jit_ms`、`dialect_lowered_jit_ms`、`speedup_hand_over_dialect_lowered`、`mlir_hand_dialect_aligned`。验证：MLIR unit + `test_bench_rms_matmul_mlir_jit_quick_smoke`。
- **Loop R43（2026-06-12，layout explore value verify，PR #95）**：闸门：正确性/契约层（闭合 R22 KN/TB layout search explore 后仅单 op 覆盖缺口）。新增 `LAYOUT_EXPLORE_BUILDERS`（KN split↔concat、chunk↔concat、concat→split、dual concat+matmul）与 `TB_LAYOUT_EXPLORE_BUILDERS`（TB split↔concat、concat→split）；`test_cpu_full_value_verify` 增 16 项。验证：value verify **91 passed**（planned **92**）；`cpu_verify_all_functions` **PASS**。
- **Loop R44（2026-06-12，MLIR dialect 候选链优先 lowered，PR #96）**：闸门：实验/原语层（闭合 R42/R40 dialect 路径与 hand tiled 竞争顺序）。`YIRAGE_CPU_MLIR_JIT_DIALECT=1` 时 `_rms_matmul_mlir_compile_candidates()` 先尝试 `yirage-cpu-jit-pipeline` lowered，再 raw dialect，最后 hand tiled/flat 回退。验证：MLIR unit + e2e **28 passed**；`test_dialect_enabled_prioritizes_lowered_before_hand_tiled`。
- **Loop R45（2026-06-12，bench `--mlir-jit` emit 路径 JSON，PR #97）**：闸门：感知/工具层（闭合 R44 dialect 优先后「实际命中哪条 emit」不可见）。`_rms_matmul_mlir_compile_candidates()` 返回带标签的 `(path, mlir)`；`rms_matmul_mlir_emit_path()` + `bench_jit_vs_interpreter` 增 `mlir_jit_emit_path`；bench JSON/报告输出该字段。验证：MLIR unit + `test_bench_rms_matmul_mlir_jit_quick_smoke`。
- **Loop R46（2026-06-12，dialect emit path bench smoke，PR #98）**：闸门：实验/验证层（闭合 R44/R45 dialect 优先链端到端契约）。新增 `test_bench_rms_matmul_mlir_jit_dialect_emit_path_smoke`（`YIRAGE_CPU_MLIR_JIT_DIALECT=1` 断言 `mlir_jit_emit_path=dialect_lowered`）与 `test_rms_matmul_emit_path_is_dialect_lowered_when_dialect_enabled`；抽取 `_run_rms_matmul_mlir_bench_json`。验证：MLIR unit + bench smoke **27 passed**。
- **Loop R47（2026-06-12，KN chunk↔split layout value verify，PR #99）**：闸门：正确性/契约层（闭合 R43 后 chunk 与 split 未组合验证）。新增 `kn_layout_chunk_split_first`（chunk→concat→split）与 `kn_layout_split_chunk_first`（split→chunk）三档 dim；TB 无 chunk op 仍跳过。验证：value verify **97 passed**（planned **98**）。
- **Loop R48（2026-06-12，fused hand_bgrid_tiled emit smoke，PR #100）**：闸门：实验/验证层（对称 R46 dialect_lowered）。抽取 `_superoptimize_rms_matmul_fused`；新增 `test_superoptimized_rms_matmul_emit_path_is_hand_bgrid_tiled`（无 `YIRAGE_CPU_MLIR_JIT_DIALECT` 时 superoptimize 融合图命中 `hand_bgrid_tiled`）。验证：MLIR e2e + unit。
- **Loop R49（2026-06-12，bench `--mlir-jit-fused` emit 路径，PR #101）**：闸门：感知/工具层（闭合 R48 e2e 与 bench 脱节）。`bench_fused_vs_mkl_baseline.py` 新增 `--mlir-jit-fused`（固定 `grid_m=2`/`forloop_k=2` superoptimize）；JSON 增 `mlir_jit_fused_seed`；`test_bench_rms_matmul_mlir_jit_fused_emit_path_smoke` 断言 `hand_bgrid_tiled`。
- **Loop R50（2026-06-12，fused dialect_lowered bench smoke，PR #102）**：闸门：实验/验证层（闭合 R49 hand_bgrid_tiled 与 R46 dialect 在融合种子上的组合）。新增 `test_bench_rms_matmul_mlir_jit_fused_dialect_emit_path_smoke`（`DIALECT=1` + `--mlir-jit-fused`）与 `test_superoptimized_rms_matmul_emit_path_is_dialect_lowered_when_dialect_enabled`。
- **Loop R51（2026-06-12，MLIR JIT bench JSON 联合文档，PR #103）**：闸门：感知/文档层（闭合 R46–R50 emit 契约分散）。`mlir_jit_bench_json_field_guide()` + `MUGRAPH_SOURCE_VALUES`/`MLIR_JIT_EMIT_PATH_VALUES`；bench 模块 doc、`HARDWARE_OPTIMIZATION.md` 表；`test_mlir_jit_bench_json_field_guide_contract`。
- **Loop R52（2026-06-12，TB chunk 契约占位，PR #104）**：闸门：契约/文档层（闭合 KN chunk layout 后 TB 对称缺口）。matrix 增 `tb_chunk_*` tier `unsupported`；`TB_LAYOUT_CHUNK_DEFERRED_PATTERNS`（9 项未来 value-verify 名）；`test_cpu_tb_chunk_deferred` 断言未纳入 search explore。
- **Loop R53（2026-06-12，layout explore chunk gap 表，PR #105）**：闸门：契约/文档层（闭合 R52 占位与 KN explore/value 覆盖的可审计缺口）。`layout_explore_gaps`（yaml）+ `cpu_layout_explore_gap_table()` / `cpu_layout_explore_gap_meta()`；`test_cpu_search_explore_sync` 增 KN chunk explore vs TB deferred 对齐断言；`cpu_verify_all_functions` / `cpu_certification` inventory 增 gap 字段；`HARDWARE_OPTIMIZATION.md` 文档化。验证：`test_cpu_search_explore_sync` + `test_cpu_tb_chunk_deferred`。
- **Loop R54（2026-06-12，TB chunk CPU + search explore，PR #106）**：闸门：layout 层（兑现 R52/R53 占位）。`TB_CHUNK_*` enum + `TBChunkOp` + Cython/解释器；search 栈（`op_utils`、`abstract_expr_*`、`symbolic_graph`、`irange`、`formal_verifier`、`config.cc`）；matrix/search explore 纳入 `tb_chunk_*`；`TB_LAYOUT_EXPLORE_BUILDERS` 增 9 项 chunk 组合；value verify **planned +12**（3 单 op + 9 layout）。验证：`test_cpu_search_explore_sync` + `test_cpu_tb_chunk_deferred` + `make test-cpu-value-verify`。
- **Loop R55（2026-06-13，TB layout concat_matmul value verify，PR #107）**：闸门：正确性/契约层（闭合 R54 后 KN 有 `kn_layout_concat_matmul`、TB layout explore 缺对称项）。新增 `tb_layout_concat_matmul` builder；`test_cpu_search_explore_sync` 增 KN/TB 对称断言。验证：value verify **planned +1**（**110** total planned）。
- **Loop R56（2026-06-13，formal_verifier 修复 + cert 快速路径 + inventory 110，PR #108）**：闸门：工具/契约层（闭合 VM 无法 `pip install -e` 与 R54/R55 实数未对齐）。修复 `formal_verifier` `concat_split` egg rewrite（`=>` → `<=>`）；修复 `cpu_layout_explore_gap_table` `NameError`；`cpu_inventory.planned_value_verify_count()` + `test_cpu_inventory_planned`；`cpu_certification.py --quick`；`make test-cpu-cert-quick`。验证：`test_cpu_inventory_planned` + `make test-cpu-cert-quick`；`make test-cpu-value-verify` **110 passed**（111 collected，1 skipped）。
- **Loop R57（2026-06-13，cert profile 归档 + superoptimize smoke 可完成性，PR #109）**：闸门：感知/工具层（闭合 R56「无 profile 归档」与 cert superoptimize smoke VM 超时）。新增 `cpu_cert_utils.py`（`parse_pytest_summary`、`cert_inventory_summary`）；`cpu_certification.py` 增 `mode`/`inventory`/`profile`（`value_verify_aligned`）；`--quick` 跳过 walkthrough、superoptimize smoke、native_gemm superoptimize；`cpu_verify_all_functions.py` 对齐 profile 字段；`test_cpu_superoptimize_value.py` 加 plain_matmul tractability（单点 grid、`BENCH_MINIMAL_EXPLORE`）；`make test-cpu-cert-profile`。验证：`test_cpu_superoptimize_value` **~1.5s**；`make test-cpu-cert-quick` **48 passed**；`make test-cpu-cert-profile` **110 passed / planned 110**（quick **~7s**）。
- **Loop R58（2026-06-13，native_gemm RMS tractability + cert 耗时基线，PR #110）**：闸门：感知/工具层（闭合 R57「quick 跳过 native_gemm」与 fused RMS 14M+ states 失败）。`cpu_cert_utils.py` 统一 `apply_plain_matmul_search_tractability` / `apply_rms_matmul_search_tractability`；`test_cpu_native_gemm` 三处 superoptimize 对齐 R33 e2e（单点 grid、缩小 shape）；`--quick` 恢复 native_gemm；`HARDWARE_OPTIMIZATION.md` cert 耗时基线表。验证：`native_gemm`（excl. near_mkl）**~25s**；`make test-cpu-cert-profile` **~32s**，`value_verify_aligned: true`。
- **Loop R59（2026-06-13，全量 cert stage 耗时归档，PR #111）**：闸门：感知/工具层（闭合 R58「full cert 仅 minutes 占位」）。`cert_profile_from_stages()` 增 `stage_elapsed_s`/`stages_run`/`stages_ok`；`make test-cpu-cert-full-profile`（`--skip-walkthrough` 全量）；`HARDWARE_OPTIMIZATION.md` 分 stage 基线。验证：full profile **~35s**（vv **2.8s** + contract **2.6s** + native_gemm **27s** + superoptimize **3s**），`value_verify_aligned: true`。
- **Loop R60（2026-06-13，CPU Demo 优化闭环 AGENTS + manifest，PR #112）**：闸门：文档/契约层（用户要求 demo 测试纳入 Loop）。`AGENTS.md` 增「CPU Demo 测试与优化闭环」；`cpu_demo_loop_manifest()`；`make test-cpu-demos`；`test_cpu_demo_loop.py` manifest 契约。验证：`make test-cpu-demos` + `make test-cpu-cert-quick`。
- **Loop R61（2026-06-13，walkthrough cert profile + demo_lora，PR #113）**：闸门：感知/工具层（闭合 R60 建议 walkthrough 耗时 + LoRA demo）。`business_capability_walkthrough.py` `--json`/`--quick` + `build_walkthrough_report`；cert `profile.walkthrough_substage_elapsed_s`；`make test-cpu-cert-walkthrough-profile`；`demo_lora` CPU 烟雾 + manifest。验证：`make test-cpu-demos` **11 passed**；walkthrough profile **~214s**（Ray **~197s**）。
- **Loop R62（2026-06-13，Ray walkthrough quick tractability + reference_mugraphs，PR #114）**：闸门：感知/工具层（闭合 R61 Ray 占 197s + reference demo 缺口）。quick 跳过 seq/ray 双 benchmark、单点 grid + tractability；`reference_mugraphs/rms_norm.py --quick` + manifest。验证：`make test-cpu-demos` **12 passed**；walkthrough profile **~27s**（Ray **~14s**，原 **~197s**）。
- **Loop R63（2026-06-13，MLIR bench 契约 + reference lora + cert e2e，PR #115）**：闸门：感知/契约层（闭合 R62 建议）。`validate_mlir_jit_bench_row` + `mlir_jit_bench_json_timing_contract`；`make test-cpu-mlir-bench-contract`；`reference_mugraphs/lora.py --quick`；`make test-cpu-cert-e2e-profile`（full + walkthrough quick）。验证：`make test-cpu-demos` **13 passed**；`make test-cpu-mlir-bench-contract` **4 passed**；e2e profile **~67s**。
- **Loop R64（2026-06-13，MLIR bench 归档 + concat 契约 + gated_mlp reference，PR #116）**：闸门：感知/契约层（闭合 R63 建议）。`cpu_mlir_bench_utils.py` + `cpu_mlir_bench_profile.py`（`YIRAGE_MLIR_BENCH_PROFILE_JSON_*`）；`validate_concat_matmul_bench_row` + `mlir_jit_applicable`；`reference_mugraphs/gated_mlp.py --quick`；`make test-cpu-mlir-bench-profile`。验证：`make test-cpu-demos` **14 passed**；`make test-cpu-mlir-bench-contract` **10 passed**；mlir bench profile concat 契约 **PASS**（`USE_MLIR=0` 时 rms 契约 `contract_skipped`）。
- **Loop R65（2026-06-13，reference plain/chain + cert e2e mlir profile，PR #117）**：闸门：感知/契约层（闭合 R64 建议）。`reference_mugraphs/plain_matmul.py` + `matmul_chain.py --quick`；`run_mlir_bench_profile()` 纳入 `cpu_certification.py` full 模式；cert `profile.mlir_bench_profile_*`。验证：`make test-cpu-demos` **16 passed**；`make test-cpu-mlir-bench-contract`（含 e2e smoke）；e2e profile 含 mlir stage。
- **Loop R66（2026-06-13，loop-close 聚合 + concat reference + bench 四 workload 对齐，PR #118）**：闸门：文档/契约层（闭合 R65 建议）。`make test-cpu-loop-close`；`cpu_loop_close_manifest()` + `cpu_bench_workload_reference_map()`；`reference_mugraphs/concat_matmul.py --quick`；MLIR 可用时 rms `contract_ok` 必过（`contract_skipped=False`）。验证：`make test-cpu-demos` **19 passed**；`make test-cpu-loop-close` 全绿。
- **Loop R67（2026-06-13，loop-close JSON 归档 + shape 契约 + MLIR CI job，PR #119）**：闸门：感知/工具层（闭合 R66 建议）。`scripts/cpu_loop_close.py`（`YIRAGE_CPU_LOOP_CLOSE_JSON_*`）；`cpu_bench_reference_shape_contract()`；`make test-cpu-loop-close-profile`；`.github/workflows/cpu-mlir-jit-contract.yml`（`USE_MLIR=1` rms timing）。验证：`make test-cpu-demos` **21 passed**；`make test-cpu-loop-close-profile` **~40s**。
- **Loop R68（2026-06-13，共享 shape 常量 + nightly archive + MLIR dialect CI，PR #120）**：闸门：感知/工具层（闭合 R67 建议）。`scripts/cpu_bench_shapes.py`；reference demos import `reference_quick_dims`；`make test-cpu-loop-close-archive`；`.github/workflows/cpu-loop-close-nightly.yml`；MLIR CI `test-cpu-mlir-dialect-smoke`。验证：`make test-cpu-demos` **22 passed**；`test_cpu_bench_shapes` + profile。
- **Loop R69（2026-06-13，bench 单源 shape + nightly artifact + fused dialect smoke，PR #121）**：闸门：感知/工具层（闭合 R68 建议）。`bench_fused_vs_mkl_baseline._workloads` → `bench_shape_tuple()`；`cpu_loop_close.py --output` + nightly artifact upload；`test-cpu-mlir-dialect-smoke` 覆盖 unfused+fused `dialect_emit_path_smoke`。验证：`test_cpu_bench_shapes` **6 passed**；`make test-cpu-demos` **22 passed**。
- **Loop R70（2026-06-13，full shape 契约 + archive 解析回归 + MLIR profile artifact，PR #122）**：闸门：感知/工具层（闭合 R69 建议）。`test_bench_workloads_use_shared_full_shapes`；`validate_loop_close_archive` + `parse_loop_close_json`；`cpu_mlir_bench_profile.py --output`；MLIR CI 上传 bench profile JSON。验证：`test_cpu_bench_shapes` **8 passed**；`make test-cpu-demos` **27 passed**。
- **Loop R71（2026-06-13，archive 校验 + bench shapes 字段契约 + MLIR CI bundle，PR #123）**：闸门：感知/工具层（闭合 R70 建议）。nightly `validate_loop_close_archive`（`report.ok` 必 true）；`bench_shape_label` + `validate_bench_json_row_shapes` 纳入 MLIR profile；`validate_loop_close_archive.py` CLI；MLIR CI 单 artifact bundle（dialect log + profile JSON + manifest）；`make validate-cpu-loop-close-archive`。验证：`test_cpu_bench_shapes`；`make test-cpu-mlir-bench-contract`；`make test-cpu-loop-close-profile`。
- **Loop R72（2026-06-13，archive 回归 + bench shape CI gate + MLIR bundle 解析，PR #124）**：闸门：感知/工具层（闭合 R71 建议）。`validate_loop_close_archive` 校验归档 `mlir_bench_profile.rows` shape；nightly 模拟下载后二次 validate；`validate_mlir_bench_profile_archive` + `validate_mlir_ci_bundle` CLI；MLIR CI bundle 上传前 gate。验证：`test_cpu_mlir_ci_bundle`；`make test-cpu-demos`；`make test-cpu-loop-close-profile`。
- **Loop R73（2026-06-13，PR archive validate + full shape gate + shape_validation_errors 回归，PR #125）**：闸门：感知/工具层（闭合 R72 建议）。`.github/workflows/cpu-loop-close-pr.yml` quick archive + validate；full archive `run_mlir_bench_profile(quick=False)` + validate `bench_quick=False`；`shape_validation_errors` 字段一致性；MLIR CI / Makefile `test-cpu-loop-close-profile-validate`。验证：`test_cpu_loop_close` + `test_cpu_mlir_ci_bundle`；`make test-cpu-loop-close-profile-validate`。
- **Loop R74（2026-06-13，artifact 元数据 bench_quick + PR archive 上传 + 耗时基线，PR #126）**：闸门：感知/文档层（闭合 R73 建议）。`loop_close_archive_metadata` + `validate_loop_close_archive.py --metadata-output`；nightly/PR 上传 `.meta.json` sidecar；`HARDWARE_OPTIMIZATION.md` full archive stage 耗时表。验证：`test_cpu_loop_close` metadata tests；`make test-cpu-loop-close-profile-validate`。
- **Loop R75（2026-06-13，metadata schema + full stage 超时 + MLIR manifest bench_quick，PR #127）**：闸门：感知/工具层（闭合 R74 建议）。`validate_loop_close_archive_metadata` CLI + schema 单测；nightly 下载回归读 meta + `--check-stage-timeouts`；MLIR bundle manifest `bench_quick` 与 profile 对齐。验证：`test_cpu_loop_close` + `test_cpu_mlir_ci_bundle`；`make test-cpu-mlir-ci-bundle`。
- **Loop R76（2026-06-13，PR meta 下载回归 + soft-warning + archive 哈希，PR #128）**：闸门：感知/工具层（闭合 R75 建议）。PR workflow 模拟下载 validate meta + hash；`stage_timeout_warnings` 写入 meta sidecar；`archive_sha256` 校验。验证：`test_cpu_loop_close` hash/warning tests；`make test-cpu-loop-close-profile-validate`。
- **Loop R77（2026-06-13，soft limit 单源 + timeout alert 占位 + MLIR profile digest，PR #129）**：闸门：感知/工具层（闭合 R76 建议）。`loop_close_timing_contract()` 单源 soft/ceiling；`emit_loop_close_timeout_alert.py` nightly 占位；MLIR bundle `profile_sha256`。验证：`test_cpu_loop_close` timing/alert；`test_cpu_mlir_ci_bundle` digest。
- **Loop R78（2026-06-13，docs timing sync + PR alert + MLIR manifest v2，PR #130）**：闸门：感知/文档层（闭合 R77 建议）。`test_hardware_optimization_timing_contract_sync`；PR workflow emit timeout alert；`mlir_ci_bundle_manifest_v2` schema 单测。验证：`make test-cpu-mlir-ci-bundle`；`make test-cpu-demos`。
- **Loop R79（2026-06-13，meta warning count + timing doc render + MLIR manifest v3 shape 摘要，PR #131）**：闸门：感知/工具层（闭合 R78 建议）。`stage_timeout_warning_count` + `emit_loop_close_timeout_alert.py --annotate-metadata`；`loop_close_timing_markdown_table()` + `render_loop_close_timing_doc.py`；`mlir_ci_bundle_manifest_v3` 含 `shape_validation_errors_summary`。验证：`make test-cpu-mlir-ci-bundle` **16 passed**；`make test-cpu-demos` **43 passed**。
- **Loop R80（2026-06-13，metadata v2 post-alert + timing doc --write + MLIR bundle 下载回归，PR #132）**：闸门：感知/工具层（闭合 R79 建议）。`loop_close_artifact_metadata_v2` + `--require-alert-annotation`；`render_loop_close_timing_doc.py --write`；MLIR CI 模拟下载 validate shape 摘要。验证：`make test-cpu-mlir-ci-bundle` **19 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R81（2026-06-13，timeout_alert_pending + Makefile timing doc + MLIR nightly 下载回归，PR #133）**：闸门：感知/工具层（闭合 R80 建议）。metadata 预置 `timeout_alert_pending`；`make check/render-loop-close-timing-doc`；`.github/workflows/cpu-mlir-ci-nightly.yml` 下载回归 validate shape 摘要。验证：`make test-cpu-mlir-ci-bundle` **21 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R82（2026-06-13，共享 MLIR bundle 脚本 + metadata docs 契约 + PR timing gate，PR #136）**：闸门：感知/工具层（闭合 R81 建议）。`build_mlir_ci_bundle.py` + `regression_validate_mlir_ci_bundle.py`；`test_loop_close_metadata_doc_contract`；PR `make check-loop-close-timing-doc`。验证：`make test-cpu-mlir-ci-bundle` **26 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R83（2026-06-13，Makefile build-mlir + loop-close regression 脚本 + metadata 字段表，PR #137）**：闸门：感知/工具层（闭合 R82 建议）。`make build-mlir-ci-bundle`；`regression_validate_loop_close_archive.py` nightly/PR；`render_loop_close_metadata_doc.py` 字段表单源。验证：`make test-cpu-mlir-ci-bundle` **30 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R84（2026-06-13，合并 docs check + PR gate + Makefile MLIR/regression，PR #138）**：闸门：感知/文档层（闭合 R83 建议）。`make check-loop-close-docs`；PR workflow docs gate；MLIR/nightly/loop-close workflow 统一 Makefile 构建与 regression。验证：`make test-cpu-mlir-ci-bundle` **31 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R85（2026-06-13，MLIR/nightly docs gate + manifest-only smoke，PR #139）**：闸门：感知/文档层（闭合 R84 建议）。MLIR PR + loop-close nightly `make check-loop-close-docs`；`make smoke-build-mlir-ci-bundle-manifest`。验证：`make test-cpu-mlir-ci-bundle` **33 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R86（2026-06-13，MLIR nightly docs + post-alert Makefile regression + MANIFEST_ONLY 文档，PR #140）**：闸门：感知/文档层（闭合 R85 建议）。MLIR nightly docs gate；nightly/PR post-alert `REQUIRE_ALERT_ANNOTATION=1` regression；`smoke-build-mlir-ci-bundle-manifest` → `MANIFEST_ONLY=1` 别名 + docs 表。验证：`make test-cpu-mlir-ci-bundle` **34 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R87（2026-06-13，post-alert artifact 上传 + MLIR PR manifest smoke + post-alert Makefile 别名，PR #141）**：闸门：感知/工具层（闭合 R86 建议）。loop-close nightly 上传 `downloaded-regression-post-alert/`；MLIR PR `smoke-build-mlir-ci-bundle-manifest` CI smoke；`validate-loop-close-metadata-post-alert` Makefile 别名。验证：`make test-cpu-mlir-ci-bundle` **35 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R88（2026-06-13，PR post-alert artifact + MLIR nightly manifest smoke + PR 路径触发，PR #142）**：闸门：感知/工具层（闭合 R87 建议）。loop-close PR 上传 `downloaded-regression-post-alert/`；MLIR nightly manifest-only smoke；PR workflow 路径触发 post-alert 契约相关文件。验证：`make test-cpu-mlir-ci-bundle` **38 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R89（2026-06-13，pre-alert artifact 上传 + MLIR nightly 路径触发 + workflow 契约文档，PR #143）**：闸门：感知/工具层（闭合 R88 建议）。loop-close PR/nightly 上传 `downloaded-regression/`；`cpu-mlir-ci-nightly.yml` pull_request 路径含 manifest smoke 相关文件；`HARDWARE_OPTIMIZATION.md` 记录 `test-cpu-mlir-ci-bundle` workflow 契约测试。验证：`make test-cpu-mlir-ci-bundle` **41 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R90（2026-06-13，MLIR regression 独立 artifact + pre-alert Makefile 别名 + nightly 路径对称，PR #144）**：闸门：感知/工具层（闭合 R89 建议）。MLIR PR/nightly 独立上传 `downloaded-regression-mlir/`；`validate-loop-close-metadata-pre-alert`；loop-close nightly pull_request 路径与 PR 对称。验证：`make test-cpu-mlir-ci-bundle` **45 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R91（2026-06-13，MLIR download 别名 + nightly push 路径 + artifact 名称契约，PR #145）**：闸门：感知/工具层（闭合 R90 建议）。`validate-mlir-ci-metadata-download`；loop-close nightly `push` main 路径与 PR 对称；MLIR workflow 契约测试断言独立 artifact 名称。验证：`make test-cpu-mlir-ci-bundle` **47 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R92（2026-06-13，MLIR nightly push/PR 路径对称 + artifact 文档表 + download smoke，PR #146）**：闸门：感知/文档层（闭合 R91 建议）。`cpu-mlir-ci-nightly.yml` push/PR 路径与 MLIR JIT contract 对称；`HARDWARE_OPTIMIZATION.md` CI artifact 名称表；`test_cpu_mlir_ci_bundle` download alias 端到端 smoke。验证：`make test-cpu-mlir-ci-bundle` **51 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R93（2026-06-13，JIT push paths + artifact manifest 单源 + workflow make 对齐，PR #147）**：闸门：感知/文档层（闭合 R92 建议）。`cpu-mlir-jit-contract.yml` push paths；`cpu_ci_artifact_manifest()` + `cpu_ci_workflow_make_target_manifest()`；workflow/Makefile 对齐契约测试。验证：`make test-cpu-mlir-ci-bundle` **57 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R94（2026-06-13，CI artifact render + workflow/make 映射表 + check-loop-close-docs，PR #148）**：闸门：感知/文档层（闭合 R93 建议）。`render_loop_close_ci_artifact_doc.py --check/--write` 并入 `check-loop-close-docs`；MLIR+loop-close artifact 标记块渲染；workflow 步骤 human name ↔ make target 文档表。验证：`make test-cpu-mlir-ci-bundle` **61 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R95（2026-06-13，CI artifact intro 单源 + PR push paths + marker 契约，PR #149）**：闸门：感知/文档层（闭合 R94 建议）。`render_loop_close_ci_artifact_doc.py --write` 同步 intro 行；PR/nightly push paths 含 render 脚本；intro/marker 契约测试。验证：`make test-cpu-mlir-ci-bundle` **64 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R96（2026-06-13，timing/metadata intro 单源 + PR push 对称 + CI render 幂等，PR #150）**：闸门：感知/文档层（闭合 R95 建议）。timing/metadata intro 行单源 + render `--check/--write`；`cpu-loop-close-pr` push 与 pull_request 完全对称；CI artifact render 幂等单测。验证：`make test-cpu-mlir-ci-bundle` **70 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R97（2026-06-13，render 幂等 helper + intro 单源统一测试 + helpers 表，PR #151）**：闸门：感知/文档层（闭合 R96 建议）。`loop_close_doc_render_write_specs()` + `loop_close_doc_intro_line_specs()` 统一三份 render 脚本 intro+table 契约测试；`cpu-mlir-jit-contract` push/pull_request 路径对称（已有契约）；`HARDWARE_OPTIMIZATION.md` helpers 表记录 render/intro 单源。验证：`make test-cpu-mlir-ci-bundle` **66 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R98（2026-06-13，bundle 契约文档表 + render smoke 合并 + nightly 路径对称，PR #152）**：闸门：感知/文档层（闭合 R97 建议）。`cpu_mlir_ci_bundle_test_contract_manifest()` + docs 表；`make smoke-check-loop-close-docs` 合并三份 render ``--check``；`cpu-mlir-ci-nightly` push/pull_request 完全对称断言。验证：`make test-cpu-mlir-ci-bundle` **65 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R99（2026-06-13，smoke 步骤契约 + render check 路径触发 + JIT docs gate，PR #153）**：闸门：感知/文档层（闭合 R98 建议）。`loop_close_docs_smoke_make_target()` 单源 + workflow docs gate 契约测试；`loop_close_doc_render_check_specs()` 纳入 CI path 触发断言；MLIR JIT contract docs gate smoke 同步测试。验证：`make test-cpu-mlir-ci-bundle` **68 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R100（2026-06-13，docs gate 工作流文档 + Makefile 路径触发 + JIT push 对称，PR #154）**：闸门：感知/文档层（闭合 R99 建议）。`loop_close_ci_docs_gate_workflows()` + `loop_close_docs_smoke_path_triggers()` 纳入 bundle 契约表；CI path 含 Makefile smoke 片段断言；`cpu-mlir-jit-contract` push 与 `cpu-mlir-ci-nightly` pull_request 路径对称。验证：`make test-cpu-mlir-ci-bundle` **70 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R101（2026-06-13，统一路径触发 + MLIR 对称文档表 + docs gate 步骤名，PR #155）**：闸门：感知/文档层（闭合 R100 建议）。`loop_close_ci_doc_render_path_triggers()` 合并 smoke/write 路径单测；`cpu_mlir_ci_workflow_path_symmetry_doc_rows()` 文档表；`loop_close_ci_docs_gate_step_names()` 单源 + workflow 契约。验证：`make test-cpu-mlir-ci-bundle` **69 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R102（2026-06-13，path symmetry render 块 + loop-close 对称扩展 + make intro，PR #156）**：闸门：感知/文档层（闭合 R101 建议）。`cpu_ci_workflow_path_symmetry_doc_rows()` 纳入 ``render_loop_close_ci_artifact_doc`` marker 块；loop-close PR/nightly 路径对称行；workflow/make intro 引用 ``loop_close_ci_docs_gate_step_names()``。验证：`make test-cpu-mlir-ci-bundle` **72 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R103（2026-06-13，path symmetry intro 统一 + PR push 对称行 + render 幂等，PR #157）**：闸门：感知/文档层（闭合 R102 建议）。path symmetry 纳入 ``loop_close_doc_intro_line_specs``；``cpu-loop-close-pr`` push ↔ nightly pull_request 文档/契约行；``cpu_ci_path_symmetry_doc_markdown_table()`` 幂等单测。验证：`make test-cpu-mlir-ci-bundle` **74 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R104（2026-06-13，render write 子块断言 + nightly push 对称行 + Loop 编号同步，PR #158）**：闸门：感知/文档层（闭合 R103 建议）。``loop_close_doc_render_write_block_specs()`` 含 path symmetry 子块；nightly push ↔ PR pull_request 文档行；``loop_close_doc_bundle_loop_revision()`` 同步 intro/契约表。验证：`make test-cpu-mlir-ci-bundle` **74 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R105（2026-06-13，bundle 契约 render 块 + 交叉对称 pytest 文档 + Loop R105 intro，PR #159）**：闸门：感知/文档层（闭合 R104 建议）。``loop_close_doc_render_write_block_specs()`` 含 bundle 契约 + render write block 子块；PR push ↔ nightly pull_request 交叉对称 contract 含 pytest 名；``render_loop_close_ci_artifact_doc`` 同步 bundle/render 表；marker 标签不含 HTML 注释避免嵌套误匹配。验证：`make test-cpu-mlir-ci-bundle` **81 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R106（2026-06-13，revision 扩展 timing/metadata + paired marker 查找 + helpers 范围，PR #160）**：闸门：感知/文档层（闭合 R105 建议）。``loop_close_doc_bundle_loop_revision()`` → R106 驱动全部 intro（含 timing/metadata）；``find_loop_close_doc_marker_span`` / ``replace_loop_close_doc_marker_block`` paired 查找；Makefile helpers 表 ``loop_close_doc_makefile_helpers_loop_range()``。验证：`make test-cpu-mlir-ci-bundle` **84 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R107（2026-06-13，helpers intro render + timing marker 解析 + bundle 契约，PR #161）**：闸门：感知/文档层（闭合 R106 建议）。``loop_close_doc_makefile_helpers_loop_range()`` 纳入 bundle 契约表；``render_loop_close_ci_artifact_doc`` 同步 Makefile helpers intro（``LOOP_CLOSE_MAKEFILE_HELPERS_TABLE`` marker）；``parse_hardware_optimization_timing_table`` 改用 marker 块提取。验证：`make test-cpu-mlir-ci-bundle` **87 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R108（2026-06-13，metadata marker 解析 + helpers render 块 + intro 注册表，PR #162）**：闸门：感知/文档层（闭合 R107 建议）。``parse_hardware_optimization_metadata_doc_table`` marker 块提取；Makefile helpers 表纳入 ``loop_close_doc_render_write_block_specs`` + 单源 ``loop_close_doc_makefile_helpers_doc_rows()``；``LOOP_CLOSE_DOC_INTRO_LINE_TABLE`` 文档化 ``loop_close_doc_intro_line_specs()``（含 ``loop_close_doc_makefile_helpers_loop_range()``）。验证：`make test-cpu-mlir-ci-bundle` **90 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R109（2026-06-13，metadata fields marker 段 + helpers parse 契约 + intro 幂等，PR #163）**：闸门：感知/文档层（闭合 R108 建议）。``parse_hardware_optimization_metadata_doc_fields`` 仅解析 metadata marker 段（schema 在 intro）；``parse_hardware_optimization_makefile_helpers_table`` 纳入 bundle 契约；intro line registry 纳入 render write 幂等子块断言。验证：`make test-cpu-mlir-ci-bundle` **93 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R110（2026-06-13，intro schema 列 + metadata marker section 契约 + paired render，PR #164）**：闸门：感知/文档层（闭合 R109 建议）。``loop_close_metadata_doc_marker_section()`` 纳入 bundle 契约；intro line 表增加 Schema 列（metadata 行 ``LOOP_CLOSE_METADATA_DOC_SCHEMA``）；``render_loop_close_ci_artifact_doc`` 改用 ``replace_loop_close_doc_marker_block`` paired 查找。验证：`make test-cpu-mlir-ci-bundle` **95 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R111（2026-06-13，intro marker section cross-ref + render write replace_fn 列 + schema 解析契约，PR #165）**：闸门：感知/文档层（闭合 R110 建议）。``loop_close_doc_intro_line_doc_rows()`` 增加 Marker section 列（metadata → ``loop_close_metadata_doc_marker_section()``）；render write 子块表增加 Replace function 列（ci_artifact → ``replace_loop_close_doc_marker_block()``）；intro 表 schema 列 marker 块专用 parse 单测。验证：`make test-cpu-mlir-ci-bundle` **97 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R112（2026-06-13，replace_fn dispatch + intro marker_section manifest + render write parse 契约，PR #166）**：闸门：感知/文档层（闭合 R111 建议）。``resolve_loop_close_doc_render_write_block_replace_fn`` / ``apply_loop_close_doc_render_write_block_replace`` 单源 dispatch；``render_loop_close_ci_artifact_doc.replace_marker_block`` 按 spec 路由；bundle 契约增加 intro parse + render write parse + dispatch 行；3/4 列 render write parse decoy 单测。验证：`make test-cpu-mlir-ci-bundle` **101 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R113（2026-06-13，write/apply e2e + marker_section decoy + manifest row count 闸门，PR #167）**：闸门：感知/文档层（闭合 R112 建议）。``loop_close_doc_render_write_specs()`` 各 write 与 ``apply_loop_close_doc_render_write_block_replace`` 端到端；intro ``marker_section_fn`` decoy 单测；``cpu_mlir_ci_bundle_test_contract_manifest_row_count()`` bundle 行数 sync 闸门。验证：`make test-cpu-mlir-ci-bundle` **104 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R114（2026-06-13，check/write cross-ref + helpers row count + intro/bundle parity，PR #168）**：闸门：感知/文档层（闭合 R113 建议）。``loop_close_doc_render_check_write_crossref_rows()`` 文档化 check↔write↔replace_fn；Makefile helpers 表纳入 manifest row count + smoke cross-ref；``loop_close_doc_intro_line_doc_row_count()`` intro/bundle 行数 parity 单测。验证：`make test-cpu-mlir-ci-bundle` **108 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R115（2026-06-13，crossref doc 表 + helpers intro row count + path trigger parity，PR #169）**：闸门：感知/文档层（闭合 R114 建议）。``LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE`` 独立 doc 表 + render write 子块；Makefile helpers ``loop_close_doc_intro_line_doc_row_count()`` cross-ref；``loop_close_ci_doc_render_path_triggers_crossref_scripts()`` parity 单测。验证：`make test-cpu-mlir-ci-bundle` **111 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R116（2026-06-13，crossref decoy parse + bundle intro row count + block_count parity，PR #170）**：闸门：感知/文档层（闭合 R115 建议）。crossref marker 块 decoy parse；bundle intro 含 ``loop_close_doc_intro_line_doc_row_count()``；``loop_close_doc_render_check_write_crossref_block_count_parity()`` 与 render write 子块 block_count 对齐。验证：`make test-cpu-mlir-ci-bundle` **114 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R117（2026-06-13，crossref block counts intro + sync gate + 5-col parse，PR #171）**：闸门：感知/文档层（闭合 R116 建议）。crossref intro 含 ``loop_close_doc_render_write_block_counts_by_write_spec()``；``cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()`` 联合闸门；crossref parse 5/6 列向后兼容。验证：`make test-cpu-mlir-ci-bundle` **117 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R118（2026-06-13，helpers sync gate + blocks summary parity + intro manifest 文案，PR #172）**：闸门：感知/文档层（闭合 R117 建议）。Makefile helpers ``test-cpu-mlir-ci-bundle`` 含 sync gate + intro row count；``loop_close_doc_render_check_write_crossref_blocks_summary_parity()``；manifest 联合 sync gate 文案含 intro row count。验证：`make test-cpu-mlir-ci-bundle` **118 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R119（2026-06-13，crossref parity intro + check sync gate + manifest/blocks 联合闸门，PR #173）**：闸门：感知/文档层（闭合 R118 建议）。crossref intro 含 ``blocks_summary_parity``；``check-loop-close-docs`` helpers 含 sync gate；``cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok()`` 联合单测。验证：`make test-cpu-mlir-ci-bundle` **119 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R120（2026-06-13，helpers manifest/blocks sync + crossref decoy/parity 联合 + render check 钩子，PR #174）**：闸门：感知/文档层（闭合 R119 建议）。Makefile helpers ``check-loop-close-docs`` / ``test-cpu-mlir-ci-bundle`` 含 ``cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok()``；crossref marker decoy + ``blocks_summary_parity`` 联合单测；``loop_close_ci_artifact_doc_bundle_sync_gate_check()`` 纳入 manifest + ``render_loop_close_ci_artifact_doc.py`` ``--check``。验证：`make test-cpu-mlir-ci-bundle` **120 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R121（2026-06-13，smoke sync gate cross-ref + manifest decoy + render --check subprocess，PR #175）**：闸门：感知/文档层（闭合 R120 建议）。Makefile helpers smoke 行 + helpers intro 含 ``loop_close_ci_artifact_doc_bundle_sync_gate_check()``；bundle manifest marker decoy parse；``render_loop_close_ci_artifact_doc.py`` ``--doc-path`` + sync gate 失败 subprocess 契约；helpers/bundle intro render check 钩子 parity。验证：`make test-cpu-mlir-ci-bundle` **124 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R122（2026-06-13，timing/metadata --doc-path + smoke stderr + crossref intro parity，PR #176）**：闸门：感知/文档层（闭合 R121 建议）。timing/metadata render ``--doc-path`` parity；``loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()`` + smoke subprocess 三脚本输出断言；manifest decoy + sync gate 全文档联合单测；crossref intro 与 helpers smoke 行 render check 钩子 parity。验证：`make test-cpu-mlir-ci-bundle` **129 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R123（2026-06-13，failure snippet manifest + --doc-path helpers + smoke stderr，PR #177）**：闸门：感知/文档层（闭合 R122 建议）。manifest 新增 ``loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()``；Makefile helpers timing/metadata/ci_artifact ``--doc-path`` cross-ref；smoke 三脚本序列 + sync gate stderr capsys/subprocess；helpers/bundle/crossref 三向 intro parity。验证：`make test-cpu-mlir-ci-bundle` **132 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R124（2026-06-13，crossref failure snippet + Check script --doc-path + make smoke 失败，PR #178）**：闸门：感知/文档层（闭合 R123 建议）。crossref intro 含 failure snippet；``loop_close_doc_render_check_script_doc_crossref()`` Check script 列 ``--doc-path``；``make check-loop-close-docs`` bad doc 端到端失败 subprocess；manifest 行数与 helpers intro ``--doc-path`` parity。验证：`make test-cpu-mlir-ci-bundle` **133 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R125（2026-06-13，crossref script manifest + parse suffix compat + smoke snippet，PR #179）**：闸门：感知/文档层（闭合 R124 建议）。manifest ``loop_close_doc_render_check_script_doc_crossref()``；``normalize_loop_close_doc_render_check_script_doc_label()`` parse backward compat；helpers smoke failure snippet cross-ref；``make check-loop-close-docs`` force-fail env 断言 stderr snippet。验证：`make test-cpu-mlir-ci-bundle` **134 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R126（2026-06-13，force-fail env manifest/helpers + crossref decoy normalize，PR #180）**：闸门：感知/文档层（闭合 R125 建议）。manifest ``loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env()`` + ``force_fail_enabled()``；helpers check/smoke/crossref intro force-fail cross-ref；crossref decoy + legacy normalize 联合单测；smoke 成功路径 assert env 未设置。验证：`make test-cpu-mlir-ci-bundle` **135 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R127（2026-06-13，force-fail doc crossref manifest + helpers unset parity + make check success，PR #181）**：闸门：感知/文档层（闭合 R126 建议）。manifest ``loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()``；helpers intro ``must be unset on smoke`` parity；crossref legacy+suffix 双路径 decoy；``make check-loop-close-docs`` 成功 subprocess assert env 未继承。验证：`make test-cpu-mlir-ci-bundle` **137 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R128（2026-06-13，crossref doc crossref intro + check unset parity + mixed parse sync gate + dual subprocess，PR #182）**：闸门：感知/文档层（闭合 R127 建议）。crossref intro 显式 ``loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()`` ``= {doc_crossref()}``；``check-loop-close-docs`` helpers 行 ``must be unset on smoke`` parity；mixed legacy/suffix parse 与 doc 表 sync gate 联合单测；smoke+check 成功 subprocess 双断言 force-fail env。验证：`make test-cpu-mlir-ci-bundle` **139 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R129（2026-06-13，force-fail parity helper + stripped subprocess env + mixed parse subprocess，PR #183）**：闸门：感知/文档层（闭合 R128 建议）。``loop_close_doc_force_fail_crossref_and_check_row_parity_ok()`` + fragment helpers 合并 crossref intro/check 行 parity；``loop_close_doc_force_fail_env_stripped_subprocess_env()`` 共享 smoke/check 成功 subprocess 契约；``loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_markdown_table()`` + 全文档 sync gate 端到端 subprocess。验证：`make test-cpu-mlir-ci-bundle` **142 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R130（2026-06-13，manifest helpers cross-ref + mixed check full chain + three-way intro parity，PR #184）**：闸门：感知/文档层（闭合 R129 建议）。``loop_close_doc_makefile_helpers_manifest_new_helpers_crossref_ok()`` Makefile helpers test 行 cross-ref R129 manifest；mixed parse ``loop_close_doc_render_check_subprocess_argv_chain()`` ``make check-loop-close-docs`` 全链路 subprocess；``loop_close_doc_force_fail_three_way_intro_parity_ok()`` bundle intro 三向 force-fail parity。验证：`make test-cpu-mlir-ci-bundle` **145 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R131（2026-06-13，manifest helpers parity merge + smoke subprocess + intro registry parity gate，PR #185）**：闸门：感知/文档层（闭合 R130 建议）。``loop_close_doc_makefile_helpers_manifest_helpers_parity_ok()`` 合并 manifest/helpers 表 parity 单测；mixed check full chain 扩展 ``loop_close_docs_smoke_check_make_subprocess_argv()`` smoke subprocess；intro line registry 新增 Intro parity gate 列 + ``loop_close_doc_intro_line_three_way_parity_ok()``。验证：`make test-cpu-mlir-ci-bundle` **146 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R132（2026-06-13，intro parity decoy parse + bundle manifest parity merge + mixed smoke argv helper，PR #186）**：闸门：感知/文档层（闭合 R131 建议）。intro line registry 6 列 Intro parity gate decoy parse 单测；``loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok()`` 合并 manifest/helpers 与 bundle intro cross-ref；``loop_close_doc_mixed_parse_check_and_smoke_subprocess_argv()`` 共享 argv chain + smoke subprocess。验证：`make test-cpu-mlir-ci-bundle` **149 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R133（2026-06-13，intro manifest parity column + doc patch helper + purpose fragment subprocess，PR #187）**：闸门：感知/文档层（闭合 R132 建议）。intro registry 新增 Manifest parity gate 列 + ``loop_close_doc_intro_line_bundle_manifest_parity_ok()``；``loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_patched_doc_text()`` 共享 doc patch；``loop_close_doc_makefile_helpers_test_row_manifest_parity_doc_parity_ok()`` purpose fragment 端到端 subprocess。验证：`make test-cpu-mlir-ci-bundle` **154 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R134（2026-06-13，7-col intro decoy + manifest parity three-way + mixed subprocess plan，PR #188）**：闸门：感知/文档层（闭合 R133 建议）。7 列 intro registry parity gates decoy parse + 6 列 backward compat；``loop_close_doc_manifest_parity_three_way_ok()`` 合并 manifest parity gate 与 bundle intro fragment；``loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_plan()`` 联合 patched doc + purpose fragment subprocess。验证：`make test-cpu-mlir-ci-bundle` **156 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R135（2026-06-13，intro three-way manifest gate + check make subprocess + combined decoy，PR #189）**：闸门：感知/文档层（闭合 R134 建议）。intro registry Manifest parity gate 列改为 ``loop_close_doc_manifest_parity_three_way_ok()``；``loop_close_doc_check_loop_close_docs_make_subprocess_argv()`` + mixed plan 扩展 canonical ``make check-loop-close-docs`` 断言；7 列 parity gates 与 marker section decoy 联合 parse 单测。验证：`make test-cpu-mlir-ci-bundle` **156 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R136（2026-06-13，merged manifest three-way + full smoke/check argv + triple decoy，PR #190）**：闸门：感知/文档层（闭合 R135 建议）。``loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()`` 合并 intro registry manifest gate 与 bundle intro 三向断言；``loop_close_doc_mixed_parse_full_smoke_and_check_subprocess_argv_batches()`` + mixed plan ``full_argv_batches`` 全链路 smoke+check subprocess；7 列 + marker section + schema 三 decoy 联合 parse。验证：`make test-cpu-mlir-ci-bundle` **157 passed**；`make test-cpu-demos` **44 passed**。
- **Loop R137 感知建议**：merged manifest helper 纳入 intro doc intro 单行断言 helper；mixed plan canonical doc ``--doc-path`` check subprocess；intro registry 8 列 backward compat parse。
- **协议（2026-06-09）**：本地验证通过后 Agent **自动合并** PR，合并后立即感知扫描 backlog 并开下一轮。
- **协议（2026-06-09，框架对齐）**：每轮策略前必过 [执行前闸门](#执行前闸门框架目标与优化价值每轮必做) — 借鉴 PyTorch/TensorFlow CPU 分层（原语→BLAS、图级→融合），用四轮自问 + 可选 `eval_optimization_value` 避免低价值微优化。


---

## Cursor Cloud specific instructions（Serving 主轨 + MetaX MACA 支撑）

YiRage is a **library**（Python + native C++/Cython + RuntimeFusion / FusionPlan 实现 + `.maca` kernels），不是长驻 Web 应用。主迭代见 [RuntimeFusion 闭环](#runtimefusion-闭环主轨2026-07-27)。

### 双环境模型

| 环境 | 用途 | 后端 |
|------|------|------|
| **Cloud Agent CPU VM** | 编辑代码、契约单测、开 PR、RF API 外壳 | `cpu` / 无卡 pytest |
| **NVIDIA GPU**（团队机） | Serving S1+ FusionCapsule / RF.step / vLLM hybrid | `cuda` |
| **MetaX GPU SSH VM** | MACA 构建、支撑轨 bench | `maca` |

**合并闸门**：Serving/RF 功能宣称须 **real torch pytest** 绿；MACA kernel/search 改动须 MetaX 验证；Cloud CPU 绿 **不能** 替代。

**Serving 验证**：见 [Serving 验证禁令](#serving-验证禁令严禁2026-07-27-起永久有效) — **严禁** `demo/serving/*smoke*.py`、`--contract-only`、NumPy stub cert。

### MetaX GPU VM 一次性设置

Prerequisites（通常已预装）：MACA SDK `/opt/maca`、mcPytorch（`/opt/conda/bin/python3`）、`mxcc`、`mx-smi`。

```bash
ssh -p 32222 'root+<vm-id>@<host>'   # 凭据见团队密钥库，勿提交仓库

export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}
export PATH=${MACA_PATH}/mxgpu_llvm/bin:$PATH

# 验证
mx-smi
/opt/conda/bin/python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
which mxcc
```

### After `git pull`（MetaX VM）

```bash
cd /workspace   # YiRage 根目录
git submodule update --init --recursive
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:build/abstract_subexpr/release:build/formal_verifier/release:${LD_LIBRARY_PATH:-}
export PATH=$HOME/.local/bin:/usr/local/cargo/bin:${MACA_PATH}/mxgpu_llvm/bin:$PATH
export YIRAGE_BACKEND=maca
export PYTHONPATH=.

/opt/conda/bin/python3 -m pip install z3-solver numpy graphviz tqdm
YIRAGE_BACKEND=maca /opt/conda/bin/python3 -m pip install -e ".[dev]" --no-build-isolation
```

Rust helpers（若链接失败）：

```bash
(cd src/search/abstract_expr/abstract_subexpr && cargo build --release --target-dir ../../../../build/abstract_subexpr)
(cd src/search/verification/formal_verifier_equiv && cargo build --release --target-dir ../../../../build/formal_verifier)
```

`config.cmake` 参考（`USE_MACA ON`，`USE_CUDA OFF`）— 见 [docs/maca_quick_start.md](docs/maca_quick_start.md)。

### Lint / test / run（MACA）

| Task | 命令 | 环境 |
|------|------|------|
| Lint | `make lint` | CPU VM OK |
| MACA 配置单测 | `pytest tests/python/test_backends.py -k maca -v` | CPU VM OK |
| MACA 烟雾 | `demo/demo_maca_optimization.py` | **MetaX VM** |
| superoptimize smoke | `demo/maca_superopt_test.py` | **MetaX VM** |
| Fusion bench | `benchmark/maca_vs_pytorch.py` | **MetaX VM** |
| Verify import | `python3 -c "import yirage as yr; print(yr.__version__, yr.get_available_backends())"` | 构建后 |

### Gotchas（MACA）

- **`MACA_PATH` / `LD_LIBRARY_PATH`**：缺 `mcruntime` 时 `import yirage.core` 失败；须含 `${MACA_PATH}/lib` 与 `mxgpu_llvm/lib`。
- **mcPytorch 非 stock PyTorch**：须用 `/opt/conda/bin/python3`；`assert "metax" in torch.__version__.lower()`。
- **`torch.cuda.*` 即 MACA**：mcPytorch 通过 CUDA API 暴露 MetaX GPU；tensor `device="cuda"`。
- **64-thread warp**：`MACA_WARP_SIZE=64`；`block_dims` 探索须为 64 倍数（`get_maca_search_config()`）。
- **`mxcc` 编译**：改 `src/kernel/maca/*.maca` 后需重建 `yirage_runtime` 或 `pip install -e .`。
- **SSH 用户名**：`root+<vm-id>@host`，非裸 `root@host`。
- **FusionPlan cache（legacy 路径）**：`superoptimize(..., use_persistent_cache=True)` 现仍落在 `~/.yirage/mugraphs/`（`MuGraphStore`）；对外称 FusionPlan 缓存，符号改名 Chore 另开。
- **同后端优化**：Search/profile/execute 均同一 `backend`；见 [docs/HARDWARE_OPTIMIZATION.md](docs/HARDWARE_OPTIMIZATION.md)。
- **CPU 闭环归档**：`make test-cpu-cert` 等仍可用于 CPU 回归，但 **不是** Serving / MACA 合并闸门。
- **Serving 反模式**：不要只换 Attention 算子，也不要宣传 Mirage/MPK/µGraph；**严禁** `*smoke*` demo / contract-only / stub cert（见 [Serving 验证禁令](#serving-验证禁令严禁2026-07-27-起永久有效)）。
- **旧 R27**：offline latency 归档已降级；勿自动开 R27 分支除非 RF 明确依赖。

### Running MACA demos

```bash
export MACA_PATH=/opt/maca YIRAGE_BACKEND=maca PYTHONPATH=.
PY=/opt/conda/bin/python3
$PY demo/demo_maca_optimization.py
$PY demo/maca_superopt_test.py
$PY benchmark/maca_vs_pytorch.py
$PY benchmark/end-to-end/maca/llama_maca.py   # 较慢
```

See [docs/maca_quick_start.md](docs/maca_quick_start.md), [docs/maca_complete_guide.md](docs/maca_complete_guide.md), [docs/INSTALLATION.md](docs/INSTALLATION.md) § MetaX MACA.

### CPU 闭环（归档，非默认）

CPU 认证与 demo 清单仍维护于 `docs/cpu_support_matrix.yaml`、`make test-cpu-cert`、`cpu_demo_loop_manifest()`；详见上文 [历史归档](#历史归档cpu-无限优化闭环)。
