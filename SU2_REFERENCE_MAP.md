# SU(2) FFT + delay update 参考代码

2026-10-16：已阅读本目录全部 10 个不以 `2` 开头的 .jl 文件，以及 README.md、system.csv、上级 MEMORY.md。仅静态阅读，未运行或修改算法。

## 用户确认的用途与边界

- 不以 `2` 开头的 .jl 文件是 SU(2) 吸引 Hubbard DQMC 的 FFT + delay update 参考实现，保留原版。
- 以 `2` 开头的对应文件（如 2main.jl）是构建新三分量代码并实现 FFT + delay update 的目标。旧 MEMORY.md 中“*2.jl”应按本次明确的前缀 `2` 理解。
- 旧 Fortran 的物理/HS/三分量局域矩阵约定参考 ../../3component_DQMC/CODE_MAP.md；本套 Julia 提供优化架构参考。两者更新所处的 Green 时间切面和矩阵乘法次序不能直接混用。
- 本次仅列出 2*.jl 的大小，未读入或改动其内容：2initialize.jl、2lattices.jl 已非空，其余八个仍为空。旧记忆“全部为空”已过时。

## 调用链

main → initialize → lattices + utilities → multiply → Block；main 另 include montecarlo、measure、postprocess。

- main.jl：MPI 初始化，Float64/ComplexF64 类型；热化/测量奇偶步交替扫描；OutputAverage。
- initialize.jl：读取 system.csv，构造晶格、稠密动能指数、HS 表、FFT/LDR/测量/delay 工作区及 MuTuner。
- montecarlo.jl：UpdateIsingKet! 每 P0 片重算 G；每片更新前 MeasureEqualTime!；UpdateBond! 先移动 Green 切面，遍历格点场，再移至下一切面。
- utilities.jl：四态 HS、FFT 初始化、各工作区、CalculateUDV!、CalculateGreenFunc!、化学势调节。
- multiply.jl：传播正逆左右乘法、LDR 稳定化、三种稳定逆、FFT 左右乘法。
- measure.jl：能量、双占据、配对、电荷、Kagome 子晶格 CDW、等时/非等时磁响应、准粒子权重。
- postprocess.jl：rank 内归一化、Fourier 和关联比、MPI 汇总、文本/JLD2 输出。
- Block.jl：BlockDiagonal 与左右 mul!；test.jl：仅打印 Makesquare 的小矩阵。

## SU(2) 与局域更新

- GreenFunc 是 Nlayer×Nlayer 的单自旋矩阵；相同自旋块的简并通过 Ns 因子和 detR^Ns 使用。不能把 Ns 改为 3 就得到混合三分量 Green 算法。
- Nhs=1，NIsingState=4；η、γ 为四态 HS 表，eVB=exp(η)，eMVB=exp(-η)，IsingΔ[new,old]=eVB[new]/eVB[old]-1。
- 接受率包含 exp(-(ηnew-ηold)*Ns/2)*γnew/γold*detR^Ns；按绝对值接受，负实比更新 WeightSign。
- CalculateMetroplis 先用缓存修正 Gii，再计算 1+(1-Gii)*Δ。
- DelayMats 保存计数 i、v(Nlayer,nd)、vᵗ(nd,Nlayer)；当前有效矩阵为 G_eff=G_base-v*vᵗ。UpdateLocal! 使用已有缓存恢复当前所需列和行，新接受一次增加一列/行；UpdateDelay! 用 mul!(G,v,vᵗ,-1,1) 批量写回。
- nd 计数接受的更新，达到 nd 或到格点扫描末尾就 flush；随后计数清零。后续三分量版本需要块修正，重新推导接受矩阵及缓存递推，不能照搬标量 Δ 或取三次幂。
- UpdateLocal! 的形式为 G'=G-G[:,i]*[Δ/(1+(1-Gii)Δ)]*(I-G)[i,:]；旧 Fortran 的形式为 G'=G-(I-G)[:,S]*Δ*M^-1*G[S,:]。必须结合 WarpUpdate* 的切面解释差异。

## FFT 与稳定化

- InitializeFFT_kagome 构造 L² 个 3×3 动量块；这里的 3 是 Kagome 子晶格数，不是自旋/颜色数。
- 实空间向量 reshape 为 (3,L²)，原胞内三个子晶格连续排列；对每个子晶格做 L×L 二维 FFT。与旧 Fortran 按颜色整块排列的基底不同。
- lmulK_block!：逐列 inverse FFT → 动量块左乘 → FFT；rmulK_block!：逐行 FFT → 动量块右乘 → inverse FFT；返回实部。使用预建 FFT plan 和复数缓冲 M₁/M₂。
- FFTDiagonal 保存正/逆整步和半步传播；Kpμ 另外乘 exp(t*μμ[1])，其中 t 实际存 t0*Δτ*步长系数。非单位 t0 下化学势单位应核对。
- rmul_kv!/lmul_kv! 实现 A*eK*eV / eK*eV*A；逆传播严格逆序。FFT 同时用于切面推进、LDR 累积和测量半步变换。
- LDR_R!/LDR_L! 维护 StableLinearAlgebra 的 LDR；Inv_IpA!、Inv_IpUV!、inv_invUpV! 通过 D 的 min/max 尺度拆分和 LU 求逆处理稳定化，不能用裸矩阵连乘替代。
- FFT 优化动能传播，不使所有稳定化求逆都变为 O(N²logN)；delay update 将多次外积写回改为矩阵乘法，仍需核查整体性能。

## 测量和统计约定

- G2=eKHalf^-1*G*eKHalf；测量公式基于单自旋 G 且假定两自旋块相同。
- Pair!=0.5*(Gij²+(Gji-δij)²)*sign；CDW!=Ns²*(1-Gii)*(1-Gjj)+Ns*(δij-Gji)*Gij。
- Charge 累加全格点全分量粒子数；每 rank 的 κ=β/Nlayer*(sum(<n_i n_j>)-<N>²)，分子先按本 rank 平均符号重加权。
- MeasureEMS! 实际系数是 3β/(2Nlayer)*Tr(G-G²)，注释规范化不能替代实际代码。
- MPIAverage 返回 rank 均值和 rank 样本标准差，没有除 sqrt(ncpurank)，不是均值标准误。关联比等非线性量在 rank 内先计算再平均。
- 非等时 G 在采样过程中先累积，postprocess 再对平均 G 做乘积计算磁响应；这不自动等于 Wick 乘积的 MC 平均，迁移前需单独核查。

## 当前目录快照的限制（未修复）

- lattices.jl 仅定义 Makesquare；initialize 调用 MakeSquare/MakeTriangular/MakeHoneycomb/MakeKagome。存在大小写不一致及缺失函数，不能声称当前 main 已可运行。
- system.csv 当前为 latticetype=5，而 initialize 仅有 1–4 分支；其他当前值含 L=4、Ns=2、U=4、t0=1、β=30、Interval=300、P0=5、uneq=false。物理吸引势符号与代码 U 输入/能量正号的关系尚未推导核实，不能套用 Fortran U<0 约定。
- FFT、CDW 和 Fourier 仍硬编码 Kagome 三子晶格；当前方格构造不能直接接入这一套流程。
- nd=2^min(6,floor(Int,log2(Nlayer/20)))：小尺寸可能给出负指数，低成本测试前需处理合法批大小。
- _logmax 对严格零误差会取 log(0)，之后转 Integer 的行为需核查。
- CalculateGreenFunc! 访问 Fs[2]/Fs[end-1]，小系统虚时间分组不能只看 round(Interval/P0) 就认定合法。
- 现有非等时估计、符号重加权、HS 恒等式及 μ 调节后的缓存一致性均未在此次阅读中验证。

后续回答函数位置时重新核对行号；在 2*.jl 中实现前先确定模型、基底顺序、局域 HS 块和 Green 切面，再复用 FFT/LDR/delay 框架。
