# 测试脚本说明

## 设置实验环境

```bash
# 1. 设置实验环境 (需要 sudo)
# ------------------------------------------------------------------
echo "正在设置实验环境..."
# 将 CPU 调速器设置为性能模式
sudo cpupower frequency-set -g performance
# 锁定频率到 4.0 GHz (4000MHz)
sudo cpupower frequency-set -u 4000MHz -d 4000MHz

# 检查当前频率确认设置成功
echo "当前 CPU 频率信息:"
cpupower frequency-info | grep "current CPU frequency"
```

## 测试不同线程和不同尺寸的矩阵乘法

```bash
# 2. 运行测试脚本
# ------------------------------------------------------------------
echo "开始运行测试..."
./find_optimal_threads.sh
# or
# 使用 taskset 将整个脚本进程绑定到前 8 个核心 (0-7)
# 这样脚本里生成的所有子进程也会继承这个绑定
sudo taskset -c 0-7 bash scripts/find_optimal_threads.sh
```

结果在 [output/outputresult/thread_scaling.csv](output/outputresult/thread_scaling.csv)中
结论：

1. **小矩阵波动与开销 (Small Matrix Instability)**:
   - **现象**: 对于 64x64 等小尺寸，多线程（尤其是 OpenMP 8线程）表现出**极高的方差（High Variance）**，标准差甚至接近均值。
   - **原因**: 
     - **计算量微乎其微**: 64x64 的浮点运算仅需几微秒即可完成。
     - **OpenMP 开销占主导**: 线程的创建(Fork)、唤醒(Wakeup)和屏障同步(Barrier)本身就需要几微秒到几十微秒。
     - **调度抖动**: 在如此短的时间窗口内，操作系统的任何微小调度延迟（如中断）都会被放大为巨大的性能波动。
   - **结论**: **绝对禁止对小矩阵使用多线程**。推荐在矩阵小于 256x256 时强制使用 `OMP_NUM_THREADS=1`。

   > **Case Study: 64x64 Anomaly (8 vs 16 Threads)**
   > **Question**: Why is 8 threads (on 8 cores) unstable with high variance, while 16 threads is faster and more stable?
   > **Explanation**:
   > 1.  **8 Threads (Active Spin)**: OpenMP runtime often uses "Active Spin" (busy waiting) when threads <= cores. Threads aggressively check for work. If one thread is slightly delayed by OS noise, other 7 threads spin and consume CPU cycles uselessly, amplifying the delay and causing huge variance.
   > 2.  **16 Threads (Passive Wait)**: When threads > cores, OpenMP/OS realizes CPU is scarce. Threads are forced to "Yield" or "Sleep" (Passive Wait) instead of spinning. This allows the OS scheduler to fill gaps more efficiently. Paradoxically for such tiny workloads, this "Yield" strategy avoids the rigid penalty of busy-waiting, leading to lower variance and better average time.

2. **统计学修正 (Statistical Methodology)**:
   - **观察**: 原始数据中存在偶发的极大值（Spikes）。
   - **优化**: 采用 **截断平均数 (Trimmed Mean)**，通常去除头尾各 **5% ~ 10%** 的极值。
   - **原理**: 
     - **去除冷启动 (Remove Cold Start)**: 最快的几次（或刚开始的几次）可能受限于 CPU 频率提升、缓存未预热等影响。
     - **过滤系统噪声 (Filter OS Jitter)**: 最慢的几次往往是因为操作系统中断（IRQ）、后台进程抢占等非算法因素导致的。
     - **注意**: 在高性能计算（HPC）测试工况下，去掉 10% 属于**保守但安全**的做法，能有效过滤偶发波动（Outliers）而不损失主体特征。如果数据非常稳定（如大矩阵），可以放宽到 1-2%。

3. **核心绑定与过载 (Core Binding & Oversubscription)**:
   - **环境**: 实验使用了 `taskset -c 0-7`，严格限制程序只能在 8 个物理核心上运行。
   - **关键发现**: 最佳线程数**严格锁定为 8**。
   - **过载惩罚 (Penalty)**: 当开启 16 线程时，操作系统必须在 8 个核心上轮流调度 16 个线程（Context Switching）。
     - 对于 **BLAS**（高度优化，流水线极满）：任何上下文切换都是纯粹的损耗，导致性能**下降约 40%**（从 0.05s 变慢到 0.08s 2048x2048）。
     - 这一现象完美验证了高性能计算中的**"不要超额订阅 (Do not oversubscribe)"**原则。

4. **BLAS vs 手写 OpenMP 的行为差异**:
   - **BLAS**: 对 CPU 利用率极高（AVX/FMA指令），对核数限制极其敏感，过载即崩溃。
   - **手写 OpenMP**:
     - 我们的实现（`dgemm_omp`）虽然也并行了，但单线程流水线效率不如 BLAS（存在指令气泡/等待）。
     - 因此，当 16 线程挤在 8 核上切换时，线程切换的延迟有时恰好能填补流水线的气泡（Latency Hiding），导致在大尺寸下 16 线程并没有像 BLAS 那样显著变慢，甚至微快。
     - **注意**: 这不代表代码好，反而说明单线程优化还有空间（没有吃满 CPU）。

5. **大矩阵推荐 (Large Matrices)**:
   - 对于 >= 512x512 的矩阵，多线程收益显著。
   - 在本实验受限环境下，推荐配置为 **8 线程**。

6. **总结建议 (Summary Recommendations)**:
   - **OpenMP**:
     - **小矩阵 (< 128x128)**: 强制单线程 `OMP_NUM_THREADS=1`。
     - **中等矩阵 (256x256 to 512x512)**: 可用4线程（既有明显的优化，又不会因为各种原因导致的不稳定）。
     - **大矩阵 (>= 1024x1024)**: 推荐使用与物理核心数相等的线程数（本例为 8 线程）。
     - **之后的实验我们将默认使用 4 线程和 8 线程进行测试**。
   - **BLAS**: 对 CPU 利用率极高（AVX/FMA指令），对核数限制极其敏感，过载即崩溃，因此我们使用核数相同的线程数（本例为 8 线程）。

```bash
# 运行制图脚本
python3 scripts/plot_scaling.py
```

结果在 [output/outputresult/scaling_plot.png](output/outputresult/scaling_plot.png) 和 [output/outputresult/scaling_speedup_plot.png](output/outputresult/scaling_speedup_plot.png) 中

## 测试不同优化级别对矩阵乘法的影响

```bash
# 2. 运行不同优化级别测试脚本
# ------------------------------------------------------------------
echo "开始运行不同优化级别测试..."
./scripts/benchmark_large.sh
# or
# 使用 taskset 将整个脚本进程绑定到前 8 个核心 (0-7)
# 这样脚本里生成的所有子进程也会继承这个绑定
sudo taskset -c 0-7 bash scripts/benchmark_large.sh
```

结果在 [output/outputresult/impl_comparison.csv](output/outputresult/impl_comparison.csv) 中

```bash
# 运行制图脚本 (生成对比图和加速比图)
python3 scripts/plot_comparison.py
```

结果在 [output/outputresult/comparison_grid_plot.png](output/outputresult/comparison_grid_plot.png) 和 [output/outputresult/comparison_speedup_grid.png](output/outputresult/comparison_speedup_grid.png) 中

结论：

1. **性能阶梯 (Performance Hierarchy)**:
   - **朴素实现 (`ijk`)**: 极慢，随着矩阵尺寸增大呈现 $O(N^3)$ 指数级爆炸。在 2048x2048 时耗时数秒。
   - **循环重排 (`ikj`)**: 仅通过改变循环顺序利用缓存局部性，即可获得显著加速。
   - **手写 OpenMP**: 多线程并行带来了数量级的提升。
   - **BLAS (OpenBLAS)**: 利用 SIMD 指令和汇编级优化，提供了极致性能，比朴素实现快 **1200倍以上**。

2. **可视化策略 (Linear Scale & Adaptive Units)**:
   - 我们使用了 **线性坐标 (Linear Scale)** 而非对数坐标，以最直观的方式展示了优化前后的巨大鸿沟（`ijk` 的柱子高耸入云，而优化后的实现几乎贴地）。
   - 为了解决量级跨度大的问题，作图脚本采用了 **自适应单位 (Adaptive Units)**：
     - 小矩阵 (64x64) 使用 **微秒 (us)**。
     - 大矩阵 (2048x2048) 使用 **秒 (s)**。
     - 这种处理方式兼顾了微观波动和宏观趋势的可读性。

3. **加速比 (Speedup)**:
   - 加速比图清晰地展示了优化的“台阶”。最大的矩阵下，BLAS 的加速比能达到 **1200x**，这有力地证明了算法优化比硬件堆砌更重要。



## 恢复环境

```bash
# 恢复为节能或调度模式或不变保持性能模式 (通常是 powersave 或 schedutil, Fedora 常用 powersave)
sudo cpupower frequency-set -g powersave
# 恢复频率范围 (根据您的 CPU: 421MHz - 5386MHz)
sudo cpupower frequency-set -d 421MHz -u 5386MHz

echo "环境已恢复。"
```

## 其他脚本：生成 PlantUML 图

```bash
python3 encode_plantuml.py output/thread_scaling.csv
```
