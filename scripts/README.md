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

结果在 [output/thread_scaling.csv](output/thread_scaling.csv)中
结论：

1. **小矩阵 (< 256x256)**:
   - **BLAS**: 单线程效果最佳。多线程带来的调度开销远大于计算收益。
   - **Recommendation**: 推理小批量数据时，建议设置 `OMP_NUM_THREADS=1`。

2. **大矩阵 (>= 512x512)**:
   - 多线程并行带来显著加速。

3. **环境限制的关键发现**:
   - 在 `taskset -c 0-7` (限制前8个核心) 的严格实验环境下，**最佳线程数为 8**。
   - **过载现象**: 当强行开启 16 线程（`OMP_NUM_THREADS=16`）时，由于物理核心只有 8 个，操作系统必须进行频繁的上下文切换。这导致性能**反而下降**（例如 BLAS 2048x2048 耗时从 8线程的 0.08s 增加到 16线程的 0.11s）。只有在无绑核限制时，16 线程才可能有优势。

4. **OpenMP 实现的数据解读**:
   - 我们的手写 OpenMP 版本（0.57s @ 8线程 -> 0.51s @ 16线程）在超线程/过载情况下**略有提升**，但这并不代表效率高。
   - **原因**: 手写的代码没有像 BLAS 那样极致利用 CPU 流水线（指令级并行度较低），因此当 16 个线程挤在 8 个核上时，线程切换有时能填补流水线的气泡（Latency Hiding），带来微小的吞吐量提升。
   - **对比**: 相比之下，BLAS 把 CPU 算力吃满了，任何额外的线程切换都是纯粹的损耗，所以 BLAS 对核数限制更敏感。

```bash
# 运行制图脚本
python3 scripts/plot_scaling.py
```

结果在 [output/scaling_plot.png](output/scaling_plot.png)中

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
