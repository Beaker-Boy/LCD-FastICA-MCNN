import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nptdms import TdmsFile
from scipy.io import savemat
from scipy.signal import butter, lfilter
import time
import os
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging

# Import PSO optimizer
try:
    from pso_optimizer import (
        calculate_spectral_entropy,
        detect_interference_frequencies,
        adaptive_update_frequency,
        calculate_fitness,
        optimize_lcd_fastica_params,
        PSO_Optimizer
    )
    PSO_AVAILABLE = True
    logger_pso = logging.getLogger(__name__)
    logger_pso.info("PSO优化器模块加载成功")
except ImportError:
    PSO_AVAILABLE = False
    logger_pso = logging.getLogger(__name__)
    logger_pso.warning("pso_optimizer模块未找到，PSO优化功能不可用")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional decomposition libraries
try:
    from vmdpy import VMD
    VMD_AVAILABLE = True
    logger.info("VMD library loaded successfully")
except ImportError:
    VMD_AVAILABLE = False
    logger.warning("vmdpy not installed. VMD method will not be available.")

try:
    from PyEMD import EEMD
    EEMD_AVAILABLE = True
    logger.info("EEMD library loaded successfully")
except ImportError:
    EEMD_AVAILABLE = False
    logger.warning("PyEMD not installed. EEMD method will not be available.")

try:
    from PyLMD import LMD
    LMD_AVAILABLE = True
    logger.info("LMD library loaded successfully")
except ImportError:
    LMD_AVAILABLE = False
    logger.warning("PyLMD not installed. LMD method will not be available.")


def linear_transform(x, t, m, a=1.0):
    """
    线性变换函数，用于构建基线
    
    Args:
        x: 信号值
        t: 时间向量
        m: 均值
        a: 插值参数（控制基线构建的平滑度）
    
    Returns:
        baseline: 基线信号
    """
    dt = t.max() - t.min()
    if dt == 0:
        return np.zeros_like(x)  # 防止除以零
    # 使用参数a调整插值权重
    return (x - m) * ((t - t.min()) / dt) ** a + m


def local_characteristic_scale_decomposition(x, t, num_components=10, interpolation_param=1.0):
    """
    局部特征尺度分解函数
    
    Args:
        x: 输入信号
        t: 时间向量
        num_components: 需要分解的分量数量
        interpolation_param: 插值参数a（控制基线构建的平滑度），默认1.0
    
    Returns:
        isc_components: ISC分量列表
    """
    m = np.mean(x)  # 计算信号的平均值
    x_centered = x - m  # 去中心化（不修改原始x）
    isc_components = []
    max_iterations = 1000  # 设置最大迭代次数以防止无限循环
    
    current_signal = x_centered.copy()
    
    for iteration in tqdm(range(max_iterations), desc=f"LCD Decomposition (a={interpolation_param:.2f})"):
        # 寻找极值点
        extrema_indices = np.where((np.diff(np.sign(np.diff(current_signal)))) != 0)[0] + 1
        
        if len(extrema_indices) < 4:  # 确保有足够的极值点
            break
        
        # 构建基线（使用插值参数a）
        baseline = np.zeros_like(current_signal)
        for i in range(len(extrema_indices) // 2):
            max_idx = extrema_indices[2 * i]
            min_idx = extrema_indices[2 * i + 1]
            segment_x = current_signal[max_idx:min_idx]
            segment_t = t[max_idx:min_idx]
            baseline[max_idx:min_idx] = linear_transform(segment_x, segment_t, m, a=interpolation_param)
        
        # 提取ISC分量
        isc = current_signal - baseline
        isc_components.append(isc)
        
        # 检查是否已经达到所需的ISC分量数量或满足极值单调性判据
        if len(isc_components) >= num_components:
            break
        
        # 更新信号
        current_signal = baseline - np.mean(isc)
    
    return isc_components

def extreme_monotonicity_criterion(x):
    """
    极值单调性判据函数
    """
    # 找出信号极大值点和极小值点，并保证极大值严格为正，极小值严格为负
    maxima = []
    minima = []
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] > x[i + 1]:
            maxima.append((i, x[i]))
        elif x[i] < x[i - 1] and x[i] < x[i + 1]:
            minima.append((i, x[i]))

    # 确定由极大值点所产生的二级极值点，包括二级极大和极小值点
    secondary_maxima = []
    secondary_minima = []
    for max_idx, max_val in maxima:
        left_min_idx = None
        right_min_idx = None
        for min_idx, min_val in minima:
            if min_idx < max_idx and (left_min_idx is None or min_idx > left_min_idx):
                left_min_idx = min_idx
            if min_idx > max_idx and (right_min_idx is None or min_idx < right_min_idx):
                right_min_idx = min_idx
        if left_min_idx is not None and right_min_idx is not None:
            secondary_maxima.append((max_idx, max_val))
            secondary_minima.append((left_min_idx, x[left_min_idx]))
            secondary_minima.append((right_min_idx, x[right_min_idx]))

    # 将二级极大值点与其前后相邻两个极值点（极大或者极小值点）进行绝对值大小比较，取三者中较大者作为新的二级极大值点
    new_secondary_maxima = []
    for max_idx, max_val in secondary_maxima:
        prev_idx = max_idx - 1 if max_idx > 0 else None
        next_idx = max_idx + 1 if max_idx < len(x) - 1 else None
        prev_val = x[prev_idx] if prev_idx is not None else -float('inf')
        next_val = x[next_idx] if next_idx is not None else -float('inf')
        new_max_val = max(max_val, abs(prev_val), abs(next_val))
        new_secondary_maxima.append((max_idx, new_max_val))

    # 将二级极小值点与其前后相邻两个极值点（极大或者极小值点）进行绝对值大小比较，取三者中较小者作为新的二级极小值点
    new_secondary_minima = []
    for min_idx, min_val in secondary_minima:
        prev_idx = min_idx - 1 if min_idx > 0 else None
        next_idx = min_idx + 1 if min_idx < len(x) - 1 else None
        prev_val = x[prev_idx] if prev_idx is not None else float('inf')
        next_val = x[next_idx] if next_idx is not None else float('inf')
        new_min_val = min(min_val, abs(prev_val), abs(next_val))
        new_secondary_minima.append((min_idx, new_min_val))

    # 对所有极小值点取绝对值，使得所有极值点序列均为正
    new_secondary_minima = [(idx, abs(val)) for idx, val in new_secondary_minima]

    # 找出由相邻两个二级极值点分割所确定的极值点序列，并判断单调性
    all_extrema = [(0, 0)] + new_secondary_minima + new_secondary_maxima + [(len(x) - 1, 0)]
    all_extrema.sort(key=lambda x: x[0])
    is_monotonic = True
    for i in range(1, len(all_extrema) - 1):
        if all_extrema[i][1] > all_extrema[i - 1][1] and all_extrema[i][1] > all_extrema[i + 1][1]:
            is_monotonic = False
            break
        elif all_extrema[i][1] < all_extrema[i - 1][1] and all_extrema[i][1] < all_extrema[i + 1][1]:
            is_monotonic = False
            break

    return is_monotonic

def vmd_decomposition(signal, fs, K=5, alpha=2000, tau=0.001):
    """
    变分模态分解 (VMD)
    
    Args:
        signal: 输入信号
        fs: 采样频率
        K: 模态数量
        alpha: 惩罚因子
        tau: 噪声容忍度
    
    Returns:
        imfs: 分解得到的本征模态函数数组
    
    Raises:
        MemoryError: 当信号过长可能导致内存不足时
        ValueError: 当参数不合理时
    """
    if not VMD_AVAILABLE:
        raise ImportError("vmdpy 库未安装，无法执行 VMD 分解")
    
    # Validate parameters before processing
    signal_length = len(signal)
    max_recommended_length = 500000  # 50万采样点
    
    if signal_length > max_recommended_length:
        # Calculate estimated memory requirement
        estimated_memory_gb = K * signal_length * 16 / (1024**3)
        
        raise MemoryError(
            f"⚠️  VMD分解内存风险检测！\n"
            f"   当前信号长度: {signal_length:,} 采样点\n"
            f"   推荐最大值: {max_recommended_length:,} 采样点\n"
            f"   预估内存需求: ~{estimated_memory_gb:.1f} GB (K={K})\n\n"
            f"解决方案：\n"
            f"  1. 在GUI中减少 '最大采样点数' 至 ≤ {max_recommended_length:,}\n"
            f"  2. 减少 '分解分量数' 至 3-10 之间\n"
            f"  3. 使用其他分解方法（如 LCD）处理长信号\n\n"
            f"提示：VMD算法在频域操作，对长信号内存需求极高。\n"
            f"      对于 {signal_length:,} 点的信号，即使 K=5 也需要约 {5 * signal_length * 16 / (1024**3):.1f} GB 内存。"
        )
    
    # Validate K parameter
    if K > 20:
        logger.warning(f"⚠️  警告：K={K} 过大，VMD通常使用 K=3-10。过大的K值会导致计算缓慢且可能不稳定。")
        K = min(K, 20)
        logger.info(f"已将 K 调整为 {K}")
    
    DC = 0
    init = 1
    tol = 1e-7
    
    logger.info(f"VMD分解配置: 信号长度={signal_length:,}, K={K}, alpha={alpha}")
    
    try:
        # 执行 VMD 分解
        imfs, u, u_hat = VMD(signal, alpha, tau, K, DC, init, tol)
    except MemoryError:
        raise MemoryError(
            f"VMD分解内存不足！信号长度={signal_length:,}, K={K}\n"
            f"建议：\n"
            f"  1. 减少 max_samples 参数（推荐 ≤ 500,000）\n"
            f"  2. 减少 num_components/K 参数（推荐 3-10）\n"
            f"  3. 使用其他分解方法（如 LCD）"
        )
    except Exception as e:
        raise RuntimeError(f"VMD分解失败：{str(e)}")
    
    print(f"VMD 分解完成，生成 {imfs.shape[0]} 个 IMF 分量")
    
    # 计算重构信号和相关系数
    reconstructed_signal = np.sum(imfs, axis=0)
    correlation_coefficient = np.corrcoef(signal, reconstructed_signal)[0, 1]
    print(f"VMD 重构信号与原信号的相关系数：{correlation_coefficient:.4f}")
    
    # 打印每个 IMF 的相关信息
    for i in range(imfs.shape[0]):
        corr = np.corrcoef(signal, imfs[i])[0, 1]
        print(f"IMF {i+1} 相关系数：{corr:.4f}")
    
    return imfs

def eemd_decomposition(signal, fs, max_imf=3, trials=10, noise_width=0.05, parallel=False):
    """
    集成经验模态分解 (EEMD)
    
    Args:
        signal: 输入信号
        fs: 采样频率
        max_imf: 最大 IMF 数量
        trials: EEMD集成试验次数（默认10，原默认100太慢）
        noise_width: 添加噪声的标准差（默认0.05）
        parallel: 是否启用并行计算（默认False）
    
    Returns:
        imfs: 分解得到的本征模态函数数组
    
    Raises:
        MemoryError: 当信号过长可能导致内存不足时
        ValueError: 当参数不合理时
    """
    if not EEMD_AVAILABLE:
        raise ImportError("PyEMD 库未安装，无法执行 EEMD 分解")
    
    # Validate signal length before processing
    signal_length = len(signal)
    max_recommended_length = 500000  # 50万采样点
    
    if signal_length > max_recommended_length:
        estimated_memory_gb = trials * max_imf * signal_length * 8 / (1024**3)
        
        raise MemoryError(
            f"⚠️  EEMD分解内存风险检测！\n"
            f"   当前信号长度: {signal_length:,} 采样点\n"
            f"   推荐最大值: {max_recommended_length:,} 采样点\n"
            f"   预估内存需求: ~{estimated_memory_gb:.1f} GB (trials={trials}, max_imf={max_imf})\n\n"
            f"解决方案：\n"
            f"  1. 在GUI中减少 '最大采样点数' 至 ≤ {max_recommended_length:,}\n"
            f"  2. 减少 '分解分量数' 至 3-8 之间\n"
            f"  3. 减少 trials 参数至 10-20\n"
            f"  4. 使用其他分解方法（如 LCD）处理长信号\n\n"
            f"提示：EEMD需要进行 {trials} 次集成试验，每次生成最多 {max_imf} 个IMF，\n"
            f"      总内存需求约为 trials × max_imf × signal_length × 8 bytes。"
        )
    
    # Validate parameters
    if trials > 50:
        logger.warning(f"⚠️  警告：trials={trials} 过大，会导致计算非常缓慢。推荐值：10-20")
        trials = min(trials, 50)
        logger.info(f"已将 trials 调整为 {trials}")
    
    if max_imf > 15:
        logger.warning(f"⚠️  警告：max_imf={max_imf} 过大，通常使用 3-8。过大的值会导致计算缓慢。")
        max_imf = min(max_imf, 15)
        logger.info(f"已将 max_imf 调整为 {max_imf}")
    
    # Performance optimization: reduce trials from default 100 to 10-20
    # This significantly speeds up computation while maintaining good decomposition quality
    logger.info(f"EEMD配置: 信号长度={signal_length:,}, trials={trials}, noise_width={noise_width}, max_imf={max_imf}, parallel={parallel}")
    
    try:
        eemd = EEMD(trials=trials, noise_width=noise_width, parallel=parallel, max_imf=max_imf)
        imfs = eemd(signal)
    except MemoryError:
        raise MemoryError(
            f"EEMD分解内存不足！信号长度={signal_length:,}, trials={trials}, max_imf={max_imf}\n"
            f"建议：\n"
            f"  1. 减少 max_samples 参数（推荐 ≤ 500,000）\n"
            f"  2. 减少 trials 参数（推荐 10-20）\n"
            f"  3. 减少 max_imf 参数（推荐 3-8）\n"
            f"  4. 使用其他分解方法（如 LCD）"
        )
    except Exception as e:
        raise RuntimeError(f"EEMD分解失败：{str(e)}")
    
    print(f"EEMD 分解完成，生成 {imfs.shape[0]} 个 IMF 分量")
    
    # 计算重构信号和相关系数
    reconstructed_signal = np.sum(imfs, axis=0)
    correlation_coefficient = np.corrcoef(signal, reconstructed_signal)[0, 1]
    print(f"EEMD 重构信号与原信号的相关系数：{correlation_coefficient:.4f}")
    
    # 打印每个 IMF 的相关信息
    for i in range(imfs.shape[0]):
        corr = np.corrcoef(signal, imfs[i])[0, 1]
        print(f"IMF {i+1} 相关系数：{corr:.4f}")
    
    return imfs

def lmd_decomposition(signal, fs, max_num_pf=6, max_smooth_iteration=8, max_envelope_iteration=100):
    """
    局部均值分解 (LMD)
    
    Args:
        signal: 输入信号
        fs: 采样频率
        max_num_pf: 最大PF分量数量（默认6，原默认8）
        max_smooth_iteration: 最大平滑迭代次数（默认8，原默认12）
        max_envelope_iteration: 最大包络迭代次数（默认100，原默认200）
    
    Returns:
        imfs: 分解得到的本征模态函数数组
    
    Raises:
        MemoryError: 当信号过长可能导致内存不足时
        ValueError: 当参数不合理时
    """
    if not LMD_AVAILABLE:
        raise ImportError("PyLMD 库未安装，无法执行 LMD 分解")
    
    # Validate signal length before processing
    signal_length = len(signal)
    max_recommended_length = 500000  # 50万采样点
    
    if signal_length > max_recommended_length:
        raise MemoryError(
            f"⚠️  LMD分解内存风险检测！\n"
            f"   当前信号长度: {signal_length:,} 采样点\n"
            f"   推荐最大值: {max_recommended_length:,} 采样点\n\n"
            f"解决方案：\n"
            f"  1. 在GUI中减少 '最大采样点数' 至 ≤ {max_recommended_length:,}\n"
            f"  2. 减少 '分解分量数' 至 4-8 之间\n"
            f"  3. 使用其他分解方法（如 LCD）处理长信号\n\n"
            f"提示：LMD算法需要多次迭代平滑和包络计算，长信号会导致计算缓慢且内存占用高。"
        )
    
    # Validate parameters
    if max_num_pf > 10:
        logger.warning(f"⚠️  警告：max_num_pf={max_num_pf} 过大，通常使用 4-8。过大的值会导致计算缓慢。")
        max_num_pf = min(max_num_pf, 10)
        logger.info(f"已将 max_num_pf 调整为 {max_num_pf}")
    
    # Performance optimization: reduce iteration limits to speed up computation
    logger.info(f"LMD配置: 信号长度={signal_length:,}, max_num_pf={max_num_pf}, max_smooth_iter={max_smooth_iteration}, max_env_iter={max_envelope_iteration}")
    
    try:
        lmd = LMD(
            max_num_pf=max_num_pf,
            max_smooth_iteration=max_smooth_iteration,
            max_envelope_iteration=max_envelope_iteration
        )
        result = lmd.lmd(signal)
        # LMD 返回的结果是元组，第一个元素是 IMF 数组
        imfs = result[0]
    except MemoryError:
        raise MemoryError(
            f"LMD分解内存不足！信号长度={signal_length:,}, max_num_pf={max_num_pf}\n"
            f"建议：\n"
            f"  1. 减少 max_samples 参数（推荐 ≤ 500,000）\n"
            f"  2. 减少 max_num_pf 参数（推荐 4-8）\n"
            f"  3. 使用其他分解方法（如 LCD）"
        )
    except Exception as e:
        raise RuntimeError(f"LMD分解失败：{str(e)}")
    
    print(f"LMD 分解完成，生成 {imfs.shape[0]} 个 PF 分量")
    
    # 计算重构信号和相关系数
    reconstructed_signal = np.sum(imfs, axis=0)
    correlation_coefficient = np.corrcoef(signal, reconstructed_signal)[0, 1]
    print(f"LMD 重构信号与原信号的相关系数：{correlation_coefficient:.4f}")
    
    # 打印每个 PF 分量的相关信息
    for i in range(imfs.shape[0]):
        corr = np.corrcoef(signal, imfs[i])[0, 1]
        print(f"PF {i+1} 相关系数：{corr:.4f}")
    
    return imfs