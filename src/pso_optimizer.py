"""
PSO双阈值动态优化策略模块

实现基于粒子群优化（Particle Swarm Optimization）的LCD与FastICA关键参数自适应调整。

核心功能：
1. 谱熵计算 - 评估信号复杂度
2. 干扰频率自动检测 - 识别强干扰频率成分
3. 适应度函数计算 - 最大化故障特征增强效果
4. PSO优化器 - 搜索最优参数组合 (a, tol)

作者：LCD-FastICA-MCNN 团队
版本：v2.3
"""

import numpy as np
from typing import cast, Tuple, List, Optional, Dict
from scipy import fft as scipy_fft
from scipy.signal import find_peaks
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def calculate_spectral_entropy(signal: np.ndarray, fs: float) -> float:
    """
    计算信号的谱熵（公式 3-26）
    
    谱熵 Hs 用于衡量信号的复杂度，Hs 越大表示信号越复杂。
    
    Args:
        signal: 输入信号（一维数组）
        fs: 采样频率（Hz）
    
    Returns:
        spectral_entropy: 谱熵值（标量）
    
    References:
        公式 (3-26): H_s = -Σ P(f_i) log P(f_i)
    """
    # 计算FFT
    N = len(signal)
    Y = scipy_fft.fft(signal)
    
    # 计算功率谱密度（仅取正频率部分）
    Y_half = Y[:N//2]
    P = np.abs(cast(np.ndarray, Y_half))**2 / N
    P = P / np.sum(P)  # 归一化为概率分布
    
    # 避免 log(0)
    P = P[P > 0]
    
    # 计算谱熵
    spectral_entropy = -np.sum(P * np.log2(P))
    
    logger.debug(f"谱熵计算完成: H_s = {spectral_entropy:.4f}")
    
    return float(spectral_entropy)


def detect_interference_frequencies(
    signal: np.ndarray, 
    fs: float,
    fault_frequencies: Optional[List[float]] = None,
    num_peaks: int = 10,
    min_prominence: float = 0.1
) -> List[float]:
    """
    自动检测强干扰频率（如转频及其谐波）
    
    通过频谱峰值检测，排除故障特征频率后的高能量频率点。
    
    Args:
        signal: 输入信号（一维数组）
        fs: 采样频率（Hz）
        fault_frequencies: 已知的故障特征频率列表（Hz），可选
        num_peaks: 检测的峰值数量
        min_prominence: 最小突出度阈值（归一化幅值的比例）
    
    Returns:
        interference_freqs: 干扰频率列表（Hz）
    """
    # 计算FFT
    N = len(signal)
    Y = scipy_fft.fft(signal)
    freqs = scipy_fft.fftfreq(N, d=1/fs)
    
    # 仅取正频率部分
    positive_mask = freqs > 0
    freqs_pos = freqs[positive_mask]
    Y_positive = Y[cast(np.ndarray, positive_mask)]
    magnitude = np.abs(cast(np.ndarray, Y_positive))
    
    # 归一化幅值
    magnitude_norm = magnitude / np.max(magnitude)
    
    # 检测峰值
    peaks, properties = find_peaks(
        magnitude_norm,
        prominence=min_prominence,
        distance=max(10, N // 1000)  # 最小峰间距
    )
    
    # 按幅值排序
    peak_magnitudes = magnitude_norm[peaks]
    sorted_indices = np.argsort(peak_magnitudes)[::-1]
    top_peaks = peaks[sorted_indices[:num_peaks]]
    
    # 获取对应的频率
    interference_freqs = freqs_pos[top_peaks].tolist()
    
    # 排除故障特征频率（如果提供）
    if fault_frequencies is not None:
        tolerance = 5.0  # Hz，频率容差
        filtered_freqs = []
        for freq in interference_freqs:
            is_fault_freq = False
            for fault_freq in fault_frequencies:
                if abs(freq - fault_freq) < tolerance:
                    is_fault_freq = True
                    break
            if not is_fault_freq:
                filtered_freqs.append(freq)
        interference_freqs = filtered_freqs
    
    logger.debug(f"检测到 {len(interference_freqs)} 个干扰频率: {interference_freqs[:5]}")
    
    return interference_freqs


def calculate_fitness(
    ica_components: np.ndarray,
    fs: float,
    fault_frequencies: List[float],
    interference_frequencies: List[float],
    frequency_tolerance: float = 5.0
) -> float:
    """
    计算适应度函数 F（公式 3-23）
    
    目标：最大化故障特征频率幅值占比
    F = ΣA_i / (ΣA_i + ΣB_j)
    
    Args:
        ica_components: ICA分离后的独立分量数组 (n_components, length)
        fs: 采样频率（Hz）
        fault_frequencies: 故障特征频率列表（Hz）
        interference_frequencies: 干扰频率列表（Hz）
        frequency_tolerance: 频率匹配容差（Hz）
    
    Returns:
        fitness: 适应度值 F ∈ [0, 1]，越大越好
    
    References:
        公式 (3-23): F = ΣA_i / (ΣA_i + ΣB_j)
        其中 A_i 为故障特征频率幅值，B_j 为干扰频率幅值
    """
    if ica_components.ndim == 1:
        ica_components = ica_components.reshape(1, -1)
    
    n_components = ica_components.shape[0]
    N = ica_components.shape[1]
    
    # 计算所有分量的平均频谱
    avg_spectrum = np.zeros(N // 2)
    for i in range(n_components):
        Y = scipy_fft.fft(ica_components[i])
        avg_spectrum += np.abs(cast(np.ndarray, Y[:N//2]))
    avg_spectrum /= n_components
    
    # 获取频率轴
    freqs = scipy_fft.fftfreq(N, d=1/fs)[:N//2]
    
    # 计算故障特征频率的幅值 A_i
    total_fault_amplitude = 0.0
    for fault_freq in fault_frequencies:
        # 找到最接近的频率索引
        freq_indices = np.where(
            np.abs(freqs - fault_freq) < frequency_tolerance
        )[0]
        if len(freq_indices) > 0:
            total_fault_amplitude += np.sum(avg_spectrum[freq_indices])
    
    # 计算干扰频率的幅值 B_j
    total_interference_amplitude = 0.0
    for interf_freq in interference_frequencies:
        freq_indices = np.where(
            np.abs(freqs - interf_freq) < frequency_tolerance
        )[0]
        if len(freq_indices) > 0:
            total_interference_amplitude += np.sum(avg_spectrum[freq_indices])
    
    # 计算适应度 F
    denominator = total_fault_amplitude + total_interference_amplitude
    if denominator == 0:
        fitness = 0.0
    else:
        fitness = total_fault_amplitude / denominator
    
    logger.debug(
        f"适应度计算: F = {fitness:.4f}, "
        f"A_fault = {total_fault_amplitude:.4f}, "
        f"B_interference = {total_interference_amplitude:.4f}"
    )
    
    return float(fitness)


class PSO_Optimizer:
    """
    粒子群优化器（Particle Swarm Optimizer）
    
    用于优化LCD插值参数a和FastICA收敛阈值tol。
    
    核心机制：
    - 每个粒子代表一组参数 (a, tol)
    - 通过个体极值 p_best 和全局极值 g_best 引导搜索
    - 惯性权重 w 线性递减
    - 自适应更新频率（基于谱熵 Hs）
    """
    
    def __init__(
        self,
        n_particles: int = 20,
        max_iterations: int = 30,
        c1: float = 2.0,
        c2: float = 2.0,
        w_start: float = 0.9,
        w_end: float = 0.4,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        convergence_tol: float = 0.01,
        random_state: Optional[int] = None
    ):
        """
        初始化PSO优化器
        
        Args:
            n_particles: 粒子数量
            max_iterations: 最大迭代次数
            c1: 个体学习因子
            c2: 全局学习因子
            w_start: 初始惯性权重
            w_end: 最终惯性权重
            param_bounds: 参数边界字典
                默认: {'a': (0.5, 2.0), 'tol': (-6, -3)}  # tol使用对数尺度
            convergence_tol: 收敛容差（g_best波动阈值）
            random_state: 随机种子
        """
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.c1 = c1
        self.c2 = c2
        self.w_start = w_start
        self.w_end = w_end
        
        # 参数边界（tol使用对数尺度以避免数值问题）
        if param_bounds is None:
            self.param_bounds = {
                'a': (0.5, 2.0),      # LCD插值参数
                'log_tol': (-6, -3)    # FastICA收敛阈值的对数（10^-6 到 10^-3）
            }
        else:
            self.param_bounds = param_bounds
        
        self.convergence_tol = convergence_tol
        
        # 设置随机种子
        if random_state is not None:
            np.random.seed(random_state)
        
        # 初始化粒子状态
        self.particles: Optional[np.ndarray] = None  # 粒子位置 (n_particles, 2)
        self.velocities: Optional[np.ndarray] = None  # 粒子速度 (n_particles, 2)
        self.p_best: Optional[np.ndarray] = None  # 个体最优位置
        self.p_best_fitness: Optional[np.ndarray] = None  # 个体最优适应度
        self.g_best: Optional[np.ndarray] = None  # 全局最优位置
        self.g_best_fitness: float = -np.inf  # 全局最优适应度
        
        # 收敛历史
        self.convergence_history: List[float] = []
        self.g_best_history: List[np.ndarray] = []
    
    def initialize_particles(self):
        """初始化粒子位置和速度"""
        n_dim = 2  # (a, log_tol)
        
        # 随机初始化位置
        self.particles = np.random.uniform(
            low=[self.param_bounds['a'][0], self.param_bounds['log_tol'][0]],
            high=[self.param_bounds['a'][1], self.param_bounds['log_tol'][1]],
            size=(self.n_particles, n_dim)
        )
        
        # 初始化速度为0
        self.velocities = np.zeros((self.n_particles, n_dim))
        
        # 初始化个体最优
        self.p_best = self.particles.copy()
        self.p_best_fitness = np.full(self.n_particles, -np.inf)
        
        logger.info(f"PSO粒子初始化完成: {self.n_particles} 个粒子")
    
    def clamp_position(self, position: np.ndarray) -> np.ndarray:
        """
        约束粒子位置在边界内
        
        Args:
            position: 粒子位置 (2,)
        
        Returns:
            clamped_position: 约束后的位置
        """
        clamped = position.copy()
        clamped[0] = np.clip(clamped[0], 
                            self.param_bounds['a'][0], 
                            self.param_bounds['a'][1])
        clamped[1] = np.clip(clamped[1], 
                            self.param_bounds['log_tol'][0], 
                            self.param_bounds['log_tol'][1])
        return clamped
    
    def update_velocity_and_position(self, iteration: int):
        """
        更新粒子速度和位置（公式 3-24, 3-25）
        
        Args:
            iteration: 当前迭代次数（用于计算惯性权重）
        """
        # 确保粒子已初始化
        if self.particles is None or self.velocities is None:
            raise RuntimeError("粒子未初始化，请先调用 initialize_particles()")
        
        # 线性递减惯性权重
        w = self.w_start - (self.w_start - self.w_end) * iteration / self.max_iterations
        
        # 随机数
        r1 = np.random.rand(self.n_particles, 2)
        r2 = np.random.rand(self.n_particles, 2)
        
        # 更新速度（公式 3-24）
        # v_{k+1} = w * v_k + c1 * r1 * (p_best - x_k) + c2 * r2 * (g_best - x_k)
        if self.p_best is not None and self.g_best is not None:
            self.velocities = (
                w * self.velocities +
                self.c1 * r1 * (self.p_best - self.particles) +
                self.c2 * r2 * (self.g_best - self.particles)
            )
        
        # 限制速度范围（防止粒子飞离）
        max_velocity = 0.5 * (
            np.array([
                self.param_bounds['a'][1] - self.param_bounds['a'][0],
                self.param_bounds['log_tol'][1] - self.param_bounds['log_tol'][0]
            ])
        )
        self.velocities = np.clip(self.velocities, -max_velocity, max_velocity)
        
        # 更新位置（公式 3-25）
        # x_{k+1} = x_k + v_{k+1}
        self.particles = self.particles + self.velocities
        
        # 约束位置在边界内
        for i in range(self.n_particles):
            self.particles[i] = self.clamp_position(self.particles[i])
    
    def evaluate_fitness(
        self,
        fitness_func,
        signal: np.ndarray,
        fs: float,
        fault_frequencies: List[float],
        interference_frequencies: List[float]
    ) -> np.ndarray:
        """
        评估所有粒子的适应度
        
        Args:
            fitness_func: 适应度计算函数
                签名: fitness_func(a, tol, signal, fs, fault_freqs, interf_freqs) -> float
            signal: 输入信号
            fs: 采样频率
            fault_frequencies: 故障特征频率
            interference_frequencies: 干扰频率
        
        Returns:
            fitness_values: 所有粒子的适应度值
        """
        fitness_values = np.zeros(self.n_particles)
        
        for i in range(self.n_particles):
            # 获取参数
            if self.particles is not None:
                a = self.particles[i, 0]
                log_tol = self.particles[i, 1]
                tol = 10 ** log_tol  # 转换回线性尺度
                
                try:
                    # 计算适应度
                    fitness = fitness_func(a, tol, signal, fs, fault_frequencies, interference_frequencies)
                    fitness_values[i] = fitness
                except Exception as e:
                    logger.warning(f"粒子 {i} 适应度计算失败: {str(e)}")
                    fitness_values[i] = 0.0
        
        return fitness_values
    
    def optimize(
        self,
        fitness_func,
        signal: np.ndarray,
        fs: float,
        fault_frequencies: List[float],
        interference_frequencies: List[float],
        progress_callback=None
    ) -> Tuple[float, float, Dict]:
        """
        执行PSO优化
        
        Args:
            fitness_func: 适应度计算函数
            signal: 输入信号
            fs: 采样频率
            fault_frequencies: 故障特征频率列表（Hz）
            interference_frequencies: 干扰频率列表（Hz）
            progress_callback: 进度回调函数 callback(iteration, max_iterations, fitness)
        
        Returns:
            best_a: 最优LCD插值参数
            best_tol: 最优FastICA收敛阈值
            optimization_info: 优化信息字典
        """
        logger.info(f"开始PSO优化: {self.n_particles} 个粒子, {self.max_iterations} 次迭代")
        
        # 初始化粒子
        self.initialize_particles()
        
        # 优化循环
        for iteration in range(self.max_iterations):
            # 评估适应度
            fitness_values = self.evaluate_fitness(
                fitness_func, signal, fs, fault_frequencies, interference_frequencies
            )
            
            # 更新个体最优
            if self.p_best_fitness is not None and self.p_best is not None and self.particles is not None:
                for i in range(self.n_particles):
                    if fitness_values[i] > self.p_best_fitness[i]:
                        self.p_best_fitness[i] = fitness_values[i]
                        self.p_best[i] = self.particles[i].copy()
            
            # 更新全局最优
            if self.particles is not None:
                best_idx = np.argmax(fitness_values)
                if fitness_values[best_idx] > self.g_best_fitness:
                    self.g_best_fitness = fitness_values[best_idx]
                    self.g_best = self.particles[best_idx].copy()
            
            # 记录收敛历史
            self.convergence_history.append(self.g_best_fitness)
            if self.g_best is not None:
                self.g_best_history.append(self.g_best.copy())
            
            # 进度回调
            if progress_callback:
                progress_callback(iteration + 1, self.max_iterations, self.g_best_fitness)
            
            if self.g_best is not None:
                logger.debug(
                    f"Iteration {iteration + 1}/{self.max_iterations}: "
                    f"g_best_fitness = {self.g_best_fitness:.4f}, "
                    f"a = {self.g_best[0]:.4f}, "
                    f"tol = {10**self.g_best[1]:.2e}"
                )
            
            # 检查收敛
            if iteration > 5:
                recent_fitness = self.convergence_history[-5:]
                fitness_variation = (max(recent_fitness) - min(recent_fitness)) / (abs(max(recent_fitness)) + 1e-10)
                if fitness_variation < self.convergence_tol:
                    logger.info(f"PSO提前收敛于迭代 {iteration + 1}")
                    break
            
            # 更新速度和位置
            self.update_velocity_and_position(iteration)
        
        # 获取最优参数
        if self.g_best is None:
            raise RuntimeError("PSO优化失败：全局最优解未找到")
        
        best_a = self.g_best[0]
        best_tol = 10 ** self.g_best[1]  # 转换回线性尺度
        
        optimization_info = {
            'best_a': best_a,
            'best_tol': best_tol,
            'best_fitness': self.g_best_fitness,
            'convergence_history': self.convergence_history,
            'n_iterations': len(self.convergence_history),
            'param_bounds': self.param_bounds
        }
        
        logger.info(
            f"PSO优化完成: 最优参数 a* = {best_a:.4f}, tol* = {best_tol:.2e}, "
            f"F* = {self.g_best_fitness:.4f}"
        )
        
        return best_a, best_tol, optimization_info


def adaptive_update_frequency(spectral_entropy: float) -> int:
    """
    根据谱熵动态调整参数更新频率
    
    Args:
        spectral_entropy: 信号谱熵值
    
    Returns:
        update_interval: 更新间隔（每多少个批次更新一次参数）
            Hs高 → 间隔短（频繁更新）
            Hs低 → 间隔长（减少更新）
    """
    # 谱熵典型范围：2-12（取决于信号复杂度）
    # 归一化到 [0, 1]
    H_normalized = np.clip((spectral_entropy - 2.0) / 10.0, 0.0, 1.0)
    
    # 映射到更新间隔 [1, 10]
    # Hs高（复杂）→ 间隔=1（每次更新）
    # Hs低（简单）→ 间隔=10（每10次更新）
    update_interval = int(np.round(1 + 9 * (1 - H_normalized)))
    
    logger.debug(f"谱熵 H_s = {spectral_entropy:.4f}, 更新间隔 = {update_interval}")
    
    return update_interval


def optimize_lcd_fastica_params(
    signal: np.ndarray,
    fs: float,
    fault_frequencies: Optional[List[float]] = None,
    pso_config: Optional[Dict] = None,
    progress_callback=None
) -> Tuple[float, float, Dict]:
    """
    便捷函数：优化LCD和FastICA参数
    
    Args:
        signal: 输入信号（一维数组）
        fs: 采样频率（Hz）
        fault_frequencies: 故障特征频率列表（Hz），如 [50, 100, 150]
        pso_config: PSO配置字典，可选
        progress_callback: 进度回调函数
    
    Returns:
        best_a: 最优LCD插值参数
        best_tol: 最优FastICA收敛阈值
        optimization_info: 优化信息字典
    
    Example:
        >>> signal = np.load('data/signal.npy')
        >>> a, tol, info = optimize_lcd_fastica_params(signal, fs=20000, fault_freqs=[50, 100])
        >>> print(f"最优参数: a={a:.4f}, tol={tol:.2e}")
    """
    # 默认PSO配置
    if pso_config is None:
        pso_config = {
            'n_particles': 20,
            'max_iterations': 30,
            'c1': 2.0,
            'c2': 2.0,
            'w_start': 0.9,
            'w_end': 0.4,
            'random_state': 42
        }
    
    # 计算谱熵
    logger.info("计算信号谱熵...")
    spectral_entropy = calculate_spectral_entropy(signal, fs)
    
    # 检测干扰频率
    logger.info("检测干扰频率...")
    interference_freqs = detect_interference_frequencies(signal, fs, fault_frequencies)
    
    # 如果没有提供故障频率，使用空列表
    if fault_frequencies is None:
        fault_frequencies = []
        logger.warning("未提供故障特征频率，适应度计算可能不准确")
    
    # 定义适应度计算函数（需要导入LCD和FastICA）
    def fitness_func(a, tol, sig, sampling_rate, fault_freqs, interf_freqs):
        """
        适应度计算函数（简化版）
        
        注意：这里需要调用实际的LCD-FastICA处理流程，
        为简化起见，这里使用频谱分析作为代理适应度。
        实际使用时应替换为完整的处理管道。
        """
        # 这里简化处理：直接使用原始信号的频谱特征
        # 实际应用中应该：
        # 1. 使用参数a执行LCD分解
        # 2. 使用参数tol执行FastICA
        # 3. 计算独立分量的适应度
        
        # 简化版：计算信号在故障频率处的能量占比
        N = len(sig)
        Y = scipy_fft.fft(sig)
        Y_half = Y[:N//2]
        Y_magnitude = np.abs(cast(np.ndarray, Y_half))
        freqs = scipy_fft.fftfreq(N, d=1/fs)[:N//2]
        
        # 故障频率能量
        fault_energy = 0.0
        for f in fault_freqs:
            idx = np.argmin(np.abs(freqs - f))
            fault_energy += Y_magnitude[idx]
        
        # 干扰频率能量
        interference_energy = 0.0
        for f in interf_freqs:
            idx = np.argmin(np.abs(freqs - f))
            interference_energy += Y_magnitude[idx]
        
        # 适应度
        total_energy = fault_energy + interference_energy
        if total_energy == 0:
            return 0.0
        return fault_energy / total_energy
    
    # 创建PSO优化器
    optimizer = PSO_Optimizer(**pso_config)
    
    # 执行优化
    best_a, best_tol, info = optimizer.optimize(
        fitness_func=fitness_func,
        signal=signal,
        fs=fs,
        fault_frequencies=fault_frequencies,
        interference_frequencies=interference_freqs,
        progress_callback=progress_callback
    )
    
    # 添加谱熵信息
    info['spectral_entropy'] = spectral_entropy
    info['interference_frequencies'] = interference_freqs
    info['update_interval'] = adaptive_update_frequency(spectral_entropy)
    
    return best_a, best_tol, info


if __name__ == "__main__":
    """
    测试脚本：验证PSO优化器功能
    """
    print("=" * 60)
    print("PSO优化器测试")
    print("=" * 60)
    
    # 生成测试信号
    fs = 20000
    t = np.linspace(0, 1, fs)
    
    # 故障特征频率：50 Hz, 100 Hz
    # 干扰频率：60 Hz（工频）, 120 Hz
    fault_freqs = [50.0, 100.0]
    interference_freqs = [60.0, 120.0]
    
    signal = (
        1.0 * np.sin(2 * np.pi * 50 * t) +
        0.5 * np.sin(2 * np.pi * 100 * t) +
        0.3 * np.sin(2 * np.pi * 60 * t) +
        0.2 * np.sin(2 * np.pi * 120 * t) +
        0.1 * np.random.randn(len(t))
    )
    
    print(f"\n测试信号参数:")
    print(f"  采样率: {fs} Hz")
    print(f"  长度: {len(signal)} 采样点")
    print(f"  故障频率: {fault_freqs} Hz")
    print(f"  干扰频率: {interference_freqs} Hz")
    
    # 测试谱熵计算
    print("\n" + "-" * 60)
    print("测试1: 谱熵计算")
    H_s = calculate_spectral_entropy(signal, fs)
    print(f"谱熵 H_s = {H_s:.4f}")
    
    # 测试干扰频率检测
    print("\n" + "-" * 60)
    print("测试2: 干扰频率检测")
    detected_interference = detect_interference_frequencies(
        signal, fs, fault_frequencies=fault_freqs
    )
    print(f"检测到 {len(detected_interference)} 个干扰频率:")
    print(f"  {detected_interference}")
    
    # 测试自适应更新频率
    print("\n" + "-" * 60)
    print("测试3: 自适应更新频率")
    update_interval = adaptive_update_frequency(H_s)
    print(f"谱熵 H_s = {H_s:.4f} → 更新间隔 = {update_interval}")
    
    # 测试PSO优化（简化版）
    print("\n" + "-" * 60)
    print("测试4: PSO优化（简化适应度函数）")
    
    def progress_callback(iteration, max_iter, fitness):
        if iteration % 5 == 0:
            print(f"  迭代 {iteration}/{max_iter}: F = {fitness:.4f}")
    
    best_a, best_tol, info = optimize_lcd_fastica_params(
        signal=signal,
        fs=fs,
        fault_frequencies=fault_freqs,
        pso_config={'n_particles': 10, 'max_iterations': 15, 'random_state': 42},
        progress_callback=progress_callback
    )
    
    print(f"\n优化结果:")
    print(f"  最优参数 a* = {best_a:.4f}")
    print(f"  最优参数 tol* = {best_tol:.2e}")
    print(f"  最优适应度 F* = {info['best_fitness']:.4f}")
    print(f"  迭代次数 = {info['n_iterations']}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
