import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nptdms import TdmsFile
from scipy.io import savemat
from scipy.signal import butter, lfilter
import time
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging

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


def linear_transform(x, t, m):
    """
    线性变换函数，用于构建基线
    """
    dt = t.max() - t.min()
    if dt == 0:
        return np.zeros_like(x)  # 防止除以零
    return (x - m) * (t - t.min()) / dt + m

def local_characteristic_scale_decomposition(x, t, num_components=10):
    """
    局部特征尺度分解函数
    """
    m = np.mean(x)  # 计算信号的平均值
    x -= m  # 去中心化
    isc_components = []
    max_iterations = 1000  # 设置最大迭代次数以防止无限循环
    
    for iteration in tqdm(range(max_iterations), desc="Local Characteristic Scale Decomposition"):
        # 寻找极值点
        extrema_indices = np.where((np.diff(np.sign(np.diff(x)))) != 0)[0] + 1
        
        if len(extrema_indices) < 4:  # 确保有足够的极值点
            break
        
        # 构建基线
        baseline = np.zeros_like(x)
        for i in range(len(extrema_indices) // 2):
            max_idx = extrema_indices[2 * i]
            min_idx = extrema_indices[2 * i + 1]
            baseline[max_idx:min_idx] = linear_transform(x[max_idx:min_idx], t[max_idx:min_idx], m)
        
        # 提取ISC分量
        isc = x - baseline
        isc_components.append(isc)
        
        # 检查是否已经达到所需的ISC分量数量或满足极值单调性判据
        if len(isc_components) >= num_components: # or extreme_monotonicity_criterion(isc):
            break
        
        # 更新信号
        x = baseline - np.mean(isc)
    
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
    """
    if not VMD_AVAILABLE:
        raise ImportError("vmdpy 库未安装，无法执行 VMD 分解")
    
    DC = 0
    init = 1
    tol = 1e-7
    
    # 执行 VMD 分解
    imfs, u, u_hat = VMD(signal, alpha, tau, K, DC, init, tol)
    
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

def eemd_decomposition(signal, fs, max_imf=3):
    """
    集成经验模态分解 (EEMD)
    
    Args:
        signal: 输入信号
        fs: 采样频率
        max_imf: 最大 IMF 数量
    
    Returns:
        imfs: 分解得到的本征模态函数数组
    """
    if not EEMD_AVAILABLE:
        raise ImportError("PyEMD 库未安装，无法执行 EEMD 分解")
    
    eemd = EEMD(max_imf=max_imf)
    imfs = eemd(signal)
    
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

def lmd_decomposition(signal, fs):
    """
    局部均值分解 (LMD)
    
    Args:
        signal: 输入信号
        fs: 采样频率
    
    Returns:
        imfs: 分解得到的本征模态函数数组
    """
    if not LMD_AVAILABLE:
        raise ImportError("PyLMD 库未安装，无法执行 LMD 分解")
    
    lmd = LMD()
    result = lmd.lmd(signal)
    # LMD 返回的结果是元组，第一个元素是 IMF 数组
    imfs = result[0]
    
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

def select_correlated_components(imfs, signal, threshold=0.5):
    """
    根据相关系数选择有效的分解分量
    
    Args:
        imfs: 分解得到的本征模态函数数组
        signal: 原始信号
        threshold: 相关系数阈值
    
    Returns:
        selected_imfs: 选中的分量数组
    """
    # 计算每个 IMF 与原始信号的相关系数
    correlation_coeffs = [np.corrcoef(signal, imfs[i])[0, 1] for i in range(len(imfs))]
    correlation_coeffs = np.array(correlation_coeffs)
    
    # 选择相关系数大于阈值的 IMF
    selected_indices = [i for i, coeff in enumerate(correlation_coeffs) if coeff > threshold]
    selected_imfs = imfs[selected_indices]
    
    # 如果没有相关系数大于阈值的 IMF，则选择相关系数最大的那个 IMF
    if selected_imfs.shape[0] == 0:
        max_corr_index = np.argmax(correlation_coeffs)
        selected_imfs = imfs[max_corr_index:max_corr_index + 1]
        print(f"没有相关系数大于 {threshold} 的分量，选择相关系数最大的分量 (索引：{max_corr_index})")
    
    return selected_imfs


if __name__ == "__main__":
    """
    测试脚本：验证 VMD、EEMD、LMD 方法是否可用
    """
    print("=" * 60)
    print("信号处理方法可用性检查")
    print("=" * 60)
    
    print(f"\nVMD 方法：{'✓ 可用' if VMD_AVAILABLE else '✗ 不可用 (需要安装 vmdpy)'}")
    print(f"EEMD 方法：{'✓ 可用' if EEMD_AVAILABLE else '✗ 不可用 (需要安装 PyEMD)'}")
    print(f"LMD 方法：{'✓ 可用' if LMD_AVAILABLE else '✗ 不可用 (需要安装 PyLMD)'}")
    
    print("\n" + "=" * 60)
    print("可用的处理方法列表:")
    print("  - None (无处理)")
    print("  - LCD (局部特征尺度分解)")
    print("  - FastICA (快速独立成分分析)")
    if VMD_AVAILABLE:
        print("  - VMD (变分模态分解)")
    if EEMD_AVAILABLE:
        print("  - EEMD (集成经验模态分解)")
    if LMD_AVAILABLE:
        print("  - LMD (局部均值分解)")
    print("=" * 60)
    
    # 生成一个简单的测试信号
    print("\n生成测试信号...")
    fs = 20000  # 采样率 20kHz
    t = np.linspace(0, 1, fs)  # 1 秒的信号
    
    # 创建一个包含多个频率成分的复合信号
    f1, f2, f3 = 50, 200, 500  # 信号频率
    test_signal = (0.5 * np.sin(2 * np.pi * f1 * t) + 
                   0.3 * np.sin(2 * np.pi * f2 * t) + 
                   0.2 * np.sin(2 * np.pi * f3 * t))
    
    # 添加一些噪声
    noise = 0.1 * np.random.randn(len(t))
    test_signal += noise
    
    print(f"测试信号长度：{len(test_signal)} 采样点")
    print(f"测试信噪比：{np.std(test_signal - noise) / np.std(noise):.2f}")
    
    # 测试 VMD 方法
    if VMD_AVAILABLE:
        print("\n" + "-" * 60)
        print("测试 VMD 分解...")
        try:
            imfs_vmd = vmd_decomposition(test_signal, fs, K=5)
            print(f"✓ VMD 分解成功，得到 {imfs_vmd.shape[0]} 个 IMF 分量")
            selected_vmd = select_correlated_components(imfs_vmd, test_signal)
            print(f"✓ 选择 {selected_vmd.shape[0]} 个相关分量")
        except Exception as e:
            print(f"✗ VMD 测试失败：{e}")
    
    # 测试 EEMD 方法
    if EEMD_AVAILABLE:
        print("\n" + "-" * 60)
        print("测试 EEMD 分解...")
        try:
            imfs_eemd = eemd_decomposition(test_signal, fs, max_imf=5)
            print(f"✓ EEMD 分解成功，得到 {imfs_eemd.shape[0]} 个 IMF 分量")
            selected_eemd = select_correlated_components(imfs_eemd, test_signal)
            print(f"✓ 选择 {selected_eemd.shape[0]} 个相关分量")
        except Exception as e:
            print(f"✗ EEMD 测试失败：{e}")
    
    # 测试 LMD 方法
    if LMD_AVAILABLE:
        print("\n" + "-" * 60)
        print("测试 LMD 分解...")
        try:
            imfs_lmd = lmd_decomposition(test_signal, fs)
            print(f"✓ LMD 分解成功，得到 {imfs_lmd.shape[0]} 个 PF 分量")
            selected_lmd = select_correlated_components(imfs_lmd, test_signal)
            print(f"✓ 选择 {selected_lmd.shape[0]} 个相关分量")
        except Exception as e:
            print(f"✗ LMD 测试失败：{e}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

def fast_ica_processing(file_path, output_path, sampling_rate, num_components=10, 
                       max_samples=None, progress_callback=None):
    """
    执行 LCD-FASTICA 处理的主函数（向后兼容接口）
    
    Args:
        file_path: 输入信号文件路径 (.npy)
        output_path: 输出文件路径 (.mat)
        sampling_rate: 采样率 (Hz)
        num_components: 分解分量数量
        max_samples: 最大采样点数
        progress_callback: 进度回调函数 callback(current_step, total_steps, message)
    """
    def report_progress(step, message):
        if progress_callback:
            progress_callback(step, 4, message)
        logger.info(message)
    
    try:
        report_progress(1, "读取 NPY 文件")
        part_channel_data = np.load(file_path)
        if max_samples is not None:
            part_channel_data = part_channel_data[:max_samples]
        
        np_channel_data = part_channel_data
        fs = sampling_rate
        N = len(np_channel_data)
        n = np.arange(N)
        t = n / fs

        # Design low-pass filter
        report_progress(2, "应用低通滤波器")
        cutoff = 5000
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        
        # Fix: Handle scipy.signal.butter return value safely
        butter_result = butter(5, normal_cutoff, btype='low', analog=False)
        # Ensure we get exactly 2 values (b, a)
        if isinstance(butter_result, tuple) and len(butter_result) == 2:
            b, a = butter_result
        else:
            raise RuntimeError("butter() returned unexpected format")
        
        x = lfilter(b, a, np_channel_data)

        report_progress(3, f"执行 LCD 分解 (分量数：{num_components})")
        isc_components = local_characteristic_scale_decomposition(x, t, num_components=num_components)

        # Fix: Replace print statements with logging
        logger.info(f"生成的 ISC 分量数量：{len(isc_components)}")
        logger.debug(f"原信号：Min={np.min(x)}, Max={np.max(x)}, Mean={np.mean(x)}")

        # Calculate and log frequency information
        Y = np.fft.fft(x, 2 ** int(np.ceil(np.log2(len(x))))) / len(x)
        f = fs / 2 * np.linspace(0, 1, len(Y) // 2)
        max_freq_x = f[np.argmax(2 * np.abs(Y[:len(Y) // 2]))]
        logger.debug(f"原信号的最大频率：{max_freq_x} Hz")

        for i, isc in enumerate(isc_components):
            Y_isc = np.fft.fft(isc, 2 ** int(np.ceil(np.log2(len(isc))))) / len(isc)
            max_freq_isc = f[np.argmax(2 * np.abs(Y_isc[:len(Y_isc) // 2]))]
            logger.debug(f"ISC Component {i+1}: Min={np.min(isc)}, Max={np.max(isc)}, Mean={np.mean(isc)}")
            logger.debug(f"ISC Component {i+1} 的最大频率：{max_freq_isc} Hz")

        correlation_coefficients = [np.corrcoef(x, isc)[0, 1] for isc in isc_components]
        for i, corr in enumerate(correlation_coefficients):
            logger.debug(f"ISC Component {i+1} 与原信号的相关系数：{corr}")

        reconstructed_signal = np.sum(isc_components, axis=0)
        reconstruction_error = np.linalg.norm(x - reconstructed_signal, ord=2)
        logger.info(f"重构误差：{reconstruction_error}")

        # Fix: Use column_stack with proper list argument
        isc_components_array = np.column_stack(isc_components)
        isc_component_1 = isc_components_array[:, 0]
        S = np.vstack((isc_component_1, x)).T

        imputer = SimpleImputer(strategy='mean')
        S_imputed = imputer.fit_transform(S)
        
        # Fix: Remove redundant centering and whitening - FastICA handles this internally
        # Only keep imputation as FastICA requires clean data
        logger.debug(f"预处理完成，形状：{S_imputed.shape}")

        report_progress(4, "执行 FastICA 分离")
        num_components = 2
        ica = FastICA(n_components=num_components, random_state=0, tol=1e-4, max_iter=500)

        # Fix: Remove unsafe dynamic attribute access - use standard tqdm without callback
        logger.debug("开始 FastICA 迭代...")
        signal_ica = ica.fit_transform(S_imputed)
        logger.info(f"FastICA 完成，输出 {signal_ica.shape[1]} 个独立成分")

        ica_result_dict = {
            'ICA_Components': signal_ica,
            'Estimated_Mixing_Matrix': ica.mixing_
        }
        savemat(output_path, ica_result_dict)
        print(f'ICA results saved successfully to {output_path}')
        
        report_progress(4, "处理完成")

    except Exception as e:
        logger.error(f"fast_ica_processing 失败：{str(e)}")
        raise

def process_signal_pipeline(file_path, output_path, sampling_rate, processing_methods, 
                           max_samples=None, num_components=10, progress_callback=None):
    """
    灵活的信号处理管道，支持单一分解方法 + 可选 FastICA
    
    Args:
        file_path: 输入信号文件路径 (.npy)
        output_path: 输出文件路径 (.mat)
        sampling_rate: 采样率 (Hz)
        processing_methods: 处理方法列表，仅允许 ['分解方法'] 或 ['分解方法', 'FastICA']
        max_samples: 最大采样点数
        num_components: 分解/分离的分量数量
        progress_callback: 进度回调函数 callback(current_step, total_steps, message)
    
    Returns:
        processed_signal: 处理后的信号数组
        processing_info: 处理信息字典
    
    Raises:
        ValueError: 当处理方法组合不合法时
        ImportError: 当所需依赖未安装时
        RuntimeError: 当处理过程失败时
    """
    def report_progress(step, message):
        """Helper function to report progress"""
        if progress_callback:
            progress_callback(step, len(processing_methods), message)
        logger.info(f"Step {step}/{len(processing_methods)}: {message}")
    
    # Validate processing methods combination
    decomposition_methods = ['LCD', 'VMD', 'EEMD', 'LMD']
    allowed_methods = decomposition_methods + ['FastICA']
    
    # Check 1: Validate all methods are allowed
    for method in processing_methods:
        if method not in allowed_methods:
            raise ValueError(f"不支持的处理方法：{method}")
    
    # Check 2: Ensure only one decomposition method
    decomp_count = sum(1 for method in processing_methods if method in decomposition_methods)
    if decomp_count == 0:
        raise ValueError("必须选择一种分解方法（LCD、VMD、EEMD 或 LMD）")
    if decomp_count > 1:
        raise ValueError("只允许使用一种分解方法，禁止多种分解方法串联")
    
    # Check 3: Ensure FastICA is only used after decomposition method
    if 'FastICA' in processing_methods:
        if processing_methods[0] == 'FastICA':
            raise ValueError("FastICA 必须在分解方法之后使用")
        if len(processing_methods) > 2:
            raise ValueError("最多只能使用两种方法：一种分解方法 + FastICA")
    
    # Read signal
    report_progress(0, "读取信号文件")
    try:
        signal = np.load(file_path)
    except Exception as e:
        raise RuntimeError(f"读取 NPY 文件失败：{str(e)}")
    
    if max_samples is not None:
        signal = signal[:max_samples]
    
    fs = sampling_rate
    N = len(signal)
    t = np.arange(N) / fs
    
    current_signal = signal.copy()
    processing_steps = []
    
    # Apply processing methods sequentially
    for idx, method in enumerate(processing_methods):
        try:
            if method == 'LCD':
                report_progress(idx + 1, f"执行 LCD 分解 (目标分量数：{num_components})")
                isc_components = local_characteristic_scale_decomposition(
                    current_signal, t, num_components=num_components
                )
                
                # Limit the number of components to prevent dimension explosion
                n_components_to_use = min(len(isc_components), num_components)
                isc_components = isc_components[:n_components_to_use]
                
                current_signal = np.column_stack(isc_components)
                processing_steps.append(f"LCD({n_components_to_use})")
                logger.info(f"LCD 完成，生成 {n_components_to_use} 个分量")
                
            elif method == 'VMD':
                if not VMD_AVAILABLE:
                    raise ImportError("vmdpy 库未安装，无法执行 VMD 分解。请运行：pip install vmdpy")
                
                report_progress(idx + 1, f"执行 VMD 分解 (模态数：{num_components})")
                imfs = vmd_decomposition(current_signal, fs, K=num_components)
                
                selected_imfs = select_correlated_components(imfs, current_signal)
                n_components_to_use = min(len(selected_imfs), num_components)
                selected_imfs = selected_imfs[:n_components_to_use]
                
                # Fix: Ensure selected_imfs is a list of arrays before column_stack
                current_signal = np.column_stack(list(selected_imfs))
                processing_steps.append(f"VMD({n_components_to_use})")
                logger.info(f"VMD 完成，选中 {n_components_to_use} 个分量")
                
            elif method == 'EEMD':
                if not EEMD_AVAILABLE:
                    raise ImportError("PyEMD 库未安装，无法执行 EEMD 分解。请运行：pip install PyEMD")
                
                report_progress(idx + 1, f"执行 EEMD 分解 (最大 IMF 数：{num_components})")
                imfs = eemd_decomposition(current_signal, fs, max_imf=num_components)
                
                selected_imfs = select_correlated_components(imfs, current_signal)
                n_components_to_use = min(len(selected_imfs), num_components)
                selected_imfs = selected_imfs[:n_components_to_use]
                
                # Fix: Ensure selected_imfs is a list of arrays before column_stack
                current_signal = np.column_stack(list(selected_imfs))
                processing_steps.append(f"EEMD({n_components_to_use})")
                logger.info(f"EEMD 完成，选中 {n_components_to_use} 个分量")
                
            elif method == 'LMD':
                if not LMD_AVAILABLE:
                    raise ImportError("PyLMD 库未安装，无法执行 LMD 分解。请运行：pip install PyLMD")
                
                report_progress(idx + 1, "执行 LMD 分解")
                imfs = lmd_decomposition(current_signal, fs)
                
                selected_imfs = select_correlated_components(imfs, current_signal)
                n_components_to_use = min(len(selected_imfs), num_components)
                selected_imfs = selected_imfs[:n_components_to_use]
                
                # Fix: Ensure selected_imfs is a list of arrays before column_stack
                current_signal = np.column_stack(list(selected_imfs))
                processing_steps.append(f"LMD({n_components_to_use})")
                logger.info(f"LMD 完成，选中 {n_components_to_use} 个分量")
                
            elif method == 'FastICA':
                # Check if signal has enough channels for FastICA
                if current_signal.ndim == 1 or current_signal.shape[1] < 2:
                    logger.warning("单通道信号无法执行 FastICA，跳过此步骤")
                    processing_steps.append("FastICA_Skipped")
                    continue
                
                report_progress(idx + 1, f"执行 FastICA 分离 (分量数：{min(num_components, current_signal.shape[1])})")
                
                # Handle NaN values
                imputer = SimpleImputer(strategy='mean')
                signal_imputed = imputer.fit_transform(current_signal)
                
                # FastICA handles centering and whitening internally
                # Only standardize if needed for numerical stability
                ica_n_components = min(num_components, signal_imputed.shape[1])
                ica = FastICA(n_components=ica_n_components, random_state=0, tol=1e-4, max_iter=500)
                signal_ica = ica.fit_transform(signal_imputed)
                
                current_signal = signal_ica
                processing_steps.append(f"FastICA({current_signal.shape[1]})")
                logger.info(f"FastICA 完成，输出 {current_signal.shape[1]} 个成分")
        
        except Exception as e:
            logger.error(f"处理步骤 {method} 失败：{str(e)}")
            raise RuntimeError(f"处理方法 '{method}' 执行失败：{str(e)}") from e
    
    # Ensure output is 2D array
    if current_signal.ndim == 1:
        current_signal = current_signal.reshape(-1, 1)
    
    processing_info = {
        'input_shape': signal.shape,
        'output_shape': current_signal.shape,
        'processing_steps': processing_steps,
        'sampling_rate': sampling_rate,
        'time_vector': t
    }
    
    # Save results
    ica_result_dict = {
        'ICA_Components': current_signal,
        'Processing_Info': processing_info,
        'Processing_Steps': '_'.join(processing_steps)
    }
    
    try:
        savemat(output_path, ica_result_dict)
        logger.info(f'处理结果已保存至 {output_path}')
    except Exception as e:
        raise RuntimeError(f"保存 MAT 文件失败：{str(e)}") from e
    
    report_progress(len(processing_methods), "处理完成")
    
    return current_signal, processing_info
