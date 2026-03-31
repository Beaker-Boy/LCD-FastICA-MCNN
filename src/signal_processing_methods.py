# -*- coding: utf-8 -*-
"""
signal_processing_methods.py

This module contains various signal decomposition methods for the project.
It is assumed that the necessary libraries are installed:
- vmdpy
- EMD-signal (for PyEMD)
- pylmd (for PyLMD)

You can install them using pip:
pip install vmdpy
pip install EMD-signal
pip install pylmd
"""

import numpy as np
from tqdm import tqdm

# --- Helper function from original lcd_fastica.py ---
def linear_transform(x, t, m):
    """
    线性变换函数，用于构建基线
    """
    dt = t.max() - t.min()
    if dt == 0:
        return np.zeros_like(x)  # 防止除以零
    return (x - m) * (t - t.min()) / dt + m

# --- LCD Method ---
def do_lcd(signal, t, num_components=10):
    """
    Performs Local Characteristic Scale Decomposition (LCD).
    Adapted from the original lcd_fastica.py.

    Args:
        signal (np.ndarray): The input 1D signal.
        t (np.ndarray): The time vector corresponding to the signal.
        num_components (int): The number of components to decompose into.

    Returns:
        np.ndarray: A 2D array of shape (N, K) where N is signal length
                    and K is the number of components.
    """
    x = signal.copy()
    m = np.mean(x)
    x -= m  # a
    isc_components = []
    max_iterations = 1000

    for _ in tqdm(range(max_iterations), desc="Local Characteristic Scale Decomposition"):
        extrema_indices = np.where((np.diff(np.sign(np.diff(x)))) != 0)[0] + 1
        
        if len(extrema_indices) < 4:
            break
        
        baseline = np.zeros_like(x)
        for i in range(len(extrema_indices) // 2):
            max_idx = extrema_indices[2 * i]
            min_idx = extrema_indices[2 * i + 1]
            baseline[max_idx:min_idx] = linear_transform(x[max_idx:min_idx], t[max_idx:min_idx], m)
        
        isc = x - baseline
        isc_components.append(isc)
        
        if len(isc_components) >= num_components:
            break
        
        x = baseline - np.mean(isc)
    
    if not isc_components:
        return np.array([]).reshape(len(signal), 0)

    return np.column_stack(isc_components)

# --- VMD Method ---
def do_vmd(signal, num_components=5):
    """
    Performs Variational Mode Decomposition (VMD).

    Args:
        signal (np.ndarray): The input 1D signal.
        num_components (int): The number of modes to decompose into (K).

    Returns:
        np.ndarray: A 2D array of shape (N, K) containing the modes.
    """
    try:
        from vmdpy import VMD
    except ImportError:
        raise ImportError("VMD library not found. Please install it using 'pip install vmdpy'")

    # VMD parameters
    alpha = 2000  # moderate bandwidth constraint
    tau = 0.001   # noise-tolerance (no strict fidelity enforcement)
    K = num_components
    DC = 0        # no DC part imposed
    init = 1      # initialize omegas uniformly
    tol = 1e-7

    # Run VMD
    imfs, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
    
    # The VMD library returns (K, N), we need (N, K)
    return imfs.T

# --- EEMD Method ---
def do_eemd(signal, num_components=3):
    """
    Performs Ensemble Empirical Mode Decomposition (EEMD).

    Args:
        signal (np.ndarray): The input 1D signal.
        num_components (int): The maximum number of IMFs to find.

    Returns:
        np.ndarray: A 2D array of shape (N, K) containing the IMFs.
    """
    try:
        from PyEMD import EEMD
    except ImportError:
        raise ImportError("PyEMD library not found. Please install it using 'pip install EMD-signal'")

    eemd = EEMD(max_imf=num_components)
    imfs = eemd(signal)

    # PyEMD returns (K, N), we need (N, K)
    return imfs.T

# --- LMD Method ---
def do_lmd(signal, num_components=None):
    """
    Performs Local Mean Decomposition (LMD).
    Note: num_components is not directly used by PyLMD's lmd method,
    but is kept for consistent function signature.

    Args:
        signal (np.ndarray): The input 1D signal.
        num_components (int): Not used by the library, but kept for consistency.

    Returns:
        np.ndarray: A 2D array of shape (N, K) containing the PFs.
    """
    try:
        from PyLMD import LMD
    except ImportError:
        raise ImportError("PyLMD library not found. Please install it using 'pip install pylmd'")

    lmd = LMD()
    # The lmd method returns a tuple (imfs, residue)
    imfs, _ = lmd.lmd(signal)

    # PyLMD returns (K, N), we need (N, K)
    return imfs.T
