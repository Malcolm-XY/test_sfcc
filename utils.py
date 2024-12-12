# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 21:18:58 2024

@author: 18307
"""
import os
import pandas
import numpy
import h5py
import scipy

def get_label():
    # path
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    
    path_labels = os.path.join(path_parent, 'data', 'SEED', 'channel features', 'labels.txt')
    
    # read txt; original channel distribution
    labels = pandas.read_csv(path_labels, sep='\t', header=None).to_numpy().flatten()
    
    print('Labels Reading Done')
    
    return labels

def raed_labels(path_txt):
    # read txt; original channel distribution
    labels = pandas.read_csv(path_txt, sep='\t', header=None).to_numpy().flatten()
    
    print('Labels Reading Done')
    
    return labels

def get_distribution():
    # path
    path_current = os.getcwd()
   
    # path_smap
    path_smap = os.path.join(path_current, 'mapping', 'smap.txt')
    # path_channel order
    path_order = os.path.join(path_current, 'mapping', 'biosemi62_64_channels_original_distribution.txt')
    
    # read smap, channel order from .txt
    smap, order = pandas.read_csv(path_smap, sep='\t', header=None), pandas.read_csv(path_order, sep='\t')
    smap, order = numpy.array(smap), numpy.array(order['channel'])
    
    return smap, order

def get_cmdata(feature, experiment):
    # path
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    
    # path_data
    path_data = os.path.join(path_parent, 'data', 'SEED', 'functional connectivity', feature, experiment + '.mat')
    
    # cmdata
    cmdata = read_mat_t(path_data)
    
    return cmdata

def read_mat(path_file):
    # 确保文件存在
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")

    try:
        # 尝试以 HDF5 格式读取文件
        with h5py.File(path_file, 'r') as f:
            print("HDF5 format detected.")
            key = list(f.keys())[0]  # 默认获取第一个键
            mat_data = numpy.array(f[key])

    except OSError:
        # 如果不是 HDF5 格式，尝试使用 scipy.io.loadmat
        print("Not an HDF5 format.")
        mat_data = scipy.io.loadmat(path_file)
        keys = [key for key in mat_data.keys() if not key.startswith('__')]
        if not keys:
            raise ValueError("No valid keys found in the MAT file.")
        mat_data = mat_data[keys[0]]  # 根据实际需求选择键

    # 确保返回值为 numpy 数组
    mat_data = numpy.array(mat_data)

    # 数据重塑
    mat_data = cmdata_reshaper(mat_data)

    return mat_data

def cmdata_reshaper(mat_data):
    """
    Reshapes mat_data to ensure the last two dimensions are square (n1 == n2).
    Automatically handles transposing and validates the shape.
    """
    import numpy

    MAX_ITER = 10  # 最大迭代次数，防止死循环
    iteration = 0

    while iteration < MAX_ITER:
        samples, n1, n2 = mat_data.shape
        if n1 == n2:
            break  # 如果满足条件，直接退出
        elif n1 != n2:
            mat_data = numpy.transpose(mat_data, axes=(2, 0, 1))  # 转置调整维度
        iteration += 1

    else:
        raise ValueError("Failed to reshape mat_data into (samples, n1, n2) with n1 == n2 after multiple attempts.")

    return mat_data

def read_mat_t(path_file):
    # 确保文件存在
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")

    try:
        # 尝试以 HDF5 格式读取文件
        with h5py.File(path_file, 'r') as f:
            print("HDF5 format detected.")
            # 提取所有键值及其数据
            mat_data = {key: numpy.array(f[key]) for key in f.keys()}

    except OSError:
        # 如果不是 HDF5 格式，尝试使用 scipy.io.loadmat
        print("Not an HDF5 format.")
        mat_data = scipy.io.loadmat(path_file)
        # 排除系统默认的键
        mat_data = {key: mat_data[key] for key in mat_data.keys() if not key.startswith('__')}

    # 数据重塑（如果需要）
    reshaped_data = {key: cmdata_reshaper_t(data) for key, data in mat_data.items()}

    return reshaped_data

def cmdata_reshaper_t(mat_data):
    """
    Reshapes mat_data to ensure the last two dimensions are square (n1 == n2).
    Automatically handles transposing and validates the shape.
    """
    MAX_ITER = 10  # 最大迭代次数，防止死循环
    iteration = 0

    while iteration < MAX_ITER:
        if mat_data.ndim == 3:
            samples, n1, n2 = mat_data.shape
            if n1 == n2:
                break  # 如果满足条件，直接退出
            else:
                mat_data = numpy.transpose(mat_data, axes=(2, 0, 1))  # 转置调整维度
        iteration += 1

    else:
        raise ValueError("Failed to reshape mat_data into (samples, n1, n2) with n1 == n2 after multiple attempts.")

    return mat_data
