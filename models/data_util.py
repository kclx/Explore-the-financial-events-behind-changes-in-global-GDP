import math

import numpy as np
from matplotlib import pyplot as plt


def dataset_split(data, start, end, step, axis=0):
    """
    用于切分DataFrame数据
    :param data: 被切分的DataFrame数据集
    :param start: 从何处开始进行切分
    :param end: 切分到什么位置
    :param step: 每次切分的步长
    :param axis: 按照行(0)/列(1)进行切分
    :return: DataFrame字典
    """
    if axis == 0:
        if data.shape[0] >= end and start >= 0 and step > 0:
            split_count = math.ceil((end - start) / step)
            if (end - start) % step == 0:
                flag = True
            else:
                flag = False
            split_dataset_array = {}
            for i in range(split_count):
                if i == split_count - 1 and not flag:
                    split_dataset_array['dataset_' + str(i)] = data.iloc[(start + step * i):end, :]
                    return split_dataset_array
                split_dataset_array['dataset_' + str(i)] = data.iloc[(start + step * i):(start + step * (i + 1)), :]
            return split_dataset_array
        else:
            return None
    elif axis == 1:
        if data.shape[1] >= end and start >= 0 and step > 0:
            split_count = math.ceil((end - start) / step)
            if (end - start) % step == 0:
                flag = True
            else:
                flag = False
            split_dataset_array = {}
            for i in range(split_count):
                if i == split_count - 1 and not flag:
                    split_dataset_array['dataset_' + str(i)] = data.iloc[:, (start + step * i):end]
                    return split_dataset_array
                split_dataset_array['dataset_' + str(i)] = data.iloc[:, (start + step * i):(start + step * (i + 1))]
            return split_dataset_array
        else:
            return None
    else:
        return None


def switch_growth_rate(data, axis=0):
    """
    计算增长率
    在给定的axis的维度上的特征会-1，因为最初的数据没有增长率
    :param data: 需要被计算的数据
    :param axis: 按照行(0)/列(1)进行计算
    :return: 返回一个新的ndarray对象，存储的是对应的增长率
    """
    if axis == 0:
        end = data.shape[1] - 1
        new_data = np.empty((data.shape[0], end))
        for i in range(data.shape[0]):
            test = data[i, 1:] - data[i, :end]
            new_data[i, :] = test / data[i, 1:]
        new_data = np.nan_to_num(new_data)
        return new_data
    elif axis == 1:
        end = data.shape[0] - 1
        new_data = np.empty((end, data.shape[1]))
        for i in range(data.shape[1]):
            test = data[1:, i] - data[:end, i]
            new_data[:, i] = test / data[:end, i]
        return new_data


def plot_data(x, y, go, to, one_picture_count=5, marker='+', label=None, ylim=None):
    """
    画折线图
    :param x: x轴的数据
    :param y: y轴的数据
    :param go: 从哪里开始
    :param to: 到哪里结束
    :param one_picture_count: 一张图中包含几条数据集（几条折线）
    :param marker: 数据点的标识
    :param label: 数据的标识
    :param ylim: y轴的距离
    :return:
    """
    if x.shape != y.shape:
        return
    if go >= 0 and to <= x.shape[1]:
        size = to - go
        count = (size // one_picture_count) + 1
    else:
        return
    picture_size = count // 4 + 1
    plt.figure(figsize=(24, 8))
    for i in range(1, count + 1):
        plt.subplot(picture_size, 4, i)
        data_end = go + (one_picture_count * i)
        data_start = go + (one_picture_count * (i - 1))
        if data_end > to:
            data_end = to
            one_picture_count = size % one_picture_count
            if one_picture_count == 0:
                plt.show()
                return
        for j in range(one_picture_count):
            plt.plot(x[:, data_start + j], y[:, data_start + j], marker=marker, label=label[data_start + j])
            if not ylim is None:
                plt.ylim(ylim)
        plt.legend(loc='upper left')
    plt.show()
    return


def split_from_arrange_values(dataset, arrange, name):
    """
    切分数据集
    :param dataset: 被切分的数据集
    :param arrange: 根据某一列进行切分
    :param name: 重新命名
    :return: 切分后得到的数据集字典
    """
    split_dataset_dict = {}
    arrange_values = set(dataset[arrange])
    for i in arrange_values:
        new = dataset[dataset[arrange] == i]
        # test_data=world_country_gdp_usd[world_country_gdp_usd['year']==1960]
        split_dataset_dict[name + '_' + str(i)] = new
    return split_dataset_dict
