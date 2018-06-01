# coding:utf-8
"""
plt 的数据输入输出模块
creatd: 02/06/2018
@author: anning
"""
import os
import numpy as np
import h5py
import yaml
from configobj import ConfigObj


class ReadHDF5(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.rad1 = np.empty(shape=(0))  # S1_FovRadMean
        self.rad1_std = np.empty(shape=(0))  # S1_FovRadStd
        self.rad2 = np.empty(shape=(0))  # S2_FovRadMean
        self.rad2_std = np.empty(shape=(0))  # S2_FovRadStd

        self.tbb1 = np.empty(shape=(0))  # S1_FovTbbMean
        self.tbb1_std = np.empty(shape=(0))  # S1_FovTbbStd
        self.tbb2 = np.empty(shape=(0))  # S2_FovTbbMean
        self.tbb2_std = np.empty(shape=(0))  # S2_FovTbbStd

        self.ref1 = np.empty(shape=(0))  # S1_FovRefMean
        self.ref1_std = np.empty(shape=(0))  # S1_FovRefStd
        self.ref2 = np.empty(shape=(0))  # S2_FovRefMean
        self.ref2_std = np.empty(shape=(0))  # S2_FovRefStd

        self.dn1 = np.empty(shape=(0))  # S1_FovDnMean
        self.dn1_std = np.empty(shape=(0))  # S1_FovDnStd
        self.dn2 = np.empty(shape=(0))  # S2_FovDnMean
        self.dn2_std = np.empty(shape=(0))  # S2_FovDnStd

        self.lat1 = np.empty(shape=(0))  # S1_Lat
        self.lon1 = np.empty(shape=(0))  # S1_Lon
        self.lat2 = np.empty(shape=(0))  # S2_Lat
        self.lon2 = np.empty(shape=(0))  # S2_Lon

        self.time = np.empty(shape=(0))  # S1_Time

    def LoadData(self, i_file, channels):
        if isinstance(channels, str):
            channels = [channels]
        else:
            channels = channels
        noError = True
        with h5py.File(i_file, 'r') as hdf5File:
            for channel in channels:
                try:
                    channel_group = hdf5File[channel]
                except StandardError:
                    noError = False
                    return noError
                rad1 = channel_group.get('S1_FovRadMean', np.empty(shape=(0)))[:]  # S1_FovRadMean
                self.rad1 = np.append(self.rad1, rad1)
                rad1_std = channel_group.get('S1_FovRadStd', np.empty(shape=(0)))[:]  # S1_FovRadStd
                self.rad1_std = np.append(self.rad1_std, rad1_std)
                rad2 = channel_group.get('S2_FovRadMean', np.empty(shape=(0)))[:]  # S2_FovRadMean
                self.rad2 = np.append(self.rad2, rad2)
                rad2_std = channel_group.get('S2_FovRadStd', np.empty(shape=(0)))[:]  # S2_FovRadStd
                self.rad2_std = np.append(self.rad2_std, rad2_std)

                tbb1 = channel_group.get('S1_FovTbbMean', np.empty(shape=(0)))[:]  # S1_FovTbbMean
                self.tbb1 = np.append(self.tbb1, tbb1)
                tbb1_std = channel_group.get('S1_FovTbbStd', np.empty(shape=(0)))[:]  # S1_FovTbbStd
                self.tbb1_std = np.append(self.tbb1_std, tbb1_std)
                tbb2 = channel_group.get('S2_FovTbbMean', np.empty(shape=(0)))[:]  # S2_FovTbbMean
                self.tbb2 = np.append(self.tbb2, tbb2)
                tbb2_std = channel_group.get('S2_FovTbbStd', np.empty(shape=(0)))[:]  # S2_FovTbbStd
                self.tbb2_std = np.append(self.tbb2_std, tbb2_std)

                ref1 = channel_group.get('S1_FovRefMean', np.empty(shape=(0)))[:]  # S1_FovRefMean
                self.ref1 = np.append(self.ref1, ref1)
                ref1_std = channel_group.get('S1_FovRefStd', np.empty(shape=(0)))[:]  # S1_FovRefStd
                self.ref1_std = np.append(self.ref1_std, ref1_std)
                ref2 = channel_group.get('S2_FovRefMean', np.empty(shape=(0)))[:]  # S2_FovRefMean
                self.ref2 = np.append(self.ref2, ref2)
                ref2_std = channel_group.get('S2_FovRefStd', np.empty(shape=(0)))[:]  # S2_FovRefStd
                self.ref2_std = np.append(self.ref2_std, ref2_std)

                dn1 = channel_group.get('S1_FovDnMean', np.empty(shape=(0)))[:]  # S1_FovDnMean
                self.dn1 = np.append(self.dn1, dn1)
                dn1_std = channel_group.get('S1_FovDnStd', np.empty(shape=(0)))[:]  # S1_FovDnStd
                self.dn1_std = np.append(self.dn1_std, dn1_std)
                dn2 = channel_group.get('S2_FovDnMean', np.empty(shape=(0)))[:]  # S2_FovDnMean
                self.dn2 = np.append(self.dn2, dn2)
                dn2_std = channel_group.get('S2_FovDnStd', np.empty(shape=(0)))[:]  # S2_FovDnStd
                self.dn2_std = np.append(self.dn2_std, dn2_std)

                lat1 = channel_group.get('S1_Lat', np.empty(shape=(0)))[:]  # S1_Lat
                self.lat1 = np.append(self.lat1, lat1)
                lon1 = channel_group.get('S1_Lon', np.empty(shape=(0)))[:]  # S1_Lon
                self.lon1 = np.append(self.lon1, lon1)
                lat2 = channel_group.get('S2_Lat', np.empty(shape=(0)))[:]  # S2_Lat
                self.lat2 = np.append(self.lat2, lat2)
                lon2 = channel_group.get('S2_Lon', np.empty(shape=(0)))[:]  # S2_Lon
                self.lon2 = np.append(self.lon2, lon2)

                time = channel_group.get('S1_Time', np.empty(shape=(0)))[:]  # S1_Time
                self.time = np.append(self.time, time)

        return noError


def loadYamlCfg(iFile):

    if not os.path.isfile(iFile):
        print("No yaml: %s" % (iFile))
        return None
    try:
        with open(iFile, 'r') as stream:
            plt_cfg = yaml.load(stream)
    except IOError:
        print('plt yaml not valid: %s' % iFile)
        plt_cfg = None

    return plt_cfg


def load_yaml_file(in_file):
    """
    加载 Yaml 文件
    :param in_file:
    :return: Yaml 类
    """
    if not os.path.isfile(in_file):
        print "File is not exist: {}".format(in_file)
        return None
    try:
        with open(in_file, 'r') as stream:
            yaml_data = yaml.load(stream)
    except IOError as why:
        print why
        print "Load yaml file error."
        yaml_data = None

    return yaml_data


if __name__ == '__main__':
    loadYamlCfg('sss')
