# coding: utf-8
"""
Created on 2016年1月6日
读取匹配后的 HDF5 文件，画散点回归图，生成 abr 文件

@author: duxiang, zhangtao, anning
"""
import os
import sys
import calendar
from datetime import datetime
from multiprocessing import Pool, Lock

import numpy as np
from matplotlib.ticker import MultipleLocator

from configobj import ConfigObj
from dateutil.relativedelta import relativedelta
from numpy.lib.polynomial import polyfit
from numpy.ma.core import std, mean
from numpy.ma.extras import corrcoef

from DV import dv_pub_3d
from DV.dv_pub_3d import plt, mpl, mdates, Basemap
from DV.dv_pub_3d import bias_information, day_data_write, get_bias_data, get_cabr_data, set_tick_font
from DV.dv_pub_3d import FONT0, FONT_MONO, FONT1
from DM.SNO.dm_sno_cross_calc_map import RED, BLUE, EDGE_GRAY, ORG_NAME, mpatches
from PB.CSC.pb_csc_console import LogServer
from PB import pb_time, pb_io
from PB.pb_time import is_day_timestamp_and_lon

from plt_io import ReadHDF5, loadYamlCfg

lock = Lock()


def run(pair, ymd, is_monthly):
    """
    pair: sat1+sensor1_sat2+sensor2
    ymd: str YYYYMMDD
    """
    # 提取参数中的卫星信息和传感器信息
    part1, part2 = pair.split("_")
    sat1, sensor1 = part1.split("+")
    sat2, sensor2 = part2.split("+")

    # 判断是静止卫星还是动态卫星
    if "FY2" in part1 or "FY4" in part1:
        Type = "GEOLEO"
    elif "FY3" in part1:
        Type = "LEOLEO"
    else:
        Log.error("Cant distinguish the satellite type")
        return

    # 加载绘图配置文件
    plt_cfg_file = os.path.join(MainPath, "%s_%s_3d.yaml" % (sensor1, sensor2))
    plt_cfg = loadYamlCfg(plt_cfg_file)
    if plt_cfg is None:
        Log.error("Not find the config file: {}".format(plt_cfg_file))
        return

    Log.info(u"----- Start Drawing Regression-Pic, PAIR: {}, YMD: {} -----".format(pair, ymd))

    for each in plt_cfg["regression"]:
        dict_cabr = {}
        dict_cabr_d = {}
        dict_cabr_n = {}
        dict_bias = {}
        dict_bias_d = {}
        dict_bias_n = {}

        # 需要回滚的天数
        if is_monthly:
            PERIOD = calendar.monthrange(int(ymd[:4]), int(ymd[4:6]))[1]  # 当月天数
            ymd = ymd[:6] + "%02d" % PERIOD  # 当月最后一天
        else:
            PERIOD = plt_cfg[each]["days"]

        # must be in "all", "day", "night"
        Day_Night = ["all", "day", "night"]
        if "time" in plt_cfg[each].keys():
            Day_Night = plt_cfg[each]["time"]
            for t in Day_Night:
                if t not in ["all", "day", "night"]:
                    Day_Night.remove(t)

        for idx, chan in enumerate(plt_cfg[each]["chan"]):
            Log.info(u"Start Drawing {} Channel {}".format(each, chan))
            oneHDF5 = ReadHDF5()
            num_file = PERIOD
            for daydelta in xrange(PERIOD):
                cur_ymd = pb_time.ymd_plus(ymd, -daydelta)
                hdf5_name = "COLLOC+%sIR,%s_C_BABJ_%s.hdf5" % (Type, pair, cur_ymd)
                filefullpath = os.path.join(MATCH_DIR, pair, hdf5_name)
                if not os.path.isfile(filefullpath):
                    Log.info(u"File not found: {}".format(filefullpath))
                    num_file -= 1
                    continue
                if not oneHDF5.LoadData(filefullpath, chan):
                    Log.error("Error occur when reading %s of %s" % (chan, filefullpath))
            if num_file == 0:
                Log.error(u"No file found.")
                continue
            elif num_file != PERIOD:
                Log.error(u"{} of {} file(s) found.".format(num_file, PERIOD))

            if is_monthly:
                str_time = ymd[:6]
                cur_path = os.path.join(MRA_DIR, pair, str_time)
            else:
                str_time = ymd
                cur_path = os.path.join(DRA_DIR, pair, str_time)

            # delete 0 in std
            if len(oneHDF5.rad1_std) > 0.0001:  # TODO: 有些极小的std可能是异常值，而导致权重极大，所以 std>0 改成 std>0.0001
                deletezeros = np.where(oneHDF5.rad1_std > 0.0001)
                oneHDF5.rad1_std = oneHDF5.rad1_std[deletezeros]
                oneHDF5.rad1 = oneHDF5.rad1[deletezeros] if len(
                    oneHDF5.rad1) > 0 else oneHDF5.rad1
                oneHDF5.rad2 = oneHDF5.rad2[deletezeros] if len(
                    oneHDF5.rad2) > 0 else oneHDF5.rad2
                oneHDF5.tbb1 = oneHDF5.tbb1[deletezeros] if len(
                    oneHDF5.tbb1) > 0 else oneHDF5.tbb1
                oneHDF5.tbb2 = oneHDF5.tbb2[deletezeros] if len(
                    oneHDF5.tbb2) > 0 else oneHDF5.tbb2
                oneHDF5.time = oneHDF5.time[deletezeros] if len(
                    oneHDF5.time) > 0 else oneHDF5.time
                oneHDF5.lon1 = oneHDF5.lon1[deletezeros] if len(
                    oneHDF5.lon1) > 0 else oneHDF5.lon1
                oneHDF5.lon2 = oneHDF5.lon2[deletezeros] if len(
                    oneHDF5.lon2) > 0 else oneHDF5.lon2
            if len(oneHDF5.ref1_std) > 0.0001:
                deletezeros = np.where(oneHDF5.ref1_std > 0.0001)
                oneHDF5.ref1_std = oneHDF5.ref1_std[deletezeros]
                oneHDF5.ref1 = oneHDF5.ref1[deletezeros] if len(
                    oneHDF5.ref1) > 0 else oneHDF5.ref1
                oneHDF5.ref2 = oneHDF5.ref2[deletezeros] if len(
                    oneHDF5.ref2) > 0 else oneHDF5.ref2
                oneHDF5.dn1 = oneHDF5.dn1[deletezeros] if len(
                    oneHDF5.dn1) > 0 else oneHDF5.dn1
                oneHDF5.dn2 = oneHDF5.dn1[deletezeros] if len(
                    oneHDF5.dn2) > 0 else oneHDF5.dn2
                oneHDF5.time = oneHDF5.time[deletezeros] if len(
                    oneHDF5.time) > 0 else oneHDF5.time
                oneHDF5.lon1 = oneHDF5.lon1[deletezeros] if len(
                    oneHDF5.lon1) > 0 else oneHDF5.lon1
                oneHDF5.lon2 = oneHDF5.lon2[deletezeros] if len(
                    oneHDF5.lon2) > 0 else oneHDF5.lon2

            # find out day and night
            if ("day" in Day_Night or "night" in Day_Night) and len(oneHDF5.time) > 0:
                vect_is_day = np.vectorize(is_day_timestamp_and_lon)
                day_index = vect_is_day(oneHDF5.time, oneHDF5.lon1)
                night_index = np.logical_not(day_index)
            else:
                day_index = None
                night_index = None

            # 将每个对通用的属性值放到对循环，每个通道用到的属性值放到通道循环
            # get threhold, unit, names...
            xname, yname = each.split("-")
            xname_l = plt_cfg[each]["x_name"]
            xunit = plt_cfg[each]["x_unit"]
            xlimit = plt_cfg[each]["x_range"][idx]
            xmin, xmax = xlimit.split("-")
            xmin = float(xmin)
            xmax = float(xmax)
            yname_l = plt_cfg[each]["y_name"]
            yunit = plt_cfg[each]["y_unit"]
            ylimit = plt_cfg[each]["y_range"][idx]
            ymin, ymax = ylimit.split("-")
            ymin = float(ymin)
            ymax = float(ymax)

            weight = None
            if "rad" in xname:
                x = oneHDF5.rad1
            elif "tbb" in xname:
                x = oneHDF5.tbb1
            elif "ref" in xname:
                x = oneHDF5.ref1
            elif "dn" in xname:
                x = oneHDF5.dn1
            else:
                Log.error("Can't plot %s" % each)
                continue
            if "rad" in yname:
                y = oneHDF5.rad2
            elif "tbb" in yname:
                y = oneHDF5.tbb2
            elif "ref" in yname:
                y = oneHDF5.ref2
            else:
                Log.error("Can't plot %s" % each)
                continue

            if "rad" in xname and "rad" in yname:
                if len(oneHDF5.rad1_std) > 0:
                    weight = oneHDF5.rad1_std
                o_name = "RadCalCoeff"
            elif "tbb" in xname and "tbb" in yname:
                o_name = "TBBCalCoeff"
            elif "ref" in xname and "ref" in yname:
                if len(oneHDF5.ref1_std) > 0:
                    weight = oneHDF5.ref1_std
                o_name = "CorrcCoeff"
            elif "dn" in xname and "ref" in yname:
                o_name = "CalCoeff"

            # 画对角线
            if xname == yname:
                diagonal = True
            else:
                diagonal = False

            if "all" in Day_Night and o_name not in dict_cabr:
                dict_cabr[o_name] = {}
                dict_bias[xname] = {}
            if "day" in Day_Night and o_name not in dict_cabr_d:
                dict_cabr_d[o_name] = {}
                dict_bias_d[xname] = {}
            if "night" in Day_Night and o_name not in dict_cabr_n:
                dict_cabr_n[o_name] = {}
                dict_bias_n[xname] = {}

            # 对样本点数量进行判断，如果样本点少于 100 个，则不进行绘制
            if x.size < 100:
                Log.error("Not enough match point to draw: {}, {}".format(each, chan))
                if "all" in Day_Night:
                    dict_cabr[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                    dict_bias[xname][chan] = [np.NaN, np.NaN]
                if "day" in Day_Night:
                    dict_cabr_d[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                    dict_bias_d[xname][chan] = [np.NaN, np.NaN]
                if "night" in Day_Night:
                    dict_cabr_n[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                    dict_bias_n[xname][chan] = [np.NaN, np.NaN]
                continue

            # regression starts
            if "all" in Day_Night:
                o_file = os.path.join(cur_path,
                                      "%s_%s_%s_ALL_%s" % (
                                          pair, o_name, chan, str_time))
                print("x_all, y_all", len(x), len(y))
                abr, bias = plot(x, y, weight, o_file,
                                 num_file, part1, part2, chan, str_time,
                                 xname, xname_l, xunit, xmin, xmax,
                                 yname, yname_l, yunit, ymin, ymax,
                                 diagonal, is_monthly)
                if abr:
                    dict_cabr[o_name][chan] = abr
                else:
                    dict_cabr[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                if bias:
                    dict_bias[xname][chan] = bias
                else:
                    dict_bias[xname][chan] = [np.NaN, np.NaN]

            # ------- day ----------
            if "day" in Day_Night:
                if day_index is not None and np.where(day_index)[0].size > 10:
                    o_file = os.path.join(cur_path,
                                          "%s_%s_%s_Day_%s" % (
                                              pair, o_name, chan, str_time))
                    x_d = x[day_index]
                    y_d = y[day_index]
                    w_d = weight[day_index] if weight is not None else None
                    print("x_all, y_all", len(x), len(y))
                    print("x_day, y_day", len(x_d), len(y_d))
                    abr, bias = plot(x_d, y_d, w_d, o_file,
                                     num_file, part1, part2, chan, str_time,
                                     xname, xname_l, xunit, xmin, xmax,
                                     yname, yname_l, yunit, ymin, ymax,
                                     diagonal, is_monthly)
                    if abr:
                        dict_cabr_d[o_name][chan] = abr
                    else:
                        dict_cabr_d[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                    if bias:
                        dict_bias_d[xname][chan] = bias
                    else:
                        dict_bias_d[xname][chan] = [np.NaN, np.NaN]
                else:
                    dict_cabr_d[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                    dict_bias_d[xname][chan] = [np.NaN, np.NaN]
            # ---------night ------------
            if "night" in Day_Night:
                if night_index is not None and np.where(night_index)[0].size > 10:
                    o_file = os.path.join(cur_path, "%s_%s_%s_Night_%s" % (
                        pair, o_name, chan, str_time))
                    x_n = x[night_index]
                    y_n = y[night_index]
                    w_n = weight[night_index] if weight is not None else None
                    print("x_all, y_all", len(x), len(y))
                    print("x_night, y_night", len(x_n), len(y_n))
                    abr, bias = plot(x_n, y_n, w_n, o_file,
                                     num_file, part1, part2, chan, str_time,
                                     xname, xname_l, xunit, xmin, xmax,
                                     yname, yname_l, yunit, ymin, ymax,
                                     diagonal, is_monthly)
                    if abr:
                        dict_cabr_n[o_name][chan] = abr
                    else:
                        dict_cabr_n[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                    if bias:
                        dict_bias_n[xname][chan] = bias
                    else:
                        dict_bias_n[xname][chan] = [np.NaN, np.NaN]
                else:
                    dict_cabr_n[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                    dict_bias_n[xname][chan] = [np.NaN, np.NaN]
            oneHDF5.clear()

        # write txt
        lock.acquire()
        channel = plt_cfg[each]["chan"]
        if "all" in Day_Night:
            for o_name in dict_cabr:
                write_bias(channel, part1, part2, xname, ymd,
                           dict_bias, "ALL")
                write_cabr(channel, part1, part2, o_name, ymd,
                           dict_cabr, "ALL")
        if "day" in Day_Night:
            for o_name in dict_cabr_d:
                write_bias(channel, part1, part2, xname, ymd,
                           dict_bias_d, "Day")
                write_cabr(channel, part1, part2, o_name, ymd,
                           dict_cabr_d, "Day")
        if "night" in Day_Night:
            for o_name in dict_cabr_n:
                write_bias(channel, part1, part2, xname, ymd,
                           dict_bias_n, "Night")
                write_cabr(channel, part1, part2, o_name, ymd,
                           dict_cabr_n, "Night")
        lock.release()


def write_cabr(channel, part1, part2, o_name, ymd,
               dict_cabr, day_or_night):
    """
    生成 CABR 数据文件
    :param channel:
    :param part1:
    :param part2:
    :param o_name:
    :param ymd:
    :param dict_cabr:
    :param day_or_night:
    :return:
    """
    for chan in channel:
        o_path = os.path.join(ABR_DIR, "%s_%s" % (part1, part2), "CABR")
        file_name_monthly = os.path.join(
            o_path, "%s_%s_%s_%s_%s_Monthly.txt" % (
                part1, part2, o_name, chan, day_or_night))
        file_name_daily = os.path.join(
            o_path, "%s_%s_%s_%s_%s_Daily.txt" % (
                part1, part2, o_name, chan, day_or_night))

        # 写入日数据
        title_daily = ("{:15}  " * 5).format("YMD", "Count", "Slope", "Intercept", "RSquared") + "\n"
        c, a, b, r = dict_cabr[o_name][chan]
        data_daily = ("{:15}  " * 5).format(ymd, c, a, b, r) + "\n"
        day_data_write(title_daily, data_daily, file_name_daily)

        # 写入月数据
        cabr_data = get_cabr_data(file_name_daily)
        title_monthly = ("{:15}  " * 8).format("YMD", "Count", "Slope", "Slope_STD",
                                               "Intercept", "Intercept_STD", "RSquared", "RSquared_STD") + "\n"
        date_data = cabr_data["date"]

        if len(date_data) <= 2:  # 如果小于2天的数据，不计算月平均
            continue

        c_data = cabr_data["count"]
        s_data = np.vstack(cabr_data["slope"])
        i_data = np.vstack(cabr_data["intercept"])
        r_data = np.vstack(cabr_data["rsquared"])

        abr_data = np.concatenate((s_data, i_data, r_data), axis=1)  # 合并 a b c 数据

        count_monthly = month_count(date_data, c_data)  # 计算月总数
        abr_monthly = month_average(date_data, abr_data)[:, 1:]  # 计算月平均和STD
        data_monthly = np.concatenate((count_monthly, abr_monthly), axis=1)

        for data in data_monthly:
            ymd_monthly = data[0]
            count_monthly = data[1]
            slope_mean_monthly = data[2]
            intercept_mean_monthly = data[3]
            rsquared_mean_monthly = data[4]

            slope_std_monthly = data[5]
            intercept_std_monthly = data[6]
            rsquared_std_monthly = data[7]

            data_str = ("{:15}  " * 8).format(ymd_monthly, count_monthly, slope_mean_monthly, slope_std_monthly,
                                              intercept_mean_monthly, intercept_std_monthly,
                                              rsquared_mean_monthly, rsquared_std_monthly) + "\n"
            day_data_write(title_monthly, data_str, file_name_monthly)
        print file_name_daily
        print file_name_monthly


def write_bias(channel, part1, part2, xname, ymd,
               dict_bias, day_or_night):
    """
    生成 RMD 日数据文件和月数据文件
    :param channel:
    :param part1:
    :param part2:
    :param xname:
    :param ymd:
    :param dict_bias:
    :param day_or_night:
    :return:
    """
    if not (xname in ["ref", "tbb"]):
        return

    for chan in channel:
        o_path = os.path.join(ABR_DIR, "%s_%s" % (part1, part2), "BIAS")
        file_name_monthly = os.path.join(
            o_path, "%s_%s_%s_%s_%s_Monthly.txt" % (
                part1, part2, xname.upper(), chan, day_or_night))
        file_name_daily = os.path.join(
            o_path, "%s_%s_%s_%s_%s_Daily.txt" % (
                part1, part2, xname.upper(), chan, day_or_night))

        # 写入日数据
        bias = dict_bias[xname][chan][0]
        md = dict_bias[xname][chan][1]
        title_daily = ("{:15}    " * 3).format("YMD", "Bias", "MD") + "\n"
        data_daily = ("{:15}    " * 3).format(ymd, bias, md) + "\n"
        day_data_write(title_daily, data_daily, file_name_daily)

        # 写入月数据
        title_monthly = ("{:15}    " * 5).format("YMD", "Bias", "Bias_STD", "MD", "MD_STD") + "\n"
        bias_data = get_bias_data(file_name_daily)
        date_data = bias_data["date"]
        bias_day_data = np.vstack(bias_data["bias"])
        md_day_data = np.vstack(bias_data["md"])

        if len(date_data) <= 2:  # 如果小于2天的数据，不计算月平均
            continue

        bias_md_data = np.concatenate((bias_day_data, md_day_data), axis=1)
        data_monthly = month_average(date_data, bias_md_data)  # 计算月平均和STD

        for data in data_monthly:
            ymd_monthly = data[0]
            bias_mean_monthly = data[1]
            md_mean_monthly = data[2]
            bias_std_monthly = data[3]
            md_std_monthly = data[4]

            data_str = ("{:15}    " * 5).format(ymd_monthly, bias_mean_monthly, bias_std_monthly,
                                                md_mean_monthly, md_std_monthly) + "\n"
            day_data_write(title_monthly, data_str, file_name_monthly)
        print file_name_daily
        print file_name_monthly


def month_count(day_date, data_day):
    """
    由日数据生成月总计数据
    :param day_date: 日期：datetime 实例
    :param data_day: 日数据
    :return:
    """
    month_datas = []
    ymd_start = day_date[0]  # 第一天日期
    ymd_end = day_date[-1]  # 最后一天日期
    date_start = ymd_start - relativedelta(
        days=(ymd_start.day - 1))  # 第一个月第一天日期

    while date_start <= ymd_end:
        # 当月最后一天日期
        date_end = date_start + relativedelta(months=1) - relativedelta(days=1)

        # 查找当月所有数据
        month_idx = np.where(np.logical_and(day_date >= date_start,
                                            day_date <= date_end))

        data_month = data_day[month_idx]
        not_nan_idx = np.isfinite(data_month)  # 清除 nan 数据
        data_month = data_month[not_nan_idx]

        ymd_data = date_start.strftime("%Y%m%d")
        count_data = np.sum(data_month)

        data = [ymd_data, count_data.tolist()]

        month_datas.append(data)

        date_start = date_start + relativedelta(months=1)

    return np.array(month_datas)


def month_average(day_date, day_data):
    """
    由日数据生成月平均数据
    :param day_date: 日期：datetime 实例
    :param day_data: 日数据
    :return: [[ymd], [mean], [mean1], [std], [std1]]
    """
    month_datas = {
        "ymd": [],
        "mean_data": [],
        "std_data": [],
    }
    ymd_start = day_date[0]  # 第一天日期
    ymd_end = day_date[-1]  # 最后一天日期
    date_start = ymd_start - relativedelta(
        days=(ymd_start.day - 1))  # 第一个月第一天日期

    while date_start <= ymd_end:
        # 当月最后一天日期
        date_end = date_start + relativedelta(months=1) - relativedelta(days=1)

        # 查找当月所有数据
        month_idx = np.where(np.logical_and(day_date >= date_start,
                                            day_date <= date_end))

        data_month = day_data[month_idx]
        ymd_data = date_start.strftime("%Y%m%d")

        data_month = np.ma.masked_invalid(data_month)  # mask nan 数据
        mean_data = np.mean(data_month, axis=0)
        std_data = np.std(data_month, axis=0)

        # 如果被掩码掩盖掉，用 nan 替换
        if np.ma.is_masked(mean_data):
            idx_masked = mean_data.mask
            mean_data = np.asarray(mean_data)
            mean_data[idx_masked] = np.NaN
        if np.ma.is_masked(std_data):
            idx_masked = std_data.mask
            std_data = np.asarray(std_data)
            std_data[idx_masked] = np.NaN

        month_datas.get("ymd").append([ymd_data])
        month_datas.get("mean_data").append(mean_data.tolist())
        month_datas.get("std_data").append(std_data.tolist())

        date_start = date_start + relativedelta(months=1)

    ymd_data = np.vstack(month_datas.get("ymd"))
    mean_data = np.vstack(month_datas.get("mean_data"))
    std_data = np.vstack(month_datas.get("std_data"))
    month_datas = np.concatenate((ymd_data, mean_data, std_data), axis=1)

    return month_datas


def plot(x, y, weight, o_file, num_file, part1, part2, chan, ymd,
         xname, xname_l, xunit, xmin, xmax, yname, yname_l, yunit, ymin, ymax,
         is_diagonal, is_monthly):
    plt.style.use(os.path.join(dvPath, "dv_pub_regression.mplstyle"))

    # 过滤 正负 delta+8 倍 std 的杂点 ------------------------
    w = 1.0 / weight if weight is not None else None
    RadCompare = G_reg1d(x, y, w)
    reg_line = x * RadCompare[0] + RadCompare[1]
    delta = np.abs(y - reg_line)
    mean_delta = np.mean(delta)
    std_delta = np.std(delta)
    max_y = reg_line + mean_delta + std_delta * 8
    min_y = reg_line - mean_delta - std_delta * 8

    idx = np.logical_and(y < max_y, y > min_y)
    x = x[idx]
    y = y[idx]
    w = w[idx] if weight is not None else None
    # -----------------------------------------
    RadCompare = G_reg1d(x, y, w)
    length_rad = len(x)

    bias_and_md = []  # 当 bias 没有被计算的时候，不输出 bias
    if not is_monthly and is_diagonal:
        # return [len(x), RadCompare[0], RadCompare[1], RadCompare[4]]
        fig = plt.figure(figsize=(14, 4.5))
        # fig.subplots_adjust(top=0.92, bottom=0.11, left=0.045, right=0.985)
        ax1 = plt.subplot2grid((1, 3), (0, 0))
        ax2 = plt.subplot2grid((1, 3), (0, 1))
        ax3 = plt.subplot2grid((1, 3), (0, 2))
        # 图片 Title
        titleName = "%s-%s" % (xname.upper(), yname.upper())
        title = "{} Regression {} Days {}_{} {} {}".format(
            titleName, num_file, part1, part2, chan, ymd)

        # 画回归图 -----------------------------------------------
        print "draw regression"
        regress_xmin = xmin
        regress_xmax = xmax
        regress_ymin = ymin
        regress_ymax = ymax
        regress_axislimit = {
            "xlimit": (regress_xmin, regress_xmax),
            "ylimit": (regress_ymin, regress_ymax),
        }

        if xunit != "":
            xlabel = "{} {} (${}$)".format(part1, xname_l, xunit)
        else:
            xlabel = "{} {}".format(part1, xname_l)

        if yunit != "":
            ylabel = "{} {} (${}$)".format(part2, yname_l, yunit)
        else:
            ylabel = "{} {}".format(part2, yname_l)

        regress_label = {
            "xlabel": xlabel, "ylabel": ylabel, "fontsize": 14,
        }

        if xname == "tbb":
            regress_locator = {"locator_x": (None, None),
                               "locator_y": (None, 5)}
        elif xname == "ref":
            regress_locator = {"locator_x": (None, None),
                               "locator_y": (None, 5)}
        else:
            regress_locator = None

        regress_annotate = {
            "left": ["{:10}: {:7.4f}".format("Slope", RadCompare[0]),
                     "{:10}: {:7.4f}".format("Intercept", RadCompare[1]),
                     "{:10}: {:7.4f}".format("Cor-Coef", RadCompare[4]),
                     "{:10}: {:7d}".format("Number", length_rad)],
            "fontsize": 14,
        }

        regress_tick = {"fontsize": 14, }

        regress_diagonal = {"line_color": "#808080", "line_width": 1.2}

        regress_regressline = {"line_color": "r", "line_width": 1.2}

        scatter_point = {"scatter_alpha": 0.8}

        dv_pub_3d.draw_regression(
            ax1, x, y, label=regress_label, ax_annotate=regress_annotate, tick=regress_tick,
            axislimit=regress_axislimit, locator=regress_locator,
            diagonal=regress_diagonal, regressline=regress_regressline,
            scatter_point=scatter_point,
        )

        # 画偏差分布图 ---------------------------------------------
        print "draw distribution"

        distri_xmin = xmin
        distri_xmax = xmax
        if xname == "tbb":
            distri_ymin = -4
            distri_ymax = 4
        elif xname == "ref":
            distri_ymin = -0.08
            distri_ymax = 0.08
        else:
            distri_ymin = None
            distri_ymax = None
        distri_limit = {
            "xlimit": (distri_xmin, distri_xmax),
            "ylimit": (distri_ymin, distri_ymax),
        }

        # x y 轴标签
        xlabel = "{}".format(xname_l)
        if xname == "tbb":
            ylabel = "{} bias {}_{} ".format(xname.upper(), part1, part2, )
        elif xname == "ref":
            ylabel = "{} bias {}_{} ".format(xname.capitalize(), part1, part2, )
        else:
            ylabel = "{} bias {}_{} ".format(xname, part1, part2, )
        distri_label = {
            "xlabel": xlabel, "ylabel": ylabel, "fontsize": 14,
        }

        # 获取 MeanBias 信息
        bias_range = 0.15
        boundary = xmin + (xmax - xmin) * 0.15
        bias_info = bias_information(x, y, boundary, bias_range)

        # 格式化 MeanBias 信息
        info_lower = "MeanBias(<={:d}%Range)=\n    {:.4f}±{:.4f}@{:.4f}".format(
            int(bias_range * 100), bias_info.get("md_lower"),
            bias_info.get("std_lower"), bias_info.get("mt_lower"))
        info_greater = "MeanBias(>{:d}%Range)=\n    {:.4f}±{:.4f}@{:.4f}".format(
            int(bias_range * 100), bias_info.get("md_greater"),
            bias_info.get("std_greater"), bias_info.get("mt_greater"))

        # 绝对偏差和相对偏差信息 TBB=250K  REF=0.25
        ab = RadCompare
        a = ab[0]
        b = ab[1]
        if xname == "tbb":
            bias = 250 - (250 * a + b)
            bias_info_md = "TBB Bias(250K):{:.4f}K".format(bias)
        elif xname == "ref":
            bias = (0.25 - (0.25 * a + b)) / 0.25 * 100
            bias_info_md = "Relative Bias(REF0.25):{:.4f}%".format(bias)
        else:
            bias = np.NaN  # RMD or TBB bias
            bias_info_md = ""
        bias_and_md.append(bias)

        # Range Mean : 偏差图的 MD 信息
        if xname == "tbb":
            md_greater = bias_info.get("md_greater", np.NaN)
            md = md_greater
        elif xname == "ref":
            md_greater = bias_info.get("md_greater")
            mt_greater = bias_info.get("mt_greater")
            if md_greater is not None and mt_greater is not None:
                md = (md_greater / mt_greater) * 100
            else:
                md = np.NaN
        else:
            md = np.NaN
        bias_and_md.append(md)

        # 添加注释信息
        distri_annotate = {"left": [], "leftbottom": [], "right": [], "fontsize": 14, }

        distri_annotate.get("left").append(bias_info_md)
        distri_annotate.get("leftbottom").append(info_lower)
        distri_annotate.get("leftbottom").append(info_greater)

        # 添加 tick 信息
        distri_tick = {"fontsize": 14, }

        # 添加间隔数量
        if xname == "tbb":
            distri_locator = {"locator_x": (None, None), "locator_y": (8, 5)}
        elif xname == "ref":
            distri_locator = {"locator_x": (None, None), "locator_y": (8, 5)}
        else:
            distri_locator = None

        # y=0 线配置
        zeroline = {"line_color": "#808080", "line_width": 1.0}

        # 偏差点配置
        scatter_delta = {
            "scatter_marker": "o", "scatter_size": 5, "scatter_alpha": 0.8,
            "scatter_linewidth": 0, "scatter_zorder": 100,
            "scatter_color": BLUE,
        }

        # 偏差回归线配置
        regressline = {"line_color": "r", "line_width": 1.2}
        dv_pub_3d.draw_distribution(ax2, x, y, label=distri_label,
                                        ax_annotate=distri_annotate,
                                        tick=distri_tick,
                                        axislimit=distri_limit,
                                        locator=distri_locator,
                                        zeroline=zeroline,
                                        scatter_delta=scatter_delta,
                                        regressline=regressline,
                                        )

        # 画直方图 --------------------------------------------------
        histogram_xmin = xmin
        histogram_xmax = xmax
        histogram_axislimit = {
            "xlimit": (histogram_xmin, histogram_xmax),
        }

        histogram_xlabel = "{}".format(xname_l)
        histogram_ylabel = "match point numbers"
        histogram_label = {
            "xlabel": histogram_xlabel, "ylabel": histogram_ylabel, "fontsize": 14,
        }

        # 添加间隔数量
        if xname == "tbb":
            histogram_locator = {"locator_x": (None, None),
                                 "locator_y": (None, 5)}
        elif xname == "ref":
            histogram_locator = {"locator_x": (None, None),
                                 "locator_y": (None, 5)}
        else:
            histogram_locator = None

        histogram_x = {
            "label": part1, "color": "red", "alpha": 0.4, "bins": 100, "fontsize": 14,
        }
        histogram_y = {
            "label": part2, "color": "blue", "alpha": 0.4, "bins": 100,"fontsize": 14,
        }

        histogram_tick = {"fontsize": 14, }

        dv_pub_3d.draw_histogram(
            ax3, x, label=histogram_label, locator=histogram_locator, tick=histogram_tick,
            axislimit=histogram_axislimit, histogram=histogram_x,
        )

        dv_pub_3d.draw_histogram(
            ax3, y, label=histogram_label, locator=histogram_locator, tick=histogram_tick,
            axislimit=histogram_axislimit, histogram=histogram_y,
        )

    elif not is_monthly and not is_diagonal:
        fig = plt.figure(figsize=(4.5, 4.5))
        # fig.subplots_adjust(top=0.89, bottom=0.13, left=0.15, right=0.91)
        ax1 = plt.subplot2grid((1, 1), (0, 0))
        # 图片 Title
        titleName = "%s-%s" % (xname.upper(), yname.upper())
        title = "{} Regression {} Days\n{}_{} {} {}".format(
            titleName, num_file, part1, part2, chan, ymd)
        # 画回归图 ----------------------------------------------------
        print "draw regression"
        regress_xmin = xmin
        regress_xmax = xmax
        regress_ymin = ymin
        regress_ymax = ymax
        regress_axislimit = {
            "xlimit": (regress_xmin, regress_xmax),
            "ylimit": (regress_ymin, regress_ymax),
        }

        if xunit != "":
            xlabel = "{} {} (${}$)".format(part1, xname_l, xunit)
        else:
            xlabel = "{} {}".format(part1, xname_l)

        if yunit != "":
            ylabel = "{} {} (${}$)".format(part2, yname_l, yunit)
        else:
            ylabel = "{} {}".format(part2, yname_l)

        regress_label = {
            "xlabel": xlabel, "ylabel": ylabel, "fontsize": 14,
        }

        if xname == "tbb":
            regress_locator = {"locator_x": (5, None), "locator_y": (5, 5)}
        elif xname == "ref":
            regress_locator = {"locator_x": (None, None),
                               "locator_y": (None, 5)}
        elif xname == "dn":
            regress_locator = {"locator_x": (5, None),
                               "locator_y": (None, 5)}
        else:
            regress_locator = None

        regress_annotate = {
            "left": ["{:10}: {:7.4f}".format("Slope", RadCompare[0]),
                     "{:10}: {:7.4f}".format("Intercept", RadCompare[1]),
                     "{:10}: {:7.4f}".format("Cor-Coef", RadCompare[4]),
                     "{:10}: {:7d}".format("Number", length_rad)],
            "fontsize": 14,
        }

        regress_tick = {"fontsize": 14, }

        regress_diagonal = {"line_color": "#808080", "line_width": 1.2}

        regress_regressline = {"line_color": "r", "line_width": 1.2}

        scatter_point = {"scatter_alpha": 0.8}

        dv_pub_3d.draw_regression(
            ax1, x, y, label=regress_label, ax_annotate=regress_annotate, tick=regress_tick,
            axislimit=regress_axislimit, locator=regress_locator,
            diagonal=regress_diagonal, regressline=regress_regressline,
            scatter_point=scatter_point,
        )
    elif is_monthly:
        o_file = o_file + "_density"

        fig = plt.figure(figsize=(4.5, 4.5))
        # fig.subplots_adjust(top=0.89, bottom=0.13, left=0.15, right=0.91)
        ax1 = plt.subplot2grid((1, 1), (0, 0))
        # 图片 Title Label
        titleName = "%s-%s" % (xname.upper(), yname.upper())
        title = "{} Regression {} Days\n{}_{} {} {}".format(
            titleName, num_file, part1, part2, chan, ymd)
        # 画密度图 -----------------------------------------------------
        print "draw density"
        density_xmin = xmin
        density_xmax = xmax
        density_ymin = ymin
        density_ymax = ymax
        density_axislimit = {
            "xlimit": (density_xmin, density_xmax),
            "ylimit": (density_ymin, density_ymax),
        }

        if xunit != "":
            xlabel = "{} {} (${}$)".format(part1, xname_l, xunit)
        else:
            xlabel = "{} {}".format(part1, xname_l)

        if yunit != "":
            ylabel = "{} {} (${}$)".format(part2, yname_l, yunit)
        else:
            ylabel = "{} {}".format(part2, yname_l)

        density_label = {
            "xlabel": xlabel, "ylabel": ylabel, "fontsize": 14,
        }

        if xname == "tbb":
            density_locator = {"locator_x": (5, None), "locator_y": (5, 5)}
        elif xname == "ref":
            density_locator = {"locator_x": (None, None),
                               "locator_y": (None, 5)}
        elif xname == "dn":
            density_locator = {"locator_x": (5, None),
                               "locator_y": (None, 5)}
        else:
            density_locator = None

        density_annotate = {
            "left": ["{:10}: {:7.4f}".format("Slope", RadCompare[0]),
                     "{:10}: {:7.4f}".format("Intercept", RadCompare[1]),
                     "{:10}: {:7.4f}".format("Cor-Coef", RadCompare[4]),
                     "{:10}: {:7d}".format("Number", length_rad)],
            "fontsize": 14,
        }

        density_tick = {"fontsize": 14, }

        density_diagonal = {"line_color": "#808080", "line_width": 1.2}

        density_regressline = {"line_color": "r", "line_width": 1.2}

        density = {
            "size": 5, "marker": "o", "alpha": 1
        }
        dv_pub_3d.draw_regression(
            ax1, x, y, label=density_label, ax_annotate=density_annotate, tick=density_tick,
            axislimit=density_axislimit, locator=density_locator,
            diagonal=density_diagonal, regressline=density_regressline,
            density=density,
        )
    else:
        print "::::::No output Pic {} : ".format(ymd)
        return
    # 自动调整子图间距
    plt.tight_layout()

    if isMonthly or not is_diagonal:
        fig.subplots_adjust(bottom=0.12, top=0.86)
    else:
        fig.subplots_adjust(top=0.90)

    FONT1.set_size(14)
    fig.suptitle(title, fontsize=14, fontproperties=FONT1)
    pb_io.make_sure_path_exists(os.path.dirname(o_file))
    fig.savefig(o_file, dpi=100)
    print o_file + ".png"
    print "-" * 50
    fig.clear()
    plt.close()

    return [len(x), RadCompare[0], RadCompare[1],
            RadCompare[4]], bias_and_md  # num, a, b, r, bias and md


def G_reg1d(xx, yy, ww=None):
    """
    description needed
    ww: weights
    """
    rtn = []
    ab = polyfit(xx, yy, 1, w=ww)
    rtn.append(ab[0])
    rtn.append(ab[1])
    rtn.append(std(yy) / std(xx))
    rtn.append(mean(yy) - rtn[2] * mean(xx))
    r = corrcoef(xx, yy)
    rr = r[0, 1] * r[0, 1]
    rtn.append(rr)
    return rtn


######################### 程序全局入口 ##############################
if __name__ == "__main__":
    # 获取程序参数接口
    args = sys.argv[1:]
    help_info = \
        u"""
        [参数样例1]：SAT1+SENSOR1_SAT2+SENSOR2  YYYYMMDD-YYYYMMDD
        [参数样例2]：处理所有卫星对
        """
    if "-h" in args:
        print help_info
        sys.exit(-1)

    # 获取程序所在位置，拼接配置文件
    MainPath, MainFile = os.path.split(os.path.realpath(__file__))
    ProjPath = os.path.dirname(MainPath)
    omPath = os.path.dirname(ProjPath)
    dvPath = os.path.join(os.path.dirname(omPath), "DV")
    cfgFile = os.path.join(ProjPath, "cfg", "global.cfg")

    # 配置不存在预警
    if not os.path.isfile(cfgFile):
        print (u"配置文件不存在 %s" % cfgFile)
        sys.exit(-1)

    # 载入配置文件
    inCfg = ConfigObj(cfgFile)
    MATCH_DIR = inCfg["PATH"]["MID"]["MATCH_DATA"]
    DRA_DIR = inCfg["PATH"]["OUT"]["DRA"]
    MRA_DIR = inCfg["PATH"]["OUT"]["MRA"]
    ABR_DIR = inCfg["PATH"]["OUT"]["ABR"]
    LogPath = inCfg["PATH"]["OUT"]["LOG"]
    Log = LogServer(LogPath)

    # 开启进程池
    threadNum = inCfg["CROND"]["threads"]
    # threadNum = "10"
    pool = Pool(processes=int(threadNum))

    if len(args) == 2:
        Log.info(u"手动日月回归分析程序开始运行-----------------------------")
        satPair = args[0]
        str_time = args[1]
        date_s, date_e = pb_time.arg_str2date(str_time)
        isMonthly = False
        if len(str_time) == 13:
            isMonthly = True
            timeStep = relativedelta(months=1)
        elif len(str_time) == 17:
            timeStep = relativedelta(days=1)
        else:
            print "time format error  yyyymmdd-yyyymmdd or yyyymm-yyyymm"
            sys.exit(-1)
        # 定义参数List，传参给线程池
        args_List = []

        while date_s <= date_e:
            ymd_day = date_s.strftime("%Y%m%d")
            # run(satPair, ymd_day, isMonthly)
            pool.apply_async(run, (satPair, ymd_day, isMonthly))
            date_s = date_s + timeStep

        pool.close()
        pool.join()

    elif len(args) == 0:
        Log.info(u"自动日月回归分析程序开始运行 -----------------------------")
        rolldays = inCfg["CROND"]["rolldays"]
        pairLst = inCfg["PAIRS"].keys()

        for satPair in pairLst:
            ProjMode1 = len(inCfg["PAIRS"][satPair]["colloc_exe"])
            if ProjMode1 == 0:
                continue
            for rdays in rolldays:
                isMonthly = False
                ymd_day = (datetime.utcnow() - relativedelta(
                    days=int(rdays))).strftime("%Y%m%d")
                pool.apply_async(run, (satPair, ymd_day, isMonthly))
            # 增加一个月的作业,默认当前月和上一个月
            isMonthly = True
            ymd_month = (datetime.utcnow() - relativedelta(
                days=int(rolldays[0]))).strftime("%Y%m%d")
            ymd_last_month = (datetime.utcnow() - relativedelta(months=1)).strftime(
                "%Y%m%d")
            pool.apply_async(run, (satPair, ymd_month, isMonthly))
            pool.apply_async(run, (satPair, ymd_last_month, isMonthly))

        pool.close()
        pool.join()
    else:
        print "args: error"
        sys.exit(-1)
