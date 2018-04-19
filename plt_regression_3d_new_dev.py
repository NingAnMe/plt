# coding: utf-8
'''
Created on 2016年1月6日
读取匹配后的 HDF5 文件，画散点回归图，生成 abr 文件

@author: duxiang, zhangtao, anning
'''
import os, sys, calendar
from PB.pb_time import get_local_time

from configobj import ConfigObj
from dateutil.relativedelta import relativedelta
from numpy.lib.polynomial import polyfit
from numpy.ma.core import std, mean
from numpy.ma.extras import corrcoef
import numpy as np

from PB import pb_time, pb_io
from PB.CSC.pb_csc_console import LogServer
from DV import dv_pub_3d
from DV.dv_pub_3d import FONT0, bias_information, day_data_write, draw_distribution
from multiprocessing import Pool, Lock
from DM.SNO.dm_sno_cross_calc_map import RED, BLUE, EDGE_GRAY, ORG_NAME
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
from plt_io import ReadHDF5, loadYamlCfg

lock = Lock()


def run(pair, ymd, isMonthly):
    """
    pair: sat1+sensor1_sat2+sensor2
    ymd: str YYYYMMDD
    """
    part1, part2 = pair.split('_')
    sat1, sensor1 = part1.split('+')
    sat2, sensor2 = part2.split('+')

    if 'FY2' in part1 or 'FY4' in part1:
        Type = "GEOLEO"
    elif 'FY3' in part1:
        Type = "LEOLEO"
    else:
        return

    # load yaml config file
    plt_cfg_file = os.path.join(MainPath, '%s_%s_3d.yaml' % (sensor1, sensor2))
    plt_cfg = loadYamlCfg(plt_cfg_file)
    if plt_cfg is None:
        return

    Log.info(u"----- Start Drawing Regression-Pic, PAIR: {}, YMD: {} -----".format(pair, ymd))

    for each in plt_cfg['regression']:
        dict_cabr = {}
        dict_cabr_d = {}
        dict_cabr_n = {}
        dict_md = {}
        dict_md_d = {}
        dict_md_n = {}

        # 需要回滚的天数
        if isMonthly:
            PERIOD = calendar.monthrange(int(ymd[:4]), int(ymd[4:6]))[1]  # 当月天数
            ymd = ymd[:6] + '%02d' % PERIOD  # 当月最后一天
        else:
            PERIOD = plt_cfg[each]['days']

        # must be in 'all', 'day', 'night'
        Day_Night = ['all', 'day', 'night']
        if 'time' in plt_cfg[each].keys():
            Day_Night = plt_cfg[each]['time']
            for t in Day_Night:
                if t not in ['all', 'day', 'night']:
                    Day_Night.remove(t)

        for idx, chan in enumerate(plt_cfg[each]['chan']):
            Log.info(u"Start Drawing {} Channel {}".format(each, chan))
            oneHDF5 = ReadHDF5()
            num_file = PERIOD
            for daydelta in xrange(PERIOD):
                cur_ymd = pb_time.ymd_plus(ymd, -daydelta)
                hdf5_name = 'COLLOC+%sIR,%s_C_BABJ_%s.hdf5' % (Type, pair, cur_ymd)
                filefullpath = os.path.join(MATCH_DIR, pair, hdf5_name)
                if not os.path.isfile(filefullpath):
                    Log.info(u"File not found: {}".format(filefullpath))
                    num_file -= 1
                    continue
                if not oneHDF5.LoadData(filefullpath, chan):
                    Log.error('Error occur when reading %s of %s' % (chan, filefullpath))
            if num_file == 0:
                Log.error(u"No file found.")
                continue
            elif num_file != PERIOD:
                Log.error(u"{} of {} file(s) found.".format(num_file, PERIOD))

            if isMonthly:
                str_time = ymd[:6]
                cur_path = os.path.join(MRA_DIR, pair, str_time)
            else:
                str_time = ymd
                cur_path = os.path.join(DRA_DIR, pair, str_time)

            # delete 0 in std
            if len(oneHDF5.rad1_std) > 0.0001:  # TODO: 有些极小的std可能是异常值，而导致权重极大，所以 std>0 改成 std>0.0001
                deletezeros = np.where(oneHDF5.rad1_std > 0.0001)
                oneHDF5.rad1_std = oneHDF5.rad1_std[deletezeros]
                oneHDF5.rad1 = oneHDF5.rad1[deletezeros] if len(oneHDF5.rad1) > 0 else oneHDF5.rad1
                oneHDF5.rad2 = oneHDF5.rad2[deletezeros] if len(oneHDF5.rad2) > 0 else oneHDF5.rad2
                print('chan1', chan, len(oneHDF5.tbb1))
                oneHDF5.tbb1 = oneHDF5.tbb1[deletezeros] if len(oneHDF5.tbb1) > 0 else oneHDF5.tbb1
                print('chan1', chan, len(oneHDF5.tbb1))
                oneHDF5.tbb2 = oneHDF5.tbb2[deletezeros] if len(oneHDF5.tbb2) > 0 else oneHDF5.tbb2
                oneHDF5.time = oneHDF5.time[deletezeros] if len(oneHDF5.time) > 0 else oneHDF5.time
            if len(oneHDF5.ref1_std) > 0.0001:
                deletezeros = np.where(oneHDF5.ref1_std > 0.0001)
                oneHDF5.ref1_std = oneHDF5.ref1_std[deletezeros]
                oneHDF5.ref1 = oneHDF5.ref1[deletezeros] if len(oneHDF5.ref1) > 0 else oneHDF5.ref1
                oneHDF5.ref2 = oneHDF5.ref2[deletezeros] if len(oneHDF5.ref2) > 0 else oneHDF5.ref2
                oneHDF5.dn1 = oneHDF5.dn1[deletezeros] if len(oneHDF5.dn1) > 0 else oneHDF5.dn1
                oneHDF5.time = oneHDF5.time[deletezeros] if len(oneHDF5.time) > 0 else oneHDF5.time

            # find out day and night
            if ('day' in Day_Night or 'night' in Day_Night) and len(oneHDF5.time) > 0:
                jd = oneHDF5.time / 24. / 3600.  # jday from 1993/01/01 00:00:00
                hour = ((jd - jd.astype('int8')) * 24).astype('int8')
                day_index = (hour < 10)  # utc hour<10 is day
                night_index = np.logical_not(day_index)
            else:
                day_index = None
                night_index = None
            print('3')
            # 将每个对通用的属性值放到对循环，每个通道用到的属性值放到通道循环
            # get threhold, unit, names...
            xname, yname = each.split('-')
            xname_l = plt_cfg[each]['x_name']
            xunit = plt_cfg[each]['x_unit']
            xlimit = plt_cfg[each]['x_range'][idx]
            xmin, xmax = xlimit.split('-')
            xmin = float(xmin)
            xmax = float(xmax)
            yname_l = plt_cfg[each]['y_name']
            yunit = plt_cfg[each]['y_unit']
            ylimit = plt_cfg[each]['y_range'][idx]
            ymin, ymax = ylimit.split('-')
            ymin = float(ymin)
            ymax = float(ymax)

            weight = None
            if 'rad' in xname:
                x = oneHDF5.rad1
            elif 'tbb' in xname:
                x = oneHDF5.tbb1
            elif 'ref' in xname:
                x = oneHDF5.ref1
            elif 'dn' in xname:
                x = oneHDF5.dn1
            else:
                Log.error("Can't plot %s" % each)
                continue
            if 'rad' in yname:
                y = oneHDF5.rad2
            elif 'tbb' in yname:
                y = oneHDF5.tbb2
            elif 'ref' in yname:
                y = oneHDF5.ref2
            else:
                Log.error("Can't plot %s" % each)
                continue

            if 'rad' in xname and 'rad' in yname:
                if len(oneHDF5.rad1_std) > 0:
                    weight = oneHDF5.rad1_std
                o_name = 'RadCalCoeff'
            elif 'tbb' in xname and 'tbb' in yname:
                o_name = 'TBBCalCoeff'
            elif 'ref' in xname and 'ref' in yname:
                if len(oneHDF5.ref1_std) > 0:
                    weight = oneHDF5.ref1_std
                o_name = 'CorrcCoeff'
            elif 'dn' in xname and 'ref' in yname:
                o_name = 'CalCoeff'

            # 画对角线
            if xname == yname:
                diagonal = True
            else:
                diagonal = False

            if 'all' in Day_Night and o_name not in dict_cabr:
                dict_cabr[o_name] = {}
                dict_md[xname] = {}
            if 'day' in Day_Night and o_name not in dict_cabr_d:
                dict_cabr_d[o_name] = {}
                dict_md_d[xname] = {}
            if 'night' in Day_Night and o_name not in dict_cabr_n:
                dict_cabr_n[o_name] = {}
                dict_md_n[xname] = {}

            if x.size < 10:
                Log.error("Not enough match point to draw.")
                if 'all' in Day_Night:
                    dict_cabr[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                    dict_md[xname][chan] = np.NaN
                if 'day' in Day_Night:
                    dict_cabr_d[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                    dict_md_d[xname][chan] = np.NaN
                if 'night' in Day_Night:
                    dict_cabr_n[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                    dict_md_n[xname][chan] = np.NaN
                continue
            print('2')
            # regression starts
            if 'all' in Day_Night:
                o_file = os.path.join(cur_path,
                                      '%s_%s_%s_ALL_%s' % (pair, o_name, chan, str_time))
                abr, bias = plot(x, y, weight, o_file,
                           num_file, part1, part2, chan, str_time,
                           xname, xname_l, xunit, xmin, xmax,
                           yname, yname_l, yunit, ymin, ymax,
                           diagonal, isMonthly)
                if abr:
                    dict_cabr[o_name][chan] = abr
                else:
                    dict_cabr[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                if bias:
                    dict_md[xname][chan] = bias
                else:
                    dict_md[xname][chan] = np.NaN
            print('4')
            print(each, chan)
            # ------- day ----------
            if 'day' in Day_Night:
                print('10')
                if day_index is not None and np.where(day_index)[0].size > 10:
                    print('11')
                    o_file = os.path.join(cur_path,
                                          '%s_%s_%s_Day_%s' % (pair, o_name, chan, str_time))
                    print('6')
                    print('x, y, day_all', len(x), len(y), len(day_index))
                    x_d = x[day_index]
                    y_d = y[day_index]
                    print('8')
                    w_d = weight[day_index] if weight is not None else None
                    print('9')
                    print('x, y, day_index', len(x_d), len(y_d), len(day_index))
                    abr, bias = plot(x_d, y_d, w_d, o_file,
                               num_file, part1, part2, chan, str_time,
                               xname, xname_l, xunit, xmin, xmax,
                               yname, yname_l, yunit, ymin, ymax,
                               diagonal, isMonthly)
                    print('7')
                    if abr:
                        dict_cabr_d[o_name][chan] = abr
                    else:
                        dict_cabr_d[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                    if bias:
                        dict_md_d[xname][chan] = bias
                    else:
                        dict_md_d[xname][chan] = np.NaN
                else:
                    dict_cabr_d[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                    dict_md_d[xname][chan] = np.NaN
            print('5')
            # ---------night ------------
            if 'night' in Day_Night:
                if night_index is not None and np.where(night_index)[0].size > 10:
                    o_file = os.path.join(cur_path, '%s_%s_%s_Night_%s' % (
                        pair, o_name, chan, str_time))

                    x_n = x[night_index]
                    y_n = y[night_index]
                    w_n = weight[night_index] if weight is not None else None
                    abr, bias = plot(x_n, y_n, w_n, o_file,
                               num_file, part1, part2, chan, str_time,
                               xname, xname_l, xunit, xmin, xmax,
                               yname, yname_l, yunit, ymin, ymax,
                               diagonal, isMonthly)
                    if abr:
                        dict_cabr_n[o_name][chan] = abr
                    else:
                        dict_cabr_n[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                    if bias:
                        dict_md_n[xname][chan] = bias
                    else:
                        dict_md_n[xname][chan] = np.NaN
                else:
                    dict_cabr_n[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                    dict_md_n[xname][chan] = np.NaN
            oneHDF5.clear()
        print('1')
        # write txt
        lock.acquire()
        channel = plt_cfg[each]['chan']
        if 'all' in Day_Night:
            for o_name in dict_cabr:
                writeTxt(channel, part1, part2, o_name, str_time, dict_cabr,
                         'ALL', isMonthly)
                write_md(channel, part1, part2, xname, ymd,
                         dict_md, 'ALL', isMonthly)
        if 'day' in Day_Night:
            for o_name in dict_cabr_d:
                writeTxt(channel, part1, part2, o_name, str_time, dict_cabr_d,
                         'Day', isMonthly)
                write_md(channel, part1, part2, xname, ymd,
                         dict_md_d, 'Day', isMonthly)
        if 'night' in Day_Night:
            for o_name in dict_cabr_n:
                writeTxt(channel, part1, part2, o_name, str_time, dict_cabr_n,
                         'Night', isMonthly)
                write_md(channel, part1, part2, xname, ymd,
                         dict_md_n, 'Night', isMonthly)
        lock.release()


def write_md(channel, part1, part2, xname, ymd,
             dict_md, day_or_night, is_monthly):
    """
    生成 RMD 文件
    :param channel:
    :param part1:
    :param part2:
    :param o_name:
    :param ymd:
    :param data:
    :param DayOrNight:
    :param isMonthly:
    :return:
    """
    if not (xname in ["ref", "tbb"]):
        return

    for chan in channel:
        o_path = os.path.join(ABR_DIR, '%s_%s' % (part1, part2), "MD")
        file_name_monthly = os.path.join(
            o_path, '%s_%s_%s_%s_%s_Monthly.txt' % (
                part1, part2, xname.upper(), chan, day_or_night))
        file_name_daily = os.path.join(
            o_path, '%s_%s_%s_%s_%s_Daily.txt' % (
                part1, part2, xname.upper(), chan, day_or_night))
        title = 'date   MD\n'
        data = "{}  {}\n".format(ymd, dict_md[xname][chan])

        day_data_write(title, data, file_name_daily)

        md_data = load_day_md(file_name_daily)
        data_monthly = month_average(md_data)
        with open(file_name_monthly, 'w') as f:
            f.write(title)
            f.writelines(data_monthly)

        print file_name_daily
        print file_name_monthly


def load_day_md(md_file):
    """
    读取日的 MD 文件，返回 np.array
    :param md_file:
    :return:
    """
    names = ('date', 'md',)
    formats = ('object', 'f4')
    print md_file
    data = np.loadtxt(md_file,
                      converters={0: lambda x: datetime.strptime(x, "%Y%m%d")},
                      dtype={'names': names,
                             'formats': formats},
                      skiprows=1, ndmin=1)
    return data


def month_average(day_data):
    """
    由 EXT 日数据生成 EXT 月平均数据
    :param day_data: EXT 日数据
    :return:
    """
    month_datas = []
    ymd_start = day_data['date'][0]  # 第一天日期
    ymd_end = day_data['date'][-1]  # 最后一天日期
    date_start = ymd_start - relativedelta(days=(ymd_start.day - 1))  # 第一个月第一天日期

    while date_start <= ymd_end:
        # 当月最后一天日期
        date_end = date_start + relativedelta(months=1) - relativedelta(days=1)

        # 查找当月所有数据
        day_date = day_data['date']
        month_idx = np.where(np.logical_and(day_date >= date_start,
                                            day_date <= date_end))

        avg_month = day_data['md'][month_idx]
        not_nan_idx = np.isfinite(avg_month)
        avg_month = avg_month[not_nan_idx]

        ymd_data = date_start.strftime('%Y%m%d')
        avg_data = avg_month.mean()

        data = "{}  {}\n".format(ymd_data, avg_data)

        month_datas.append(data)

        date_start = date_start + relativedelta(months=1)

    return month_datas


def writeTxt(channel, part1, part2, o_name, ymd,
             dict_cabr, DayOrNight, isMonthly):
    """
    生成abr文件
    ymd: YYYYMMDD or YYYYMM
    """
    if len(ymd) == 6:
        ymd = ymd + '01'
    if isMonthly:
        FileName = os.path.join(ABR_DIR, '%s_%s' % (part1, part2),
                                '%s_%s_%s_%s_Monthly.txt' % (part1, part2, o_name, DayOrNight))
    else:
        FileName = os.path.join(ABR_DIR, '%s_%s' % (part1, part2),
                                '%s_%s_%s_%s_%s.txt' % (part1, part2, o_name, DayOrNight, ymd[:4]))

    title_line = 'YMD       '
    newline = ''
    for chan in channel:
        title_line = title_line + '  Count(%s) Slope(%s) Intercept(%s) RSquared(%s)' % (chan, chan, chan, chan)
        newline = newline + '  %10d  %-10.6f  %-10.6f  %-10.6f' % (tuple(dict_cabr[o_name][chan]))
    newline = newline + '\n'  # don't forget to end with \n

    allLines = []
    titleLines = []
    DICT_TXT = {}

    pb_io.make_sure_path_exists(os.path.dirname(FileName))

    # 写十行头信息
    titleLines.append('Sat1: %s\n' % part1)
    titleLines.append('Sat2: %s\n' % part2)
    if isMonthly:
        titleLines.append('TimeRange: since launch\n')
        titleLines.append('           Monthly\n')
    else:
        titleLines.append('TimeRange: %s\n' % ymd[:4])
        titleLines.append('           Daily\n')
    titleLines.append('Day or Night: %s\n' % DayOrNight)
    titleLines.append('Calc time : %s\n' % get_local_time().strftime('%Y.%m.%d %H:%M:%S'))
    titleLines.append('\n')
    titleLines.append('\n')
    titleLines.append(title_line + '\n')
    titleLines.append('-' * len(title_line) + '\n')

    #
    if os.path.isfile(FileName) and os.path.getsize(FileName) != 0:
        fp = open(FileName, 'r')
        for i in xrange(10):
            fp.readline()  # 跳过头十行
        Lines = fp.readlines()
        fp.close()
        # 使用字典特性，保证时间唯一，读取数据
        for Line in Lines:
            DICT_TXT[Line[:8]] = Line[8:]
        # 添加或更改数据
        DICT_TXT[ymd] = newline
        # 按照时间排序
        newLines = sorted(DICT_TXT.iteritems(), key=lambda d: d[0], reverse=False)

        for i in xrange(len(newLines)):
            allLines.append(str(newLines[i][0]) + str(newLines[i][1]))
    else:
        allLines.append(ymd + newline)

    fp = open(FileName, 'w')
    fp.writelines(titleLines)
    fp.writelines(allLines)
    fp.close()


def plot(x, y, weight, o_file, num_file, part1, part2, chan, ymd,
         xname, xname_l, xunit, xmin, xmax, yname, yname_l, yunit, ymin, ymax,
         is_diagonal, isMonthly):
    plt.style.use(os.path.join(dvPath, 'dv_pub_regression.mplstyle'))
    print 'right 1'

    # 过滤 正负 delta+8倍std 的杂点 ------------------------
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
    print 'right 2'
    bias = None  # 当 bias 没有被计算的时候，不输出 bias
    if not isMonthly and is_diagonal:
        # return [len(x), RadCompare[0], RadCompare[1], RadCompare[4]]
        fig = plt.figure(figsize=(14, 4.5))
        fig.subplots_adjust(top=0.92, bottom=0.11, left=0.045, right=0.985)
        ax1 = plt.subplot2grid((1, 3), (0, 0))
        ax2 = plt.subplot2grid((1, 3), (0, 1))
        ax3 = plt.subplot2grid((1, 3), (0, 2))
        # 图片 Title
        titleName = '%s-%s' % (xname.upper(), yname.upper())
        title = '{} Regression {} Days {}_{} {} {}'.format(
            titleName, num_file, part1, part2, chan, ymd)

        # 画回归图 -----------------------------------------------
        print 'draw regression'
        regress_xmin = xmin
        regress_xmax = xmax
        regress_ymin = ymin
        regress_ymax = ymax
        regress_axislimit = {
            "xlimit": (regress_xmin, regress_xmax),
            "ylimit": (regress_ymin, regress_ymax),
        }

        if xunit != "":
            xlabel = '{} {} (${}$)'.format(part1, xname_l, xunit)
        else:
            xlabel = '{} {}'.format(part1, xname_l)

        if yunit != "":
            ylabel = '{} {} (${}$)'.format(part2, yname_l, yunit)
        else:
            ylabel = '{} {}'.format(part2, yname_l)

        regress_label = {
            "xlabel": xlabel, "ylabel": ylabel,
        }

        if xname == "tbb":
            regress_locator = {"locator_x": (5, 5), "locator_y": (5, 5)}
        elif xname == "ref":
            regress_locator = {"locator_x": (None, None), "locator_y": (None, 5)}

        regress_annotate = {
                "left": ['{:15}: {:7.4f}'.format('Slope', RadCompare[0]),
                         '{:15}: {:7.4f}'.format('Intercept', RadCompare[1]),
                         '{:15}: {:7.4f}'.format('Cor-Coef', RadCompare[4]),
                         '{:15}: {:7d}'.format('Number', length_rad)]
        }

        regress_diagonal = {"line_color": '#808080', "line_width": 1.2}

        regress_regressline = {"line_color": 'r', "line_width": 1.2}

        scatter_point = {"scatter_alpha": 0.8}

        dv_pub_3d.draw_regression(
            ax1, x, y, regress_label, ax_annotate=regress_annotate,
            axislimit=regress_axislimit, locator=regress_locator,
            diagonal=regress_diagonal, regressline=regress_regressline,
            scatter_point=scatter_point,
        )

        # 画偏差分布图 ---------------------------------------------
        print 'draw distribution'

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
        ylabel = "{} minus {} {} bias".format(part1, part2, xname)
        distri_label = {
            "xlabel": xlabel, "ylabel": ylabel,
        }

        # 获取 MeanBias 信息
        boundary = xmin + (xmax - xmin) * 0.15
        bias_info = bias_information(x, y, boundary)

        # 绝对偏差和相对偏差信息 TBB=250K  REF=0.25
        bias = np.NaN  # RMD or TBB bias
        bias_info_md = ''
        ab = RadCompare
        a = ab[0]
        b = ab[1]
        if xname == 'tbb':
            bias = 250 - (250 * a + b)
            bias_info_md = "TBB Bias (250K) : {:.4f} K".format(bias)
        elif xname == 'ref':
            bias = (0.25 - (0.25 * a + b)) / 0.25 * 100
            bias_info_md = "Relative Bias (REF 0.25) : {:.4f} %".format(bias)

        # 添加注释信息
        distri_annotate = {"left": [], "right": []}
        distri_annotate.get("left").append(bias_info.get("info_lower"))
        distri_annotate.get("left").append(bias_info.get("info_greater"))
        distri_annotate.get("left").append(bias_info_md)

        # 添加间隔数量
        if xname == "tbb":
            distri_locator = {"locator_x": (5, None), "locator_y": (8, 5)}
        elif xname == 'ref':
            distri_locator = {"locator_x": (None, None), "locator_y": (8, 5)}

        # y=0 线配置
        zeroline = {"line_color": '#808080', "line_width": 1.0}

        # 偏差点配置
        scatter_delta = {
            "scatter_marker": 'o', "scatter_size": 5, "scatter_alpha": 0.8,
            "scatter_linewidth": 0, "scatter_zorder": 100,
            "scatter_color": BLUE,
        }

        # 偏差回归线配置
        regressline = {"line_color": 'r', "line_width": 1.2}
        print "ddddd1"
        dv_pub_3d.draw_distribution(ax2, x, y, label=distri_label,
                                    ax_annotate=distri_annotate,
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
            "xlabel": histogram_xlabel, "ylabel": histogram_ylabel,
        }

        # 添加间隔数量
        if xname == "tbb":
            histogram_locator = {"locator_x": (5, None), "locator_y": (None, 5)}
        elif xname == "ref":
            histogram_locator = {"locator_x": (None, None), "locator_y": (None, 5)}

        histogram_x = {
            "label": part1, "color": "red", "alpha": 0.4, "bins": 100
        }
        histogram_y = {
            "label": part2, "color": "blue", "alpha": 0.4, "bins": 100
        }

        dv_pub_3d.draw_histogram(
            ax3, x, label=histogram_label, locator=histogram_locator,
            axislimit=histogram_axislimit, histogram=histogram_x,
        )

        dv_pub_3d.draw_histogram(
            ax3, y, label=histogram_label, locator=histogram_locator,
            axislimit=histogram_axislimit, histogram=histogram_y,
        )
        print 'right 4'
    elif not isMonthly and not is_diagonal:
        fig = plt.figure(figsize=(4.5, 4.5))
        fig.subplots_adjust(top=0.89, bottom=0.13, left=0.15, right=0.91)
        ax1 = plt.subplot2grid((1, 1), (0, 0))
        # 图片 Title
        titleName = '%s-%s' % (xname.upper(), yname.upper())
        title = '{} Regression {} Days\n{}_{} {} {}'.format(
            titleName, num_file, part1, part2, chan, ymd)
        # 画回归图 ----------------------------------------------------
        print 'draw regression'
        regress_xmin = xmin
        regress_xmax = xmax
        regress_ymin = ymin
        regress_ymax = ymax
        regress_axislimit = {
            "xlimit": (regress_xmin, regress_xmax),
            "ylimit": (regress_ymin, regress_ymax),
        }

        if xunit != "":
            xlabel = '{} {} (${}$)'.format(part1, xname_l, xunit)
        else:
            xlabel = '{} {}'.format(part1, xname_l)

        if yunit != "":
            ylabel = '{} {} (${}$)'.format(part2, yname_l, yunit)
        else:
            ylabel = '{} {}'.format(part2, yname_l)

        regress_label = {
            "xlabel": xlabel, "ylabel": ylabel,
        }

        if xname == "tbb":
            regress_locator = {"locator_x": (5, None), "locator_y": (5, 5)}
        elif xname == "ref":
            regress_locator = {"locator_x": (None, None), "locator_y": (None, 5)}
        elif xname == "dn":
            regress_locator = {"locator_x": (5, None),
                               "locator_y": (None, 5)}

        regress_annotate = {
            "left": ['{:15}: {:7.4f}'.format('Slope', RadCompare[0]),
                     '{:15}: {:7.4f}'.format('Intercept', RadCompare[1]),
                     '{:15}: {:7.4f}'.format('Cor-Coef', RadCompare[4]),
                     '{:15}: {:7d}'.format('Number', length_rad)]
        }

        regress_diagonal = {"line_color": '#808080', "line_width": 1.2}

        regress_regressline = {"line_color": 'r', "line_width": 1.2}

        scatter_point = {"scatter_alpha": 0.8}

        dv_pub_3d.draw_regression(
            ax1, x, y, regress_label, ax_annotate=regress_annotate,
            axislimit=regress_axislimit, locator=regress_locator,
            diagonal=regress_diagonal, regressline=regress_regressline,
            scatter_point=scatter_point,
        )
    elif isMonthly:
        o_file = o_file + "_density"

        fig = plt.figure(figsize=(4.5, 4.5))
        fig.subplots_adjust(top=0.89, bottom=0.13, left=0.15, right=0.91)
        ax1 = plt.subplot2grid((1, 1), (0, 0))
        # 图片 Title Label
        titleName = '%s-%s' % (xname.upper(), yname.upper())
        title = '{} Regression {} Days\n{}_{} {} {}'.format(
            titleName, num_file, part1, part2, chan, ymd)
        # 画密度图 -----------------------------------------------------
        print 'draw density'
        density_xmin = xmin
        density_xmax = xmax
        density_ymin = ymin
        density_ymax = ymax
        density_axislimit = {
            "xlimit": (density_xmin, density_xmax),
            "ylimit": (density_ymin, density_ymax),
        }

        if xunit != "":
            xlabel = '{} {} (${}$)'.format(part1, xname_l, xunit)
        else:
            xlabel = '{} {}'.format(part1, xname_l)

        if yunit != "":
            ylabel = '{} {} (${}$)'.format(part2, yname_l, yunit)
        else:
            ylabel = '{} {}'.format(part2, yname_l)

        density_label = {
            "xlabel": xlabel, "ylabel": ylabel,
        }

        if xname == "tbb":
            density_locator = {"locator_x": (5, None), "locator_y": (5, 5)}
        elif xname == "ref":
            density_locator = {"locator_x": (None, None), "locator_y": (None, 5)}
        if xname == "dn":
            density_locator = {"locator_x": (5, None),
                               "locator_y": (None, 5)}

        density_annotate = {
            "left": ['{:15}: {:7.4f}'.format('Slope', RadCompare[0]),
                     '{:15}: {:7.4f}'.format('Intercept', RadCompare[1]),
                     '{:15}: {:7.4f}'.format('Cor-Coef', RadCompare[4]),
                     '{:15}: {:7d}'.format('Number', length_rad)]
        }

        density_diagonal = {"line_color": '#808080', "line_width": 1.2}

        density_regressline = {"line_color": 'r', "line_width": 1.2}

        density = {
            "size": 5, "marker": "o", "alpha": 1
        }
        dv_pub_3d.draw_regression(
            ax1, x, y, density_label, ax_annotate=density_annotate,
            axislimit=density_axislimit, locator=density_locator,
            diagonal=density_diagonal, regressline=density_regressline,
            density=density,
        )
    else:
        print 'No output Pic {} : '.format(ymd)
        return

    print 'right 3'
    fig.suptitle(title, fontsize=11, fontproperties=FONT0)
    pb_io.make_sure_path_exists(os.path.dirname(o_file))
    fig.savefig(o_file, dpi=100)
    print o_file
    fig.clear()
    plt.close()

    return [len(x), RadCompare[0], RadCompare[1], RadCompare[4]], bias  # num, a, b, r, md


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

# 获取程序参数接口
args = sys.argv[1:]
help_info = \
    u'''
    [参数样例1]：SAT1+SENSOR1_SAT2+SENSOR2  YYYYMMDD-YYYYMMDD
    [参数样例2]：处理所有卫星对
    '''
if '-h' in args:
    print help_info
    sys.exit(-1)

# 获取程序所在位置，拼接配置文件
MainPath, MainFile = os.path.split(os.path.realpath(__file__))
ProjPath = os.path.dirname(MainPath)
omPath = os.path.dirname(ProjPath)
dvPath = os.path.join(os.path.dirname(omPath), 'DV')
cfgFile = os.path.join(ProjPath, 'cfg', 'global.cfg')

# 配置不存在预警
if not os.path.isfile(cfgFile):
    print (u'配置文件不存在 %s' % cfgFile)
    sys.exit(-1)

# 载入配置文件
inCfg = ConfigObj(cfgFile)
MATCH_DIR = inCfg['PATH']['MID']['MATCH_DATA']
DRA_DIR = inCfg['PATH']['OUT']['DRA']
MRA_DIR = inCfg['PATH']['OUT']['MRA']
ABR_DIR = inCfg['PATH']['OUT']['ABR']
LogPath = inCfg['PATH']['OUT']['LOG']
Log = LogServer(LogPath)

# 开启进程池
threadNum = inCfg['CROND']['threads']
pool = Pool(processes=int(threadNum))

if len(args) == 2:
    Log.info(u'手动日月回归分析程序开始运行-----------------------------')
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
        print 'time format error  yyyymmdd-yyyymmdd or yyyymm-yyyymm'
        sys.exit(-1)
    # 定义参数List，传参给线程池
    args_List = []

    while date_s <= date_e:
        ymd = date_s.strftime('%Y%m%d')
        run(satPair, ymd, isMonthly)
        # pool.apply_async(run, (satPair, ymd, isMonthly))
        date_s = date_s + timeStep

    pool.close()
    pool.join()

elif len(args) == 0:
    Log.info(u'自动日月回归分析程序开始运行 -----------------------------')
    rolldays = inCfg['CROND']['rolldays']
    pairLst = inCfg['PAIRS'].keys()

    for satPair in pairLst:
        ProjMode1 = len(inCfg['PAIRS'][satPair]['colloc_exe'])
        if ProjMode1 == 0:
            continue
        for rdays in rolldays:
            isMonthly = False
            ymd = (datetime.utcnow() - relativedelta(days=int(rdays))).strftime('%Y%m%d')
            pool.apply_async(run, (satPair, ymd, isMonthly))
        # 增加一个月的作业,默认当前月和上一个月
        isMonthly = True
        ymd = (datetime.utcnow() - relativedelta(days=int(rolldays[0]))).strftime('%Y%m%d')
        ymdLast = (datetime.utcnow() - relativedelta(months=1)).strftime('%Y%m%d')
        pool.apply_async(run, (satPair, ymd, isMonthly))
        pool.apply_async(run, (satPair, ymdLast, isMonthly))

    pool.close()
    pool.join()
else:
    print 'args: error'
    sys.exit(-1)
