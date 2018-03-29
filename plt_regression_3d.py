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
from DV import dv_pub_legacy
from multiprocessing import Pool, Lock

import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
from plt_io import ReadHDF5, loadYamlCfg

lock = Lock()


def run(pair, ymd, isMonthly):
    '''
    pair: sat1+sensor1_sat2+sensor2
    ymd: str YYYYMMDD
    '''
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
                oneHDF5.tbb1 = oneHDF5.tbb1[deletezeros] if len(oneHDF5.tbb1) > 0 else oneHDF5.tbb1
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
            if 'day' in Day_Night and o_name not in dict_cabr_d:
                dict_cabr_d[o_name] = {}
            if 'night' in Day_Night and o_name not in dict_cabr_n:
                dict_cabr_n[o_name] = {}

            if x.size < 10:
                Log.error("Not enough match point to draw.")
                if 'all' in Day_Night:
                    dict_cabr[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                if 'day' in Day_Night:
                    dict_cabr_d[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                if 'night' in Day_Night:
                    dict_cabr_n[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                continue
            # regression starts
            if 'all' in Day_Night:
                o_file = os.path.join(cur_path,
                                      '%s_%s_%s_ALL_%s' % (pair, o_name, chan, str_time))
                abr = plot(x, y, weight, o_file,
                           num_file, part1, part2, chan, str_time,
                           xname, xname_l, xunit, xmin, xmax,
                           yname, yname_l, yunit, ymin, ymax,
                           diagonal, isMonthly)
                if abr:
                    dict_cabr[o_name][chan] = abr
                else:
                    dict_cabr[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
            print(each, chan)
            # ------- day ----------
            if 'day' in Day_Night:

                if day_index is not None and np.where(day_index)[0].size > 10:

                    o_file = os.path.join(cur_path,
                                          '%s_%s_%s_Day_%s' % (pair, o_name, chan, str_time))

                    print('x, y, day_index', len(x), len(y), len(day_index))
                    x_d = x[day_index]
                    y_d = y[day_index]

                    w_d = weight[day_index] if weight is not None else None

                    abr = plot(x_d, y_d, w_d, o_file,
                               num_file, part1, part2, chan, str_time,
                               xname, xname_l, xunit, xmin, xmax,
                               yname, yname_l, yunit, ymin, ymax,
                               diagonal, isMonthly)

                    if abr:
                        dict_cabr_d[o_name][chan] = abr
                    else:
                        dict_cabr_n[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                else:
                    dict_cabr_d[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]

            # ---------night ------------
            if 'night' in Day_Night:
                if night_index is not None and np.where(night_index)[0].size > 10:
                    o_file = os.path.join(cur_path, '%s_%s_%s_Night_%s' % (
                        pair, o_name, chan, str_time))

                    x_n = x[night_index]
                    y_n = y[night_index]
                    w_n = weight[night_index] if weight is not None else None
                    abr = plot(x_n, y_n, w_n, o_file,
                               num_file, part1, part2, chan, str_time,
                               xname, xname_l, xunit, xmin, xmax,
                               yname, yname_l, yunit, ymin, ymax,
                               diagonal, isMonthly)
                    if abr:
                        dict_cabr_n[o_name][chan] = abr
                    else:
                        dict_cabr_n[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
                else:
                    dict_cabr_n[o_name][chan] = [0, np.NaN, np.NaN, np.NaN]
            oneHDF5.clear()

        # write txt
        lock.acquire()
        channel = plt_cfg[each]['chan']
        if 'all' in Day_Night:
            for o_name in dict_cabr:
                writeTxt(channel, part1, part2, o_name, str_time, dict_cabr,
                         'ALL', isMonthly)
        if 'day' in Day_Night:
            for o_name in dict_cabr_d:
                writeTxt(channel, part1, part2, o_name, str_time, dict_cabr_d,
                         'Day', isMonthly)
        if 'night' in Day_Night:
            for o_name in dict_cabr_n:
                writeTxt(channel, part1, part2, o_name, str_time, dict_cabr_n,
                         'Night', isMonthly)
        lock.release()


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
         diagonal, isMonthly):
    plt.style.use(os.path.join(dvPath, 'dv_pub_legacy.mplstyle'))

    titleName = '%s-%s' % (xname.upper(), yname.upper())
    if xunit != "":
        xlabel = '{} {} (${}$)'.format(part1, xname_l, xunit)
    else:
        xlabel = '{} {}'.format(part1, xname_l)

    if yunit != "":
        ylabel = '{} {} (${}$)'.format(part2, yname_l, yunit)
    else:
        ylabel = '{} {}'.format(part2, yname_l)

    DictTitle_rad = {
        'xlabel': xlabel,
        'ylabel': ylabel,
        'title': '{} Regression {} Days {}_{} {} {}'.format(
            titleName, num_file, part1, part2, chan, ymd)}

    w = 1.0 / weight if weight is not None else None
    RadCompare = G_reg1d(x, y, w)
    # 过滤 正负 delta+4倍std 的杂点 ------------------------
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

    pb_io.make_sure_path_exists(os.path.dirname(o_file))

    if diagonal:
        dv_pub_legacy.draw_Scatter_Bar(
            x, y,
            o_file, DictTitle_rad,
            [['{:15}: {:7.4f}'.format('Slope', RadCompare[0]),
              '{:15}: {:7.4f}'.format('Intercept', RadCompare[1]),
              '{:15}: {:7.4f}'.format('Cor-Coef', RadCompare[4]),
              '{:15}: {:7d}'.format('Number', length_rad)]], '',
            part1, part2, xname, xname_l,
            xmin, xmax, ymin, ymax)

    else:
        dv_pub_legacy.draw_Scatter(
            x, y,
            o_file, DictTitle_rad,
            [['{:15}: {:7.4f}'.format('Slope', RadCompare[0]),
             '{:15}: {:7.4f}'.format('Intercept', RadCompare[1]),
             '{:15}: {:7.4f}'.format('Cor-Coef', RadCompare[4]),
             '{:15}: {:7d}'.format('Number', length_rad)]], '',
            xmin, xmax, ymin, ymax, diagonal)
    if isMonthly:
        o_file = o_file + "_density"
        x = extraction_point(x, 10000, 3)
        y = extraction_point(y, 10000, 3)
        dv_pub_legacy.draw_density(
            x, y,
            o_file, DictTitle_rad,
            [['{:15}: {:7.4f}'.format('Slope', RadCompare[0]),
             '{:15}: {:7.4f}'.format('Intercept', RadCompare[1]),
             '{:15}: {:7.4f}'.format('Cor-Coef', RadCompare[4]),
             '{:15}: {:7d}'.format('Number', length_rad)]], '',
            xmin, xmax, ymin, ymax, diagonal)

    return [len(x), RadCompare[0], RadCompare[1], RadCompare[4]]  # num, a, b, r


def G_reg1d(xx, yy, ww=None):
    '''
    description needed
    ww: weights
    '''
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


def extraction_point(lyst, counts, times):
    """
    对一个一维 numpy 列表进行数据抽取，直到数据的数量小于某个值
    :param lyst: 一维 numpy 列表
    :param counts: 需要控制的数量
    :param times: 每次迭代减少的倍数
    :return:
    """
    if len(lyst) > counts:
        idx = [x for x in xrange(0, len(lyst), times)]
        lyst = lyst[idx]
        if len(lyst) > counts:
            lyst = extraction_point(lyst, counts, times)
    return lyst


# def isDay(sec):
#     '''
#     not used
#     '''
#     mjd = sec / 24. / 3600. + 48988.
#     dt = datetime.datetime(1858, 11, 17) + relativedelta(days=mjd)
#     return dt.hour == 0

######################### 程序全局入口 ##############################
if __name__ == "__main__":
    # ###### test start
    # run('FY3D+MERSI_METOP-A+IASI', '20180101-20180101', False)
    # ###### test end

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
            pool.apply_async(run, (satPair, ymd, isMonthly))
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
