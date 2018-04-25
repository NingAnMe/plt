# coding: utf-8
"""
Created on 2016年1月6日
读取匹配后的HDF5文件，画散点回归图，生成abr文件

@author: duxiang, zhangtao, anning
"""
import calendar
import os
import sys
from datetime import datetime
from multiprocessing import Pool

import numpy as np

from configobj import ConfigObj
from dateutil.relativedelta import relativedelta
from numpy.lib.polynomial import polyfit
from numpy.ma.core import std, mean
from numpy.ma.extras import corrcoef

from DV.dv_pub_3d import plt, add_annotate, set_tick_font, FONT0, FONT_MONO,\
    draw_distribution, draw_bar, draw_histogram, bias_information, FONT1
from PB import pb_time, pb_io
from DM.SNO.dm_sno_cross_calc_map import RED, BLUE, EDGE_GRAY, ORG_NAME
from PB.CSC.pb_csc_console import LogServer
from plt_io import ReadHDF5, loadYamlCfg


def run(pair, ymd):
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

    # load yaml
    plt_cfg_file = os.path.join(MainPath, '%s_%s_3d.yaml' % (sensor1, sensor2))
    plt_cfg = loadYamlCfg(plt_cfg_file)
    if plt_cfg is None:
        return

    PERIOD = calendar.monthrange(int(ymd[:4]), int(ymd[4:6]))[1]  # 当月天数
    ym = ymd[:6]
    ymd = ym + '%02d' % PERIOD  # 当月最后一天

    Log.info(u"----- Start Drawing Monthly TBBias Analysis Pic, PAIR: {}, YMD: {}" \
             u" -----".format(pair, ymd))
    for each in plt_cfg['monthly_staistics']:

        # Day_Night must be in 'all', 'day', 'night'
        Day_Night = ['all', 'day', 'night']  # default
        if 'time' in plt_cfg[each].keys():
            Day_Night = plt_cfg[each]['time']
            for i in Day_Night:
                if i not in ['all', 'day', 'night']:
                    Day_Night.remove(i)

        for idx, chan in enumerate(plt_cfg[each]['chan']):
            Log.info(u"Start Drawing {} Channel {}".format(each, chan))
            oneHDF5 = ReadHDF5()
            # load Matched HDF5
            num_file = PERIOD
            for daydelta in xrange(PERIOD):
                cur_ymd = pb_time.ymd_plus(ymd, -daydelta)
                nc_name = 'COLLOC+%sIR,%s_C_BABJ_%s.hdf5' % (Type, pair, cur_ymd)
                filefullpath = os.path.join(MATCH_DIR, pair, nc_name)
                if not os.path.isfile(filefullpath):
                    Log.info(u"HDF5 not found: {}".format(filefullpath))
                    num_file -= 1
                    continue
                if not oneHDF5.LoadData(filefullpath, chan):
                    Log.error('Error occur when reading %s of %s' % (chan, filefullpath))

            if num_file == 0:
                Log.error(u"No file found.")
                continue
            elif num_file != PERIOD:
                Log.info(u"{} of {} file(s) found.".format(num_file, PERIOD))

            # 输出目录
            cur_path = os.path.join(MBA_DIR, pair, ymd[:6])

            # find out day and night
            if ('day' in Day_Night or 'night' in Day_Night) and len(oneHDF5.time) > 0:
                jd = oneHDF5.time / 24. / 3600.  # jday from 1993/01/01 00:00:00
                hour = ((jd - jd.astype('int8')) * 24).astype('int8')
                day_index = (hour < 10)  # utc hour<10 is day
                night_index = np.logical_not(day_index)
            else:
                day_index = None
                night_index = None
            # get threhold, unit, names...
            xname, yname = each.split('-')
            bias = xname
            xname_l = xname.upper()
            xunit = plt_cfg[each]['x_unit']
            xlimit = plt_cfg[each]['x_range'][idx]
            xmin, xmax = xlimit.split('-')
            xmin = float(xmin)
            xmax = float(xmax)
            # get x
            dset_name = xname + "1"
            if hasattr(oneHDF5, dset_name):
                x = getattr(oneHDF5, dset_name)
            else:
                Log.error("Can't plot, no %s in HDF5 class" % dset_name)
                continue
            # get y
            dset_name = yname + "2"
            if hasattr(oneHDF5, dset_name):
                y = getattr(oneHDF5, dset_name)
            else:
                Log.error("Can't plot, no %s in HDF5 class" % dset_name)
                continue
            if 'rad' == bias:
                o_name = 'RadBiasMonthStats'
            elif 'tbb' == bias:
                o_name = 'TBBiasMonthStats'
            elif 'ref' == bias:
                o_name = 'RefBiasMonthStats'
            else:
                o_name = 'DUMMY'
            if x.size < 10:
                Log.error("Not enough match point to draw.")
                continue

            # 获取 std
            weight = None
            if 'rad' in xname and 'rad' in yname:
                if len(oneHDF5.rad1_std) > 0:
                    weight = oneHDF5.rad1_std
            elif 'tbb' in xname and 'tbb' in yname:
                weight = None
            elif 'ref' in xname and 'ref' in yname:
                if len(oneHDF5.ref1_std) > 0:
                    weight = oneHDF5.ref1_std
            elif 'dn' in xname and 'ref' in yname:
                weight = None

            # rad-specified regression starts
            reference_list = []
            if 'reference' in plt_cfg[each]:
                reference_list = plt_cfg[each]['reference'][idx]
            if 'all' in Day_Night:
                o_file = os.path.join(cur_path,
                                      '%s_%s_%s_ALL_%s' % (pair, o_name, chan, ym))
                plot(x, y, weight, o_file,
                     part1, part2, chan, ym, 'ALL', reference_list,
                     xname, xname_l, xunit, xmin, xmax)

            # ------- day ----------
            if 'day' in Day_Night:
                if day_index is not None and np.where(day_index)[0].size > 10:
                    # rad-specified
                    o_file = os.path.join(cur_path,
                                          '%s_%s_%s_Day_%s' % (pair, o_name, chan, ym))
                    x_d = x[day_index]
                    y_d = y[day_index]
                    w_d = weight[day_index] if weight is not None else None
                    plot(x_d, y_d, w_d, o_file,
                         part1, part2, chan, ym, 'Day', reference_list,
                         xname, xname_l, xunit, xmin, xmax)
            if 'night' in Day_Night:
                # ---------night ------------
                if night_index is not None and np.where(night_index)[0].size > 10:
                    # rad-specified
                    o_file = os.path.join(cur_path,
                                          '%s_%s_%s_Night_%s' % (pair, o_name, chan, ym))
                    x_n = x[night_index]
                    y_n = y[night_index]
                    w_n = weight[day_index] if weight is not None else None
                    plot(x_n, y_n, w_n, o_file,
                         part1, part2, chan, ym, 'Night', reference_list,
                         xname, xname_l, xunit, xmin, xmax)


def plot(x, y, weight, picPath,
         part1, part2, chan, ym, DayOrNight, reference_list,
         xname, xname_l, xunit, xmin, xmax):
    """
    x: 参考卫星传感器数据
    y: FY数据
    """
    plt.style.use(os.path.join(dvPath, 'dv_pub_regression.mplstyle'))
    if xname_l == "TBB":
        xname_l = "TB"

    # 过滤 正负 delta+8倍std 的杂点 ------------
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

    if xname == "tbb":
        step = 5
    else:
        step = 0.1

    # 计算回归信息： 斜率，截距，R
    RadCompare = G_reg1d(x, y, w)

    # 开始绘图
    fig = plt.figure(figsize=(6, 5))
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0))
    # 图片 Title
    title = '%s Bias Monthly Statistics\n%s Minus %s  %s  %s' % \
            (xname_l, part1, part2, chan, DayOrNight)

    # plot 偏差分布图 -------------------------------------------------
    # x y 轴范围
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

    distri_locator = {
        "locator_x": (None, None), "locator_y": (8, 5)
    }
    # Distri label
    distri_label = {}
    if xunit != "":
        ylabel = 'D{}({})'.format(xname_l, xunit)
    else:
        ylabel = "D{}".format(xname_l)
    distri_label["ylabel"] = ylabel

    ref_temp = reference_list[0]  # 获取拟合系数

    # 获取 MeanBias 信息
    bias_range = 0.15
    boundary = xmin + (xmax - xmin) * 0.15
    bias_info = bias_information(x, y, boundary, bias_range)

    # 绝对偏差和相对偏差信息 TBB=250K  REF=0.25
    ab = RadCompare
    a = ab[0]
    b = ab[1]
    if xname == 'tbb':
        bias_info_md = "TBB Bias ({} K) : {:.4f} K".format(
            ref_temp, ref_temp - (ref_temp * a + b))
    elif xname == 'ref':
        bias_info_md = "Relative Bias (REF {}) : {:.4f} %".format(
            ref_temp, (ref_temp - (ref_temp * a + b)) / ref_temp * 100)
    else:
        bias_info_md = ""

    # 配置注释信息
    distri_annotate = {"left": [bias_info.get("info_lower"),
                                bias_info.get("info_greater"),
                                bias_info_md]}
    # 注释线配置
    if xname == "tbb":
        avxline = {
            'line_x': ref_temp, 'line_color': '#4cd964', 'line_width': 0.7,
            'word': str(ref_temp) + xunit, 'word_color': EDGE_GRAY,
            'word_size': 6, 'word_location': (ref_temp, -3.5)
        }
    elif xname == "ref":
        avxline = {
            'line_x': ref_temp, 'line_color': '#4cd964', 'line_width': 0.7,
            'word': str(ref_temp) + xunit, 'word_color': EDGE_GRAY,
            'word_size': 6, 'word_location': (ref_temp, -0.07)
        }
    else:
        avxline = None
        distri_annotate = None

    # y=0 线配置
    zeroline = {"line_color": '#808080', "line_width": 1.0}

    # 偏差点配置
    scatter_delta = {
        "scatter_marker": 'o', "scatter_size": 1.5, "scatter_alpha": 0.5,
        "scatter_linewidth": 0, "scatter_zorder": 100, "scatter_color": BLUE,
    }
    # 偏差 fill 配置
    scatter_fill = {
        "fill_marker": 'o-', "fill_size": 6, "fill_alpha": 0.5,
        "fill_linewidth": 0.6, "fill_zorder": 50, "fill_color": RED,
        "fill_step": step,
    }

    draw_distribution(ax1, x, y, label=distri_label, ax_annotate=distri_annotate,
                      axislimit=distri_limit, locator=distri_locator,
                      zeroline=zeroline,
                      scatter_delta=scatter_delta,
                      avxline=avxline,
                      scatter_fill=scatter_fill,
                      )

    # 绘制 Bar 图 -------------------------------------------------
    bar_xmin = distri_xmin
    bar_xmax = distri_xmax
    bar_ymin = 0
    bar_ymax = 7

    bar_limit = {
        "xlimit": (bar_xmin, bar_xmax),
        "ylimit": (bar_ymin, bar_ymax),
    }

    if xname == "tbb":
        bar_locator = {
            "locator_x": (None, None), "locator_y": (7, 5)
        }
    elif xname == "ref":
        bar_locator = {
            "locator_x": (None, None), "locator_y": (7, 5)
        }
    else:
        bar_locator = None

    # bar 的宽度
    if xname == "tbb":
        width = 3
    elif xname == "ref":
        width = 0.07
    else:
        width = 1
    # bar 配置
    bar = {
        "bar_width": width, "bar_color": BLUE, "bar_linewidth": 0,
        "text_size": 6, "text_font": FONT_MONO, "bar_step": step,
    }

    bar_annotate = {
        "left": ['Total Number: %7d' % len(x)]
    }
    bar_label = {
        "xlabel": '%s %s' % (part2, xname_l) + (
            '($%s$)' % xunit if xunit != "" else ""),
        "ylabel": 'Number of sample points\nlog (base = 10)'
    }

    draw_bar(ax2, x, y, label=bar_label, ax_annotate=bar_annotate,
             axislimit=bar_limit, locator=bar_locator,
             bar=bar,
             )

    # ---------------
    plt.tight_layout()
    # 将 ax1 的 xticklabels 设置为不可见
    plt.setp(ax1.get_xticklabels(), visible=False)

    # 子图的底间距
    fig.subplots_adjust(bottom=0.16, top=0.90)
    FONT1.set_size(11)
    fig.suptitle(title, fontsize=11, fontproperties=FONT1)
    fig.text(0.6, 0.02, '%s' % ym, fontsize=11, fontproperties=FONT0)
    fig.text(0.8, 0.02, ORG_NAME, fontsize=11, fontproperties=FONT0)
    # ---------------

    pb_io.make_sure_path_exists(os.path.dirname(picPath))
    fig.savefig(picPath)
    plt.close()
    fig.clear()


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


def get_bar_data(xx, delta, Tmin, Tmax, step):
    T_seg = []
    mean_seg = []
    std_seg = []
    sampleNums = []
    for i in np.arange(Tmin, Tmax, step):
        idx = np.where(np.logical_and(xx >= i, xx < (i + step)))[0]

        if idx.size > 0:
            DTb_block = delta[idx]
        else:
            continue

        mean1 = mean(DTb_block)
        std1 = std(DTb_block)

        idx1 = np.where((abs(DTb_block - mean1) < std1))[0]  # 去掉偏差大于std的点
        if idx1.size > 0:
            DTb_block = DTb_block[idx1]
            mean_seg.append(mean(DTb_block))
            std_seg.append(std(DTb_block))
            sampleNums.append(len(DTb_block))
        else:
            mean_seg.append(0)
            std_seg.append(0)
            sampleNums.append(0)
        T_seg.append(i + step / 2.)

    return np.array(T_seg), np.array(mean_seg), np.array(std_seg), np.array(sampleNums)


######################### 程序全局入口 ##############################

# 获取程序参数接口
args = sys.argv[1:]
help_info = \
    u'''
        【参数1】：FY3A+MERSI_AQUA+MODIS(样例，具体参见global.cfg 标签PAIRS下的标识)
        【参数2】：yyyymmdd-yyyymmdd
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
omdFile = os.path.join(ProjPath, 'cfg', 'dm_odm.cfg')

# 配置不存在预警
if not os.path.isfile(cfgFile):
    print (u'配置文件不存在 %s' % cfgFile)
    sys.exit(-1)

# 载入配置文件
inCfg = ConfigObj(cfgFile)
MATCH_DIR = inCfg['PATH']['MID']['MATCH_DATA']
MBA_DIR = inCfg['PATH']['OUT']['MBA']
LogPath = inCfg['PATH']['OUT']['LOG']
Log = LogServer(LogPath)

# 获取开机线程的个数，开启线程池。
threadNum = inCfg['CROND']['threads']
pool = Pool(processes=int(threadNum))

if len(args) == 2:
    Log.info(u'手动月统计绘图程序开始运行-----------------------------')
    satPair = args[0]
    str_time = args[1]
    date_s, date_e = pb_time.arg_str2date(str_time)

    while date_s <= date_e:
        ymd = date_s.strftime('%Y%m%d')
        # run(satPair, ymd)
        pool.apply_async(run, (satPair, ymd))
        date_s = date_s + relativedelta(months=1)
    pool.close()
    pool.join()
elif len(args) == 0:
    Log.info(u'自动月统计绘图程序开始运行 -----------------------------')
    rolldays = inCfg['CROND']['rolldays']
    pairLst = inCfg['PAIRS'].keys()
    # 定义参数List，传参给线程池
    args_List = []
    for satPair in pairLst:
        ProjMode1 = len(inCfg['PAIRS'][satPair]['colloc_exe'])
        if ProjMode1 == 0:
            continue
        # 增加一个月的作业,默认当前月和上一个月
        ymd = (datetime.utcnow()).strftime('%Y%m%d')
        ymdLast = (datetime.utcnow() - relativedelta(months=1)).strftime('%Y%m%d')
        pool.apply_async(run, (satPair, ymd))
        pool.apply_async(run, (satPair, ymdLast))
    pool.close()
    pool.join()
else:
    print 'args: FY3A+MERSI_AQUA+MODIS yyyymmdd-yyyymmdd '
    sys.exit(-1)
