# coding: utf-8
'''
Created on 2016年1月6日
读取匹配后的HDF5文件，画散点回归图，生成abr文件

@author: duxiang, zhangtao, anning
'''
import numpy as np
from multiprocessing import Pool , Lock
import os, sys, calendar
from configobj import ConfigObj
from dateutil.relativedelta import relativedelta
from matplotlib.ticker import MultipleLocator
from numpy.lib.polynomial import polyfit
from numpy.ma.core import std, mean
from numpy.ma.extras import corrcoef
from DM.SNO.dm_sno_cross_calc_map import RED, BLUE, EDGE_GRAY, ORG_NAME
from DV.dv_pub_legacy import plt, add_annotate, set_tick_font, FONT0, FONT_MONO
from PB import pb_time, pb_io
from PB.CSC.pb_csc_console import LogServer
from datetime import datetime
from plt_io import ReadHDF5, loadYamlCfg

def run(pair, ymd):
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
                if day_index is not None and np.where(day_index)[0].size > 10 :
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
                if night_index is not None and np.where(night_index)[0].size > 10 :
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
    if xname_l == "TBB": xname_l = "TB"
    xlim_min = xmin
    xlim_max = xmax

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

    # 修改偏差值为国内减国外： x - y
    delta = x - y

    if xname == "tbb":
        step = 5
    else:
        step = 0.1
    T_seg, mean_seg, std_seg, sampleNums = get_bar_data(x, delta, xlim_min, xlim_max, step)

    RadCompare = G_reg1d(x, y)
    a, b = RadCompare[0], RadCompare[1]

    fig = plt.figure(figsize=(6, 5))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1)

    # format the Xticks
    ax1.set_xlim(xlim_min, xlim_max)

    # format the Yticks
    if xname == "tbb":
        ax1.set_ylim(-4, 4)
        ax1.yaxis.set_major_locator(MultipleLocator(2))
        ax1.yaxis.set_minor_locator(MultipleLocator(1))
    elif xname == "ref":
        ax1.set_ylim(-0.08, 0.08)
        ax1.yaxis.set_major_locator(MultipleLocator(0.02))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax2.set_ylim(0, 7)
    ax2.yaxis.set_major_locator(MultipleLocator(1))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.2))

    title = '%s Bias Monthly Statistics\n%s Minus %s  %s  %s' % \
            (xname_l, part1, part2, chan, DayOrNight)
    # plot ax1 -------------------------------------------------
    plt.sca(ax1)
    strlist = [[]]
    for ref_temp in reference_list:
        plt.axvline(x=ref_temp, color='#4cd964', lw=0.7)
        ax1.annotate(str(ref_temp) + xunit, (ref_temp, -3.5),
                     va="top", ha="center", color=EDGE_GRAY,
                     size=6, fontproperties=FONT_MONO)
        strlist[0].append("%s Bias %s: %6.3f" %
                          (xname_l, str(ref_temp) + xunit, ref_temp * a + b - ref_temp))
    strlist[0].append('Total Number: %7d' % len(x))
    plt.plot(x, delta, 'o', ms=1.5,
             markerfacecolor=BLUE, alpha=0.5,
             mew=0, zorder=10)
    plt.plot(T_seg, mean_seg, 'o-',
             ms=6, lw=0.6, c=RED,
             mew=0, zorder=50)
    plt.fill_between(T_seg, mean_seg - std_seg, mean_seg + std_seg,
                     facecolor=RED, edgecolor=RED,
                     interpolate=True, alpha=0.4, zorder=100)

    ylabel = 'D%s' % (xname_l) + ('($%s$)' % xunit if xunit != "" else "")
    plt.ylabel(ylabel, fontsize=11, fontproperties=FONT0)
    plt.grid(True)
    plt.title(title, fontsize=12, fontproperties=FONT0)
    set_tick_font(ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # point number -------------------------------------------------
    plt.sca(ax2)
    if xname == "tbb":
        width = 3
    elif xname == "ref":
        width = 0.07
    plt.bar(T_seg, np.log10(sampleNums), width=width, align="center",
            color=BLUE, linewidth=0)
    for i, T in enumerate(T_seg):
        if sampleNums[i] > 0:
            plt.text(T, np.log10(sampleNums[i]) + 0.2, '%d' % int(sampleNums[i]), ha="center",
                     fontsize=6, fontproperties=FONT_MONO)

    add_annotate(ax2, strlist, 'left', EDGE_GRAY, 9)
    plt.ylabel('Number of sample points\nlog (base = 10)', fontsize=11, fontproperties=FONT0)
    xlabel = '%s %s' % (part2, xname_l) + ('($%s$)' % xunit if xunit != "" else "")
    plt.xlabel(xlabel, fontsize=11, fontproperties=FONT0)
    plt.grid(True)
    set_tick_font(ax2)

    #---------------
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.16)

#     circle1 = mpatches.Circle((74, 18), 5, color=BLUE, ec=EDGE_GRAY, lw=0)
#     circle2 = mpatches.Circle((164, 18), 5, color=RED, ec=EDGE_GRAY, lw=0)
#     fig.patches.extend([circle1, circle2])
#
#     fig.text(0.15, 0.02, 'Daily', color=BLUE, fontproperties=FONT0)
#     fig.text(0.3, 0.02, 'Monthly', color=RED, fontproperties=FONT0)

    fig.text(0.6, 0.02, '%s' % ym, fontproperties=FONT0)
    fig.text(0.8, 0.02, ORG_NAME, fontproperties=FONT0)
    #---------------

    pb_io.make_sure_path_exists(os.path.dirname(picPath))
    fig.savefig(picPath)
    plt.close()
    fig.clear


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


def get_bar_data(xx, delta, Tmin, Tmax, step):
    T_seg = []
    mean_seg = []
    std_seg = []
    sampleNums = []
    for i in np.arange(Tmin, Tmax, step):
        idx = np.where(np.logical_and(xx >= i , xx < (i + step)))[0]

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
