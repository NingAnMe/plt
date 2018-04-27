# coding: utf-8
"""
Created on 2018年4月20日
读取 abr 和 rmd 的 txt 文件，画时间序列图
包括 abc, tbbias, rmd, omb 四种图

@author: zhangtao, anning
"""
import os, sys
from configobj import ConfigObj
import numpy as np
import matplotlib as mpl
from PB import pb_time, pb_io
from PB.CSC.pb_csc_console import LogServer
from DV.dv_pub_legacy import plt, mdates, set_tick_font, FONT0

from DV import dv_pub_3d_dev
from DV.dv_pub_3d_dev import get_bias_data
from datetime import datetime
from DM.SNO.dm_sno_cross_calc_map import RED, BLUE, EDGE_GRAY, ORG_NAME, mpatches
from matplotlib.ticker import MultipleLocator
from dateutil.relativedelta import relativedelta
from multiprocessing import Pool , Lock
from plt_io import loadYamlCfg


class coeff_abr(object):

    def __init__(self, pair):
        self.pair = pair
        self.path = os.path.join(ABR_DIR, pair)
        self.error = False

        # yaml config file
        sensor1 = (pair.split('_')[0]).split('+')[1]
        sensor2 = (pair.split('_')[1]).split('+')[1]

        # load yaml config file
        plt_cfg_file = os.path.join(MainPath, '%s_%s_3d.yaml' % (sensor1, sensor2))
        self.plt_cfg = loadYamlCfg(plt_cfg_file)
        if self.plt_cfg is None:
            self.error = True

    def loadCoeff(self, physical_pair, oname, DayOrNight):
        """
        load Monthly coeff to self.Coeff_M
        load Daily coeff to self.Coeff_D
        args:
        oname: for now must be 'TBBCalCoeff', in future plan may add 'RadCalCoeff'
        DayOrNight: ALL, Day, Night
        """
        if not os.path.isdir(self.path):
            self.error = True
            return

        flst = [e for e in os.listdir(self.path) if '%s_%s_%s_' % (self.pair, oname, DayOrNight) in e]

        fname = '%s_%s_%s_Monthly.txt' % (self.pair, oname, DayOrNight)
        self.Coeff_M = None
        if fname in flst:
            flst.pop(flst.index(fname))
            self.Coeff_M = self.__loadCoeffTxt_M(physical_pair, fname)
        else:
            Log.error('No Monthly Coeff TXT %s' % fname)
            self.error = True

        self.Coeff_D = self.__loadCoeffTxt_D(physical_pair, flst)

    def __loadCoeffTxt_M(self, physical_pair, fname):
        """
        read txt, return ndarray
        """
        retAry = None
        names = ['date']
        formats = ['object']
        for chan in self.plt_cfg[physical_pair]['chan']:
            names = names + ['c_%s' % chan, 'a_%s' % chan, 'b_%s' % chan, 'r_%s' % chan]
            formats = formats + ['i4', 'f4', 'f4', 'f4']

        convert = lambda x: datetime.strptime(x[:6] + '15', "%Y%m%d")  # 画点在15号
        fpath = os.path.join(self.path, fname)

        ary = np.loadtxt(fpath,
              converters={0:convert},
              dtype={'names': tuple(names),
                     'formats': tuple(formats)},
              skiprows=10, ndmin=1)
        if len(ary) == 0:
            pass
        elif len(ary[0]) == 0:
            pass
        else:
            retAry = ary

        return retAry

    def __loadCoeffTxt_D(self, physical_pair, fnameLst):
        """
        read txt, return ndarray
        """
        retAry = None
        names = ['date']
        formats = ['object']
        for chan in self.plt_cfg[physical_pair]['chan']:
            names = names + ['c_%s' % chan, 'a_%s' % chan, 'b_%s' % chan, 'r_%s' % chan]
            formats = formats + ['i4', 'f4', 'f4', 'f4']

        convert = lambda x: datetime.strptime(x, "%Y%m%d")
        fnameLst.sort()
        for eachtxt in fnameLst:
            fpath = os.path.join(self.path, eachtxt)
            ary = np.loadtxt(fpath,
                  converters={0:convert},
                  dtype={'names': tuple(names),
                         'formats': tuple(formats)},
                  skiprows=10, ndmin=1)
            if len(ary) == 0: continue
            if len(ary[0]) == 0: continue

            if retAry is None:
                retAry = ary
            else:
                retAry = np.concatenate((retAry, ary), axis=0)  # 合并多天的数据

        return retAry


def run(pair, date_s, date_e):
    """
    pair: sat1+sensor1_sat2+sensor2
    date_s: datetime of start date
            None  处理 从发星 到 有数据的最后一天
    date_e: datetime of end date
            None  处理 从发星 到 有数据的最后一天
    """
    isLaunch = False
    if date_s is None or date_e is None:
        isLaunch = True
    part1, part2 = pair.split('_')
    sat1, sensor1 = part1.split('+')
    sat2, sensor2 = part2.split('+')

    # load yaml config file
    plt_cfg_file = os.path.join(MainPath, '%s_%s_3d.yaml' % (sensor1, sensor2))
    plt_cfg = loadYamlCfg(plt_cfg_file)
    if plt_cfg is None:
        return

    for each in plt_cfg['time_series']:
        # must be in 'all', 'day', 'night'
        Day_Night = ['all', 'day', 'night']
        if 'time' in plt_cfg[each].keys():
            Day_Night = plt_cfg[each]['time']
            for i in Day_Night:
                if i not in ['all', 'day', 'night']:
                    Day_Night.remove(i)

        co = coeff_abr(pair)
        if co.error:
            return

        # 将每个对通用的属性值放到对循环，每个通道用到的属性值放到通道循环
        xname, yname = each.split('-')
        xname_l = plt_cfg[each]['x_name']
        xunit = plt_cfg[each]['x_unit']
        yname_l = plt_cfg[each]['y_name']
        yunit = plt_cfg[each]['y_unit']

        if 'tbb' in xname and 'tbb' in yname:
            o_name = 'TBBCalCoeff'
        elif 'ref' in xname and 'ref' in yname:
            o_name = 'CorrcCoeff'
        elif 'dn' in xname and 'ref' in yname:
            o_name = 'CalCoeff'
        else:
            continue

        for DayOrNight in Day_Night:

            if DayOrNight == 'all':
                DayOrNight = DayOrNight.upper()  # all -> ALL
            else:
                DayOrNight = DayOrNight[0].upper() + DayOrNight[1:]

            co.loadCoeff(each, o_name, DayOrNight)
            if co.error: continue
            date_D = co.Coeff_D['date']
            date_M = co.Coeff_M['date']
            if isLaunch:
                if sat1 in inCfg['LUANCH_DATE']:
                    date_s = pb_time.ymd2date(str(inCfg['LUANCH_DATE'][sat1]))
                else:
                    Log.error('%s not in LUANCH_DATE of Cfg, use the first day in txt instead.')
                    date_s = date_D[0]
                date_e = date_D[-1]

            ymd_s, ymd_e = date_s.strftime('%Y%m%d'), date_e.strftime('%Y%m%d')
            Log.info(u"----- Start Drawing ABC TBBias Pic, {}, PAIR: {}, YMD: {}-{}: " \
                     u"{}-----".format(each, pair, ymd_s, ymd_e, DayOrNight))

            idx_D = np.where(np.logical_and(date_D >= date_s, date_D <= date_e))
            idx_M = np.where(np.logical_and(date_M >= pb_time.ymd2date(ymd_s[:6] + '01'), date_M <= date_e))

            for i, chan in enumerate(plt_cfg[each]['chan']):
                a_D = co.Coeff_D['a_%s' % chan]
                b_D = co.Coeff_D['b_%s' % chan]
                c_D = np.log10(co.Coeff_D['c_%s' % chan])
                a_M = co.Coeff_M['a_%s' % chan]
                b_M = co.Coeff_M['b_%s' % chan]
                c_M = np.log10(co.Coeff_M['c_%s' % chan])
                # plot slope Intercept count ------------------------
                print "plot abc : {} {} {}".format(each, DayOrNight, chan)
                title = 'Time Series of Slope Intercept & Counts  %s  %s\n(%s = Slope * %s + Intercept)' % \
                        (chan, DayOrNight,
                         part1.replace('+', '_'),
                         part2.replace('+', '_'))

                if isLaunch:
                    picPath = os.path.join(ABC_DIR, pair,
                                           '%s_%s_ABC_%s_%s_Launch.png' % (pair, o_name, chan, DayOrNight))
                else:
                    picPath = os.path.join(ABC_DIR, pair, ymd_e,
                                           '%s_%s_ABC_%s_%s_Year_%s.png' % (pair, o_name, chan, DayOrNight, ymd_e))

                # 系数坐标范围
                slope_range = plt_cfg[each]['slope_range'][i]
                slope_min, slope_max = slope_range.split('-')
                slope_min = float(slope_min)
                slope_max = float(slope_max)

                plot_abc(date_D[idx_D], a_D[idx_D], b_D[idx_D], c_D[idx_D],
                         date_M[idx_M], a_M[idx_M], b_M[idx_M], c_M[idx_M],
                         picPath, title, date_s, date_e, slope_min, slope_max,
                         each)

                # plot RMD ----------------------------------------------------
                if xname == "ref" and yname == "ref":
                    print "plot RMD : {} {} {}".format(each, DayOrNight, chan)
                    # 从 BIAS 文件中读取数据
                    bias_ref_path = os.path.join(ABR_DIR, '%s_%s' % (part1, part2),
                                                 "BIAS")
                    file_name_monthly = os.path.join(
                        bias_ref_path, '%s_%s_%s_%s_%s_Monthly.txt' % (
                            part1, part2, xname.upper(), chan, DayOrNight))
                    file_name_daily = os.path.join(
                        bias_ref_path, '%s_%s_%s_%s_%s_Daily.txt' % (
                            part1, part2, xname.upper(), chan, DayOrNight))
                    rmd_d = get_bias_data(file_name_daily)

                    rmd_m = get_bias_data(file_name_monthly)

                    date_rmd_d = rmd_d["date"]
                    data_rmd_d = rmd_d["bias"]
                    date_rmd_m = rmd_m["date"]
                    date_rmd_m = date_rmd_m + relativedelta(days=14)
                    data_rmd_m = rmd_m["bias"]
                    std_rmd_m = rmd_m["std"]
                    idx_d = np.where(
                        np.logical_and(date_rmd_d >= date_s,
                                       date_rmd_d <= date_e))
                    idx_m = np.where(
                        np.logical_and(
                            date_rmd_m >= pb_time.ymd2date(ymd_s[:6] + '01'),
                            date_rmd_m <= date_e))

                    # 根据拟合系数进行绘制
                    reference_list = plt_cfg[each]['reference'][i]
                    for ref_temp in reference_list:
                        # ref_temp_f = float(ref_temp)
                        if isLaunch:
                            pic_path = os.path.join(
                                OMB_DIR, pair, '%s_RMD_%s_%s_Launch_%d.png' % (
                                    pair, chan, DayOrNight, ref_temp * 100))
                        else:
                            # plot latest year
                            pic_path = os.path.join(
                                OMB_DIR, pair, ymd_e,
                                '%s_RMD_%s_%s_Year_%s_%d.png' % (
                                    pair, chan, DayOrNight, ymd_e,
                                    ref_temp * 100))
                        plot_rmd(
                            date_rmd_d[idx_d], data_rmd_d[idx_d],
                            date_rmd_m[idx_m], data_rmd_m[idx_m],
                            std_rmd_m[idx_m],
                            pic_path, date_s, date_e,
                            sat1, pair, chan, DayOrNight, ref_temp,
                            xname, xname_l, xunit,
                            yname, yname_l, yunit,
                        )

                # plot TBBias ------------------------
                if xname == 'tbb' and yname == 'tbb':
                    print "plot TBBias : {} {} {}".format(each, DayOrNight, chan)
                    # 从 BIAS 文件中读取数据
                    bias_tbb_path = os.path.join(ABR_DIR, '%s_%s' % (part1, part2),
                                                 "BIAS")
                    file_name_monthly = os.path.join(
                        bias_tbb_path, '%s_%s_%s_%s_%s_Monthly.txt' % (
                            part1, part2, xname.upper(), chan, DayOrNight))
                    file_name_daily = os.path.join(
                        bias_tbb_path, '%s_%s_%s_%s_%s_Daily.txt' % (
                            part1, part2, xname.upper(), chan, DayOrNight))
                    tbbias_d = get_bias_data(file_name_daily)

                    tbbias_m = get_bias_data(file_name_monthly)

                    date_tbbias_d = tbbias_d["date"]
                    data_tbbias_d = tbbias_d["bias"]
                    date_tbbias_m = tbbias_m["date"]
                    date_tbbias_m = date_tbbias_m + relativedelta(days=14)
                    data_tbbias_m = tbbias_m["bias"]
                    std_tbbias_m = tbbias_m["std"]
                    idx_d = np.where(
                        np.logical_and(date_tbbias_d >= date_s,
                                       date_tbbias_d <= date_e))
                    idx_m = np.where(
                        np.logical_and(
                            date_tbbias_m >= pb_time.ymd2date(ymd_s[:6] + '01'),
                            date_tbbias_m <= date_e))

                    # 根据拟合系数进行绘制
                    reference_list = plt_cfg[each]['reference'][i]
                    for ref_temp in reference_list:
                        # ref_temp_f = float(ref_temp)
                        if isLaunch:
                            pic_path = os.path.join(
                                OMB_DIR, pair, '%s_TBBias_%s_%s_Launch_%dK.png' % (
                                    pair, chan, DayOrNight, ref_temp))
                        else:
                            # plot latest year
                            pic_path = os.path.join(
                                OMB_DIR, pair, ymd_e,
                                '%s_TBBias_%s_%s_Year_%s_%dK.png' % (
                                    pair, chan, DayOrNight, ymd_e,
                                    ref_temp))
                        plot_tbbias(
                            date_tbbias_d[idx_d], data_tbbias_d[idx_d],
                            date_tbbias_m[idx_m], data_tbbias_m[idx_m],
                            std_tbbias_m[idx_m],
                            pic_path, date_s, date_e,
                            sat1, pair, chan, DayOrNight, ref_temp,
                            xname, xname_l, xunit,
                            yname, yname_l, yunit,
                        )

                    # plot interpolated TBBias img (obs minus backgroud) -------------
                    print "plot OMB : {} {} {}".format(each, DayOrNight, chan)
                    title = 'Brightness Temperature Correction\n%s  %s  %s' % \
                            (pair, chan, DayOrNight)

                    if isLaunch:
                        picPath = os.path.join(OMB_DIR, pair,
                                               '%s_TBBOMB_%s_%s_Launch.png' % (
                                                pair, chan, DayOrNight))
                    else:
                        picPath = os.path.join(OMB_DIR, pair, ymd_e,
                                               '%s_TBBOMB_%s_%s_Year_%s.png' % (
                                                pair, chan, DayOrNight, ymd_e))
                    plot_omb(date_D[idx_D], a_D[idx_D], b_D[idx_D],
                             picPath, title, date_s, date_e)


def plot_abc(date_D, a_D, b_D, c_D,
             date_M, a_M, b_M, c_M,
             picPath, title,
             date_s, date_e,
             slope_min, slope_max,
             var):
    plt.style.use(os.path.join(dvPath, 'dv_pub_timeseries.mplstyle'))
    fig = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(313, sharex=ax1)

    # format the Xticks
    xlim_min = pb_time.ymd2date('%04d%02d01' % (date_s.year, date_s.month))
    xlim_max = date_e
    ax1.set_xlim(xlim_min, xlim_max)

    # format the Yticks\
    # Y 轴，坐标轴范围
    if var == "tbb-tbb" or var == "ref-ref":
        ax1.set_ylim(slope_min, slope_max)
        ax1.yaxis.set_major_locator(MultipleLocator(0.01))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.002))
    elif var == "dn-ref":
        ax1.set_ylim(slope_min, slope_max)
        # 根据要求：dn-ref 的图，ax1 需要有两种坐标范围
        if slope_max >= 0.00030:
            ax1.yaxis.set_major_locator(MultipleLocator(0.00010))
            ax1.yaxis.set_minor_locator(MultipleLocator(0.00002))
        else:
            ax1.yaxis.set_major_locator(MultipleLocator(0.00002))
            ax1.yaxis.set_minor_locator(MultipleLocator(0.000004))
    if var == "tbb-tbb":
        ax2.set_ylim(-30, 30)
        ax2.yaxis.set_major_locator(MultipleLocator(10))
        ax2.yaxis.set_minor_locator(MultipleLocator(5))
    elif var == "ref-ref":
        ax2.set_ylim(-0.1, 0.1)
        ax2.yaxis.set_major_locator(MultipleLocator(0.02))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.01))
    elif var == "dn-ref":
        ax2.set_ylim(-0.08, 0.08)
        ax2.yaxis.set_major_locator(MultipleLocator(0.02))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax3.set_ylim(0, 7)
    ax3.yaxis.set_major_locator(MultipleLocator(1))
    ax3.yaxis.set_minor_locator(MultipleLocator(0.5))

    # plot ax1 -------------------------------------------------
    plt.sca(ax1)
    plt.plot(date_D, a_D, 'x', ms=5,
             markerfacecolor=None, markeredgecolor=BLUE, alpha=0.8,
             mew=0.3, label='Daily')
    plt.plot(date_M, a_M, 'o-',
             ms=4, lw=0.6, c=RED,
             mew=0, label='Monthly')
    plt.ylabel('Slope', fontsize=11, fontproperties=FONT0)
    plt.grid(True)
    plt.title(title, fontsize=12, fontproperties=FONT0)
    set_tick_font(ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # plot ax2 -------------------------------------------------
    plt.sca(ax2)
    plt.plot(date_D, b_D, 'x', ms=5,
             markerfacecolor=None, markeredgecolor=BLUE, alpha=0.8,
             mew=0.3, label='Daily')
    plt.plot(date_M, b_M, 'o-',
             ms=4, lw=0.6, c=RED,
             mew=0, label='Monthly')
    plt.ylabel('Intercept', fontsize=11, fontproperties=FONT0)
    plt.grid(True)
    set_tick_font(ax2)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # point number -------------------------------------------------
    plt.sca(ax3)

    plt.fill_between(date_D, 0, c_D,
                 edgecolor=BLUE, facecolor=BLUE, alpha=0.6)
#     plt.fill_between(date_M, 0, c_M,
#                  edgecolor=RED, facecolor=RED, alpha=0.5)
#     plt.bar(date_M, c_M, width=1, align='edge',  # "center",
#             color=RED, linewidth=0)
    plt.plot(date_M, c_M, 'o-',
             ms=4, lw=0.6, c=RED,
             mew=0, label='Monthly')
    plt.ylabel('Number of sample points\nlog (base = 10)', fontsize=11, fontproperties=FONT0)
    plt.grid(True)
    set_tick_font(ax3)

    setXLocator(ax3, xlim_min, xlim_max)

#     circle1 = mpatches.Circle((430, 563), 5, color=BLUE, ec=EDGE_GRAY, lw=0)
#     circle2 = mpatches.Circle((508, 563), 5, color=RED, ec=EDGE_GRAY, lw=0)
#     fig.patches.extend([circle1, circle2])
#
#     fig.text(0.74, 0.93, 'Daily', color=BLUE, fontproperties=FONT0)
#     fig.text(0.86, 0.93, 'Monthly', color=RED, fontproperties=FONT0)
    #---------------
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.14)

    circle1 = mpatches.Circle((74, 18), 6, color=BLUE, ec=EDGE_GRAY, lw=0)
    circle2 = mpatches.Circle((164, 18), 6, color=RED, ec=EDGE_GRAY, lw=0)
    fig.patches.extend([circle1, circle2])

    fig.text(0.15, 0.02, 'Daily', color=BLUE, fontproperties=FONT0)
    fig.text(0.3, 0.02, 'Monthly', color=RED, fontproperties=FONT0)

    ymd_s, ymd_e = date_s.strftime('%Y%m%d'), date_e.strftime('%Y%m%d')
    if ymd_s != ymd_e:
        fig.text(0.50, 0.02, '%s-%s' % (ymd_s, ymd_e), fontproperties=FONT0)
    else:
        fig.text(0.50, 0.02, '%s' % ymd_s, fontproperties=FONT0)

    fig.text(0.8, 0.02, ORG_NAME, fontproperties=FONT0)
    #---------------

    pb_io.make_sure_path_exists(os.path.dirname(picPath))
    fig.savefig(picPath)
    print picPath
    plt.close()
    fig.clear()


def plot_omb(date_D, a_D, b_D,
             picPath, title,
             date_s, date_e):
    """
    画偏差时序彩色图
    """
    plt.style.use(os.path.join(dvPath, 'dv_pub_timeseries.mplstyle'))
    if (np.isnan(a_D)).all():
        Log.error('Everything is NaN: %s' % picPath)
        return

    ylim_min, ylim_max = 210, 330
    y_res = 0.2
    x_size = (date_e - date_s).days
    yy = np.arange(ylim_min, ylim_max, y_res) + y_res / 2.  # 一列的y值
    grid = np.ones(len(date_D)) * yy.reshape(-1, 1)

    aa = a_D * np.ones((len(grid), 1))
    bb = b_D * np.ones((len(grid), 1))

    grid = grid - np.divide((grid - bb), aa)

    # zz = np.zeros((len(yy), x_size))  # 2D， 要画的值
    zz = np.full((len(yy), x_size), -65535)  # 将值填充为 - ，以前填充0
    zz = np.ma.masked_where(zz == -65535, zz)

    j = 0
    xx = []  # 一行的x值
    for i in xrange(x_size):  # 补充缺失数据的天
        date_i = date_s + relativedelta(days=i)
        xx.append(date_i)
        if j < len(date_D) and date_D[j] == date_i:
            zz[:, i] = grid[:, j]
            j = j + 1

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    norm = mpl.colors.Normalize(vmin=-4.0, vmax=4.0)
    xx = np.array(xx)
    plt.pcolormesh(xx, yy, zz, cmap='jet', norm=norm, shading='flat', zorder=0)
    plt.grid(True, zorder=10)

    xlim_min = date_s
    xlim_max = date_e
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(ylim_min, ylim_max)
    plt.ylabel('TB($K$)', fontsize=11, fontproperties=FONT0)

    # format the ticks
    setXLocator(ax, xlim_min, xlim_max)
    set_tick_font(ax)

    # title
    plt.title(title, fontsize=12, fontproperties=FONT0)

    plt.tight_layout()
    # --------------------
    fig.subplots_adjust(bottom=0.25)

    # -------add colorbar ---------
    fig.canvas.draw()
    point_bl = ax.get_position().get_points()[0]  # 左下
    point_tr = ax.get_position().get_points()[1]  # 右上
    cbar_height = 0.05
    colorbar_ax = fig.add_axes([point_bl[0] - 0.05, 0.05,
                                (point_tr[0] - point_bl[0]) / 2.2, cbar_height])

    mpl.colorbar.ColorbarBase(colorbar_ax, cmap='jet',
                               norm=norm,
                               orientation='horizontal')
    # ---font of colorbar-----------
    for l in colorbar_ax.xaxis.get_ticklabels():
        l.set_fontproperties(FONT0)
        l.set_fontsize(9)
    # ------Time and ORG_NAME----------------
    ymd_s, ymd_e = date_s.strftime('%Y%m%d'), date_e.strftime('%Y%m%d')
    if ymd_s != ymd_e:
        fig.text(0.52, 0.05, '%s-%s' % (ymd_s, ymd_e), fontproperties=FONT0)
    else:
        fig.text(0.52, 0.05, '%s' % ymd_s, fontproperties=FONT0)

    fig.text(0.82, 0.05, ORG_NAME, fontproperties=FONT0)
    # ---------------

    pb_io.make_sure_path_exists(os.path.dirname(picPath))
    plt.savefig(picPath)
    print picPath
    fig.clear()
    plt.close()


def plot_tbbias(date_d, data_d, date_m, data_m, std_m, pic_path, date_s, date_e,
                sat_name, pair, chan, day_or_night, ref_temp,
                xname, xname_l, xunit,
                yname, yname_l, yunit,
                ):
    """
    画偏差时序折线图
    """
    if (np.isnan(data_d)).all():
        Log.error('Everything is NaN: %s' % pic_path)
        return
    plt.style.use(os.path.join(dvPath, 'dv_pub_timeseries.mplstyle'))
    fig = plt.figure(figsize=(6, 4))
    # fig.subplots_adjust(top=0.88, bottom=0.11, left=0.12, right=0.97)

    ax1 = plt.subplot2grid((1, 1), (0, 0))

    # 设置 title 参数
    part1, part2 = pair.split('_')

    title = 'Time Series of Brightness Temperature Bias \n{} Minus {} {} {} {}K'.format(
        part1, part2, chan, day_or_night, ref_temp)

    # plot timeseries --------------------------------------
    # x y 轴的限制
    timeseries_xmin = pb_time.ymd2date(
        '%04d%02d01' % (date_s.year, date_s.month))
    timeseries_xmax = date_e

    if "FY2" in sat_name:
        timeseries_ymin = -8
        timeseries_ymax = 8
    elif "FY3" in sat_name:
        timeseries_ymin = -4
        timeseries_ymax = 4
    elif "FY4" in sat_name:
        timeseries_ymin = -2
        timeseries_ymax = 2
    else:
        timeseries_ymin = None
        timeseries_ymax = None
    timeseries_axislimit = {
        "xlimit": (timeseries_xmin, timeseries_xmax),
        "ylimit": (timeseries_ymin, timeseries_ymax),
    }

    # x y 轴标签
    timeseries_label = {}
    if xunit != "":
        ylabel = 'DTB({})'.format(xunit)
    else:
        ylabel = "DTB"
    timeseries_label["ylabel"] = ylabel

    # x, y 轴大刻度的数量，和小刻度的数量
    timeseries_locator = {"locator_x": (None, None), "locator_y": (8, 2)}

    # y=0 线配置
    timeseries_zeroline = {"line_color": '#808080', "line_width": 1.0}

    timeseries_daily = {
        "marker": 'x', "color": BLUE, "linewidth": None,
        "markersize": 6, "markerfacecolor": None, "markeredgecolor": BLUE,
        "alpha": 0.8, "markeredgewidth": 0.3, "label": "Daily",
    }
    dv_pub_3d_dev.draw_timeseries(
        ax1, date_d, data_d, label=timeseries_label,
        axislimit=timeseries_axislimit, locator=timeseries_locator,
        zeroline=timeseries_zeroline, timeseries=timeseries_daily,
    )
    timeseries_monthly = {
        "marker": 'o-', "color": RED, "linewidth": 0.6,
        "markersize": 5, "markerfacecolor": None, "markeredgecolor": RED,
        "alpha": 0.8, "markeredgewidth": 0, "label": "Monthly",
    }
    background_fill_timeseries = {
        'x': date_m, 'y': data_m - std_m, 'y1': data_m + std_m, "color": RED,
    }
    dv_pub_3d_dev.draw_timeseries(
        ax1, date_m, data_m, label=timeseries_label,
        axislimit=timeseries_axislimit, locator=timeseries_locator,
        zeroline=timeseries_zeroline, timeseries=timeseries_monthly,
        background_fill=background_fill_timeseries,
    )
    # --------------------
    plt.tight_layout()
    fig.suptitle(title, fontsize=11, fontproperties=FONT0)
    fig.subplots_adjust(bottom=0.2, top=0.88)

    circle1 = mpatches.Circle((74, 15), 6, color=BLUE, ec=EDGE_GRAY, lw=0)
    circle2 = mpatches.Circle((164, 15), 6, color=RED, ec=EDGE_GRAY, lw=0)
    fig.patches.extend([circle1, circle2])

    fig.text(0.15, 0.02, 'Daily', color=BLUE, fontproperties=FONT0)
    fig.text(0.3, 0.02, 'Monthly', color=RED, fontproperties=FONT0)

    ymd_s, ymd_e = date_s.strftime('%Y%m%d'), date_e.strftime('%Y%m%d')
    if ymd_s != ymd_e:
        fig.text(0.50, 0.02, '%s-%s' % (ymd_s, ymd_e), fontproperties=FONT0)
    else:
        fig.text(0.50, 0.02, '%s' % ymd_s, fontproperties=FONT0)

    fig.text(0.8, 0.02, ORG_NAME, fontproperties=FONT0)
    # ---------------
    pb_io.make_sure_path_exists(os.path.dirname(pic_path))
    plt.savefig(pic_path)
    print pic_path
    fig.clear()
    plt.close()


def plot_rmd(date_d, data_d, date_m, data_m, std_m, pic_path, date_s, date_e,
             sat_name, pair, chan, day_or_night, ref_temp,
             xname, xname_l, xunit,
             yname, yname_l, yunit,
             ):
    if (np.isnan(data_d)).all():
        Log.error('Everything is NaN: %s' % pic_path)
        return
    plt.style.use(os.path.join(dvPath, 'dv_pub_timeseries.mplstyle'))
    fig = plt.figure(figsize=(6, 4))
    # fig.subplots_adjust(top=0.88, bottom=0.11, left=0.12, right=0.97)

    ax1 = plt.subplot2grid((1, 1), (0, 0))

    # 设置 title 参数
    part1, part2 = pair.split('_')

    title = 'Time Series of REF Relative Bias \n{} Minus {} {} {} REF={}'.format(
                            part1, part2, chan, day_or_night, ref_temp)

    # plot timeseries --------------------------------------------------------
    timeseries_xmin = pb_time.ymd2date(
        '%04d%02d01' % (date_s.year, date_s.month))
    timeseries_xmax = date_e

    if "FY2" in sat_name:
        timeseries_ymin = -20
        timeseries_ymax = 20
    elif "FY3" in sat_name:
        timeseries_ymin = -20
        timeseries_ymax = 20
    elif "FY4" in sat_name:
        timeseries_ymin = -20
        timeseries_ymax = 20
    else:
        timeseries_ymin = None
        timeseries_ymax = None
    timeseries_axislimit = {
        "xlimit": (timeseries_xmin, timeseries_xmax),
        "ylimit": (timeseries_ymin, timeseries_ymax),
    }

    # x y 轴标签
    timeseries_label = {}
    if xunit != "":
        ylabel = 'Relative Bias {}'.format(xunit)
    else:
        ylabel = "Relative Bias %"
    timeseries_label["ylabel"] = ylabel

    # x, y 轴大刻度的数量，和小刻度的数量
    timeseries_locator = {"locator_x": (None, None), "locator_y": (8, 2)}

    # y=0 线配置
    timeseries_zeroline = {"line_color": '#808080', "line_width": 1.0}

    timeseries_daily = {
        "marker": 'x', "color": BLUE, "linewidth": None,
        "markersize": 6, "markerfacecolor": None, "markeredgecolor": BLUE,
        "alpha": 0.8, "markeredgewidth": 0.3, "label": "Daily",
    }
    dv_pub_3d_dev.draw_timeseries(
        ax1, date_d, data_d, label=timeseries_label,
        axislimit=timeseries_axislimit, locator=timeseries_locator,
        zeroline=timeseries_zeroline, timeseries=timeseries_daily,
    )
    timeseries_monthly = {
        "marker": 'o-', "color": RED, "linewidth": 0.6,
        "markersize": 5, "markerfacecolor": None, "markeredgecolor": RED,
        "alpha": 0.8, "markeredgewidth": 0, "label": "Monthly",
    }
    background_fill_timeseries = {
        'x': date_m, 'y': data_m - std_m, 'y1': data_m + std_m, "color": RED,
    }
    dv_pub_3d_dev.draw_timeseries(
        ax1, date_m, data_m, label=timeseries_label,
        axislimit=timeseries_axislimit, locator=timeseries_locator,
        zeroline=timeseries_zeroline, timeseries=timeseries_monthly,
        background_fill=background_fill_timeseries,
    )
    # --------------------
    plt.tight_layout()
    fig.suptitle(title, fontsize=11, fontproperties=FONT0)
    fig.subplots_adjust(bottom=0.2, top=0.88)

    circle1 = mpatches.Circle((74, 15), 6, color=BLUE, ec=EDGE_GRAY, lw=0)
    circle2 = mpatches.Circle((164, 15), 6, color=RED, ec=EDGE_GRAY, lw=0)
    fig.patches.extend([circle1, circle2])

    fig.text(0.15, 0.02, 'Daily', color=BLUE, fontproperties=FONT0)
    fig.text(0.3, 0.02, 'Monthly', color=RED, fontproperties=FONT0)

    ymd_s, ymd_e = date_s.strftime('%Y%m%d'), date_e.strftime('%Y%m%d')
    if ymd_s != ymd_e:
        fig.text(0.50, 0.02, '%s-%s' % (ymd_s, ymd_e), fontproperties=FONT0)
    else:
        fig.text(0.50, 0.02, '%s' % ymd_s, fontproperties=FONT0)

    fig.text(0.8, 0.02, ORG_NAME, fontproperties=FONT0)
    # ---------------
    pb_io.make_sure_path_exists(os.path.dirname(pic_path))
    plt.savefig(pic_path)
    print pic_path
    fig.clear()
    plt.close()


def setXLocator(ax, xlim_min, xlim_max):
    day_range = (xlim_max - xlim_min).days
#     if day_range <= 2:
#         days = mdates.HourLocator(interval=4)
#         ax.xaxis.set_major_locator(days)
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    if day_range <= 60:
        days = mdates.DayLocator(interval=(day_range / 6))
        ax.xaxis.set_major_locator(days)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    else:
        month_range = day_range / 30
        if month_range <= 12.:
            months = mdates.MonthLocator()  # every month
            ax.xaxis.set_major_locator(months)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        elif month_range <= 24.:
            months = mdates.MonthLocator(interval=2)
            ax.xaxis.set_major_locator(months)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        elif month_range <= 48.:
            months = mdates.MonthLocator(interval=4)
            ax.xaxis.set_major_locator(months)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        else:
            years = mdates.YearLocator()
            ax.xaxis.set_major_locator(years)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        if month_range <= 48:
            add_year_xaxis(ax, xlim_min, xlim_max)


def add_year_xaxis(ax, xlim_min, xlim_max):
    """
    add year xaxis
    """
    if xlim_min.year == xlim_max.year:
        ax.set_xlabel(xlim_min.year, fontsize=11, fontproperties=FONT0)
        return
    newax = ax.twiny()
    newax.set_frame_on(True)
    newax.grid(False)
    newax.patch.set_visible(False)
    newax.xaxis.set_ticks_position('bottom')
    newax.xaxis.set_label_position('bottom')
    newax.set_xlim(xlim_min, xlim_max)
    newax.xaxis.set_major_locator(mdates.YearLocator())
    newax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    newax.spines['bottom'].set_position(('outward', 20))
    newax.spines['bottom'].set_linewidth(0.6)

    newax.tick_params(which='both', direction='in')
    set_tick_font(newax)
    newax.xaxis.set_tick_params(length=5)


######################### 程序全局入口 ##############################
# 获取程序参数接口
args = sys.argv[1:]
help_info = \
    u'''
    [参数样例1]：SAT1+SENSOR1_SAT2+SENSOR2  YYYYMMDD-YYYYMMDD
    [参数样例2]：SAT1+SENSOR1_SAT2+SENSOR2
    [参数样例3]：处理所有卫星对
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
ABR_DIR = inCfg['PATH']['OUT']['ABR']
OMB_DIR = inCfg['PATH']['OUT']['OMB']
ABC_DIR = inCfg['PATH']['OUT']['ABC']
LogPath = inCfg['PATH']['OUT']['LOG']
Log = LogServer(LogPath)

# 开启进程池
threadNum = inCfg['CROND']['threads']
pool = Pool(processes=int(threadNum))

if len(args) == 2:
    Log.info(u'手动长时间序列绘图程序开始运行-----------------------------')
    satPair = args[0]
    str_time = args[1]
    date_s, date_e = pb_time.arg_str2date(str_time)
    run(satPair, date_s, date_e)
    Log.info(u'SUCCESS，手动长时间序列绘图程序-----------------------------')

elif len(args) == 1:
    Log.info(u'手动长时间序列绘图程序开始运行 -----------------------------')
    satPair = args[0]
    run(satPair, None, None)
    Log.info(u'SUCCESS，手动长时间序列绘图程序-----------------------------')

elif len(args) == 0:
    Log.info(u'自动长时间序列绘图程序开始运行 -----------------------------')
    rolldays = inCfg['CROND']['rolldays']
    pairLst = inCfg['PAIRS'].keys()
    # 定义参数List，传参给线程池
    args_List = []
    for satPair in pairLst:
        ProjMode1 = len(inCfg['PAIRS'][satPair]['colloc_exe'])
        if ProjMode1 == 0:
            continue

        for rdays in rolldays:
            date1 = datetime.utcnow() - relativedelta(days=int(rdays))
            dateYearAgo = date1 - relativedelta(years=1)
            # 处理一年的时间序列
            pool.apply_async(run, (satPair, dateYearAgo, date1))
        # 处理发星以来的时间序列
        pool.apply_async(run, (satPair, None, None))
    pool.close()
    pool.join()
    Log.info(u'SUCCESS，自动长时间序列绘图程序-----------------------------')

else:
    print 'args error'
    sys.exit(-1)
