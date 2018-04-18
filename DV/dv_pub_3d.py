# coding: utf-8
'''
Created on 2013-4-24

@author: admin
'''
import os
import matplotlib as mpl
from matplotlib.dates import AutoDateFormatter
from PB import pb_io

mpl.use('Agg')
import numpy as np
from math import floor, ceil
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.font_manager import FontProperties
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import interp as mapInterp
import matplotlib.dates as mdates
from dateutil.relativedelta import relativedelta

mpl.rcParams['legend.numpoints'] = 1
mpl.rcParams['axes.linewidth'] = 0.7
SCALE_SIZE = 10  # 刻度字体大小


def get_DS_Font(fontName='OpenSans-Regular.ttf'):
    '''
    载入字体
    'OpenSans-Regular.ttf'
    'simhei.ttf'
    '微软雅黑.ttf'
    '''
    selfpath = os.path.split(os.path.realpath(__file__))[0]
    font0 = FontProperties()
    font_path = os.path.join(selfpath, 'FNT', fontName)
    if os.path.isfile(font_path):
        font0.set_file(font_path)
        return font0
    return None


FONT0 = get_DS_Font()
FONT_MONO = get_DS_Font('DroidSansMono.ttf')


def draw_RSD_Bar(filename, x, deviationValue, titledict, tl_list, tr_list, ab=None):
    '''
    画errorBar图
    '''
    fig = plt.figure(figsize=(6, 4))
    fig.subplots_adjust(top=0.92, bottom=0.13, left=0.11, right=0.96)

    plt.grid(True)
    plt.plot(x, deviationValue, marker='D', color='blue',
             markerfacecolor='blue', markersize=1,
             markeredgecolor='none', lw=0)  # , alpha = 0.5)
    ax = plt.gca()
    xList, meanList, stdList = get_bar_data(x, deviationValue)
    er = ax.errorbar(xList, meanList, stdList, fmt='', marker='^',
                     ms=10, lw=1, mec='red', ls='-', color='red',
                     elinewidth=1, label="err", capsize=5, barsabove=True)

    if ab is not None:
        # 回归线 test
        xmin = np.min(x) - np.min(x) / 100.
        xmax = np.max(x) + np.min(x) / 100.
        # xmin=0. # np.min(x) - 0.1
        #         xmax=0.12   # np.max(x) + 0.1
        plt.xlim(xmin, xmax)
        plt.plot([xmin, xmax], [ab[0] * xmin + ab[1], ab[0] * xmax + ab[1]], color='g',
                 linewidth=1.2, zorder=100)
        ax.set_ylim((-2., 0.5))
    add_annotate(ax, tl_list, 'left')
    add_annotate(ax, tr_list, 'right')
    add_title(titledict)
    set_tick_font(ax)
    print filename
    plt.savefig(filename)
    fig.clear()
    plt.close()
    return


def get_bar_data(x, y):
    '''
    计算errorbar的位置和bar的高度
    '''
    divide = 15
    minValue = np.min(x)
    maxValue = np.max(x)
    step = (maxValue - minValue) / (divide * 1.0)
    xList = []  # errorbar x
    meanList = []  # errorbar y
    stdList = []  # errorbar height

    for i in xrange(divide):
        if i < (divide - 1):
            lines = np.where(np.logical_and(x >= (minValue + step * i), x < (minValue + step * (i + 1))))[0]
        else:
            lines = np.where(np.logical_and(x >= (minValue + step * i), x <= (minValue + step * (i + 1))))[0]
        if len(lines) > 0:
            dvalues = y[lines]
        else:
            continue

        dv_mean = np.mean(dvalues)
        dv_std = np.std(dvalues)
        lines2 = np.where(np.abs(dvalues - dv_mean) < dv_std)[0]  # <1倍std
        if len(lines2) > 0:
            t = minValue + step * i + step * 0.5
            dvalues2 = dvalues[lines2]
            meanList.append(np.mean(dvalues2))
            stdList.append(np.std(dvalues2))
            xList.append(t)

    return xList, meanList, stdList


def draw_histogram(filename, dvalues, titledict, tl_list, tr_list):
    '''
    画直方图
    '''
    fig = plt.figure(figsize=(6, 4))
    fig.subplots_adjust(top=0.92, bottom=0.13, left=0.11, right=0.96)

    hist, bins = np.histogram(dvalues, bins=80)
    hist = hist / (dvalues.size * 1.0)
    center = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0])
    plt.bar(center, hist, align='center', width=width, color='gold', alpha=0.5)

    # params = exponpow.fit(dvalues, floc=0)
    #     x = np.linspace(center.min(), center.max(), 100)
    #     pdf = exponpow.pdf(x, *params)
    #     lred = plt.plot(x, pdf, 'r--', linewidth=1)

    plt.grid(True)
    ax = plt.gca()
    add_annotate(ax, tl_list, 'left')
    add_annotate(ax, tr_list, 'right')
    add_title(titledict)
    set_tick_font(ax)
    print filename
    plt.savefig(filename)
    fig.clear()
    plt.close()
    return


def set_tick_font(ax, scale_size=SCALE_SIZE):
    """
    设定刻度的字体
    """
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontproperties(FONT0)
        tick.label1.set_fontsize(scale_size)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontproperties(FONT0)
        tick.label1.set_fontsize(scale_size)


def add_title(titledict):
    """
    添加大标题和xy轴名称
    """
    tt = plt.title(titledict['title'], fontsize=11, fontproperties=FONT0)
    tt.set_y(1.01)  # set gap space below title and subplot
    if 'xlabel' in titledict.keys() and titledict['xlabel'] != '':
        plt.xlabel(titledict['xlabel'], fontsize=11, fontproperties=FONT0)
    if 'ylabel' in titledict.keys() and titledict['ylabel'] != '':
        plt.ylabel(titledict['ylabel'], fontsize=11, fontproperties=FONT0)


def add_fig_title(_ax, fig_title, fig_gap=1.01):
    """
    添加图片标题，并且设置与子图的间距
    :param _ax:
    :param fig_title:
    :param fig_gap:
    :return:
    """
    tt = plt.title(fig_title, fontsize=11, fontproperties=FONT0)
    tt.set_y(fig_gap)  # set gap space below title and subplot


def add_xylabel(ax, xlabel, ylabel):
    """
    添加xy轴名称
    :param ax: plt.ax
    :param xlabel: 字符串
    :param ylabel: 字符串
    :return:
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=11, fontproperties=FONT0)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=11, fontproperties=FONT0)


def add_label(ax, label, local, fontsize=11, fontproperties=FONT0):
    """
    添加子图的标签
    :param ax:
    :param label:
    :param local:
    :return:
    """
    if label is None:
        return
    if local == "xlabel":
        ax.set_xlabel(label, fontsize=fontsize, fontproperties=fontproperties)
    elif local == "ylabel":
        ax.set_ylabel(label, fontsize=fontsize, fontproperties=fontproperties)


def add_annotate(ax, strlist, local, color='#303030', fontsize=11):
    """
    添加上方注释文字
    loc must be 'left' or 'right'
    格式 ['annotate1', 'annotate2']
    """
    if strlist is None:
        return
    xticklocs = ax.xaxis.get_ticklocs()
    yticklocs = ax.yaxis.get_ticklocs()

    x_step = (xticklocs[1] - xticklocs[0])
    x_toedge = x_step / 6.
    y_toedge = (yticklocs[1] - yticklocs[0]) / 6.

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if local == 'left':
        ax.text(xlim[0] + x_toedge, ylim[1] - y_toedge,
                '\n'.join(strlist), ha=local, va='top', color=color,
                fontsize=fontsize, fontproperties=FONT_MONO)
        x_toedge = x_toedge + x_step * 1.4
    elif local == 'right':
        ax.text(xlim[1] - x_toedge, ylim[1] - y_toedge,
                '\n'.join(strlist), ha=local, va='top', color=color,
                fontsize=fontsize, fontproperties=FONT_MONO)
        x_toedge = x_toedge + x_step * 1.4


def add_annotate_bak(ax, strlist, loc, color='#303030', fontsize=11):
    """
    添加上方注释文字
    loc must be 'left' or 'right'
    格式[['annotate1', 'annotate2']]
    """
    if strlist is None:
        return
    xticklocs = ax.xaxis.get_ticklocs()
    yticklocs = ax.yaxis.get_ticklocs()

    x_step = (xticklocs[1] - xticklocs[0])
    x_toedge = x_step / 6.
    y_toedge = (yticklocs[1] - yticklocs[0]) / 6.

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if loc == 'left':
        for eachCol in strlist:
            ax.text(xlim[0] + x_toedge, ylim[1] - y_toedge,
                    '\n'.join(eachCol), ha=loc, va='top', color=color,
                    fontsize=fontsize, fontproperties=FONT_MONO)
            x_toedge = x_toedge + x_step * 1.4
    elif loc == 'right':
        for eachCol in strlist:
            ax.text(xlim[1] - x_toedge, ylim[1] - y_toedge,
                    '\n'.join(eachCol), ha=loc, va='top', color=color,
                    fontsize=fontsize, fontproperties=FONT_MONO)
            x_toedge = x_toedge + x_step * 1.4


def day_data_write(title, data, outFile):
    """
    title: 标题
    data： 数据体
    outFile:输出文件
    """

    allLines = []
    DICT_D = {}
    FilePath = os.path.dirname(outFile)
    if not os.path.exists(FilePath):
        os.makedirs(FilePath)

    if os.path.isfile(outFile) and os.path.getsize(outFile) != 0:
        fp = open(outFile, 'r')
        fp.readline()
        Lines = fp.readlines()
        fp.close()
        # 使用字典特性，保证时间唯一，读取数据
        for Line in Lines:
            DICT_D[Line[:8]] = Line[8:]
        # 添加或更改数据
        Line = data
        DICT_D[Line[:8]] = Line[8:]
        # 按照时间排序

        newLines = sorted(DICT_D.iteritems(), key=lambda d: d[0], reverse=False)
        for i in xrange(len(newLines)):
            allLines.append(str(newLines[i][0]) + str(newLines[i][1]))
        fp = open(outFile, 'w')
        fp.write(title)
        fp.writelines(allLines)
        fp.close()
    else:
        fp = open(outFile, 'w')
        fp.write(title)
        fp.writelines(data)
        fp.close()


def draw_Scatter(x, y, filename, titledict, tl_list, tr_list,
                 xmin=None, xmax=None, ymin=None, ymax=None, diagonal=True):
    """
    画散点回归线图
    x: modis
    y: mersi
    d: mesi - modis
    """
    ab = np.polyfit(x, y, 1)
    if None in [xmin, xmax, ymin, ymax]:
        xmin = floor(np.min(x))
        xmax = ceil(np.max(x))
        ymin = floor(np.min(y))
        ymax = ceil(np.max(y))

    fig = plt.figure(figsize=(6, 4))
    fig.subplots_adjust(top=0.93, bottom=0.12, left=0.11, right=0.96)

    plt.grid(True)

    # 对角线
    if diagonal:
        # 设定 x y 轴的范围
        xylimMax = max(xmax, ymax)
        xylimMin = min(xmin, ymin)
        plt.xlim(xylimMin, xylimMax)
        plt.ylim(xylimMin, xylimMax)
        xmax = xylimMax
        xmin = xylimMin
        plt.plot([xylimMin, xylimMax], [xylimMin, xylimMax], color='#aaaaaa', linewidth=1)

    else:
        # 设定 x y 轴的范围
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

    # 回归线
    plt.plot([xmin, xmax], [ab[0] * xmin + ab[1], ab[0] * xmax + ab[1]],
             color='r', linewidth=1.2, zorder=100)

    # 画散点
    colorValues = 'b'
    plt.scatter(x, y, s=5, marker='o', c=colorValues, lw=0, alpha=0.4)

    ax = plt.gca()

    # 设定小刻度
    xticklocs = ax.xaxis.get_ticklocs()
    yticklocs = ax.yaxis.get_ticklocs()
    ax.xaxis.set_minor_locator(MultipleLocator((xticklocs[1] - xticklocs[0]) / 5))
    ax.yaxis.set_minor_locator(MultipleLocator((yticklocs[1] - yticklocs[0]) / 5))

    add_annotate(ax, tl_list, 'left')  # 注释文字
    add_annotate(ax, tr_list, 'right')  #  '#9300d3'
    add_title(titledict)
    set_tick_font(ax)
    plt.savefig(filename)
    fig.clear()
    plt.close()


def draw_regression(ax, x, y, x_label=None, y_label=None, ax_annotate={},
                    xmin=None, xmax=None, ymin=None, ymax=None, gab_number=5,
                    is_diagonal=False):
    """
    画回归线图
    :param ax:
    :param x:
    :param y:
    :param x_label:
    :param y_label:
    :param ax_annotate: 注释
    :param xmin:
    :param xmax:
    :param ymin:
    :param ymax:
    :param gab_number: 间隔数量
    :param is_diagonal: 是否画对角线
    :return:
    """
    if None in [xmin, xmax, ymin, ymax]:
        xmin = floor(np.min(x))
        xmax = ceil(np.max(x))
        ymin = floor(np.min(y))
        ymax = ceil(np.max(y))

    # 设定x y 轴的范围
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # 计算 a b
    ab = np.polyfit(x, y, 1)
    a = ab[0]
    b = ab[1]

    # 画对角线
    diagonal_color = '#808080'
    diagonal_lw = 1.2
    if is_diagonal:
        ax.plot([xmin, xmax], [ymin, ymax], color=diagonal_color,
                linewidth=diagonal_lw)

    # 画回归线
    ax.plot([xmin, xmax], [xmin * a + b, xmax * a + b],
            color='r', linewidth=1.2, zorder=100)

    # 画散点
    alpha_value = 0.8  # 透明度
    marker_value = 'o'  # 形状
    color_value = 'b'  # 颜色
    ax.scatter(x, y, s=5, marker=marker_value, c=color_value, lw=0, alpha=alpha_value)

    # 设定小刻度
    xticklocs = ax.xaxis.get_ticklocs()
    yticklocs = ax.yaxis.get_ticklocs()
    ax.xaxis.set_minor_locator(MultipleLocator((xticklocs[1] - xticklocs[0]) / gab_number))
    ax.yaxis.set_minor_locator(MultipleLocator((yticklocs[1] - yticklocs[0]) / gab_number))

    # 注释，格式['annotate1', 'annotate2']
    add_annotate(ax, ax_annotate.get("left"), "left", fontsize=10)
    add_annotate(ax, ax_annotate.get("right"), "right", fontsize=10)
    add_xylabel(ax, x_label, y_label)
    set_tick_font(ax)


def bias_information(x, y, percent=0.1):
    """
    # 过滤 range%10 范围的值，计算偏差信息
    # MeanBias( <= 10 % Range) = MD±Std @ MT
    # MeanBias( > 10 % Range) = MD±Std @ MT
    :param x:
    :param y:
    :param percent:
    :return: MD Std MT 偏差均值 偏差 std 样本均值
    """
    # 计算偏差
    delta = x - y

    range_percent = (delta.min() + (delta.max() - delta.min())) * percent
    idx_greater = np.where(delta > range_percent)
    delta_greater = delta[idx_greater]
    x_greater = x[idx_greater]

    idx_lower = np.where(delta <= range_percent)
    delta_lower = delta[idx_lower]
    x_lower = x[idx_lower]

    md_greater = delta_greater.mean()  # 偏差均值
    std_greater = delta_greater.std()  # 偏差 std
    mt_greater = x_greater.mean()  # 样本均值

    md_lower = delta_lower.mean()
    std_lower = delta_lower.std()
    mt_lower = x_lower.mean()

    info_lower = "MeanBias(<=10%Range)={:.4f}±{:.4f}@{:.4f}".format(
        md_lower, std_lower, mt_lower)
    info_greater = "MeanBias(>10%Range) ={:.4f}±{:.4f}@{:.4f}".format(
        md_greater, std_greater, mt_greater)

    bias_info = {"md_greater": md_greater, "std_greater": std_greater,
                 "mt_greater": mt_greater,
                 "md_lower": md_lower, "std_lower": std_lower,
                 "mt_lower": mt_lower,
                 "info_lower": info_lower, "info_greater": info_greater}

    return bias_info


def draw_distribution(ax, x, y, label=None, ax_annotate=None,
                      xmin=None, xmax=None, ymin=None, ymax=None, gab_number=8.0,
                      is_diagonal=False, avxline=None):
    """
    画偏差分布图
    :return:
    """
    if None in [xmin, xmax]:
        xmin = floor(np.min(x))
        xmax = ceil(np.max(x))
    if None in [ymin, ymax]:
        ymin = floor(np.min(y))
        ymax = ceil(np.max(y))

    # 设置 x y 坐标轴范围
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # 设置 x y 轴的刻度
    ax.xaxis.set_major_locator(MultipleLocator((xmax - xmin) / gab_number))
    ax.xaxis.set_minor_locator(MultipleLocator((xmax - xmin) / gab_number / 2))
    ax.yaxis.set_major_locator(MultipleLocator((ymax - ymin) / gab_number))
    ax.yaxis.set_minor_locator(MultipleLocator((ymax - ymin) / gab_number / 2))

    # 添加 y = 0 的线
    zeroline_width = 1.0
    zeroline_color = '#808080'
    ax.plot([xmin, xmax], [0, 0], color=zeroline_color, linewidth=zeroline_width)

    # 画偏差散点
    delta = x - y  # 计算偏差
    scatter_alpha = 0.8  # 透明度
    scatter_marker = 'o'  # 形状
    scatter_color = 'b'  # 颜色
    ax.scatter(x, delta, s=5, marker=scatter_marker, c=scatter_color, lw=0, alpha=scatter_alpha)

    # 画偏差回归线
    delt_ab = np.polyfit(x, delta, 1)
    delt_a = delt_ab[0]
    delt_b = delt_ab[1]
    regressline_x = [xmin, xmax]
    regressline_y = [xmin * delt_a + delt_b, xmax * delt_a + delt_b]

    regressline_color = 'r'
    regressline_width = 1.2

    ax.plot(regressline_x, regressline_y,
            color=regressline_color, linewidth=regressline_width, zorder=100)

    # 画 avx 注释线
    if avxline is not None:
        avxline_x = avxline.get("line_x")
        avxline_color = avxline.get("x_color")
        avxline_width = 0.7
        avxline_word = avxline.get("word")
        avxline_wordcolor = avxline.get("word_color")
        avxline_wordlocal = avxline.get("word_location")
        avxline_wordsize = avxline.get("word_size")
        ax.axvline(x=avxline_x, color=avxline_color, lw=avxline_width)
        ax.annotate(avxline_word, avxline_wordlocal,
                    va="top", ha="center", color=avxline_wordcolor,
                    size=avxline_wordsize, fontproperties=FONT_MONO)

    # 注释，格式 ['annotate1', 'annotate2']
    if ax_annotate is not None:
        font_size = 10
        add_annotate(ax, ax_annotate.get("left"), "left", fontsize=font_size)
        add_annotate(ax, ax_annotate.get("right"), "right", fontsize=font_size)

    # 标签
    if label is not None:
        add_label(ax, label.get("xlabel"), "xlabel")  # x 轴标签
        add_label(ax, label.get("ylabel"), "ylabel")  # y 轴标签

    # 设置 tick 字体
    set_tick_font(ax)


def draw_histogram(ax, x, y=None, x_label=None, y_label=None, ax_annotate={},
                   hist_label_x='X', hist_label_y='Y',
                   xmin=None, xmax=None, ymin=None, ymax=None, gab_number=8.0,
                   is_diagonal=False):
    """
    画直方图
    :param hist_label_y:
    :param hist_label_x:
    :param ax:
    :param x:
    :param y: 如果绘制 y ，需要和 x 的坐标轴相同
    :param x_label:
    :param y_label:
    :param ax_annotate:
    :param xmin:
    :param xmax:
    :param ymin:
    :param ymax:
    :param gab_number:
    :param is_diagonal:
    :return:
    """
    if ax_annotate is None:
        ax_annotate = {}
    if None in [xmin, xmax,]:
        xmin = floor(np.min(x))
        xmax = ceil(np.max(x))

    alpha = 0.4
    color_x = "red"
    hist_label_x = hist_label_x
    ax.hist(x, 100, range=(xmin, xmax), histtype='bar', color=color_x,
            label=hist_label_x, alpha=alpha)

    if y is not None:
        color_y = "blue"
        hist_label_y = hist_label_y
        ax.hist(y, 100, range=(xmin, xmax), histtype='bar', color=color_y,
                label=hist_label_y, alpha=alpha)

    # 设置标签
    ax.legend(prop={'size': 10})
    # 注释，格式['annotate1', 'annotate2']
    add_annotate(ax, ax_annotate.get("left"), "left", fontsize=10)
    add_annotate(ax, ax_annotate.get("right"), "right", fontsize=10)
    add_xylabel(ax, x_label, y_label)
    set_tick_font(ax)


def draw_Scatter_Bar(x, y, filename, titledict, tl_list, tr_list, part1, part2, xname, xname_l,
                 xmin=None, xmax=None, ymin=None, ymax=None):
    """
    画散点回归线图 和 直方图
    x: modis
    y: mersi
    """
    print 'bar 1'
    ab = np.polyfit(x, y, 1)
    if None in [xmin, xmax, ymin, ymax]:
        xmin = floor(np.min(x))
        xmax = ceil(np.max(x))
        ymin = floor(np.min(y))
        ymax = ceil(np.max(y))

    fig = plt.figure(figsize=(14, 4.5))
    fig.subplots_adjust(top=0.92, bottom=0.11, left=0.045, right=0.985)
    print 'bar 1.1'
    alpha = 0.4
    print 'bar 1.1.1'
    # 散点图 --------------------------
    try:
        ax1 = fig.add_subplot(131)
    except Exception, why:
        print why
    # 对角线
    # 设定x y 轴的范围
    print 'bar 1.2'
    xylimMax = max(xmax, ymax)
    xylimMin = min(xmin, ymin)
    ax1.set_xlim(xylimMin, xylimMax)
    ax1.set_ylim(xylimMin, xylimMax)
    xmax = xylimMax
    xmin = xylimMin
    print 'bar 2'
    ax1.plot([xylimMin, xylimMax], [xylimMin, xylimMax], color='#808080', linewidth=1.0)
    print 'bar 3'
    # 回归线
    ax1.plot([xmin, xmax], [ab[0] * xmin + ab[1], ab[0] * xmax + ab[1]],
             color='r', linewidth=1.2, zorder=100)
    print 'bar 4'
    # 画散点
    colorValues = 'b'
    ax1.scatter(x, y, s=5, marker='o', c=colorValues, lw=0, alpha=alpha)
    print 'bar 5'
    # 设定小刻度
    xticklocs = ax1.xaxis.get_ticklocs()
    yticklocs = ax1.yaxis.get_ticklocs()
    ax1.xaxis.set_minor_locator(MultipleLocator((xticklocs[1] - xticklocs[0]) / 5))
    ax1.yaxis.set_minor_locator(MultipleLocator((yticklocs[1] - yticklocs[0]) / 5))

    add_annotate(ax1, tl_list, 'left')  # 注释文字
    add_annotate(ax1, tr_list, 'right')  #  '#9300d3'
    add_xylabel(ax1, titledict["xlabel"], titledict["ylabel"])
    set_tick_font(ax1)
    print 'bar 6'
    # 直方图 ---------------------------------------

    ax2 = fig.add_subplot(133)
    ax2.grid(True)
    color = "red"
    ax2.hist(x, 100, range=(xmin, xmax), histtype='bar', color=color, label=part1, alpha=alpha)
    print 'bar 7'
    color = "blue"
    ax2.hist(y, 100, range=(xmin, xmax), histtype='bar', color=color, label=part2, alpha=alpha)
    ax2.legend(prop={'size': 10})
    add_xylabel(ax2, xname_l, "match point numbers")
    set_tick_font(ax2)

    # 20180126 徐娜要求增加偏差分布图 --------------------------
    ax3 = fig.add_subplot(132)
    ax3.grid(True)
    ax3.set_xlim(xylimMin, xylimMax)
    # format the Yticks
    if xname == "tbb":
        # 20180313 徐娜要改为（-4，4） --------------------------
        ax3.set_ylim(-4, 4)
        ax3.yaxis.set_major_locator(MultipleLocator(2))
        ax3.yaxis.set_minor_locator(MultipleLocator(0.4))
    elif xname == "ref":
        ax3.set_ylim(-0.08, 0.081)
        ax3.yaxis.set_major_locator(MultipleLocator(0.04))
        ax3.yaxis.set_minor_locator(MultipleLocator(0.008))
    delta = x - y

    # 过滤 range%10 范围的值，计算偏差信息
    # MeanBias( <= 10 % Range) = MD±Std @ MT
    # MeanBias( > 10 % Range) = MD±Std @ MT
    range_10 = (delta.min() + (delta.max() - delta.min())) * 0.1
    idx_greater_10 = np.where(delta > range_10)
    delta_greater_10 = delta[idx_greater_10]
    x_greater_10 = x[idx_greater_10]

    idx_lower_10 = np.where(delta <= range_10)
    delta_lower_10 = delta[idx_lower_10]
    x_lower_10 = x[idx_lower_10]

    md_greater_10 = delta_greater_10.mean()  # 偏差均值
    std_greater_10 = delta_greater_10.std()  # 偏差 std
    mt_greater_10 = x_greater_10.mean()  # 样本均值

    md_lower_10 = delta_lower_10.mean()
    std_lower_10 = delta_lower_10.std()
    mt_lower_10 = x_lower_10.mean()

    # 添加 y = 0 的线
    COLOR_Darkgray = '#808080'
    ax3.plot([xmin, xmax], [0, 0], color=COLOR_Darkgray, linewidth=1.0)

    # 散点
    ax3.scatter(x, delta, s=5, marker='o', c=colorValues, lw=0, alpha=alpha)
    delt_ab = np.polyfit(x, delta, 1)
    ax3.plot([xmin, xmax], [delt_ab[0] * xmin + delt_ab[1], delt_ab[0] * xmax + delt_ab[1]],
             color='r', linewidth=1.2, zorder=100)
    add_xylabel(ax3, xname_l, "%s minus %s %s bias" % (part1, part2, xname))
    set_tick_font(ax3)

    fig.suptitle(titledict['title'], fontsize=11, fontproperties=FONT0)
    plt.savefig(filename)
    fig.clear()
    plt.close()
    print 'bar 8'


def draw_Scatter_withColorbar(x, y, filename, titledict, tl_list, tr_list):
    """
    画散点回归线图
    x: modis
    y: mersi
    d: mesi - modis
    """
    ab = np.polyfit(x, y, 1)
    xmin = floor(np.min(x))
    xmax = ceil(np.max(x))
    ymin = floor(np.min(y))
    ymax = ceil(np.max(y))

    fig = plt.figure(figsize=(6, 4))
    fig.subplots_adjust(top=0.94, bottom=0.13, left=0.11, right=0.89)

    plt.grid(True)

    # 设定x y 轴的范围
    xylimMax = max(xmax, ymax)
    xylimMin = min(xmin, ymin)
    plt.xlim(xylimMin, xylimMax)
    plt.ylim(xylimMin, xylimMax)

    # 对角线
    plt.plot([xylimMin, xylimMax], [xylimMin, xylimMax], color='#808080', linewidth=1)

    # 回归线
    plt.plot([xylimMin, xylimMax], [ab[0] * xylimMin + ab[1], ab[0] * xylimMax + ab[1]], color='r',
             linewidth=1.2, zorder=100)

    # 对角线两侧区域线
    # adjust_ab = (0.1, 0.05)
    # refer_ab1 = (1 + adjust_ab[0], adjust_ab[1])
    # refer_ab2 = (1 - adjust_ab[0], -1 * adjust_ab[1])
    # plt.plot([xmin, xmax], [refer_ab1[0] * xmin + refer_ab1[1], refer_ab1[0] * xmax + refer_ab1[1]],
    #          color='#FFB5C5', linewidth=0.8)
    # plt.plot([xmin, xmax], [refer_ab2[0] * xmin + refer_ab2[1], refer_ab2[0] * xmax + refer_ab2[1]],
    #          color='#FFB5C5', linewidth=0.8)

    # 画散点
    colorValues = get_dot_color(x, y)
    norm = plt.Normalize()
    norm.autoscale(colorValues)
    plt.scatter(x, y, s=2, marker='o', c=colorValues, norm=norm, lw=0)

    ax = plt.gca()

    # 设定小刻度
    xticklocs = ax.xaxis.get_ticklocs()
    yticklocs = ax.yaxis.get_ticklocs()
    ax.xaxis.set_minor_locator(MultipleLocator((xticklocs[1] - xticklocs[0]) / 5))
    ax.yaxis.set_minor_locator(MultipleLocator((yticklocs[1] - yticklocs[0]) / 5))

    add_annotate(ax, tl_list, 'left')  # 注释文字
    add_annotate(ax, tr_list, 'right')
    add_title(titledict)
    set_tick_font(ax)
    add_colorbar_right_vertical(fig, 0, np.max(colorValues))
    print filename
    plt.savefig(filename)
    fig.clear()
    plt.close()
    return


def draw_density(ax, x, y, x_label=None, y_label=None, ax_annotate={},
                 xmin=None, xmax=None, ymin=None, ymax=None, gab_number=5,
                 is_diagonal=False):
    """
    绘制密度图
    :return:
    """
    if None in [xmin, xmax, ymin, ymax]:
        xmin = floor(np.min(x))
        xmax = ceil(np.max(x))
        ymin = floor(np.min(y))
        ymax = ceil(np.max(y))

    # 设定x y 轴的范围
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # 计算 a b
    ab = np.polyfit(x, y, 1)
    a = ab[0]
    b = ab[1]

    # 画对角线
    diagonal_color = '#808080'
    diagonal_lw = 1.2
    if is_diagonal:
        ax.plot([xmin, xmax], [ymin, ymax], color=diagonal_color,
                linewidth=diagonal_lw)

    # 画回归线
    ax.plot([xmin, xmax], [xmin * a + b, xmax * a + b],
            color='r', linewidth=1.2, zorder=100)

    # 画密度点
    pos = np.vstack([x, y])
    kernel = stats.gaussian_kde(pos)
    z = kernel(pos)
    norm = plt.Normalize()
    norm.autoscale(z)
    ax.scatter(x, y, c=z, norm=norm, s=5, marker="o", cmap=plt.cm.jet, lw=0, alpha=1)

    # 设定小刻度
    xticklocs = ax.xaxis.get_ticklocs()
    yticklocs = ax.yaxis.get_ticklocs()
    ax.xaxis.set_minor_locator(
        MultipleLocator((xticklocs[1] - xticklocs[0]) / gab_number))
    ax.yaxis.set_minor_locator(
        MultipleLocator((yticklocs[1] - yticklocs[0]) / gab_number))

    # 注释，格式['annotate1', 'annotate2']
    add_annotate(ax, ax_annotate.get("left"), "left", fontsize=10)
    add_annotate(ax, ax_annotate.get("right"), "right", fontsize=10)
    add_xylabel(ax, x_label, y_label)
    set_tick_font(ax)


def add_colorbar_right_vertical(fig, colorMin, colorMax,
                                label_format='%d',
                                cbar_width=0.04,
                                colormap='jet', font_size=SCALE_SIZE):
    '''
    add colorbar at right of pic
    '''
    fig.canvas.draw()
    ax = plt.gca()
    point_bl = ax.get_position().get_points()[0]  # 左下
    point_tr = ax.get_position().get_points()[1]  # 右上
    space = 0.015
    colorbar_ax = fig.add_axes([point_tr[0] + space,
                                point_bl[1],
                                cbar_width,
                                point_tr[1] - point_bl[1]])
    norm = mpl.colors.Normalize(vmin=colorMin, vmax=colorMax)
    mpl.colorbar.ColorbarBase(colorbar_ax, cmap=colormap,
                              norm=norm,
                              orientation='vertical', format=label_format)
    # font of colorbar
    for l in colorbar_ax.yaxis.get_ticklabels():
        l.set_fontproperties(FONT0)
        l.set_fontsize(font_size)


def get_dot_color(xlist, ylist):
    """
    取得color数组
    """
    divide = 200.  # 分成200份
    colors = np.zeros(len(xlist))
    xmin = floor(np.min(xlist))
    xmax = ceil(np.max(xlist))
    ymin = floor(np.min(ylist))
    ymax = ceil(np.max(ylist))

    xstep = (xmax - xmin) / divide
    ystep = (ymax - ymin) / divide

    x1 = xmin
    while x1 < xmax:
        x2 = x1 + xstep
        y1 = ymin
        while y1 < ymax:
            y2 = y1 + ystep
            condition = np.logical_and(xlist >= x1, xlist < x2)
            condition = np.logical_and(condition, ylist >= y1)
            condition = np.logical_and(condition, ylist < y2)

            colors[condition] = np.count_nonzero(condition)
            y1 = y2
        x1 = x2
    return colors


def draw_time_fig(time_list, data_dic, picPath, titledict, Rate, tl_list=None):
    """
    画时序图
    """
    fig_height = 4.
    # fig_width = max(int(len(time_list)* 0.8), fig_height+1.)
    fig_width = fig_height * 6. / 4.

    fig = plt.figure(figsize=(fig_width, fig_height))

    #     fig.subplots_adjust(top =0.93, bottom =0.08, left =0.025, right =0.99)
    flag = 0
    for eachkey in data_dic.keys():
        if (np.isnan(data_dic[eachkey])).all():
            continue
        # plt.plot(time_list, data_dic[eachkey],
        #                'o-', color = color_dict[eachkey], ms = 4, lw = 0.6,
        #                mew=0, label=eachkey)
        else:
            flag = 1
#             print eachkey
            plt.plot(time_list, data_dic[eachkey],
                     'o-', ms=4, lw=0.6,
                     mew=0, label=eachkey)
    #     fig.autofmt_xdate()
    #     plt.xlim(time_list[0]-relativedelta(days=1), time_list[-1]+relativedelta(days=1))
    # legend
    if flag == 1:
        legs = plt.legend(loc=1, frameon=True)  # 标注
        legs.get_frame().set_alpha(0.4)  # 透明度
        plt.setp(gca().get_legend().get_texts(), fontproperties=FONT0)  # 设置标注字体

        fr = legs.get_frame()
        light_grey = np.array([float(248) / float(255)] * 3)
        #     legend = ax.legend(frameon=True, scatterpoints=1)
        fr.set_facecolor(light_grey)
        fr.set_linewidth(0.6)

        ax = plt.gca()

        # format the ticks
        date_range = len(time_list)
        print 'date_range:', date_range
        if date_range >= 10:
            my_intervar = int(date_range / 10)
        else:
            my_intervar = 1

        if Rate == 'H':
            plt.xlim(time_list[0] - relativedelta(hours=1), time_list[-1] + relativedelta(hours=1))
            days = mdates.HourLocator(interval=my_intervar)
            ax.xaxis.set_major_locator(days)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        elif Rate == 'D':
            plt.xlim(time_list[0] - relativedelta(days=1), time_list[-1] + relativedelta(days=1))
            days = mdates.DayLocator(interval=my_intervar)
            ax.xaxis.set_major_locator(days)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        elif Rate == 'M':
            plt.xlim(time_list[0] - relativedelta(months=1), time_list[-1] + relativedelta(months=1))
            days = mdates.MonthLocator(interval=my_intervar)
            ax.xaxis.set_major_locator(days)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        elif Rate == 'Y':
            plt.xlim(time_list[0] - relativedelta(years=1), time_list[-1] + relativedelta(years=1))
            days = mdates.YearLocator()
            ax.xaxis.set_major_locator(days)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        set_tick_font(ax)

        # title
        add_title(titledict)
        # comment
        add_annotate(ax, tl_list, 'left')

        plt.grid(True)
        plt.savefig(picPath, dpi=600)
        fig.clear()
        plt.close()


# ---------------画 map-------------------
from pylab import *
# color for map
# OCEAN = '#b3d1ff'
GRAY = '#f4f3f0'
OCEAN = GRAY
GRAY_COAST = 'k'  # '#202020'
GRAY_COUNTRY = 'k'  # '#707070'


def map_base(box, delat, delon, china=False):
    '''
    地图底图
    '''
    nlat, slat, wlon, elon = box
    fig_height = 8.
    fig_width = np.floor(fig_height * (elon - wlon) / (nlat - slat) * 0.95)

    fig = plt.figure(figsize=(fig_width, fig_height))

    m = Basemap(llcrnrlon=floor(wlon), llcrnrlat=floor(slat),
                urcrnrlon=ceil(elon), urcrnrlat=ceil(nlat),
                projection='cyl', resolution='l')  # resolution = 'h' for high-res coastlines

    # 背景颜色
    m.drawmapboundary(fill_color=OCEAN)
    # fill continents, set lake color same as ocean color.
    m.fillcontinents(color=GRAY, lake_color=OCEAN, zorder=0)

    # 画 海岸线 和 国境线
    #     m.drawcoastlines(linewidth = 0.9, color='w', zorder = 1)
    m.drawcoastlines(linewidth=0.3, color=GRAY_COAST, zorder=102)
    #     m.drawcountries(linewidth = 0.9, color='w', zorder = 1) # 为国境线添加白边，美观
    #     m.drawcountries(linewidth = 0.3, color=GRAY, zorder = 2)
    m.drawcountries(linewidth=0.3, color=GRAY_COUNTRY, zorder=103)
    m.drawcountries(linewidth=0.3, color=GRAY_COUNTRY, zorder=103)

    # draw parallels
    circles = np.arange(0., 91., delat).tolist() + \
              np.arange(-delat, -91., -delat).tolist()
    m.drawparallels(circles, linewidth=0.2, labels=[1, 0, 0, 1], dashes=[1, 1], fontproperties=FONT0)

    # draw meridians
    meridians = np.arange(0., 180., delon).tolist() + \
                np.arange(-delon, -180., -delon).tolist()
    m.drawmeridians(meridians, linewidth=0.2, labels=[1, 0, 0, 1], dashes=[1, 1], fontproperties=FONT0)

    if china:
        localpath = os.path.split(os.path.realpath(__file__))[0]  # 取得本程序所在目录
        # shape
        m.readshapefile(os.path.join(localpath, 'CHN_adm/CHN_adm1'), 'province', linewidth=0.2, color='w')
        m.readshapefile(os.path.join(localpath, 'CHN_adm/CHN_adm1'), 'province', linewidth=0.2, color=GRAY_COUNTRY)
    return fig, m


def draw_map_mark_types(lonslst, latslst, colorlst, box, filename, title):
    '''
    画多个颜色的点
    '''
    fig, m = map_base(box, 2., 2., True)
    fig.subplots_adjust(top=0.94, bottom=0.06, left=0.08, right=0.95)

    if len(lonslst) == len(latslst) == len(colorlst):
        pass
    else:

        return

    for i in range(len(lonslst)):
        lons = lonslst[i]
        lats = latslst[i]
        color = colorlst[i]
        x1, y1 = m(lons, lats)
        #         m.scatter(x1, y1, s=6, c='w', marker='D', linewidths = 0, alpha=1, zorder = 9)
        m.scatter(x1, y1, s=6, c=color, marker='D', linewidths=0, alpha=1, zorder=10)

    plt.title(title, fontsize=14, fontproperties=FONT0)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    return


def draw_map_with_colormap(lons, lats, values, box, filename, title,
                           point_size=2, colormap='jet', vmin=0., vmax=1.2):
    '''
    画带色标的地图
    '''
    zm = ma.masked_where(values <= vmin, values)  # !!!!!!!!

    fig, m = map_base(box, 10., 20.)
    #     fig.subplots_adjust(top =0.94, bottom =0.06)

    # 颜色标尺
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # ## Drawing
    x1, y1 = m(lons, lats)

    #     c = m.pcolormesh(x1, y1, zm, norm = norm, rasterized=True, zorder = 10)
    c = m.scatter(x1, y1, s=point_size, marker='s', c=zm, cmap=colormap, norm=norm, lw=0)

    ax = plt.gca()
    spines = ax.spines
    for eachspine in spines:
        spines[eachspine].set_linewidth(0.7)

    tt = plt.title(title, fontsize=17, fontproperties=FONT0)
    tt.set_y(1.02)  # set gap space below title and subplot
    plt.grid(True)

    add_colorbar_right_vertical(fig, vmin, vmax, '%.2f', 0.02, colormap, 11)

    plt.savefig(filename, dpi=600)

    plt.close()
    return


def draw_map_with_colormap_interp(lons, lats, values, box, filename, title,
                                  colormap='jet', vmin=0., vmax=1.2):
    '''
    画带色标的地图(平滑) TEST
    '''
    zm = ma.masked_where(values <= vmin, values)  # !!!!!!!!

    lon_new = lons[0]
    lat_new = lats[:, 0]
    (nlat, slat, wlon, elon) = box
    ll = lonlat(nlat, slat, wlon, elon, (lon_new[1] - lon_new[0]) / 2.)
    lon_out = ll.getLons()
    lat_out = ll.getLats()
    #     print lon_new, lat_new
    #     print lat_out.shape, lon_out.shape
    result = mapInterp(values, lon_new, lat_new, \
                       lon_out, lat_out, checkbounds=False, masked=True, order=1)

    fig, m = map_base(box, 10., 20.)
    #     fig.subplots_adjust(top =0.94, bottom =0.06)

    # 颜色标尺
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # ## Drawing
    x1, y1 = m(lon_out, lat_out)
    c = m.pcolormesh(x1, y1, result, norm=norm, rasterized=True, zorder=10)
    #     cs = m.contourf(x1,y1,zm, levels = np.arange(vmin,vmax,0.2), extend='both')

    ax = plt.gca()
    spines = ax.spines
    for eachspine in spines:
        spines[eachspine].set_linewidth(0.7)

    tt = plt.title(title, fontsize=17, fontproperties=FONT0)
    tt.set_y(1.02)  # set gap space below title and subplot
    plt.grid(True)

    add_colorbar_right_vertical(fig, vmin, vmax, '%.2f', 0.02, colormap, 11)

    plt.savefig(filename)
    plt.close()
    return


class lonlat():
    '''
    等经纬度区域类
    '''

    def __init__(self, nlat, slat, wlon, elon, res, zeropoint_at='tl', edge=False):
        '''
        nlat, slat, wlon, elon: 北纬, 南纬, 西经, 东经
        res: 分辨率（度）
        zeropoint_at: 坐标0点在左上角'tl' or 左下角'bl'
        edge: False 表示经纬度代表网格中心店， True 表示经纬度代表网格边缘
        '''
        self.nlat = float(nlat)  # 北纬
        self.slat = float(slat)  # 南纬
        self.wlon = float(wlon)  # 西经
        self.elon = float(elon)  # 东经
        self.res = float(res)  # 分辨率
        self.zeropoint_at = zeropoint_at

        self.rowMax = int(round((self.nlat - self.slat) / self.res))  # 最大行数
        self.colMax = int(round((self.elon - self.wlon) / self.res))  # 最大列数

        if edge:
            self.rowMax = self.rowMax + 1
            self.colMax = self.colMax + 1

    def getLons(self):
        '''
        ret: 返回 等经纬度网格上的经度2维数组，以左下角为起点0,0
        '''
        lons = []
        for i in xrange(self.rowMax):
            lons.append(self.__getLonsRange_edge())
        return np.array(lons)

    def getLats(self):
        '''
        ret: 返回 等经纬度网格上的纬度2维数组，以左下角为起点0,0
        '''
        lats = []
        latr = self.__getLatsRange_edge()
        for i in xrange(self.rowMax):
            lats.append([latr[i]] * self.colMax)
        return np.array(lats)

    def __getLonsRange_edge(self):
        '''
        ret: 返回 等经纬度网格上的一行经度1维数组
        '''
        return np.arange(self.wlon, self.elon, self.res)

    def __getLatsRange_edge(self):
        '''
        ret: 返回 等经纬度网格上的一列纬度1维数组
        '''
        if self.zeropoint_at == 'tl':  # topleft
            return np.arange(self.nlat, self.slat, self.res * (-1))
        elif self.zeropoint_at == 'bl':  # bottomleft
            return np.arange(self.slat, self.nlat, self.res)

# def line_area_count(x,y,ab):
#     res= np.logical_and((1.1*ab[1]*x+1.1*ab[2]+0.05)>=y,(0.9*ab[1]*x+0.9*ab[2]-0.05)<=y)
#     return np.count_nonzero(res)


if __name__ == '__main__':
    plt.style.use('dv_pub_legacy.mplstyle')
#     from DM.SNO.dm_sno_cross_calc_map import RED, BLUE, EDGE_GRAY, ORG_NAME
    draw_Scatter([1, 2, 3, 4, 5], [1, 2, 6, 3, 4],
                 '1.png', {'title':'ttt'},
                 [['1234   56789', 'bbbb   bbb'], ['aaaaaaaaa', 'bbbbbbb']],
                 [])
