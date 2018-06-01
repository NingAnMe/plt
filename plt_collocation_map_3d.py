# coding: utf-8
"""
Created on 2016年1月6日
读取匹配后的 HDF5 文件，画蝴蝶图

@author: zhangtao anning
"""
import os
import sys

import numpy as np

from configobj import ConfigObj

from DV.dv_pub_3d_dev import plt, Basemap
from DV.dv_pub_3d_dev import FONT0
from DM.SNO.dm_sno_cross_calc_map import *
from PB.CSC.pb_csc_console import LogServer
from PB import pb_time, pb_io
from PB.pb_time import is_day_timestamp_and_lon

from plt_pb_io import ReadHDF5, loadYamlCfg


def run(pair, ymd):
    """
    pair: sat1+sensor1_sat2+sensor2
    ymd: YYYYMMDD
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
    plt_cfg_file = os.path.join(MAIN_PATH, "cfg", "%s_%s_3d.yaml" % (sensor1, sensor2))
    plt_cfg = loadYamlCfg(plt_cfg_file)
    if plt_cfg is None:
        Log.error("Not find the config file: {}".format(plt_cfg_file))
        return

    # 读取配置文件的信息
    PERIOD = plt_cfg['collocation_map']['days']  # 回滚天数
    chans = plt_cfg['collocation_map']['chan']  # 通道
    maptype = plt_cfg['collocation_map']['maptype']  # 需要绘制的类型

    if 'area' in maptype:  # 区域块视图
        area = plt_cfg['collocation_map']['area']
    else:
        area = None
    if 'polar' in maptype:  # 两极视图
        polar = plt_cfg['collocation_map']['polar']
    else:
        polar = None

    # 读取范围配置
    if not area and not polar:
        return
    else:
        map_range = (polar, area)

    Log.info(u"----- Start Drawing Matched Map-Pic, PAIR: {}, YMD: {}, PERIOD: {} -----".format(pair, ymd, PERIOD))

    # 读取 HDF5 文件数据
    oneHDF5 = ReadHDF5()
    num_file = PERIOD
    cur_ymd = pb_time.ymd_plus(ymd, 1)  # 回滚天数，现在为 1
    for daydelta in xrange(PERIOD):
        cur_ymd = pb_time.ymd_plus(ymd, -daydelta)
        filename = "COLLOC+%sIR,%s+%s_%s+%s_C_BABJ_%s.hdf5" % (Type, sat1,
                                                               sensor1, sat2,
                                                               sensor2, cur_ymd)
        filefullpath = os.path.join(MATCH_DIR, pair, filename)
        if not os.path.isfile(filefullpath):
            Log.info(u"File not found: {}".format(filefullpath))
            num_file -= 1
            continue

        if not oneHDF5.LoadData(filefullpath, chans):
            Log.error('Error occur when reading %s of %s' % (chans,
                                                             filefullpath))
    if num_file == 0:
        Log.error(u"No file found.")
        return
    elif num_file != PERIOD:
        Log.error(u"{} of {} file(s) found.".format(num_file, PERIOD))
    cur_path = os.path.join(DMS_DIR, pair, ymd)

    o_file = os.path.join(cur_path,
                          '%s_%s_MatchedPoints_ALL_%s' % (part1, part2, ymd))

    # find out day and night
    vect_is_day = np.vectorize(is_day_timestamp_and_lon)
    day_index = vect_is_day(oneHDF5.time, oneHDF5.lon1)
    night_index = np.logical_not(day_index)

    x = oneHDF5.lon1  # 经度数据
    y = oneHDF5.lat1  # 维度数据
    print 'date: {}, x_all: {} y_all: {} '.format(ymd, len(x), len(y))

    draw_butterfly(part1, part2, cur_ymd, ymd, x, y, o_file, map_range)
    # ------- day ----------
    if np.where(day_index)[0].size > 0:
        o_file = os.path.join(cur_path,
                              '%s-%s_MatchedPoints_Day_%s' % (part1, part2,
                                                              ymd))
        x_d = x[day_index]
        y_d = y[day_index]
        print 'date: {}, x_day: {} y_day: {} '.format(ymd, len(x_d), len(y_d))
        draw_butterfly(part1, part2, cur_ymd, ymd, x_d, y_d, o_file, map_range)
    # ---------night ------------
    if np.where(night_index)[0].size > 0:
        o_file = os.path.join(cur_path,
                              '%s-%s_MatchedPoints_Night_%s' % (part1, part2,
                                                                ymd))
        x_n = x[night_index]
        y_n = y[night_index]
        print 'date: {}, x_night: {} y_night: {} '.format(ymd, len(x_n), len(y_n))
        draw_butterfly(part1, part2, cur_ymd, ymd, x_n, y_n, o_file, map_range)
    Log.info(u"Success")


def draw_butterfly(sat1Nm, sat2Nm,
                   ymd_s, ymd_e,
                   lons, lats,
                   out_fig_file, map_range):
    """
    画 FY3X 匹配蝴蝶图
    """
    if len(lons) == 0:
        return
    plt.style.use(os.path.join(DV_PATH, 'dv_pub_map.mplstyle'))
#     COLORS = ['#4cd964', '#1abc9c', '#5ac8fa', '#007aff', '#5856d6']
    COLORS = [RED]
    if map_range[0] and map_range[1]:
        fig = plt.figure(figsize=(8, 10), dpi=100)  # china
        plt.subplots_adjust(left=0.09, right=0.93, bottom=0.12, top=0.94)
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

        # 画两极
        polar_range = map_range[0]  # 两极范围
        m = drawFig_map(ax1, "north", polar_range)
        plot_matchpoint(m, lons, lats, COLORS[0])
        m = drawFig_map(ax2, "south", polar_range)
        plot_matchpoint(m, lons, lats, COLORS[0])

        # 画区域
        area_range = map_range[1]  # 区域范围
        m = drawFig_map(ax3, "area", area_range)
        plot_matchpoint(m, lons, lats, COLORS[0])

    elif map_range[0]:
        fig = plt.figure(figsize=(8, 5), dpi=100)  # china
        plt.subplots_adjust(left=0.09, right=0.93, bottom=0.12, top=0.94)
        ax1 = plt.subplot2grid((1, 2), (0, 0))
        ax2 = plt.subplot2grid((1, 2), (0, 1))

        # 画两极
        polar_range = map_range[0]  # 两极范围
        m = drawFig_map(ax1, "north", polar_range)
        plot_matchpoint(m, lons, lats, COLORS[0])
        m = drawFig_map(ax2, "south", polar_range)
        plot_matchpoint(m, lons, lats, COLORS[0])

    elif map_range[1]:
        fig = plt.figure(figsize=(8, 5), dpi=100)  # china
        plt.subplots_adjust(left=0.09, right=0.93, bottom=0.12, top=0.94)
        ax3 = plt.subplot2grid((1, 2), (0, 0), colspan=2)

        # 画区域
        area_range = map_range[1]  # 区域范围
        m = drawFig_map(ax3, "area", area_range)
        plot_matchpoint(m, lons, lats, COLORS[0])

    # ---------legend-----------

    # 对整张图片添加文字
    TEXT_Y = 0.05
    fig.text(0.1, TEXT_Y, '%s' % sat1Nm, color=RED, fontproperties=FONT0)
    if ymd_s != ymd_e:
        fig.text(0.55, TEXT_Y, '%s-%s' % (ymd_s, ymd_e), fontproperties=FONT0)
    else:
        fig.text(0.55, TEXT_Y, '%s' % ymd_s, fontproperties=FONT0)
    fig.text(0.83, TEXT_Y, ORG_NAME, fontproperties=FONT0)

    pb_io.make_sure_path_exists(os.path.dirname(out_fig_file))
    fig.savefig(out_fig_file, dpi=100)
    print out_fig_file
    print '-' * 50
    plt.close()
    fig.clear()


def drawFig_map(ax, n_s, range):
    """
    create new figure
    """
    if n_s == "area":
        area_range = range
        llcrnrlat = area_range[0]
        urcrnrlat = area_range[1]
        llcrnrlon = area_range[2]
        urcrnrlon = area_range[3]

        m = Basemap(llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                    llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                    resolution='c', area_thresh=10000.,
                    projection='cyl', lat_ts=20., ax=ax)

        m.fillcontinents(color=GRAY)

        # draw parallels
        m.drawparallels(np.arange(90., -91., -30.), linewidth=LINE_WIDTH,
                        labels=[1, 0, 0, 1],
                        dashes=[100, .0001], color='white',
                        textcolor=EDGE_GRAY, fontproperties=TICKER_FONT)
        # draw meridians
        m.drawmeridians(np.arange(-180., 180., 30.), linewidth=LINE_WIDTH,
                        labels=[1, 0, 0, 1],
                        dashes=[100, .0001], color='white',
                        textcolor=EDGE_GRAY, fontproperties=TICKER_FONT)
        # draw parallels
        m.drawparallels(np.arange(60., -90., -30.), linewidth=LINE_WIDTH,
                        labels=[0, 0, 0, 0],
                        dashes=[100, .0001], color='white')
        # draw meridians
        m.drawmeridians(np.arange(-150., 180., 30.), linewidth=LINE_WIDTH,
                        labels=[0, 0, 0, 0],
                        dashes=[100, .0001], color='white')
        ax.set_title("Global Map", fontproperties=FONT0)

    else:
        if n_s == "north":
            north_range = range[0]
            north_range = int(north_range[0])
            m = Basemap(projection='npaeqd', boundinglat=north_range-1,
                        lon_0=0, resolution='c', ax=ax)

        elif n_s == "south":
            south_range = range[1]
            south_range = int(south_range[0])
            m = Basemap(projection='spaeqd', boundinglat=south_range+1,
                        lon_0=180, resolution='c', ax=ax)

        m.fillcontinents(color=GRAY)

        # draw parallels
        m.drawparallels(np.arange(60., 91., 10.), linewidth=LINE_WIDTH,
                        labels=[1, 0, 0, 1],
                        dashes=[100, .0001], color='white',
                        textcolor=EDGE_GRAY, fontproperties=TICKER_FONT)
        # draw meridians
        m.drawmeridians(np.arange(-180., 180., 30.), linewidth=LINE_WIDTH,
                        labels=[1, 0, 0, 1],
                        dashes=[100, .0001], color='white', latmax=90,
                        textcolor=EDGE_GRAY, fontproperties=TICKER_FONT)
        # draw parallels
        m.drawparallels(np.arange(60., -91., -10.), linewidth=LINE_WIDTH,
                        labels=[0, 0, 0, 0],
                        dashes=[100, .0001], color='white')
        # draw meridians
        m.drawmeridians(np.arange(-180., 180., 30.), linewidth=LINE_WIDTH,
                        labels=[0, 0, 0, 0],
                        dashes=[100, .0001], color='white', latmax=90)

        if n_s == "north":
            lat_labels = [(0.0, i) for i in xrange(north_range, 91, 10)]
            for lon, lat in lat_labels:
                xpt, ypt = m(lon, lat)
                ax.text(xpt - 500000, ypt - 100000, str(lat)[0:2] + u'°N',
                        fontproperties=TICKER_FONT)

            ax.set_title("Northern Hemisphere", fontproperties=FONT0)
        elif n_s == "south":
            lat_labels = [(0.0, i) for i in xrange(south_range, -91, -10)]
            for lon, lat in lat_labels:
                xpt, ypt = m(lon, lat)
                ax.text(xpt + 500000, ypt + 200000, str(lat)[1:3] + u'°S',
                        fontproperties=TICKER_FONT)
            ax.set_title("Southern Hemisphere", fontproperties=FONT0)

    return m


def plot_matchpoint(m, lons, lats, color, alpha=1):
    # plot
    markersize = 1.8
    x, y = m(lons, lats)
    m.plot(x, y, marker='o', linewidth=0, markerfacecolor=color,
           markersize=markersize,
           markeredgecolor=None, mew=0, alpha=alpha)


######################### 程序全局入口 ##############################
if __name__ == "__main__":
    # 获取程序参数接口
    ARGS = sys.argv[1:]
    HELP_INFO = \
        u"""
        [参数1]：pair 卫星对
        [参数2]：yyyymmdd 时间
        [样例]: python app.py pair yyyymmdd
        """
    if "-h" in ARGS:
        print HELP_INFO
        sys.exit(-1)

    # 获取程序所在位置，拼接配置文件
    MAIN_PATH, MAIN_FILE = os.path.split(os.path.realpath(__file__))
    PROJECT_PATH = os.path.dirname(MAIN_PATH)
    OM_PATH = os.path.dirname(PROJECT_PATH)
    DV_PATH = os.path.join(os.path.dirname(OM_PATH), "DV")
    CONFIG_FILE = os.path.join(PROJECT_PATH, "cfg", "global.cfg")

    # 配置不存在预警
    if not os.path.isfile(CONFIG_FILE):
        print (u"配置文件不存在 %s" % CONFIG_FILE)
        sys.exit(-1)

    GLOBAL_CONFIG = ConfigObj(CONFIG_FILE)
    ORBIT_DIR = GLOBAL_CONFIG['PATH']['IN']['ORBIT']
    MATCH_DIR = GLOBAL_CONFIG['PATH']['MID']['MATCH_DATA']
    DMS_DIR = GLOBAL_CONFIG['PATH']['OUT']['DMS']
    LogPath = GLOBAL_CONFIG['PATH']['OUT']['LOG']
    Log = LogServer(LogPath)

    if len(ARGS) == 2:
        Log.info(u'手动蝴蝶图绘制程序开始运行-----------------------------')
        satPair = ARGS[0]
        str_time = ARGS[1]

        run(satPair, str_time)

    else:
        print 'args error'
        sys.exit(-1)
