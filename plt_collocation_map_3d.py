# coding: utf-8
'''
Created on 2016年1月6日
读取匹配后的 HDF5 文件，画蝴蝶图

@author: zhangtao anning
'''
import os
import sys
from configobj import ConfigObj
from dateutil.relativedelta import relativedelta
import numpy as np
from mpl_toolkits.basemap import Basemap
from PB import pb_time, pb_io
from PB.pb_time import is_day_timestamp_and_lon
from PB.CSC.pb_csc_console import LogServer
from DM.SNO.dm_sno_cross_calc_map import *
from DM.SNO.dm_sno_cross_calc_core import Sat_Orbit
from multiprocessing import Pool, Lock
from datetime import datetime
from plt_io import ReadHDF5, loadYamlCfg


def run(pair, ymd):
    """
    pair: sat1+sensor1_sat2+sensor2
    ymd: YYYYMMDD
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
    print('1')
    # load yaml config file
    plt_cfg_file = os.path.join(MainPath, '%s_%s_3d.yaml' % (sensor1, sensor2))
    plt_cfg = loadYamlCfg(plt_cfg_file)
    if plt_cfg is None:
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

    if not area and not polar:
        return
    else:
        map_range = (polar, area)
        print('map_range:', map_range)

    Log.info(u"----- Start Drawing Matched Map-Pic,"
             u" PAIR: {}, YMD: {}, PERIOD: {} -----".format(pair, ymd, PERIOD))

    # 读取 HDF5 文件数据
    oneHDF5 = ReadHDF5()
    num_file = PERIOD
    cur_ymd = pb_time.ymd_plus(ymd, 1)
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
    print 'date:{}, x_all:{} y_all:{} '.format(ymd, len(x), len(y))

    draw_butterfly(part1, part2, cur_ymd, ymd, x, y, o_file, map_range)
    # ------- day ----------
    if np.where(day_index)[0].size > 0:
        o_file = os.path.join(cur_path,
                              '%s-%s_MatchedPoints_Day_%s' % (part1, part2,
                                                              ymd))
        x_d = x[day_index]
        y_d = y[day_index]
#         d_d = d[day_index]
        print 'date:{}, x_day:{} y_day:{} '.format(ymd, len(x_d), len(y_d))
        draw_butterfly(part1, part2, cur_ymd, ymd, x_d, y_d, o_file, map_range)
    # ---------night ------------
    if np.where(night_index)[0].size > 0:
        o_file = os.path.join(cur_path,
                              '%s-%s_MatchedPoints_Night_%s' % (part1, part2,
                                                                ymd))
        x_n = x[night_index]
        y_n = y[night_index]
#         d_n = d[night_index]
        print 'date:{}, x_night:{} y_night:{} '.format(ymd, len(x_n), len(y_n))
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
    plt.style.use(os.path.join(dvPath, 'dv_pub_map.mplstyle'))
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
    circle1 = mpatches.Circle((58, 36), 6, color=RED, ec=EDGE_GRAY, lw=0.3)
#     circle_lst = [circle1]
#     for i in xrange(1):
#         circle_lst.append(mpatches.Circle((219 + i * 7, 36), 6, color=COLORS[i], ec=EDGE_GRAY, lw=0.3))
#     fig.patches.extend(circle_lst)

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
    plt.close()
    fig.clear()


def drawFig_map(ax, n_s, range):
    '''
    create new figure
    '''
#     m = Basemap(llcrnrlon=50., llcrnrlat=-40., urcrnrlon=150., urcrnrlat=40., \
#         resolution='c', area_thresh=10000., projection='cyl', \
#         lat_ts=20.)
    if n_s == "area":
        print('area range', range)
        area_range = range
        llcrnrlat = area_range[0]
        urcrnrlat = area_range[1]
        llcrnrlon = area_range[2]
        urcrnrlon = area_range[3]
        print('area')
        m = Basemap(llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                    llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
                    resolution='c', area_thresh=10000.,
                    projection='cyl', lat_ts=20., ax=ax)
        print('area success')
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
        print('area 1')
    else:
        if n_s == "north":
            print('north range', range)
            north_range = range[0]
            north_range = int(north_range[0])
            m = Basemap(projection='npaeqd', boundinglat=north_range-1,
                        lon_0=0, resolution='c', ax=ax)
            print('north success')
        elif n_s == "south":
            print('south range', range)
            south_range = range[1]
            south_range = int(south_range[0])
            m = Basemap(projection='spaeqd', boundinglat=south_range+1,
                        lon_0=180, resolution='c', ax=ax)
            print('south success')

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
            for lon, lat in (lat_labels):
                xpt, ypt = m(lon, lat)
                ax.text(xpt - 500000, ypt - 100000, str(lat)[0:2] + u'°N',
                        fontproperties=TICKER_FONT)

            ax.set_title("Northern Hemisphere", fontproperties=FONT0)
        elif n_s == "south":
            lat_labels = [(0.0, i) for i in xrange(south_range, -91, -10)]
            for lon, lat in (lat_labels):
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
ORBIT_DIR = inCfg['PATH']['IN']['ORBIT']
MATCH_DIR = inCfg['PATH']['MID']['MATCH_DATA']
DMS_DIR = inCfg['PATH']['OUT']['DMS']
LogPath = inCfg['PATH']['OUT']['LOG']
Log = LogServer(LogPath)

# 获取开机线程的个数，开启线程池。
threadNum = inCfg['CROND']['threads']
pool = Pool(processes=int(threadNum))

if len(args) == 2:
    Log.info(u'手动蝴蝶图绘制程序开始运行-----------------------------')
    satPair = args[0]
    str_time = args[1]
    date_s, date_e = pb_time.arg_str2date(str_time)
    # 定义参数List，传参给线程池
    args_List = []

    while date_s <= date_e:
        ymd = date_s.strftime('%Y%m%d')
        pool.apply_async(run, (satPair, ymd))
        date_s = date_s + relativedelta(days=1)

    pool.close()
    pool.join()


elif len(args) == 0:
    Log.info(u'自动蝴蝶图绘制程序开始运行 -----------------------------')
    rolldays = inCfg['CROND']['rolldays']
    pairLst = inCfg['PAIRS'].keys()
    # 定义参数List，传参给线程池
    args_List = []
    for satPair in pairLst:
        ProjMode1 = len(inCfg['PAIRS'][satPair]['colloc_exe'])
        if ProjMode1 == 0:
            continue
        for rdays in rolldays:
            ymd = (datetime.utcnow() - relativedelta(days=int(rdays))).strftime('%Y%m%d')
            pool.apply_async(run, (satPair, ymd))

    pool.close()
    pool.join()
else:
    print 'args error'
    sys.exit(-1)
