# coding: utf-8
'''
Created on 2016年1月6日
读取匹配后的NC文件，画蝴蝶图

@author: zhangtao
'''
import os, sys, netCDF4
from configobj import ConfigObj
import numpy as np
from dateutil.relativedelta import relativedelta
from PB import pb_time, pb_io
from PB.CSC.pb_csc_console import LogServer
from DM.SNO.dm_sno_cross_calc_map import *
from DM.SNO.dm_sno_cross_calc_core import Sat_Orbit
from multiprocessing import Pool , Lock
import matplotlib.pyplot as plt
from datetime import datetime

def run(pair, ymd):
    '''
    pair: sat1+sensor1_sat2+sensor2
    ymd: YYYYMMDD
    '''
    part1, part2 = pair.split('_')
    sat1, sensor1 = part1.split('+')
    sat2, sensor2 = part2.split('+')

#     # TODO: for now only support FY3
    if 'FY3' in part1:
        Type = "LEOLEO"
    else:
        return

    # load yaml config file
    plt_cfg_file = os.path.join(MainPath, '%s_%s.yaml' % (sensor1, sensor2))
    plt_cfg = pb_io.loadYamlCfg(plt_cfg_file)
    if plt_cfg is None:
        return
#     PERIOD = plt_cfg['days']
    PERIOD = 1

    Log.info(u"----- Start Drawing Matched Map-Pic, PAIR: {}, YMD: {}, PERIOD: " \
             u"{} -----".format(pair, ymd, PERIOD))

    chan = plt_cfg['chan'][0]

    NC = ReadMatchNC()
    num_file = PERIOD
    for daydelta in xrange(PERIOD):
        cur_ymd = pb_time.ymd_plus(ymd, -daydelta)
        filename = "COLLOC+%sIR,%s+%s_%s+%s_C_BABJ_%s.NC" % (Type, sat1, sensor1, sat2, sensor2, cur_ymd)
        filefullpath = os.path.join(MATCH_DIR, pair, filename)

        if not os.path.isfile(filefullpath):
            Log.info(u"File not found: {}".format(filefullpath))
            num_file -= 1
            continue

        if NC.LoadData(filefullpath, chan) == False:
            Log.error('Error occur when reading %s of %s' % (chan, filefullpath))

    if num_file == 0:
        Log.error(u"No file found.")
        return
    elif num_file != PERIOD:
        Log.error(u"{} of {} file(s) found.".format(num_file, PERIOD))

    cur_path = os.path.join(DMS_DIR, pair, ymd)

    o_file = os.path.join(cur_path,
                          '%s_%s_MatchedPoints_ALL_%s' % (part1, part2, ymd))

    # FY2F+VISSR-MetopA+IASI_MatchedPoints_ALL_20150226.png

    # find out day and night
    jd = NC.time / 24. / 3600.  # jday from 1993/01/01 00:00:00
    hour = ((jd - jd.astype('int8')) * 24).astype('int8')
    day_index = (hour < 10)  # utc hour<10 is day
    night_index = np.logical_not(day_index)

    x = NC.lon
    y = NC.lat
#     d = NC.days.astype('uint8')  # trans to int
#     if len(d) == 0:
#         Log.error('No days info in NC.')
#         return
    draw_butterfly(part1, part2, cur_ymd, ymd, x, y, o_file)

    # ------- day ----------
    if np.where(day_index)[0].size > 0 :
        o_file = os.path.join(cur_path,
                              '%s-%s_MatchedPoints_Day_%s' % (part1, part2, ymd))
        x_d = x[day_index]
        y_d = y[day_index]
#         d_d = d[day_index]
        draw_butterfly(part1, part2, cur_ymd, ymd, x_d, y_d, o_file)
    # ---------night ------------
    if np.where(night_index)[0].size > 0 :
        o_file = os.path.join(cur_path,
                              '%s-%s_MatchedPoints_Night_%s' % (part1, part2, ymd))
        x_n = x[night_index]
        y_n = y[night_index]
#         d_n = d[night_index]
        draw_butterfly(part1, part2, cur_ymd, ymd, x_n, y_n, o_file)
    Log.info(u"Success")

class ReadMatchNC():
    def __init__(self):
        self.lat = np.empty(shape=(0))  # lat
        self.lon = np.empty(shape=(0))  # lon
        self.time = np.empty(shape=(0))

    def LoadData(self, i_file, channel):
        '''
        i_file: fullpath of input NC file
        channel: channel name
        '''
        noError = True
        ncFile = netCDF4.Dataset(i_file, 'r', format='NETCDF4')
        if channel in ncFile.groups:
            chanGroup = ncFile.groups[channel]

            vkey = 'FyLat'
            if vkey in chanGroup.variables:
                dset = chanGroup.variables[vkey]
                if dset.ndim == 2:
                    self.lat = np.concatenate((self.lat, dset[:, 0]))
                else:
                    self.lat = np.concatenate((self.lat, dset))

            vkey = 'FyLon'
            if vkey in chanGroup.variables:
                dset = chanGroup.variables[vkey]
                if dset.ndim == 2:
                    self.lon = np.concatenate((self.lon, dset[:, 0]))
                else:
                    self.lon = np.concatenate((self.lon, dset))

            vkey = 'FyTime'
            if vkey in chanGroup.variables:
                dset = chanGroup.variables[vkey]
                if dset.ndim == 2:
                    self.time = np.concatenate((self.time, dset[:, 0]))
                else:
                    self.time = np.concatenate((self.time, dset))

        else:
            noError = False
        ncFile.close()
        return noError


def draw_butterfly(sat1Nm, sat2Nm, ymd_s, ymd_e, lons, lats, out_fig_file):
    '''
    画 FY3X 匹配蝴蝶图
    '''
#     COLORS = ['#4cd964', '#1abc9c', '#5ac8fa', '#007aff', '#5856d6']
    COLORS = [RED]
    fig = plt.figure(figsize=(8, 5), dpi=100)  # china
#     plt.subplots_adjust(left=0.11, right=0.91, bottom=0.12, top=0.92)
    plt.subplots_adjust(left=0.09, right=0.93, bottom=0.12, top=0.94)
    ax = subplot(121)
    m1 = drawFig_map(ax, "north")
    plot_matchpoint(m1, lons, lats, COLORS[0])

    ax = subplot(122)
    m2 = drawFig_map(ax, "south")
    plot_matchpoint(m2, lons, lats, COLORS[0])

    # ---------legend-----------
    circle1 = mpatches.Circle((58, 36), 6, color=RED, ec=EDGE_GRAY, lw=0.3)
#     circle_lst = [circle1]
#     for i in xrange(1):
#         circle_lst.append(mpatches.Circle((219 + i * 7, 36), 6, color=COLORS[i], ec=EDGE_GRAY, lw=0.3))
#     fig.patches.extend(circle_lst)

    TEXT_Y = 0.05
    fig.text(0.1, TEXT_Y, '%s' % sat1Nm, color=RED, fontproperties=FONT0)
#     fig.text(0.34, TEXT_Y, '%s' % sat2Nm, color=BLUE, fontproperties=FONT0)
    if ymd_s != ymd_e:
        fig.text(0.55, TEXT_Y, '%s-%s' % (ymd_s, ymd_e), fontproperties=FONT0)
    else:
        fig.text(0.55, TEXT_Y, '%s' % ymd_s, fontproperties=FONT0)
    fig.text(0.83, TEXT_Y, ORG_NAME, fontproperties=FONT0)

    # 设定Map边框粗细
    spines = ax.spines
    for eachspine in spines:
        spines[eachspine].set_linewidth(0)

    pb_io.make_sure_path_exists(os.path.dirname(out_fig_file))
    fig.savefig(out_fig_file, dpi=100)
    fig.clear()
    plt.close()

def drawFig_map(ax, n_s):
    '''
    create new figure
    '''
#     m = Basemap(llcrnrlon=50., llcrnrlat=-40., urcrnrlon=150., urcrnrlat=40., \
#         resolution='c', area_thresh=10000., projection='cyl', \
#         lat_ts=20.)

    if n_s == "north":
        m = Basemap(projection='npaeqd', boundinglat=59, lon_0=0, resolution='c')
    elif n_s == "south":
        m = Basemap(projection='spaeqd', boundinglat=-59, lon_0=180, resolution='c')

    m.fillcontinents(color=GRAY)

    # draw parallels
    m.drawparallels(np.arange(60., 91., 10.), linewidth=LINE_WIDTH, labels=[1, 0, 0, 1],
                    dashes=[100, .0001], color='white',
                    textcolor=EDGE_GRAY, fontproperties=TICKER_FONT)
    # draw meridians
    m.drawmeridians(np.arange(-180., 180., 30.), linewidth=LINE_WIDTH, labels=[1, 0, 0, 1],
                    dashes=[100, .0001], color='white', latmax=90,
                    textcolor=EDGE_GRAY, fontproperties=TICKER_FONT)
    # draw parallels
    m.drawparallels(np.arange(60., -91., -10.), linewidth=LINE_WIDTH, labels=[0, 0, 0, 0],
                    dashes=[100, .0001], color='white')
    # draw meridians
    m.drawmeridians(np.arange(-180., 180., 30.), linewidth=LINE_WIDTH, labels=[0, 0, 0, 0],
                    dashes=[100, .0001], color='white', latmax=90)

    if n_s == "north":
        lat_labels = [(0.0, 90.0), (0.0, 80.0), (0.0, 70.0), (0.0, 60.0)]
        for lon, lat in (lat_labels):
            xpt, ypt = m(lon, lat)
            ax.text(xpt - 500000, ypt - 100000, str(lat)[0:2] + u'°N', fontproperties=TICKER_FONT)

        ax.set_title("Northern Hemisphere", fontproperties=FONT0)
    elif n_s == "south":
        lat_labels = [(0.0, -90.0), (0.0, -80.0), (0.0, -70.0), (0.0, -60.0)]
        for lon, lat in (lat_labels):
            xpt, ypt = m(lon, lat)
            ax.text(xpt + 500000, ypt + 200000, str(lat)[1:3] + u'°S', fontproperties=TICKER_FONT)
        ax.set_title("Southern Hemisphere", fontproperties=FONT0)

    return m

def plot_matchpoint(m, lons, lats, color, alpha=1):
    # plot
    markersize = 1.8
    x, y = m(lons, lats)
    m.plot(x, y, marker='o', linewidth=0, markerfacecolor=color, markersize=markersize,
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
