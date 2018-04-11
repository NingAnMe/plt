# coding: utf-8
'''
Created on 2016年1月6日
读取匹配后的NC文件，画蝴蝶图

@author: zhangtao
'''
import os
import sys, netCDF4
import numpy as np
from configobj import ConfigObj
from dateutil.relativedelta import relativedelta
from PB import pb_time, pb_io
from PB.CSC.pb_csc_console import LogServer
from DM.SNO.dm_sno_cross_calc_map import *
from DM.SNO.dm_sno_cross_calc_core import Sat_Orbit
from multiprocessing import Pool , Lock
from datetime import datetime

def run(pair, ymd):
    '''
    pair: sat1+sensor1_sat2+sensor2
    ymd: YYYYMMDD
    '''
    part1, part2 = pair.split('_')
    sat1, sensor1 = part1.split('+')
    sat2, sensor2 = part2.split('+')

    # TODO: for now only support FY2 FY4
    if 'FY2' in part1 or 'FY4' in part1:
        pass
    else:
        return

    # load yaml config file
    plt_cfg_file = os.path.join(MainPath, '%s_%s.yaml' % (sensor1, sensor2))
    plt_cfg = pb_io.loadYamlCfg(plt_cfg_file)
    if plt_cfg is None:
        return
    PERIOD = plt_cfg['days']

    Log.info(u"----- Start Drawing Matched Map-Pic, PAIR: {}, YMD: {}, PERIOD: " \
             u"{} -----".format(pair, ymd, PERIOD))

    fy2_orbit = Sat_Orbit(inCfg['SAT_S2L'][sat1], ymd, ORBIT_DIR)
    if fy2_orbit.error:
        return
    fy2_orbit.get_orbit(ymd)
    if fy2_orbit.error:
        return
    fy2_lon = fy2_orbit.orbit['Lon'][0]
    fy2_lat = fy2_orbit.orbit['Lat'][0]

    if len(plt_cfg['chan']) > 3:
        chan = plt_cfg['chan'][2]  # FY2 IR3 has the most matchpoint
    else:
        chan = plt_cfg['chan'][0]

    NC = ReadMatchNC()
    num_file = PERIOD
    for daydelta in xrange(PERIOD):
        cur_ymd = pb_time.ymd_plus(ymd, -daydelta)
        filename = 'COLLOC+GEOLEOIR,%s_C_BABJ_%s.NC' % (pair, cur_ymd)
        filefullpath = os.path.join(MATCH_DIR, pair, filename)

        if not os.path.isfile(filefullpath):
            Log.info(u"File not found: {}".format(filefullpath))
            num_file -= 1
            continue

        if NC.LoadData(filefullpath, chan, PERIOD - daydelta) == False:
            Log.error('Error occur when reading %s of %s' % (chan, filefullpath))

    if num_file == 0:
        Log.error(u"No file found.")
        return
    elif num_file != PERIOD:
        Log.error(u"{} of {} file(s) found.".format(num_file, PERIOD))

    cur_path = os.path.join(DMS_DIR, pair, ymd)



    # FY2F+VISSR-MetopA+IASI_MatchedPoints_ALL_20150226.png

    # find out day and night
    jd = NC.time / 24. / 3600.  # jday from 1993/01/01 00:00:00
    hour = ((jd - jd.astype('int8')) * 24).astype('int8')
    day_index = (hour < 10)  # utc hour<10 is day
    night_index = np.logical_not(day_index)

    x = NC.lon
    y = NC.lat
    d = NC.days.astype('uint8')  # trans to int
    if len(d) == 0:
        Log.error('No days info in NC.')
        return

    # ------- day ----------
    if np.where(day_index)[0].size > 0 :
        o_file = os.path.join(cur_path,
                              '%s-%s_MatchedPoints_Day_%s' % (part1, part2, ymd))
        x_d = x[day_index]
        y_d = y[day_index]
        d_d = d[day_index]
        draw_butterfly(part1, part2, cur_ymd, ymd, fy2_lon, fy2_lat, x_d, y_d, d_d, o_file)
    # ---------night ------------
    if np.where(night_index)[0].size > 0 :
        o_file = os.path.join(cur_path,
                              '%s-%s_MatchedPoints_Night_%s' % (part1, part2, ymd))
        x_n = x[night_index]
        y_n = y[night_index]
        d_n = d[night_index]
        draw_butterfly(part1, part2, cur_ymd, ymd, fy2_lon, fy2_lat, x_n, y_n, d_n, o_file)

    # ------- ALL ----------
    if np.where(day_index)[0].size != 0 and np.where(night_index)[0].size != 0:
        o_file = os.path.join(cur_path,
                              '%s_%s_MatchedPoints_ALL_%s' % (part1, part2, ymd))
        draw_butterfly(part1, part2, cur_ymd, ymd, fy2_lon, fy2_lat, x, y, d, o_file)
    Log.info(u"Success")

class ReadMatchNC():
    def __init__(self):
        self.lat = np.empty(shape=(0))  # n天的lat
        self.lon = np.empty(shape=(0))  # n天的lon
        self.time = np.empty(shape=(0))
        self.days = np.empty(shape=(0))  # 第几天

    def LoadData(self, i_file, channel, dayNum):
        '''
        i_file: fullpath of input NC file
        channel: channel name
        dayNum: 5天中的第几天
        '''
        noError = True
        ncFile = netCDF4.Dataset(i_file, 'r', format='NETCDF4')
        if channel in ncFile.groups:
            chanGroup = ncFile.groups[channel]

            vkey = 'LeoLat'
            if vkey in chanGroup.variables:
                dset = chanGroup.variables[vkey]
                if dset.ndim == 1:
                    lat = dset[:]
                else:
                    lat = dset[:, 0]
                self.lat = np.concatenate((self.lat, lat))
                self.days = np.concatenate((self.days, np.zeros_like(lat) + dayNum))
            vkey = 'LeoLon'
            if vkey in chanGroup.variables:
                dset = chanGroup.variables[vkey]
                if dset.ndim == 1:
                    self.lon = np.concatenate((self.lon, dset[:]))
                else:
                    self.lon = np.concatenate((self.lon, dset[:, 0]))
            vkey = 'LeoTime'
            if vkey in chanGroup.variables:
                dset = chanGroup.variables[vkey]
                if dset.ndim == 1:
                    self.time = np.concatenate((self.time, dset[:]))
                else:
                    self.time = np.concatenate((self.time, dset[:, 0]))

        else:
            noError = False
        ncFile.close()
        return noError


def draw_butterfly(sat1Nm, sat2Nm, ymd_s, ymd_e, fy2_lon, fy2_lat, lons, lats, days, out_fig_file):
    '''
    画 FY2X 匹配蝴蝶图
    '''
    fig = plt.figure(figsize=(8, 6), dpi=100)  # china
    ax = subplot(111)
    plt.subplots_adjust(left=0.11, right=0.91, bottom=0.12, top=0.92)

    m = drawFig_map(fig)
    maxday = np.max(days)

    COLORS = ['#d65856', '#d4d656', '#4cd964', '#1abc9c', '#5ac8fa', '#007aff', '#5856d6']

    if len(COLORS) < maxday:
        Log.error('defined COLORS not enough.')
        return

    for i in xrange(1, maxday + 1):
        idx = days == i

        lons_1day = lons[idx]
        lats_1day = lats[idx]
        plot_matchpoint(m, lons_1day, lats_1day, COLORS[i - 1])

    x, y = m(fy2_lon, fy2_lat)
    m.plot(x, y, marker='o', linewidth=0, markerfacecolor=RED, markeredgecolor=EDGE_GRAY,
           markersize=8, mew=0.2)
    # ---------legend-----------
    circle1 = mpatches.Circle((58, 36), 6, color=RED, ec=EDGE_GRAY, lw=0.3)
    circle_lst = [circle1]
    for i in xrange(maxday):
        circle_lst.append(mpatches.Circle((219 + i * 7, 36), 6, color=COLORS[i], ec=EDGE_GRAY, lw=0.3))
    fig.patches.extend(circle_lst)

    TEXT_Y = 0.05
    fig.text(0.1, TEXT_Y, '%s' % sat1Nm, color=RED, fontproperties=FONT0)
    fig.text(0.34, TEXT_Y, '%s' % sat2Nm, color=BLUE, fontproperties=FONT0)
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

def drawFig_map(fig):
    '''
    create new figure
    '''
    m = Basemap(llcrnrlon=50., llcrnrlat=-40., urcrnrlon=150., urcrnrlat=40., \
        resolution='c', area_thresh=10000., projection='cyl', \
        lat_ts=20.)

    m.fillcontinents(color=GRAY)

    # draw parallels
    m.drawparallels(np.arange(90., -91., -30.), linewidth=LINE_WIDTH, labels=[1, 0, 0, 1],
                    dashes=[100, .0001], color='white',
                    textcolor=EDGE_GRAY, fontproperties=TICKER_FONT)
    # draw meridians
    m.drawmeridians(np.arange(-180., 180., 30.), linewidth=LINE_WIDTH, labels=[1, 0, 0, 1],
                    dashes=[100, .0001], color='white',
                    textcolor=EDGE_GRAY, fontproperties=TICKER_FONT)
    # draw parallels
    m.drawparallels(np.arange(60., -90., -30.), linewidth=LINE_WIDTH, labels=[0, 0, 0, 0],
                    dashes=[100, .0001], color='white')
    # draw meridians
    m.drawmeridians(np.arange(-150., 180., 30.), linewidth=LINE_WIDTH, labels=[0, 0, 0, 0],
                    dashes=[100, .0001], color='white')

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