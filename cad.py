# Useful functions to analyze a CAD data
# FDB 6/15/2017
#CONTAINS:
#
#get_units: find unique list of units from a cad file
#
#GIS_unit: get list of units from a GIS file, specifies type of units according to "rule" and replaces rule with "rep"
#
#splitter: return a subset of the cad data in the requested time interval(passed as string with format 'dd/mm/yyyy hh:mm:ss')
#
#to_mat: returns the dataframe in a matrix format and a dictionary for the columns
#
#timer: calculates time between events. Eg: pass Arrive and Enroute to get travel time; pass Dispatch and Arrive to get response time.
# Needs a matrix (faster) and a dictionary for columns: see to_mat function
#
#save: save to xlsx or csv as specified by "form". If passing an array needs also a dictionary
#
#resp_units: gets the cad in array form, returns number of responses per unit
#
#unixt: add columns to dataframe with specified columns converted to unix time
#
#datelist_1hr(yr): create datelist in 1 hr steps for year yr
#
#queues: queing formulas for Erlang M/M/c model
# takes service rate,n of units, calls rate and number of units in service desired (x)
# returns probability of all units busy, aver. length of queue, waiting times, unit utilization rate, aver number units busy
# probability of x units busy
#
#rates_q: takes dataframe and 1 year long datelist and returns average calls rate (hr-1) and service time(hr) for the time period between hour h1 and hour h2
#needs dispatch columns and service time columns
#
#queues_byhr: takes dataframe,1 year long datelist per hr and list of units, works with cad.queues to get M/M/c quantities per hr
# needs also x1, x2, to calculate probability of a number of units between x1 and x2 busy
# returns p. of all busy, aver. leng of queue, calls per hr, n of units busy per hr, prob of x1 to x2 units busy.
#
#seconds_hrs: replace columns in seconds with the same column in hrs
#
#count_units: count units that where in service in each hour of the year needs datelist with 1hr intervals
#
#b2b calculates back to back time, needs matrix and dictionary, dispatch and available columns, name for back to back column
#
#incident_type(dfm,dc,colname,coltype,chars_ef,chars_ab,sepi,rulesep=None): fill the new coltype column with code for ems als/bls and fire
#
#utilization: returns total time busy summed over all the units, calls received in that hr, all calls (including across) hr, unique incidents
#
#busy_at_response: returns number of same kind of units busy at the time of response
#
# count_overlaps
# produces a file where each column is a number of units in service at the same time and the cell is the number of seconds
# reformat_overlaps
# reoformat the file produced by count_overlaps to have only three columns: date, number_overlapping, seconds

import numpy as np
import pandas as pd
from numba import jit
import datetime
import math


def remove_duplicates(df,c1,c2):
    df['dup'] = [str(df[c1].ix[x]) + str(df[c2].ix[x]) for x in df.index]
    df = df.ix[df.dup.drop_duplicates().index]
    df = df.drop('dup',axis=1)
    return df

def get_units(df,uname):
    # find unique list of units from a cad file
    un_global = np.unique([x for x in list(df[uname])])
    return un_global

def GIS_unit(gpath,colapp,rule,rep,sheet=0):
    # get units from a GIS file, specifies type of units according to "rule" and replaces rule with "rep"
    desc = pd.read_excel(gpath,sheetname=sheet)
    base_u = list(desc[colapp])
    all_u = np.concatenate(pd.DataFrame(base_u).dropna().as_matrix())
    amb = []
    for x in range(len(all_u)):
        if len(all_u[x]) > 0:
            if all_u[x][0:len(rule)] == rule: amb.append(all_u[x])
    amb = list(map(lambda x: x.replace(rule, rep), amb))
    #amb = [amb[x] for x in range(len(amb))]
    return amb

def splitter(df,time1,time2,col):
    # splitter: return a subset of the cad data in the requested time interval(passed as string with format 'dd/mm/yyyy hh:mm:ss')
    dft = df[np.logical_and(df[col]<pd.to_datetime(time2),df[col]>pd.to_datetime(time1))]
    return dft

def to_mat(df_r):
    # returns the dataframe in a matrix format and a dictionary for the columns
    df_mat = df_r.as_matrix()
    c = 0
    # dictionary for columns
    dict = {}
    for x in range(np.shape(df_mat)[1]):
        dict.update({df_r.columns[x]: c})
        c = c + 1
    return df_mat,dict

def incident_type(dfm,dc,colname,coltype,chars_ef,chars_ab,sepi,rulesep=None):
    #split fire/ems als/ ems bls
    # dfm matrix of data; dc dictionary of cols; colname name of relevant columns; coltype: new column with types;
    # chars_ef index of relevant characters for ems/fire; chars_ab relevant characters for als/bls; sepi index of relevant element of separated list; rulesep if needed, separator
    # 1 is ems als 2 is ems bls 3 is fire 4 is other
    if np.logical_not(rulesep is None): ai = list(map(lambda x: x.split('-'),dfm[:,dc[colname]]))
    codes = [ai[x][0] for x in range(len(ai))]
    for i in range(len(codes)):
        if codes[i][chars_ef[0]:chars_ef[1]].isdigit():
            if (codes[i][chars_ef[0]:chars_ef[1]].isdigit())<51:
                if codes[i][chars_ab]=='A' or codes[i][chars_ab]=='B' or codes[i][chars_ab]=='O':
                    dfm[i,dc[coltype]] = 2
                else:
                    dfm[i, dc[coltype]] = 1
            else:
                dfm[i, dc[coltype]] = 3
        else:
            if codes[i][chars_ef[0]].isdigit():
                if codes[i][chars_ef[0]+1] == 'A' or codes[i][chars_ef[0]+1] == 'B' or codes[i][chars_ef[0]+1] == 'O':
                    dfm[i, dc[coltype]] = 2
                else:
                    dfm[i, dc[coltype]] = 1
            else:
                if codes[i][chars_ef[0]] == 'F':  dfm[i, dc[coltype]] = 3
    return dfm



@jit
def timer(arr,travcol,arrcol,enrcol,dict):
    # calculates time between events. Eg: pass Arrive and Enroute to get travel time; pass Dispatch and Arrive to get response time.
    # Needs a matrix (faster) and a dictionary for columns: see to_mat function
    if type(arr) != np.ndarray:
        print('Stop, convert your dataframe to matrix first. Use to_mat function provided in this module.')
    else:
        for x in range(len(arr)):
            arr[x,dict[travcol]] = (pd.to_datetime(arr[x,dict[arrcol]])-pd.to_datetime(arr[x,dict[enrcol]]))/np.timedelta64(1,'s')
        # set missing times to -1.
        iw = np.where(np.logical_or(arr[:,dict[arrcol]]=='00:00:00',arr[:,dict[enrcol]]=='00:00:00'))
        arr[iw,dict[travcol]]= -1.
    return arr


@jit
def first_arriving_s(arr,dict,orderc,arrivec,incc):
    arr[0, dict[orderc]] = 1
    for x in np.arange(1, len(arr), 1):
        if arr[x, dict[arrivec]] != '00:00:00':
            if arr[x, dict[incc]] == arr[x - 1, dict[incc]]:
                arr[x, dict[orderc]] = arr[x - 1, dict[orderc]] + 1
            else:
                arr[x, dict[orderc]] = 1
    return arr

@jit
def b2b(arr,dict,unit,sn,en,incid,bb):
    # calculates back to back time, needs matrix and dictionary, dispatch and available columns, name for back to back column
    for x in np.unique(arr[:,dict[unit]]):
        iw = np.where(arr[:,dict[unit]]==x)[0]
        for ind in range(len(iw)):
            if ind==0 or arr[iw[ind-1], dict[en]]==0:
                arr[iw[ind],dict[bb]] = -1.
            else:
                if (arr[iw[ind],dict[incid]])!=(arr[iw[ind-1],dict[incid]]) :arr[iw[ind], dict[bb]] =(arr[iw[ind], dict[sn]]-arr[iw[ind-1], dict[en]])
    return arr


def save(ob,nameo,form,dict=None):
# save to xlsx or csv as specified by form. If passing an array needs also a dictionary
    if type(ob) == np.ndarray and str(type(dict))!="<class 'dict'>":
        print('Need to pass a dictionary when passing an array.')
    else:
        if type(ob) == np.ndarray:
            ndfm = pd.DataFrame(ob, columns=list(dict.keys()))
        else:
            ndfm = ob
        if form == 'csv': ndfm.to_csv(nameo)
        if form == 'xlsx':
            writer = pd.ExcelWriter(nameo)
            ndfm.to_excel(writer, sheet_name='Sheet1')
            writer.save()
        if form!='xlsx' and form!='csv': print('format not supported')

def resp_units(arr,dict,colname,units):
    # gets the cad in array form, returns number of responses per unit
    count=[]
    for u in units:
        count.append([u,len(arr[np.where(arr[:,dict[colname]]==u)])])
    return count

def utilization(dfm,datelist,dict,namedisp,nameav,nameinc):
    dts = dfm[:, dict[namedisp]]
    ats = dfm[:, dict[nameav]]
    ub = np.zeros(len(datelist))
    utotc = np.zeros(len(datelist))
    uinc = np.zeros(len(datelist))
    busy_time = np.zeros(len(datelist))
    for x in range(len(datelist) - 1):
        iw = np.logical_and(dts < datelist[x + 1], dts > datelist[x])
        iw_s = np.logical_or(np.logical_and(dts < datelist[x], ats > datelist[x]),
                             np.logical_and(dts > datelist[x], dts < datelist[x + 1]))
        start_end = [[max(dts[y], datelist[x]), min(ats[y], datelist[x + 1])] for y in np.where(iw_s)[0]]
        if (len(start_end) > 0):
            startime, endtime = list(map(list, zip(*start_end)))
            startime = np.array(startime)
            endtime = np.array(endtime)
            # total time busy summed over all the units
            if type(endtime[0]-startime[0]) != np.float64: busy_time[x] = sum(np.array(list(map(datetime.timedelta.total_seconds, endtime - startime))))
            if type(endtime[0]-startime[0]) == np.float64: busy_time[x] = sum(endtime - startime)
            # calls received in that hr
            ub[x] = sum(iw)
            # all calls (including across) hr
            utotc[x] = sum(iw_s)
            # unique incidents
            uinc[x] = len(np.unique(dfm[iw_s, dict[nameinc]]))
    return ub, utotc, uinc, busy_time

def datelist_1hr(yr,d1,m1,d2,m2):
    # create datelist in 1 hr steps for year yr
    t1 = pd.datetime(yr,m1,d1)
    t2 = pd.datetime(yr,m2,d2)
    days = (t2-t1).days+1
    time_0 = pd.datetime(yr, m1, d1, 0, 0, 0)
    fr = 3600
    dl= pd.date_range(time_0, freq=str(fr) + 's', periods=int(days * 24 * 60 * 60 / fr)).tolist()
    return dl

def datelist_1hr_multiyears(yr1, yr2, d1, m1, d2, m2,part):
    # create datelist in 1 hr steps for year yr
    t1 = pd.datetime(int(yr1), int(m1), int(d1))
    t2 = pd.datetime(int(yr2), int(m2), int(d2))
    days = (t2 - t1).days + 1
    time_0 = pd.datetime(yr1, m1, d1, 0, 0, 0)
    fr = int(3600*part)
    dl = pd.date_range(time_0, freq=str(fr) + 's', periods=int(days * 24 * 60 * 60 / fr)).tolist()
    return dl

def unixt(df,cols):
    # add columns with columns in cols in unix time
    for j in cols:
        df[j+'_seconds']= pd.DatetimeIndex(df[j]).astype(np.int64)/1e9
    return df


# ###
#
# def unixt(df,cols):
# # add columns with columns in cols in unix time
#     for j in cols:
#         df[j+'_seconds']=0.
#         for x in df.index:
#             df[j+'_seconds'].ix[x]= df[j].ix[x].value/1e9
#     return df

@jit
def count_units(dfm,dict,datelist,disp,available,unit):
    # count units that where in service in each time frame of the year
    # needs datelist with 1hr or part of hour intervals
    if type(dfm) != np.ndarray: print('Stop, convert your dataframe to matrix first. Use to_mat function provided in this module.')
    # if (datelist[1]-datelist[0]).value/1e9 != 3600: print('WARNING: datelist not binned by hour!')
    ub = np.zeros(len(datelist))
    for x in range(len(datelist)-1):
        iw = np.logical_or(np.logical_and(dfm[:,dict[disp]]<datelist[x],dfm[:,dict[available]]>datelist[x]),np.logical_and(dfm[:,dict[disp]]>datelist[x],dfm[:,dict[disp]]<datelist[x+1]))
        ub[x] = len(np.unique(dfm[iw,dict[unit]]))
    return ub

@jit
def travelt_byhr(dfm,dict,datelist,disp,available,arrcol,enrcol):
    # get travel time by hr
    if type(dfm) != np.ndarray: print('Stop, convert your dataframe to matrix first. Use to_mat function provided in this module.')
    if (datelist[1]-datelist[0]).value/1e9 != 3600: print('WARNING: datelist not binned by hour!')
    tth = np.zeros(len(datelist))
    for x in range(len(datelist)-1):
        iw = np.logical_or(np.logical_and(dfm[:,dict[disp]]<datelist[x],dfm[:,dict[available]]>datelist[x]),np.logical_and(dfm[:,dict[disp]]>datelist[x],dfm[:,dict[disp]]<datelist[x+1]))
        if len(np.where(iw)[0]) > 0:
            tth[x] = np.mean(dfm[iw,dict[arrcol]]-dfm[iw,dict[enrcol]]).value/1e9
        else:
            tth[x] = 0.
    return tth

def time_overlaps(arr,datelist,dc,sn,en,inc,amb):
    o = []
    flag=[]
    ovll=[]
    for i in range(len(datelist)-1):
        bound_s = datelist[i].value/1e9
        bound_e = datelist[i+1].value/1e9
        alloc=[]
        for j in range(0,len(arr),1):
            if (arr[j,dc[sn]]<=bound_s and arr[j,dc[en]]>bound_s) or (arr[j,dc[sn]]>bound_s and arr[j,dc[sn]]<bound_e):
                alloc.append(j)
        if len(alloc)>0:
            mat = np.zeros([len(alloc),int((datelist[1]-datelist[0]).value/1e9)])
            for x in range(len(alloc)):
                if x<len(alloc)-1: y =x+1
                else: y=0
                if 1>0:#arr[alloc[x], dc[inc]]  !=  arr[alloc[y], dc[inc]]:
                    ind_s = int(max(arr[alloc[x],dc[sn]] - bound_s,0))
                    ind_e = int(min(arr[alloc[x],dc[en]] - bound_s,int((datelist[1]-datelist[0]).value/1e9)))
                    mat[x,ind_s:ind_e] = 1.
            ovl = np.array([np.sum(mat[:,x]) for x in range(len(mat[0,:]))])
            if ovl.max()>len(amb): print(i,'Maximum number of units exceeded, check your data'),flag.append(i)
            o.append([len(ovl[ovl==x]) for x in range(int(len(amb)+1))])
        else:
            o.append([0. for x in range(int(len(amb))+1)])
    return o,flag


# def count_overlaps(arr,dc,sn,en,incname,units):
# # count all the times when ALL the specified units are engaged at the same time
#     arr     = arr[np.argsort(arr[:,dc[sn]])]
#     clist   = []
#     partial = []
#     count   = 0
#     for j in range(0,len(arr)-2,1):
#         bound_e = arr[j,dc[en]]
#         i=j+1
#         overl_incs = []
#         # print(len(arr),i,arr[i,dc[sn]],arr[i,dc[incname]])
#         existing=[]
#         c=1
#         while arr[i,dc[sn]]<bound_e:
#             overl_incs.append(arr[i, dc[incname]])
#             if len(existing)>0:
#                 ex_count = [1 if (arr[i, dc[sn]]-arr[x,dc[en]]).value<0 else 0 for x in existing ]
#                 c = 2+sum(ex_count)
#             else:
#                 c = 2
#             existing.append(i)
#             i=i+1
#         if (c==len(units)):
#             count = count+1
#             clist.append([arr[j,dc[incname]],count,overl_incs])
#         elif c>0:
#             partial.append([c,arr[j,dc[incname]],overl_incs])
#     return clist,partial

def count_overlaps(arr,dc,sn,en,incname,units):
# count all the times when ALL the specified units are engaged at the same time
    arr     = arr[np.argsort(arr[:,dc[sn]])]
    clist   = []
    partial = []
    count   = 0
    for j in range(0,len(arr)-2,1):
        bound_e = arr[j,dc[en]]
        i=j+1
        overl_incs = []
        # print(len(arr),i,arr[i,dc[sn]],arr[i,dc[incname]])
        existing=[]
        c=1
        while arr[i,dc[sn]]<bound_e and i<len(arr)-1:
            if len(existing)>0:
                ex_count = [1 if (arr[i, dc[sn]]-arr[x,dc[en]]).value<0 else 0 for x in existing ]
                c = 2+sum(ex_count)
                overl_incs.append(arr[i, dc[incname]])
            else:
                c = 2
                overl_incs.append(arr[i, dc[incname]])
            existing.append(i)
            i=i+1
        if (c==len(units)):
            count = count+1
            clist.append([arr[j,dc[incname]],count,overl_incs])
        elif c>0:
            partial.append([c,arr[j,dc[incname]],overl_incs])
    return clist,partial

# c,p= count_overlaps(arr,dc,sn,en,incname,units)

def busy_at_response(arr,dc,disp,clear,inc,unit,arv,enr):
    inserv = []
# return number of same kind of units busy at the time of response
    for i in range(1,len(arr),1):
        c = 0
        for j in range(max(0,i-500),i):
            if (arr[j,dc[disp]]<arr[i,dc[disp]]) and (arr[j,dc[clear]]>arr[i,dc[disp]]):
                c = c + 1
        inserv.append([arr[i,dc[inc]],arr[i,dc[disp]],arr[i,dc[unit]],c,(arr[i,dc[arv]]-arr[i,dc[enr]])/np.timedelta64(1,'s')])
    return inserv

def rates_q(dfa,datelist,h1,h2,sn,Serv_Time):
    # takes 1 year datelist and returns average calls rate (hr-1) and service time(hr)
    # for the time period between hour h1 and hour h2, needs dispatch columns and service time columns
    days = 365
    cday   = np.zeros(days)
    serday = np.zeros(days)
    for i in range(days):
        s1 = datelist[h1+i*24].value/1e9
        s2 = datelist[h2+i*24].value/1e9
        # calls per day in that time slot
        cday[i] = len(dfa[np.logical_and(dfa[sn]>s1,dfa[sn] <s2)])
        # average serv time per call in that time slot
        serday[i] = dfa[np.logical_and(dfa[sn]>s1,dfa[sn] <s2)][Serv_Time].sum()/cday[i]
# aver per hr
    c_rhr =  cday.mean()/(h2-h1)
    s_rhr =  serday.mean()/60
    return c_rhr,s_rhr

def queues(srate,nunits,callsrate,x):
    # queing formulas for Erlang M/M/c model
    # takes service rate,n of units, calls rate and number of units in service desired (x)
    # returns probability of all units busy, aver. length of queue, waiting times, unit utilization rate, aver number units busy
    # probability of x units busy
    # service rate in hr-1
    s_rate =  srate
    # unis
    nn = nunits
    cc = callsrate
    r = cc/nn/s_rate
    # all units busy
    kk = np.arange(0,100,1)
    kf = np.array(list(map(math.factorial,kk)))
    #http://web.mst.edu/~gosavia/queuing_formulas.pdf
    #http://www.cs.wayne.edu/~hzhang/courses/7290/Lectures/15%20-%20M-M-Star%20Queues.pdf
    p0 = 1/(np.sum((r*nn)**kk[0:int(nn)]/kf[0:int(nn)])+ nn*r/math.factorial(nn)/(1-r))
    pw = p0*(nn*r)**nn/math.factorial(nn)/(1-r)
    # p of x custom in system
    px = (nn*r)**x/np.array(list(map(math.factorial,x)))*p0
    lq = pw*r/(1.-r)
    wt = pw*r/cc/(1.-r)
    return pw,lq,wt,r,pw*r/(1.-r)+nn*r,px

def queues_byhr(dfa,datelist,sn,Serv_time,nambulances,x1,x2):
# takes dataframe,1 year long datelist per hr and list of units, works with cad.queues to get M/M/c quantities per hr
# needs also x1, x2, to calculate probability of a number of units between x1 and x2 busy
# returns p. of all busy, aver. leng of queue, calls per hr, n of units busy per hr, prob of x1 to x2 units busy
    #if len(datelist)!=8760: print('WARNING: your datelist seems to be shorter than 1 year'),len(datelist)
    days = int(len(datelist)/24)
    cday = np.zeros(days)
    serday = np.zeros(days)
    prob_a = []
    pxx_a = []
    cperh = []
    insysh = []
    queued = []
    sperh = []
    for t in range(24):
        for i in range(days-1):
            tt = t + 1
            s1 = datelist[t + i * 24].value / 1e9
            s2 = datelist[tt + i * 24].value / 1e9
            # calls per day in that time slot
            cday[i] = len(dfa[np.logical_and(dfa[sn] > s1, dfa[sn] < s2)])
            # average serv time per call in that time slot
            serday[i] = dfa[np.logical_and(dfa[sn] > s1, dfa[sn] < s2)][Serv_time].sum() / cday[i]
            # aver per hr
        serdnan = serday[np.logical_not(np.isnan(serday))]
        c_rhr = cday.mean()
        s_rhr = serdnan.mean() / 60
        #u_rh = c_rhr * s_rhr / len(ambulances) / 60
        probw, lengq, waitt, rho, insys, pxx = queues(60. / s_rhr, nambulances, c_rhr, np.arange(x1,x2))
        cperh.append(c_rhr)
        sperh.append(s_rhr)
        insysh.append(insys)
        prob_a.append(probw)
        queued.append(lengq)
        pxx_a.append(pxx.sum())
    return prob_a,queued,cperh,sperh,insysh,pxx_a

def seconds_hrs(filein,fileout,cols):
    # replace columns in seconds with the same column in hrs
    df = pd.read_csv(filein)
    df[df.columns[cols]] = df[df.columns[cols]] / 60. / 60.
    df.to_csv(fileout)

def merger(names,nameout=None):
    # merge cad files, eg from different years
    form = names[0].split('.')[-1]
    if form == 'xlsx':
        fr = pd.read_excel
        this = fr(names[0])
    if form == 'csv':
        fr  = pd.read_csv
        this = fr(names[0],error_bad_lines=False,encoding='iso-8859-1')
    for i in names[1::]:
        if form =='csv': u = fr(i,error_bad_lines=False,encoding='iso-8859-1')
        if form =='xlsx': u = fr(i)
        uc = pd.concat([this,u], ignore_index=True)
        this = uc
    if nameout: uc.to_csv(nameout)
    return uc

def reformat_overlaps(dfm, dc, datecol):
    co = [x for x in list(dc.keys()) if x != datecol]
    l = []
    for d in range(len(dfm)):  # dfm[:,dc[datecol]][0:100]:
        for n in co:
            l.append([dfm[d, dc[datecol]], n, dfm[d, dc[n]]])
    dfl = pd.DataFrame.from_records(l, columns=['date', 'number_overlapping', 'seconds'])
    return dfl


        # @jit
# def utilization_2(dfm,yr,dict,namedisp,nameav,nameinc):
#     time_0 = pd.datetime(yr, 1, 1, 0, 0, 0)
#     fr = 3600
#     datelist = pd.date_range(time_0, freq=str(fr) + 's', periods=int(365 * 24 * 60 * 60 / fr)).tolist()
#     dts = dfm[:, dict[namedisp]]
#     ats = dfm[:, dict[nameav]]
#     disp_sort = np.argsort(dfm[:, dict[namedisp]])
#     dfm = dfm[disp_sort]
#     ub = np.zeros(len(datelist))
#     utotc = np.zeros(len(datelist))
#     uinc = np.zeros(len(datelist))
#     iw = np.zeros(len(dfm))
#     iw_s = np.zeros(len(dfm))
#     busy_time = np.zeros(len(datelist))
#     for x in range(len(datelist) - 1):
#         start_end = [[0, 0]]
#         for y in range(len(dfm)):
#             if dts[y]<datelist[x+1]:
#                 if dts[y] > datelist[x]: iw[y]=1
#                 if (dts[y] < datelist[x] and ats[y] > datelist[x]) or (dts[y] > datelist[x] and dts[y] < datelist[x + 1]): iw_s[y]=1
#         for y in np.where(iw_s)[0]:
#             if dts[y] >  datelist[x]:
#                 ss = dts[y]
#             else:
#                 ss = datelist[x]
#             if ats[y] > datelist[x+1]:
#                 ee = ats[y]
#             else:
#                 ee = datelist[x+1]
#             start_end = start_end + [[ss,ee]]
#         start_end.remove([0,0])
#     #     if (len(start_end) > 0):
#     #         startime, endtime = list(map(list, zip(*start_end)))
#     #         startime = np.array(startime)
#     #         endtime = np.array(endtime)
#     #         # total time busy summed over all the units
#     #         busy_time[x] = sum(np.array(list(map(datetime.timedelta.total_seconds, endtime - startime))))
#     #         # calls received in that hr
#     #         ub[x] = sum(iw)
#     #         # all calls (including across) hr
#     #         utotc[x] = sum(iw_s)
#     #         # unique incidents
#     #         uinc[x] = len(np.unique(dfm[np.where(iw_s)[0], dict[nameinc]]))
#     return datelist, ub, utotc, uinc, busy_time