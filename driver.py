import pandas as pd
import numpy as np
import cad
import configparser
import warnings
import sys
import traceback
import time
warnings.filterwarnings("ignore")


def travel_time(df,atscene,enroute,timeformat=None):
    ### split ems a/ems b /fire
    # df['travel'] = 0.
    # dfm,dc = cad.to_mat(df)
    dfmt = cad.timer(df, 'travel', atscene, enroute,timeformat)
    # dataf_tr = pd.DataFrame(dfmt,columns=df.columns)
    return dfmt

def response_time(df,atscene,calltime,timeformat=None):
    ### split ems a/ems b /fire
    dfmt = cad.timer(df, 'response', atscene, calltime, timeformat)
    return dfmt

def service_time(df,atscene,calltime,timeformat=None):
    dfmt = cad.timer(df, 'service', atscene, calltime, timeformat)
    return dfmt

def turnout_time(df,atscene,calltime,timeformat=None):
    dfmt = cad.timer(df, 'turnout', atscene, calltime, timeformat)
    return dfmt

# back to back time
def bb(dataframe,disp,cleared,unitname,iid):
    dataframe[disp] = pd.to_datetime(dataframe[disp])
    dataframe[cleared] = pd.to_datetime(dataframe[cleared])
    dsecs = cad.unixt(dataframe,[disp,cleared])
    dsecs['bb_time']=0.
   # dsecs['Serv_Time']=0.
    dsecs = dsecs.sort_values([disp,cleared])
    #dsecs = dsecs.ix[dsecs[disp].dropna().index]
    #dsecs = dsecs.ix[dsecs[cleared].dropna().index]
    dm,dc = cad.to_mat(dsecs)
    arr = cad.b2b(dm,dc,unitname,disp+'_seconds',cleared+'_seconds',iid,'bb_time')
    bbdf = pd.DataFrame(arr,columns=dsecs.columns)
    return bbdf

def arrive_order(df,atscene,id):
    df['order'] = 0.
    dfs = df.sort_values([id, atscene])
    dfm,dc=cad.to_mat(dfs)
#     Here split -> select fires only and engines only and run the order function
#     Here split 2->select ems and check order for all units
    dfor = cad.first_arriving_s(dfm,dc,'order',atscene,id)
    dataf_or = pd.DataFrame(dfor, columns=df.columns)
    return dataf_or

#['SelectStaffedAp', 'TravelTime', 'ArrivingOrder', 'BackToBackTimeA','DemandAnalysisE', 'DemandAnalysisL', 'DemandAnalysisM']
def gis(gisf,gisapp,giseng,giseng_cad,gislad,gislad_cad,gisamb,gisamb_cad,gisot,gisot_cad):
    print(' ')
    print('**************')
    print()
    print('Reading Units from file:',gisf)
    d = {}
    if (len(giseng)>0):
        ueng = cad.GIS_unit(gisf, gisapp, giseng, giseng_cad)
        d.update({giseng:np.array(ueng)})
    if (len(gislad) > 0):
        utr = cad.GIS_unit(gisf, gisapp, gislad, gislad_cad)
        d.update({gislad:np.array(utr)})
    if (len(gisamb) > 0):
        uresc = cad.GIS_unit(gisf, gisapp, gisamb, gisamb_cad)
        d.update({gisamb: np.array(uresc)})
    if (len(gisot) > 0):
        uot = cad.GIS_unit(gisf, gisapp, gisot, gisot_cad)
        d.update({'Other': np.array(uot)})
    udf = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
    udf.to_csv('Units.csv',index=False)
    print('Done.')
    print(' ')

def main_reader(cf):
    print(cf)
    config = configparser.ConfigParser()
    config.read(cf)
    datafile = config.get("myvars", "datafile")
    iid = config.get("myvars", "IncidentID")
    staffname = config.get("myvars", "Staff")
    atscene = config.get("myvars", "atscene")
    enroute = config.get("myvars", "enroute")
    calltime = config.get("myvars", "call")
    disp = config.get("myvars", 'dispatch')
    cleared = config.get("myvars", 'cleared')
    format_date = config.get("myvars", 'format_date')
    unitname = config.get("myvars", 'units_columns')
    firstd = config.get("myvars", 'FirstDue_column')
    appstat = config.get("myvars", 'ApparatusStation_column')
    descriptor_c = config.get("myvars", 'Type_column')
    descriptor = config.get("myvars", 'Description')
    max_lines = config.get("myvars", 'max_lines')
    part = float(config.get("myvars", 'part_hour'))
    data = datafile.split(',')
    max_lines = max_lines.strip(' ')
    if max_lines:
        max_lines = int(max_lines)
    else:
        max_lines = None
    return iid,staffname,atscene,enroute,calltime,disp,cleared,format_date,unitname,firstd,appstat,descriptor_c,descriptor,max_lines,part,data

# old version
# def unroller(df1,incnum,unitid,rolledc,rolledatt):
# # create temporaty id (unit ID + incident number)
#     dfn = df1.copy()
#     dfn['inc_temp'] = dfn[incnum] + dfn[unitid]
#     # print(dfn.head(15).inc_temp.ix[5],dfn.head(15).inc_temp.ix[6])
#     # print(rolledc)
#     rc = list(dfn[rolledc].unique())#['DISP','RESP','ONLOC','CLEAR']#list(dfn[rolledc].unique())
#     ls = '_'+rc[0]
#     # df1resp = dfn[dfn[rolledc]== rc[0]]
#     for r in rc[1::]:
#         df1onloc = dfn[dfn[rolledc] == r]
#         # print('************',r)
#         # print('A---->',df1resp.head(5))
#         # print('B---->',r,df1onloc.head(5))
#         # print(dfn.head(15))
#         # print(df1onloc.head(15))
#         dfn = dfn.set_index('inc_temp').join(df1onloc[['inc_temp', rolledc, rolledatt]].set_index('inc_temp'),
#                                               lsuffix='', rsuffix= '_'+r, how='left')
#         dfn =dfn.reset_index()
#         ls = '_'+r
#         # print('step')
#         # print(dfn.head(15))
#         # print('********')
#     dfn = dfn.drop('inc_temp',axis=1)
#     ckeeps = [c for c in dfn if not c.startswith(rolledc)]
#     return dfn[ckeeps]


def unroller(dfn,incnum,unitid,rolledc,rolledatt):
# create temporaty id (unit ID + incident number)
    dfn['inc_temp'] = dfn[incnum] + dfn[unitid]
    rc = list(dfn[rolledc].unique())
    dfred = dfn[['inc_temp',rolledc,rolledatt]]
    dfcomp = dfn.drop([rolledc,rolledatt],axis=1)
    dfcomp = dfcomp.groupby('inc_temp').first().reset_index()
    a = dfred[dfred[rolledc] ==rc[0]].reset_index()
    for r in rc[1::]:
        b = dfred[dfred[rolledc] == r].reset_index()
        a = a.set_index('inc_temp').join(b[['inc_temp', rolledc, rolledatt]].set_index('inc_temp'), lsuffix=rc[0],rsuffix=r, how='outer')
        a = a.reset_index()
        if r != rc[0]: a = a.rename(columns={rolledatt: rolledatt + r})
    a = a.set_index('inc_temp').join(dfcomp.set_index('inc_temp'),lsuffix=' ', rsuffix=' ', how='inner')
    a = a.reset_index()
    cr = [x for x in a.columns if x.startswith(rolledc)]
    try:
        remcol = cr + ['inc_temp','index']
        a = a.drop(remcol,axis=1)
    except:
        try : a.drop(['inc_temp','index'])
        except:
            pass
    return a



def preprocessing(cf,pref,delet_dup,tkv=None,tkvr=None):
    iid, staffname, atscene, enroute, calltime, disp, cleared, format_date,unitname, firstd, appstat, descriptor_c, descriptor, max_lines, part, data = main_reader(cf)
    filenumber = 0
    for d in data:
        print(' ')
        print('Reading file', d)
        if max_lines: print('   ',max_lines,' lines.')
        print('--------------------------------------- ')
        filenumber = filenumber + 1
        if d.split('.')[1] == 'csv': df = pd.read_csv(d, encoding='iso-8859-1', warn_bad_lines=False,error_bad_lines=False,converters={iid:str,unitname:str},nrows = max_lines)
        if (d.split('.')[1] == 'xls') or (d.split('.')[1] == 'xlsx'): df = pd.read_excel(d,encoding='iso-8859-1',converters={iid:str,unitname:str},nrows= max_lines)  # error_bad_lines=False,
        df[unitname] = [str(x) for x in df[unitname]]
        df[unitname] = [x.strip() for x in df[unitname]]
        if tkv and tkvr:
            print('Reshaping...')
            df = unroller(df,iid,unitname,tkv,tkvr)
            print('Done.')
        if delet_dup:
            print(' ')
            print('Deleting Duplicates...')
            df = cad.remove_duplicates(df, iid, unitname)
            print('Done.')
        print('Writing output')
        df.to_csv(pref + '_preprocess.csv', index=False)
        print('Done')


def run(s,cf,pref,gis,gisf,engine,ladder,rescue,other,overtime,overcount,erf,chief,enforce,erf_out):
    # get options
    ###
    print('***********************')
    print('Starting CAD analyzer. Version 3.0 August 2019')
    print('Contact fdebernardis@iaff.org for questions')
    #check version and license for necessary update
    this = time.time()
    expdate = pd.to_datetime('2/2/2020 9:00:00')
    if this < expdate.value/1e9:
        pass
    else:
        print('\n Invalid start. Update needed, contact developer.\n ')
        print('\n   .Terminated.    ')
        sys.exit(0)
    print('***********************')
    print('')
    ###
    try:
        iid, staffname, atscene, enroute, calltime, disp, cleared, format_date, unitname, firstd, appstat, descriptor_c, descriptor, max_lines, part, data = main_reader(cf)
        if len(s)>0 or erf:
            opt = [x[0:15].strip() for x in s]
            # print(opt)
            opt3 = ['TravelTime', 'ArrivingOrder', 'BackToBackTimeA','CoverIncidentsA']
            co= 0.
            for o in opt3: co = co + (o in opt)
            # get data file and config variables
        # now run
            filenumber = 0
            ueng = utr = uresc = uother = None
            gisf =gisf.strip(' ')
            if len(gisf)>0:
                if gisf.split('.')[1] == 'csv': units_df = pd.read_csv(gisf)
                if (gisf.split('.')[1] == 'xls') or (gisf.split('.')[1] == 'xlsx'): units_df = pd.read_excel(gisf)
                units_df.columns = [x.strip() for x in units_df.columns]
                cols = units_df.columns
                uall=[]
                if 'Engine' in cols and (engine or gis==0):
                    ueng = list(units_df['Engine'].dropna())
                    uall = uall + ueng
                if 'Truck' in cols and (ladder or gis==0):
                    utr = list(units_df['Truck'].dropna())
                    uall = uall + utr
                if 'Rescue' in cols and (rescue or gis==0):
                    uresc = list(units_df['Rescue'].dropna())
                    uall = uall + uresc
                if 'Other' in cols and (other or gis==0):
                    uother = list(units_df['Other'].dropna())
                    uall = uall + uother
            for d in data:
                print(' ')
                print('Reading file',d)
                print('--------------------------------------- ')
                filenumber = filenumber+1
                if d.split('.')[1]=='csv' : dfo = pd.read_csv(d, encoding='iso-8859-1',warn_bad_lines=False,error_bad_lines=False,nrows=max_lines)
                if (d.split('.')[1] == 'xls') or (d.split('.')[1] =='xlsx'): dfo = pd.read_excel(d, encoding='iso-8859-1',nrows=max_lines)#error_bad_lines=False,
                dfo[unitname] = [str(x) for x in dfo[unitname]]
                dfo[unitname] = [x.strip() for x in dfo[unitname]]
                if max_lines:# and max_lines.isdigit():
                    print('     Using only first',max_lines,' lines.\n')
                    # max_lines  = int(max_lines)
                    # dfo = dfo.head(max_lines)
                df  =dfo.copy()
                if gis == 1:
                    print(' ')
                    print('Selecting Staffed Units only...')
                    # reduce data to staffed units only
                    uall = [str(x) for x in uall]
                    df  = df[df[unitname].isin(uall)]
                    if 'SelectStaffedAp' in opt: df.to_csv(pref+'_CAD_staffed_only_'+'file_'+str(filenumber)+'.csv',index=False)
                if 'SelectStaffedAp' in opt and gis==0: print('You have to select "Yes" above to select staffed units.')
                checker = 0
                if 'TravelTime' in opt:
                    checker = checker + 1
                    print('Calculating Travel Time...')
                    df = travel_time(df,atscene,enroute,format_date)
                    # if checker == co: df.to_csv('testout.csv')
                    print('Done.')
                if 'Response Time (' in opt:
                    checker = checker + 1
                    print('Calculating Response Time...')
                    df = response_time(df,atscene,calltime,format_date)
                    # if checker == co: df.to_csv('testout.csv')
                    print('Done.')
                if 'Service Time (D' in opt:
                    checker = checker + 1
                    print('Calculating Service Time...')
                    df = service_time(df,cleared,disp,format_date)
                    # if checker == co: df.to_csv('testout.csv')
                    print('Done.')
                if 'Turnout Time' in opt:
                    checker = checker + 1
                    print('Calculating Turnout Time...')
                    df = turnout_time(df,enroute,disp,format_date)
                    # if checker == co: df.to_csv('testout.csv')
                    print('Done.')
                if 'ArrivingOrder' in opt:
                    checker = checker + 1
                    print('Calculating Arriving Order...')
                    df = arrive_order(df,atscene,iid)
                    # if checker == co: df.to_csv('testout.csv')
                    print('Done.')
                if 'BackToBackTimeA' in opt:
                    checker = checker + 1
                    print('Calculating Back to Back runs...')
                    df = bb(df,disp,cleared,unitname,iid)
                    df = df.drop([disp+'_seconds',cleared+'_seconds'],axis=1)
                    # if checker == co: df.to_csv('testout.csv')
                    print('Done.')
                if 'CoverIncidentsA' in opt:
                    checker = checker + 1
                    coverf = 'cover_flag'
                    locfd = 'location_first_due'
                    df[coverf] = 'N'
                    df[locfd] = -1.
                    print('Running Cover Incidents Analysis...')
                    df = df.sort_values([iid])
                    df[unitname]= [x.strip() for x in df[unitname]]
                    df = df[df[unitname].isin(ueng+utr)]
                    df = df.ix[df[cleared].dropna().index]
                    df = df.ix[df[disp].dropna().index]
                    df[disp] = pd.to_datetime(df[disp])
                    df[cleared] = pd.to_datetime(df[cleared])
                    dsel = df[df[unitname].isin(ueng+utr+uresc)]
                    arr, dc = cad.to_mat(dsel)
                    for i in range(len(arr)):
                        inc = arr[i,dc[iid]]
                    # check if first_due apparatus is on scene
                        if (arr[arr[:, dc[iid]] == inc, dc[firstd]] == arr[arr[:, dc[iid]] == inc, dc[appstat]]).sum()==0:
                    # it's not
                            timers = arr[arr[:, dc[iid]] == inc, dc[disp]].min()
                    # was the first due apparatus busy?
                            itime = np.where(np.logical_and(arr[:,dc[disp]]<=timers,arr[:,dc[cleared]]>=timers))[0]
                            otherinc = itime[arr[itime, dc[iid]] != inc]
                            icheck  = (arr[otherinc,dc[appstat]] ==  arr[arr[:, dc[iid]] == inc, dc[firstd]][0])
                            if icheck.sum()>0 :
                            #it was
                                arr[arr[:, dc[iid]] == inc, dc[coverf]] = 'Y'
                                arr[arr[:, dc[iid]] == inc, dc[locfd]] = arr[otherinc[np.where(icheck)[0]],dc[firstd]][0]
                    df = pd.DataFrame(arr,columns=dc.keys())
                    # dfcover.to_csv(pref + '_cover_incidents_'+str(filenumber)+'.csv',index=False)
                if checker!=0: df.to_csv(pref + '_CAD_edited_' + 'file_' + str(filenumber) + '.csv',index=False)
                df[disp] = pd.to_datetime(df[disp])
                df[cleared] = pd.to_datetime(df[cleared])
                df = cad.unixt(df,[disp,cleared])
                # df.to_c
                # stop
                if df[disp].max().year-df[disp].min().year> 10: print("Looks like you're using more than 10 years of data. If this is not correct check your time columns. This might slow down other functions.")
                dl = cad.datelist_1hr_multiyears(df[disp].min().year,df[disp].max().year,df[disp].min().day,df[disp].min().month,df[disp].max().day,df[disp].max().month,part)
                dlut = list(map(lambda x: x.value/1e9,dl))
                if ueng!=None: ueng = [str(x) for x in ueng]
                if utr !=None: utr = [str(x) for x in utr]
                if uresc!=None: uresc = [str(x) for x in uresc]
                if uother!=None: uother = [str(x) for x in uother]
                lu = [x for x in [ueng,utr,uresc,uother]]
                choice_d = [0,0,0,0]
                if 'DemandAnalysisE' in opt: choice_d[0]=1
                if 'DemandAnalysisT' in opt: choice_d[1]=1
                if 'DemandAnalysisR' in opt: choice_d[2]=1
                if 'DemandAnalysisO' in opt: choice_d[3]=1
                didIrun=0
    # Demand Analysis
                if sum(choice_d)!=0:
                    print('')
                    print('Running demand analysis...')
                    for ui in range(len(lu)):
                         u=lu[ui]
                         if choice_d[ui]==1 and u!=None:
                             didIrun = 100
                             if ui<3:name = u[0][0]
                             if ui==3: name = 'for Other units '
                             print(' ')
                             print('Unit:   '+name   )
                             dsel = df[df[unitname].isin(u)]
                             dsel = dsel.ix[dsel[disp].dropna().index]
                             dsel = dsel.ix[dsel[cleared].dropna().index]
                             dfm,dict = cad.to_mat(dsel)
                             print('       Computing time demand. Starting from year: ',dsel[disp].min().year)
                             chr, call, uninc, bt = cad.utilization_v2(dsel, dl, disp,cleared, iid)
                             #chr,call, uninc, bt = cad.utilization(dfm,dlut,dict,disp+'_seconds',cleared+'_seconds',iid)
                             dsel[disp] = pd.to_datetime(dsel[disp])
                             dsel[cleared]= pd.to_datetime(dsel[cleared])
                             dfm, dict = cad.to_mat(dsel)
                             print('       Computing units in service...')
                             cu,sttot = cad.count_units(dfm,dict,dl,disp,cleared,unitname,staffname)
                             dataf = {'date':dl,'calls_perhour':chr,'unique_incidents':uninc,'total_time_engaged':bt,'total_units_engaged':cu,'total staff':sttot}
                             ndf = pd.DataFrame(dataf,columns=['date','calls_perhour','unique_incidents','total_time_engaged','total_units_engaged','total staff'])
                             outname=pref + '_' + name +'_demand_time_unitcounts_file_'+str(filenumber)+'.csv'
                             ndf.to_csv(outname,index=False)
                             print('Done\n')
                    if didIrun!=100: print('Nothing to be done with current units selection. Possible causes:'
                                           '\n  * You did not select any action. '
                                           '\n  * You did not provide a units file.'
                                           '\n  * The columns in the units file have wrong names. It should be: Engines, Rescue, Truck, Other.')
    #Overlap Analysis
                choice_o = [0, 0, 0, 0]
                if 'OverlapsEngines' in opt: choice_o[0] = 1
                if 'OverlapsTrucks(' in opt: choice_o[1] = 1
                if 'OverlapsRescues' in opt: choice_o[2] = 1
                if 'OverlapsOther(s' in opt: choice_o[3] = 1
                didIruno = 0
                # Demand Analysis
                if sum(choice_o) != 0:
                    print('')
                    print('Running overlap analysis...')
                    for ui in range(len(lu)):
                         u=lu[ui]
                         if choice_o[ui]==1 and u!=None:
                             didIruno = 100
                             if ui < 3: name = u[0][0]
                             if ui==3: name = 'for Other units '
                             print(' ')
                             print('Unit:   '+name)
                             dsel = df[df[unitname].isin(u)]
                             # dn = cad.unixt(dsel,[disp,cleared])
                             dsel[disp]= pd.to_datetime(dsel[disp])
                             dsel[cleared] = pd.to_datetime(dsel[cleared])
                             dsel = cad.unixt(dsel,[disp,cleared])
                             dsel = dsel.ix[dsel[disp].dropna().index]
                             dsel = dsel.ix[dsel[cleared].dropna().index]
                             # dsel = dsel[np.logical_and(dsel[disp]>pd.to_datetime(yr+'/1/1 00:00:00'),dsel[disp]<pd.to_datetime(yr+'/12/31 23:59:59'))]
                             dsel['check']=[str(dsel[iid].ix[i])+str(dsel[unitname].ix[i])+str(dsel[disp].ix[i]) for i in dsel.index]
                             dsel = dsel.drop_duplicates('check')
                             dfm,dc = cad.to_mat(dsel)
        #     #update here units
                             if overtime or (overtime==0 and overcount==0):
                                o,f = cad.time_overlaps(dfm,dl,dc,disp+'_seconds',cleared+'_seconds',iid,u)
                                ovs = pd.DataFrame.from_records(o)
                                ovs.columns = [str(x) for x in range(int(len(u))+1)]
                                ovs['date'] = dl[0:len(o)]
                                ovs = ovs.drop(ovs.ix[f].index)
                                ovs.to_csv(pref + '_' + name + '_' + '_time_overlaps_'+str(filenumber)+'.csv',index=False)
                             if overcount or (overtime==0 and overcount==0):
                                count,partial = cad.count_overlaps(dfm,dc,disp,cleared,iid,u)
                                count_df = pd.DataFrame.from_records(count,columns=['Incident','Count','Overlapping_incidents'])
                                partial_df = pd.DataFrame.from_records(partial,columns=['Partial_Count','Incident','Overlapping_incidents'])
                                count_df.to_csv(pref + '_' + name + '_' + '_count_overlaps_' + str(filenumber) + '.csv', index=False)
                                partial_df.to_csv(pref + '_' + name + '_' + '_partial_overlaps_' + str(filenumber) + '.csv', index=False)
    # Travel Time by 1st arriving
                if 'TravelTimeByEng' in opt:
                     print('Calculating Travel Time by Engines Engaged...')
                     dft = travel_time(df,atscene,enroute)
                     df = arrive_order(dft,atscene,iid)
                     df = df[df.order == 1]
                     dfr = df[df[unitname].isin(ueng)]
                     # print('**************',ueng,len(dfr))
                     dfrs = dfr.sort_values([disp])
                     dfrs[disp] = pd.to_datetime(dfrs[disp])
                     dfrs[cleared] = pd.to_datetime(dfrs[cleared])
                     dfrs[atscene] = pd.to_datetime(dfrs[atscene])
                     dfrs[enroute] = pd.to_datetime(dfrs[enroute])
                     dfm, dc = cad.to_mat(dfrs)
                     ins = cad.busy_at_response(dfm, dc, disp, cleared, iid, unitname, atscene, enroute)
                     df_ins = pd.DataFrame.from_records(ins, columns=['Incident', 'Date', 'Unit', 'InService',
                                                                      'TravelTime'])
                     df_ins.to_csv( pref + '_busyatresponse_travel_firstarriving_' + str(filenumber) + '.csv',index=False)
        if erf:
            try:
                enforce_limit = int(enforce)
            except ValueError:
                enforce_limit = 10000
            print(' ')
            print('Running ERF analysis.')
            if enforce_limit<10000: print('     Enforce limit: ', enforce_limit)
            if chief: print('     Chief required.')
            descrip = [x.strip() for x in descriptor.split(',')]
            tdf = cad.erf_calculator(dfo,units_df, disp, iid, unitname, atscene, enroute, descriptor_c, descrip, chief,enforce_limit)
            tdf.to_csv(erf_out+'.csv', index=False)
    except FileNotFoundError:
        print('  \n Hmm...Instructions and/or Units and/or data and/or data files do not seem to exist.'
              '\n'
              '\n An Error Log has been produced.')
        with open('ErrorLog.txt', 'a') as f:
            f.write(traceback.format_exc())
    except configparser.NoSectionError:
        print('  \n Hmm...No variables found. Possible cause:'
                '\n       *  Your Instructions file name is wrong, or it is not in the proper folder.'
              '\n       * Units table file name wrong or in wrong folder.'
              '\n'
              '\n An Error Log has been produced.')
        with open('ErrorLog.txt', 'a') as f:
            f.write(traceback.format_exc())
       # sys.exit(1)
    except configparser.NoOptionError:
        print('  \n Hmm...One or more variables needed for this analysis are missing. Possible cause: '
              '\n       * Your Instructions file does not contain all the variables. Check with template.'
              '\n       * Units table file name wrong or in wrong folder.'
              '\n'
              '\n An Error Log has been produced.')
        with open('ErrorLog.txt', 'a') as f:
            f.write(traceback.format_exc())
       # sys.exit(1)
    except KeyError:
        print('  \n Hmm...One or more variables not found. Possible cause: the CAD data does not contain the columns listed in the Instructions file or a column is missing. Check column names in Instructions file (case sensitive).'
              '\n'
              '\n An Error Log has been produced.')
        with open('ErrorLog.txt', 'a') as f:
            f.write(traceback.format_exc())
       # sys.exit(1)
    except IndexError:
        print(' \n There is something wrong with units:'
              ' \n*    You might have the wrong units name file or the wrong path to the file. Remember to include the extension (eg. Units.csv).'
              ' \n*    You might have selected *Yes* to staffed units only but did not specify units type(s).'
              ' \n*    The unit column needed for this analysis might be empty.'
              '\n'
              '\n An Error Log has been produced.')
        with open('ErrorLog.txt', 'a') as f:
            f.write(traceback.format_exc())
    except ValueError:
        print(' \n Generic Error. Depending on selected actions:'
              '\n   * There might be something wrong in the CAD timestamps, such as: wrong format, numbers instead of dates, letters instead of dates, etc...'
              '\n   * You might have selected *Yes* to staffed units only but did not specify units type(s).'
          '\n\n An Error Log has been produced.')
        with open('ErrorLog.txt', 'a') as f:
            f.write(traceback.format_exc())
    except UnboundLocalError:
        print(' \n Generic Error. Units file might be missing.'
          '\n\n An Error Log has been produced.')
        with open('ErrorLog.txt', 'a') as f:
            f.write(traceback.format_exc())
    except ZeroDivisionError:
        print(' \n Division by zero. Is part_hour set to 0 in the Instructions file?'
          '\n\n An Error Log has been produced.')
        with open('ErrorLog.txt', 'a') as f:
            f.write(traceback.format_exc())
    except PermissionError:
        print(' \n Permission Error: you might be trying to overwrite a file that is already open.'
              '\n\n An Error Log has been produced.')
        with open('ErrorLog.txt', 'a') as f:
            f.write(traceback.format_exc())
    except TypeError:
        print(' \n Looks like something might be wrong with the Units file.')
        with open('ErrorLog.txt', 'a') as f:
            f.write(traceback.format_exc())
       # sys.exit(1)
    print('\n All Done.')