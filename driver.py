import pandas as pd
import numpy as np
import cad
import configparser
import warnings
import sys
import traceback
warnings.filterwarnings("ignore")


def travel_time(df,atscene,enroute):
    ### split ems a/ems b /fire
    df['travel'] = 0.
    dfm,dc = cad.to_mat(df)
    dfmt = cad.timer(dfm, 'travel', atscene, enroute, dc)
    dataf_tr = pd.DataFrame(dfmt,columns=df.columns)
#    cad.save(dataf_t,'..\LasVegas\outs\cad_types_travel_'+yr+'.csv','csv')
    return dataf_tr

def response_time(df,atscene,calltime):
    ### split ems a/ems b /fire
    df['response'] = 0.
    dfm,dc = cad.to_mat(df)
    dfmt = cad.timer(dfm, 'response', atscene, calltime, dc)
    dataf_tr = pd.DataFrame(dfmt,columns=df.columns)
#    cad.save(dataf_t,'..\LasVegas\outs\cad_types_travel_'+yr+'.csv','csv')
    return dataf_tr

def service_time(df,atscene,calltime):
    ### split ems a/ems b /fire
    df['service'] = 0.
    dfm,dc = cad.to_mat(df)
    dfmt = cad.timer(dfm, 'service', atscene, calltime, dc)
    dataf_tr = pd.DataFrame(dfmt,columns=df.columns)
#    cad.save(dataf_t,'..\LasVegas\outs\cad_types_travel_'+yr+'.csv','csv')
    return dataf_tr

def turnout_time(df,atscene,calltime):
    df['turnout'] = 0.
    dfm,dc = cad.to_mat(df)
    dfmt = cad.timer(dfm, 'turnout', atscene, calltime, dc)
    dataf_tr = pd.DataFrame(dfmt,columns=df.columns)
#    cad.save(dataf_t,'..\LasVegas\outs\cad_types_travel_'+yr+'.csv','csv')
    return dataf_tr

# back to back time
def bb(dataframe,disp,cleared,unitname,iid):
    dataframe[disp] = pd.to_datetime(dataframe[disp])
    dataframe[cleared] = pd.to_datetime(dataframe[cleared])
    dsecs = cad.unixt(dataframe,[disp,cleared])
    dsecs['bb_time']=0.
   # dsecs['Serv_Time']=0.
    dsecs = dsecs.sort_values([disp,cleared])
    dsecs = dsecs.ix[dsecs[disp].dropna().index]
    dsecs = dsecs.ix[dsecs[cleared].dropna().index]
    dm,dc = cad.to_mat(dsecs)
    arr = cad.b2b(dm,dc,unitname,disp+'_seconds',cleared+'_seconds',iid,'bb_time')
    #arr = cad.timer(dm,'Serv_Time',cleared,disp,dc)
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

def run(s,cf,pref,gis,gisf,engine,ladder,rescue,other,overtime,overcount):
    # get options
    ###
    print('***********************')
    print('Starting CAD analyzer.')
    print('Contact fdebernardis@iaff.org for questions')
    print('***********************')
    print('')
    ###
    if len(s)>0:
        opt = [x[0:15].strip() for x in s]
        # print(opt)
        opt3 = ['TravelTime', 'ArrivingOrder', 'BackToBackTimeA','CoverIncidentsA']
        co= 0.
        for o in opt3: co = co + (o in opt)
        # get data file and config variables
        try:
            config = configparser.ConfigParser()
            config.read(cf)
            datafile = config.get("myvars", "datafile")
            iid = config.get("myvars", "IncidentID")
            atscene = config.get("myvars", "atscene")
            enroute = config.get("myvars", "enroute")
            calltime = config.get("myvars","call")
            disp = config.get("myvars",'dispatch')
            cleared = config.get("myvars",'cleared')
            unitname = config.get("myvars",'units_columns')
            firstd = config.get("myvars",'FirstDue_column')
            appstat = config.get("myvars",'ApparatusStation_column')
            part = float(config.get("myvars", 'part_hour'))
            data = datafile.split(',')
        # now run
            filenumber = 0
            ueng = utr = uresc = uother = None
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
                if d.split('.')[1]=='csv' : df = pd.read_csv(d, encoding='iso-8859-1',warn_bad_lines=False,error_bad_lines=False)
                if (d.split('.')[1] == 'xls') or (d.split('.')[1] =='xlsx'): df = pd.read_excel(d, encoding='iso-8859-1')#error_bad_lines=False,
                df[unitname] = [str(x) for x in df[unitname]]
                df[unitname] = [x.strip() for x in df[unitname]]
                if gis == 1:
                    print(' ')
                    print('Selecting Staffed Units only...')
                    # reduce data to staffed units only
                    uall = [str(x) for x in uall]
                    df  = df[df[unitname].isin(uall)]
                    if 'SelectStaffedAp' in opt: df.to_csv(pref+'_CAD_staffed_only_'+'file_'+str(filenumber)+'.csv',index=False)
                if 'SelectStaffedAp' in opt and gis==0: print('You have to select "Yes" above to select staffed units.')
                checker = 0
                if 'Delete duplicat' in opt:
                    checker = checker + 1
                    print(' ')
                    print('Deleting Duplicates...')
                    df = cad.remove_duplicates(df,iid,unitname)
                    print('Done.')
                if 'TravelTime' in opt:
                    checker = checker + 1
                    print('Calculating Travel Time...')
                    df = travel_time(df,atscene,enroute)
                    # if checker == co: df.to_csv('testout.csv')
                    print('Done.')
                if 'Response Time (' in opt:
                    checker = checker + 1
                    print('Calculating Response Time...')
                    df = response_time(df,atscene,calltime)
                    # if checker == co: df.to_csv('testout.csv')
                    print('Done.')
                if 'Service Time (D' in opt:
                    checker = checker + 1
                    print('Calculating Service Time...')
                    df = service_time(df,cleared,disp)
                    # if checker == co: df.to_csv('testout.csv')
                    print('Done.')
                if 'Turnout Time' in opt:
                    checker = checker + 1
                    print('Calculating Turnout Time...')
                    df = turnout_time(df,enroute,disp)
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
                    df = df.sort([iid])
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
                             print('       Computing time demand...')
                             chr,call, uninc, bt = cad.utilization(dfm,dlut,dict,disp+'_seconds',cleared+'_seconds',iid)
                             dsel[disp] = pd.to_datetime(dsel[disp])
                             dsel[cleared]= pd.to_datetime(dsel[cleared])
                             dfm, dict = cad.to_mat(dsel)
                             print('       Computing units in service...')
                             cu = cad.count_units(dfm,dict,dl,disp,cleared,unitname)
                             dataf = {'date':dl,'calls_perhour':chr,'unique_incidents':uninc,'total_time_engaged':bt,'total_units_engaged':cu}
                             ndf = pd.DataFrame(dataf,columns=['date','calls_perhour','unique_incidents','total_time_engaged','total_units_engaged'])
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
                if '\n TravelTimeByEng' in opt:
                     print('Calculating Travel Time by Engines Engaged...')
                     dft = travel_time(df,atscene,enroute)
                     df = arrive_order(dft,atscene,iid)
                     df = df[df.order == 1]
                     dfr = df[df[unitname].isin(ueng)]
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

           # sys.exit(1)
#### Cover Incidents Analysis ####
            # if 'CoverIncidentsA' in opt:
            #     coverf = 'cover_flag'
            #     locfd = 'location_first_due'
            #     df[coverf] = 'N'
            #     df[locfd] = -1.
            #     print('Running Cover Incidents Analysis...')
            #     df = df.sort([iid])
            #     df[unitname]= [x.strip() for x in df[unitname]]
            #     df = df[df[unitname].isin(ueng+utr)]
            #     df = df.ix[df[cleared].dropna().index]
            #     df = df.ix[df[disp].dropna().index]
            #     df[disp] = pd.to_datetime(df[disp])
            #     df[cleared] = pd.to_datetime(df[cleared])
            #     dsel = df[df[unitname].isin(ueng+utr)]
            #     arr, dc = cad.to_mat(dsel)
            #     for i in range(len(arr)):
            #         inc = arr[i,dc[iid]]
            #     # check if first_due apparatus is on scene
            #         if (arr[arr[:, dc[iid]] == inc, dc[firstd]] == arr[arr[:, dc[iid]] == inc, dc[appstat]]).sum()==0:
            #     # it's not
            #             timers = arr[arr[:, dc[iid]] == inc, dc[disp]].min()
            #     # was the first due apparatus busy?
            #             itime = np.where(np.logical_and(arr[:,dc[disp]]<=timers,arr[:,dc[cleared]]>=timers))[0]
            #             otherinc = itime[arr[itime, dc[iid]] != inc]
            #             icheck  = (arr[otherinc,dc[appstat]] ==  arr[arr[:, dc[iid]] == inc, dc[firstd]][0])
            #             if icheck.sum()>0 :
            #             #it was
            #                 arr[arr[:, dc[iid]] == inc, dc[coverf]] = 'Y'
            #                 arr[arr[:, dc[iid]] == inc, dc[locfd]] = arr[otherinc[np.where(icheck)[0]],dc[firstd]][0]
            #     dfcover = pd.DataFrame(arr,columns=dc.keys())
            #     dfcover.to_csv(pref + '_cover_incidents_'+str(filenumber)+'.csv',index=False)
    print('\n All Done.')