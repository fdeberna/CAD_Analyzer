from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import driver
import pandas as pd

#https://www.homeandlearn.uk/python-database-form-tabs3.html
cols = [' ']
file = ' '
file2 = ' '
tkvar= None
tkvar_att = None
df = None
# def Select():
#     global filename
#     filename = filedialog.askopenfilename(initialdir="/", title="Select file",
#                                           filetypes=(("csv files", "*.txt"), ("all files", "*.*")))
#     return cf

def call_driver_tab1():
    global tkvar,tkvar_att
    if tkvar and tkvar_att:
        driver.preprocessing(file,pref.get(),delet_dup.get(),tkvar.get(),tkvar_att.get())
    else:
        driver.preprocessing(file, pref.get(), delet_dup.get())
    #reset tkvar
    tkvar     = None
    tkvar_att = None

def openfile_oneline():
    global cols,file
    file = filedialog.askopenfilename(initialdir=".", title="Select file",
                                          filetypes=(("text files", "*.txt"), ("all files", "*.*")))
    #get only columns name
    iid, staffname, atscene, enroute, calltime, disp, cleared, format_date,unitname, firstd, appstat, descriptor_c, descriptor, max_lines, part, data = driver.main_reader(file)
    filenumber = 0
    # currently force selection of first file
    for d in data:
        filenumber = filenumber + 1
        if d.split('.')[1] == 'csv': df = pd.read_csv(d, encoding='iso-8859-1', warn_bad_lines=False,
                                                      error_bad_lines=False,nrows=2)
        if (d.split('.')[1] == 'xls') or (d.split('.')[1] == 'xlsx'): df = pd.read_excel(d, encoding='iso-8859-1',nrows=2)
    cols = df.columns
    print('\nInstructions file loaded.')
    return cols,file,df,iid,unitname

def openfile_gis():
    global cols,file2
    file2 = filedialog.askopenfilename(initialdir=".", title="Select file",
                                          filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
    #get only columns name
    iid, staffname, atscene, enroute, calltime, disp, cleared, format_date,unitname, firstd, appstat, descriptor_c, descriptor, max_lines, part, data = driver.main_reader(file)
    filenumber = 0
    # currently force selection of first file
    for d in data:
        filenumber = filenumber + 1
        if d.split('.')[1] == 'csv': df = pd.read_csv(d, encoding='iso-8859-1', warn_bad_lines=False,
                                                      error_bad_lines=False,nrows=2)
        if (d.split('.')[1] == 'xls') or (d.split('.')[1] == 'xlsx'): df = pd.read_excel(d, encoding='iso-8859-1',nrows=2)
    cols = df.columns
    print('\nStaffing table loaded.')
    return cols,file2,df,iid,unitname


def call_driver():
        sel=[]
        selection = lstbox.curselection()
        for i in selection:
            sel.append(lstbox.get(i))
        driver.run(sel,file,pref2.get(),gis.get(),file2,engine.get(),ladder.get(),rescue.get(),other.get(),overtime.get(),overcount.get(),erf.get(),chief.get(),enforce.get(),erf_out.get())

def on_tab_selected(event):
    selected_tab = event.widget.select()
    tab_text = event.widget.tab(selected_tab, "text")
    if tab_text == "Preprocessing":
        print("\n Preprocessing tab selected")
    if tab_text == "Analysis":
        print("\n Analysis tab selected")


#### CONTINUE HERE NEED TO MOVE THIS TO DRIVER, THEN CREATE NEW DROPDOWN FOR ROLLED ATT
def change_dropdown(*args):
    global tkvar,tkvar_att
    print('You selected: ',tkvar.get(),',',tkvar_att.get())

def unroll_file_dropdown():
    global tkvar
    global tkvar_att
    ### dropdown
    print('Reshaping columns:\n')
    tkvar = StringVar(main)
    choices = cols
    tkvar.set(None) # set the default option
    popupMenu = OptionMenu(tab1, tkvar, *choices)
    Label(tab1, text="Column to reshape:").grid(row = 3, column = 2 ,sticky=W)
    popupMenu.grid(row = 3, column =3,sticky=W)
    # on change dropdown value
    # link function to change dropdown
    # tkvar.trace('w', change_dropdown(tkvar))
    tkvar_att = StringVar(main)
    tkvar_att.set(None) # set the default option
    popupMenu = OptionMenu(tab1, tkvar_att, *choices)
    Label(tab1, text="Attributes to reshape:").grid(row = 3, column = 4 ,sticky=W)
    popupMenu.grid(row = 3, column =5,sticky=W)
    # on change dropdown value
    # link function to change dropdown
    tkvar_att.trace('w', change_dropdown)



    # actually unroll the file as requested

# https://pythonspot.com/tk-file-dialogs/
main = Tk()
main.title("CAD Analyzer")
main.geometry("800x670")

tab_control = ttk.Notebook(main)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
main.bind("<<NotebookTabChanged>>", on_tab_selected)
tab_control.add(tab1, text='Preprocessing')
tab_control.add(tab2, text='Analysis')

####################################
##### TAB 1 STARTS HERE ##########
lbl1 = Label(tab1, text= 'label1')
lbl1.grid(column=0, row=0)
# Label(tab1, text="Instructions File:",font='Helvetica 10 bold').grid(row=0,column=0)
# cf = Entry(tab1)
# cf.grid(row=0, column=1,sticky='W')
Button(tab1, text="Select Instructions File",command=openfile_oneline).grid(column=0,row=0)
# Button(tab1, text="Submit",command=openfile_oneline).grid(column=10,row=0)
Label(tab1, text="Output Prefix:",font='Helvetica 10 bold').grid(row=1,column=0)
pref = Entry(tab1)
pref.grid(row=1, column=1,sticky='W')
delet_dup = IntVar()
Label(tab1, text="Delete Duplicates",font='Helvetica 10 bold').grid(row=2,column=0)
Checkbutton(tab1,text=[],variable=delet_dup,state='active',anchor=E).grid(row = 2,column=1,sticky=W)
unroll = IntVar()
Button(tab1,text="Reshape File",command=unroll_file_dropdown).grid(row = 3,column=1,sticky=W)
Button(tab1, text="Run!",command=call_driver_tab1).grid(column=1,row=4)

#################################
##### TAB 2 STARTS HERE ##########
lbl2 = Label(tab2, text='label2')
lbl2.grid(column=0, row=0)
Button(tab2, text="Select Instructions File",command=openfile_oneline).grid(column=0,row=0)
# Label(tab2, text="Instructions File:",font='Helvetica 10 bold').grid(row=0,column=0)
# cf2 = Entry(tab2)
# cf2.grid(row=0, column=1,sticky='W')
Label(tab2, text="Output Prefix:",font='Helvetica 10 bold').grid(row=3,column=0)
pref2 = Entry(tab2)
pref2.grid(row=3, column=1,sticky='W')
# Label(tab2, text="Units Name Table:",font='Helvetica 10 bold').grid(row=3,column=0)
# gisf = Entry(tab2)
# gisf.grid(row=3, column=1,sticky='W')
Button(tab2, text="Select Staffing File",command=openfile_gis).grid(column=0,row=2)
# ## GIS Section
rows = 4
gis = IntVar()
Label(tab2, text="Analyze staffed units only?",font='Helvetica 10 bold').grid(row=rows,column=0)
Checkbutton(tab2, text=['Yes'], variable=gis, state='active').grid(row = rows,column=1,sticky=W)
engine=IntVar()
ladder=IntVar()
rescue=IntVar()
other=IntVar()
Label(tab2, text="If yes, select unit tpye(s):",font='Helvetica 8 bold').grid(row=rows,column=1)
Checkbutton(tab2, text=['Engines'], variable=engine,state='active',anchor=E).grid(row = rows,column=2,sticky=W)
Checkbutton(tab2, text=['Trucks'], variable=ladder,state='active',anchor=E).grid(row = rows,column=3,sticky=W)
Checkbutton(tab2, text=['Rescues'], variable=rescue,state='active',anchor=E).grid(row = rows,column=4,sticky=W)
Checkbutton(tab2, text=['Other'], variable=other,state='active',anchor=E).grid(row = rows,column=5,sticky=W)

overcount=IntVar()
overtime=IntVar()
Label(tab2, text="For overlaps (blank if not needed):",font='Helvetica 10 bold').grid(row=rows+1,column=0)
Checkbutton(tab2, text=['Counts\0Only'], variable=overcount,state='active',anchor=E).grid(row = rows+1,column=1,sticky=W)
Checkbutton(tab2, text=['Time\0Only'], variable=overtime,state='active',anchor=E).grid(row = rows+1,column=2,sticky=W)

Label(tab2, text="What do you want to do?",font='Helvetica 10 bold').grid(row=rows+2,column=0)
frame = ttk.Frame(tab2, padding=(1, 1, 1, 1))
frame.grid(column=1, row=rows+2, sticky=(N, S, E, W))
valores = StringVar()
valores.set(["SelectStaffedApparatus", "Response Time (Call to Arrive)","Service Time (Dispatch to Clear)","Turnout Time","TravelTime", "ArrivingOrder","BackToBackTimeAnalysis","DemandAnalysisEngines(staffed only)",
             "DemandAnalysisTrucks(staffed only)","DemandAnalysisRescues(staffed only)","DemandAnalysisOther(staffed only)",
             "OverlapsEngines(staffed only)","OverlapsTrucks(staffed only)", "OverlapsRescues(staffed only)",
             "OverlapsOther(staffed only)","CoverIncidentsAnalysis","TravelTimeByEngaged(Engines Column)"])


lstbox = Listbox(frame, listvariable=valores, selectmode=MULTIPLE, width=40, height=20)
lstbox.grid(column=0, row=rows+2)
##### HERE ERF ###
Label(tab2, text="Effective Response Force Analysis?",font='Helvetica 10 bold').grid(row=rows+4,column=0)
erf = IntVar()
Checkbutton(tab2, text=['Yes'], variable=erf,state='active',anchor=W).grid(row = rows+4,column=1,sticky=W)
chief = IntVar()
Checkbutton(tab2, text="Chief Required ('Other' column)", variable=chief,state='active',anchor=E).grid(row = rows+5,column=1,sticky=W)
Label(tab2, text="Enforce Number of FF on scene:",font='Helvetica 10 bold').grid(row=rows+6,column=0)
enforce = Entry(tab2)
enforce.grid(row=rows+6, column=1,sticky='W')

Label(tab2, text="Output ERF:",font='Helvetica 10 bold').grid(row=rows+7,column=0)
erf_out = Entry(tab2)
erf_out.grid(row=rows+7, column=1,sticky='W')
btn = ttk.Button(tab2, text="Run!",command=call_driver).grid(column=1,row=rows+10)
#
# ###### END ALL TABS
#
tab_control.pack(expand=1, fill='both')
main.mainloop()





### Config file
# Label(main, text="Instructions File:",font='Helvetica 10 bold').grid(row=0,column=0)
# cf = Entry(main)
# cf.grid(row=0, column=1,sticky='W')
# Label(main, text="Output Prefix:",font='Helvetica 10 bold').grid(row=1,column=0)
# pref = Entry(main)
# pref.grid(row=1, column=1,sticky='W')
# Label(main, text="Units Name Table:",font='Helvetica 10 bold').grid(row=3,column=0)
# gisf = Entry(main)
# gisf.grid(row=3, column=1,sticky='W')
# ## GIS Section
# rows = 4
# gis = IntVar()
# Label(main, text="Analyze staffed units only?",font='Helvetica 10 bold').grid(row=rows,column=0)
# Checkbutton(main, text=['Yes'], variable=gis, state='active').grid(row = rows,column=1,sticky=W)
# #Radiobutton(main, text=['Yes'], variable=gis, value=[1],state='active').grid(row = rows,column=1,sticky=W)
# # Radiobutton(main, text=['No'], variable=gis, value=[0]).grid(row  = rows+1,column=1,sticky=W)
# # usf.grid(row=2, column=3,sticky='W')
# engine=IntVar()
# ladder=IntVar()
# rescue=IntVar()
# other=IntVar()
# Label(main, text="If yes, select unit tpye(s):",font='Helvetica 8 bold').grid(row=rows,column=1)
# Checkbutton(main, text=['Engines'], variable=engine,state='active',anchor=E).grid(row = rows,column=2,sticky=W)
# Checkbutton(main, text=['Trucks'], variable=ladder,state='active',anchor=E).grid(row = rows,column=3,sticky=W)
# Checkbutton(main, text=['Rescues'], variable=rescue,state='active',anchor=E).grid(row = rows,column=4,sticky=W)
# Checkbutton(main, text=['Other'], variable=other,state='active',anchor=E).grid(row = rows,column=5,sticky=W)
#
# overcount=IntVar()
# overtime=IntVar()
# Label(main, text="For overlaps (blank if not needed):",font='Helvetica 10 bold').grid(row=rows+1,column=0)
# Checkbutton(main, text=['Counts\0Only'], variable=overcount,state='active',anchor=E).grid(row = rows+1,column=1,sticky=W)
# Checkbutton(main, text=['Time\0Only'], variable=overtime,state='active',anchor=E).grid(row = rows+1,column=2,sticky=W)
#
# Label(main, text="What do you want to do?",font='Helvetica 10 bold').grid(row=rows+2,column=0)
# frame = ttk.Frame(main, padding=(1, 1, 1, 1))
# frame.grid(column=1, row=rows+2, sticky=(N, S, E, W))
# valores = StringVar()
# valores.set(["Delete duplicates","SelectStaffedApparatus", "Response Time (Call to Arrive)","Service Time (Dispatch to Clear)","Turnout Time","TravelTime", "ArrivingOrder","BackToBackTimeAnalysis","DemandAnalysisEngines(staffed only)",
#              "DemandAnalysisTrucks(staffed only)","DemandAnalysisRescues(staffed only)","DemandAnalysisOther(staffed only)",
#              "OverlapsEngines(staffed only)","OverlapsTrucks(staffed only)", "OverlapsRescues(staffed only)",
#              "OverlapsOther(staffed only)","CoverIncidentsAnalysis","TravelTimeByEngaged(Engines Column)"])
#
#
# lstbox = Listbox(frame, listvariable=valores, selectmode=MULTIPLE, width=40, height=20)
# lstbox.grid(column=0, row=rows+2)
# ##### HERE ERF ###
# Label(main, text="Effective Response Force Analysis?",font='Helvetica 10 bold').grid(row=rows+4,column=0)
# erf = IntVar()
# Checkbutton(main, text=['Yes'], variable=erf,state='active',anchor=W).grid(row = rows+4,column=1,sticky=W)
# chief = IntVar()
# Checkbutton(main, text="Chief Required ('Other' column)", variable=chief,state='active',anchor=E).grid(row = rows+5,column=1,sticky=W)
# Label(main, text="Enforce Number of FF on scene:",font='Helvetica 10 bold').grid(row=rows+6,column=0)
# enforce = Entry(main)
# enforce.grid(row=rows+6, column=1,sticky='W')
#
# Label(main, text="Output ERF:",font='Helvetica 10 bold').grid(row=rows+7,column=0)
# erf_out = Entry(main)
# erf_out.grid(row=rows+7, column=1,sticky='W')
#
#
# btn = ttk.Button(main, text="Run!",command=call_driver).grid(column=1,row=rows+10)
# main.mainloop()
