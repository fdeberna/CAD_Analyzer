from tkinter import *
from tkinter import ttk
import driver

def call_driver():
        sel=[]
        selection = lstbox.curselection()
        for i in selection:
            sel.append(lstbox.get(i))
        driver.run(sel,cf.get(),pref.get(),gis.get(),gisf.get(),engine.get(),ladder.get(),rescue.get(),other.get(),overtime.get(),overcount.get())#

      # # ,gisf.get(),gisapp.get(),giseng.get(),giseng_cad.get(),gislad.get(),gislad_cad.get(),gisamb.get(),gisamb_cad.get())

main = Tk()
main.title("CAD Analyzer")
main.geometry("800x470")

### Config file
Label(main, text="Instructions File:",font='Helvetica 10 bold').grid(row=0,column=0)
cf = Entry(main)
cf.grid(row=0, column=1,sticky='W')
Label(main, text="Output Prefix:",font='Helvetica 10 bold').grid(row=1,column=0)
pref = Entry(main)
pref.grid(row=1, column=1,sticky='W')
Label(main, text="Units Name Table:",font='Helvetica 10 bold').grid(row=3,column=0)
gisf = Entry(main)
gisf.grid(row=3, column=1,sticky='W')
## GIS Section
rows = 4
gis = IntVar()
Label(main, text="Analyze staffed units only?",font='Helvetica 10 bold').grid(row=rows,column=0)
Checkbutton(main, text=['Yes'], variable=gis, state='active').grid(row = rows,column=1,sticky=W)
#Radiobutton(main, text=['Yes'], variable=gis, value=[1],state='active').grid(row = rows,column=1,sticky=W)
# Radiobutton(main, text=['No'], variable=gis, value=[0]).grid(row  = rows+1,column=1,sticky=W)
# usf.grid(row=2, column=3,sticky='W')
engine=IntVar()
ladder=IntVar()
rescue=IntVar()
other=IntVar()
Label(main, text="If yes, select unit tpye(s):",font='Helvetica 8 bold').grid(row=rows,column=1)
Checkbutton(main, text=['Engines'], variable=engine,state='active',anchor=E).grid(row = rows,column=2,sticky=W)
Checkbutton(main, text=['Trucks'], variable=ladder,state='active',anchor=E).grid(row = rows,column=3,sticky=W)
Checkbutton(main, text=['Rescues'], variable=rescue,state='active',anchor=E).grid(row = rows,column=4,sticky=W)
Checkbutton(main, text=['Other'], variable=other,state='active',anchor=E).grid(row = rows,column=5,sticky=W)

overcount=IntVar()
overtime=IntVar()
Label(main, text="For overlaps (blank if not needed):",font='Helvetica 10 bold').grid(row=rows+1,column=0)
Checkbutton(main, text=['Counts\0Only'], variable=overcount,state='active',anchor=E).grid(row = rows+1,column=1,sticky=W)
Checkbutton(main, text=['Time\0Only'], variable=overtime,state='active',anchor=E).grid(row = rows+1,column=2,sticky=W)

Label(main, text="What do you want to do?",font='Helvetica 10 bold').grid(row=rows+2,column=0)
frame = ttk.Frame(main, padding=(3, 3, 5, 5))
frame.grid(column=1, row=rows+2, sticky=(N, S, E, W))



valores = StringVar()
valores.set(["Delete duplicates","SelectStaffedApparatus", "Response Time (Call to Arrive)","Service Time (Dispatch to Clear)","Turnout Time","TravelTime", "ArrivingOrder","BackToBackTimeAnalysis","DemandAnalysisEngines(staffed only)",
             "DemandAnalysisTrucks(staffed only)","DemandAnalysisRescues(staffed only)","DemandAnalysisOther(staffed only)",
             "OverlapsEngines(staffed only)","OverlapsTrucks(staffed only)", "OverlapsRescues(staffed only)",
             "OverlapsOther(staffed only)","CoverIncidentsAnalysis","TravelTimeByEngaged(Engines Column)"])


lstbox = Listbox(frame, listvariable=valores, selectmode=MULTIPLE, width=40, height=20)
lstbox.grid(column=0, row=rows+2)

btn = ttk.Button(frame, text="Run!",command=call_driver)
btn.grid(column=0)
main.mainloop()

# master = Tk()
#
# v = IntVar()
#
# Radiobutton(master, text="One", variable=v, value=1).pack(anchor=W)
# Radiobutton(master, text="Two", variable=v, value=2).pack(anchor=W)
#
# mainloop()