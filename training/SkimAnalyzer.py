import ROOT
import numpy as np
import pandas as pd

import sys
import os 

from TreeProducer import CreateRDataFrame
from samplesAndVariables import variables,background_dict,signal_dict,mva_variables

from mva_tools import load_data,CreateROC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

work_dir ='/Users/vinaykrishnan/Documents/tau_polarization/MVA/'

import uproot3


class SkimAnalyzer(CreateRDataFrame):
    def __init__(self,pT_th,sig_type,turnOn_plot):
        self.df_sig = dict()
        self.df_bkg = dict()
        self.signal_samples = signal_dict[sig_type]
        self.sig_type = sig_type
        print('The background samples which considered are, ')
        for key in background_dict.keys():
            print('Processing ......',key)
            for label,sampleitem in background_dict[key].items():
                outfile = label+'.root'
                super().__init__(sampleitem[0],outfile,sampleitem[1],sampleitem[2])
                self.df_bkg[label] = self.RDFrame
                train_outfile = work_dir+f'data/mva_ntuples/bkg/train_pT{pT_th}'+label+'.root'
                test_outfile  = work_dir+f'data/mva_ntuples/bkg/test_pT{pT_th}'+label+'.root'
                columns = ROOT.std.vector["string"](variables)
                self.df_bkg[label].Filter(f'(events % 2 ==0 && tau1_vis_pt >{pT_th} && tau1_vis_eta < 2.5 && tau1_vis_eta > -2.5)',"").Snapshot("T",train_outfile,columns)
                self.df_bkg[label].Filter(f'(events % 2 ==1 && tau1_vis_pt >{pT_th} && tau1_vis_eta < 2.5 && tau1_vis_eta > -2.5)',"").Snapshot("T",test_outfile,columns)
                os.remove(outfile)
        for label,sampleitem in self.signal_samples.items():
            outfile = label+'.root'
            train_outfile = work_dir+f'data/mva_ntuples/signal/train_pT{pT_th}'+label+'.root'
            test_outfile  = work_dir+f'data/mva_ntuples/signal/test_pT{pT_th}'+label+'.root'
            columns = ROOT.std.vector["string"](variables)
            super().__init__(sampleitem[0],outfile,sampleitem[1],sampleitem[2])
            self.df_sig[label] = self.RDFrame
            self.df_sig[label].Filter(f'(events % 2 ==0 && tau1_vis_pt >{pT_th} && tau1_vis_eta < 2.5 && tau1_vis_eta > -2.5)',"").Snapshot("T",train_outfile,columns)
            self.df_sig[label].Filter(f'(events % 2 ==1 && tau1_vis_pt >{pT_th} && tau1_vis_eta < 2.5 && tau1_vis_eta > -2.5)',"").Snapshot("T",test_outfile,columns)
            os.remove(outfile)

        if turnOn_plot == True:
            self.CreateStackPlot('tau1_vis_pt','#tau pT',20,0,3000)
            self.CreateStackPlot('met','missing E_{T}',20,0,3500)
            self.CreateStackPlot('CosTheta','cos#theta',20,-1,1)
            #self.CreateStackPlot('mT','mT',20,0,5000)
            self.CreateStackPlot('LeadChPtOverTauPt','p_{T}^{#pi}/p_{T}^{#tau}',20,0,1.1)
            self.CreateStackPlot('DeltaPtOverPt','#Delta pT/p_{T}^{#tau}',20,0,1.1)

    def CreateStackPlot(self,variable,title,nbins,xlow,xhigh):
        fill_colors = {"DYsamples":38,"WToTauNusamples":46,"TTbarsamples":9,"Dibosonsamples":7}
        legend_titles = {"DYsamples":"Z #rightarrow #tau #tau","WToTauNusamples":"W #rightarrow #tau_{L} #nu_{#tau}","TTbarsamples":"ttbar","Dibosonsamples":"VV"}
        HistoSig = []
        histSignalName = ''
        for label in self.signal_samples.keys():
            histSignalName = label+'_'+variable
            HistoSig.append(self.df_sig[label].Filter(f'(tau1_vis_pt >{pT_th} && tau1_vis_eta < 2.5 && tau1_vis_eta > -2.5)',"").Histo1D((histSignalName,histSignalName,nbins,xlow,xhigh),variable,"weight"))

        HistoBkg_set = dict()
        for key in background_dict.keys():
            HistoBkg_set[key] = []
            for label in background_dict[key].keys():
                histBkgName = key+'_'+label+'_'+variable
                HistoBkg_set[key].append(self.df_bkg[label].Filter(f'(tau1_vis_pt >{pT_th} && tau1_vis_eta < 2.5 && tau1_vis_eta > -2.5)',"").Histo1D((histBkgName,histSignalName,nbins,xlow,xhigh),variable,"weight"))
        
        # Adding Signal histograms
        histoSignal = HistoSig[0].Clone()
        for ih in range(1,len(HistoSig)):
            hist = HistoSig[ih].Clone()
            histoSignal.Add(hist)

        histoSignal.SetName(histSignalName)        

        # Adding Background histograms
        histoBackground = dict()
        histBkgNames    = dict()
        for key in background_dict.keys():
            histoBackground[key] = HistoBkg_set[key][0].Clone()
            for ih in range(1,len(HistoBkg_set[key])):
                hist = HistoBkg_set[key][ih].Clone()
                histoBackground[key].Add(hist)
            histBkgNames[key] = key+'_'+variable
            histoBackground[key].SetFillColor(fill_colors[key])
            histoBackground[key].SetName(histBkgNames[key])
        
        hreff = ROOT.TH1F("hreff","",nbins,xlow,xhigh)
        stack = ROOT.THStack("stack","")
        for key in background_dict.keys():
            stack.Add(histoBackground[key])
        
        c1 = ROOT.TCanvas('c1','',500,500)
        c1.SetLogy()
        c1.SetTickx()
        c1.SetTicky()

        histoSignal.SetLineColor(2)
        histoSignal.SetLineWidth(4)
        histoSignal.SetLineStyle(9)

        hreff.Draw("histo")
        stack.Draw("histosame")
        histoSignal.Draw("histosame")

        maxbin = -1
        if stack.GetMaximum() > histoSignal.GetMaximum():
            maxbin = stack.GetMaximum()*stack.GetMaximum()
        else:
            maxbin = histoSignal.GetMaximum()*histoSignal.GetMaximum()

        hreff.GetYaxis().SetRangeUser(0.01,maxbin)
        hreff.SetStats(0)
        
        hreff.SetTitle("")
        hreff.GetXaxis().SetTitle(title)

        plotname = self.sig_type+f'_SM_pT{pT_th}'+variable+'.png'
        sig_legend = "pp #rightarrow W'_{L} #rightarrow #tau_{L} #nu"
    
        if 'Right' in self.sig_type:
            sig_legend = "pp #rightarrow W'_{R} #rightarrow #tau_{R} #nu"

        legend = ROOT.TLegend(0.55,0.65,0.85,0.75)
        legend.SetNColumns(2)
        for key in background_dict.keys():
            legend.AddEntry(histBkgNames[key],legend_titles[key],"f")
        legend.AddEntry(histSignalName,sig_legend,"l")
        legend.SetLineWidth(0)
        legend.Draw("same")

        c1.SaveAs(work_dir+"plots/"+plotname)


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("th", help="The thereshold ",type=int)
parser.add_argument("-s","--sig_type", help="Left or Right")
parser.add_argument("-d","--plot",  help="produce the control plot",action="store_true")
parser.add_argument("-t","--train", help="boolean for BDT training",action="store_true")
parser.add_argument("-p","--predict", help="boolean for prediction",action="store_true")

args = parser.parse_args()

polarisation = args.sig_type
pT_th = args.th


signal_dir = work_dir+'data/mva_ntuples/signal/'
bkg_dir = work_dir+'data/mva_ntuples/bkg/'

if args.predict != True:
    skiming = SkimAnalyzer(pT_th,polarisation,args.plot)

if args.train == True:
    # Clean root file
    try:
        os.remove(signal_dir+f'train_{polarisation}pT{pT_th}_sig.root')
        os.remove(signal_dir+f'test_{polarisation}pT{pT_th}_sig.root')
    except OSError as error:
        print(error)

    try:
        os.mkdir(signal_dir)
    except OSError as error:
        print(error)

    
    try:
        os.remove(bkg_dir+f'train_{polarisation}pT{pT_th}_bkg.root')
        os.remove(bkg_dir+f'test_{polarisation}pT{pT_th}_bkg.root')
    except OSError as error:
        print(error)

    try:
        os.mkdir(bkg_dir)
    except OSError as error:
        print(error)
    



    # Prepare signal samples

    os.chdir('/Users/vinaykrishnan/Documents/tau_polarization/MVA/data/mva_ntuples/signal')
    os.system(f'hadd train_{polarisation}pT{pT_th}_sig.root train_pT{pT_th}*.root')
    os.system(f'hadd test_{polarisation}pT{pT_th}_sig.root test_pT{pT_th}*.root')
    os.system(f'rm train_pT{pT_th}*.root')
    os.system(f'rm test_pT{pT_th}*.root')

    # Prepare background samples 

    os.chdir('/Users/vinaykrishnan/Documents/tau_polarization/MVA/data/mva_ntuples/bkg')
    os.system(f'hadd train_{polarisation}pT{pT_th}_bkg.root train_pT{pT_th}*.root')
    os.system(f'hadd test_{polarisation}pT{pT_th}_bkg.root test_pT{pT_th}*.root')
    os.system(f'rm train_pT{pT_th}*.root')
    os.system(f'rm test_pT{pT_th}*.root')

    print("Training inputs are, ")
    print(f'train_{polarisation}pT{pT_th}_sig.root','  and   ',f'train_{polarisation}pT{pT_th}_bkg.root')

    os.chdir('/Users/vinaykrishnan/Documents/tau_polarization/MVA/training')
    xtrain, ytrain, wtrain = load_data(signal_dir+f'train_{polarisation}pT{pT_th}_sig.root', bkg_dir+f'train_{polarisation}pT{pT_th}_bkg.root')
    # estimator = XGBClassifier(
    #     objective= 'binary:logistic',
    #     nthread=4,
    #     seed=42
    # )
    # # parameters = {
    #     'max_depth': range (2, 10, 1),
    #     'n_estimators': range(60, 220, 40),
    #     'learning_rate': [0.1, 0.01, 0.05]
    # }
    # grid_search = GridSearchCV(
    #     estimator=estimator,
    #     param_grid=parameters,
    #     scoring = 'roc_auc',
    #     n_jobs = 10,
    #     cv = 10,
    #     verbose=True
    # )
    # grid_search.fit(xtrain, ytrain)
    # best_params = grid_search.best_estimator_
    # print(best_params)
    # best_params = {
    #     'max_depth': 4,
    #     'n_estimators': 200,
    #     'learning_rate': 0.01
    # }
    bdt = XGBClassifier(max_depth= 10,min_split_loss=0.001,n_estimators= 500,learning_rate = 0.01)
    bdt.fit(xtrain, ytrain, wtrain)
    ROOT.TMVA.Experimental.SaveXGBoost(bdt, "myBDT", f'tmvapT{pT_th}_{polarisation}.root',6)
    feature_importance = list(bdt.get_booster().get_score(importance_type='gain').values())
    print(feature_importance)
    importance = pd.DataFrame(data=feature_importance,index=mva_variables,columns=["score"]).sort_values(by = "score", ascending=False)
    fig = importance.nlargest(6, columns="score").plot(kind='barh', figsize = (20,10)).get_figure()
    fig.savefig(work_dir+"plots/"+f'feature_{polarisation}_importance.png')

    # Save model in TMVA format


if args.predict == True:
    xtest ,y_true, w = load_data(signal_dir+f'test_{polarisation}pT{pT_th}_sig.root', bkg_dir+f'test_{polarisation}pT{pT_th}_bkg.root')

    File = f'tmvapT{pT_th}_{polarisation}.root'
    if (ROOT.gSystem.AccessPathName(File)) :
        ROOT.Info(File+"does not exist")
        exit()

    bdt = ROOT.TMVA.Experimental.RBDT[""]("myBDT", File)

    # Make  prediction
    y_pred = bdt.Compute(xtest)
    roc_plotname = work_dir+'plots/'+f'ROC_pT{pT_th}of{polarisation}.png'
    CreateROC(y_true,y_pred,w,roc_plotname)
    dataset = [signal_dir+f'test_{polarisation}pT{pT_th}_sig.root', bkg_dir+f'test_{polarisation}pT{pT_th}_bkg.root']

    # y_pred_histos = []
    for data in dataset:
        data_df = ROOT.RDataFrame("T", data).AsNumpy()
        x = np.vstack([data_df[var] for var in mva_variables]).T
        y_pred_ar = bdt.Compute(x)

        up3_file = uproot3.open(data)
        up3_events = up3_file["T"]
        #Create new Branchs
        newBranchs = dict()
        newBranchs['mva_score'] = y_pred_ar[:,0]
        for variable in variables:
            newBranchs[variable] = up3_events.array(variable)
        outfilename = data.split('test_')[1]
        outfile = work_dir+'DataCards/'+'DataCard'+outfilename
        newfile = uproot3.recreate(outfile)
        BranchNames = dict()
        for branch,_ in newBranchs.items():
            if branch == 'events':
                BranchNames[branch] = np.int32
            else:
                BranchNames[branch] = np.float32
        newfile["tree"] = uproot3.newtree(BranchNames)
        newfile["tree"].extend(newBranchs)
        newfile.close()

    
        









        

