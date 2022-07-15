from tkinter.tix import Tree
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

import uproot3


## Please check the following variables
work_dir = '/home/vinay/private/WprimeAnalysisPart2/WprimePolarsiation'


#########

class SkimAnalyzer(CreateRDataFrame):
    def __init__(self,pT_th,sig_type,turnOn_plot):
        self.df_sig = dict()
        self.df_bkg = dict()
        self.signal_samples = signal_dict[sig_type]
        self.sig_type = sig_type
        if turnOn_plot == True:
            self.sig_type = 'control'
        print('The background samples which considered are, ')
        for key in background_dict.keys():
            print('Processing ......',key)
            for label,sampleitem in background_dict[key].items():
                outfile = label+'.root'
                super().__init__(sampleitem[0],outfile,sampleitem[1],sampleitem[2])
                self.df_bkg[label] = self.RDFrame
                train_outfile = work_dir+f'/data/mva_ntuples/bkg/train_pT{pT_th}'+label+'.root'
                test_outfile  = work_dir+f'/data/mva_ntuples/bkg/test_pT{pT_th}'+label+'.root'
                columns = ROOT.std.vector["string"](variables)
                self.df_bkg[label].Filter(f'(events % 2 ==0 && tau1_vis_pt >{pT_th} && tau1_vis_eta < 2.5 && tau1_vis_eta > -2.5)',"").Snapshot("T",train_outfile,columns)
                self.df_bkg[label].Filter(f'(events % 2 ==1 && tau1_vis_pt >{pT_th} && tau1_vis_eta < 2.5 && tau1_vis_eta > -2.5)',"").Snapshot("T",test_outfile,columns)
                os.remove(outfile)
        for label,sampleitem in self.signal_samples.items():
            outfile = label+'.root'
            train_outfile = work_dir+f'/data/mva_ntuples/signal/train_pT{pT_th}'+label+'.root'
            test_outfile  = work_dir+f'/data/mva_ntuples/signal/test_pT{pT_th}'+label+'.root'
            columns = ROOT.std.vector["string"](variables)
            super().__init__(sampleitem[0],outfile,sampleitem[1],sampleitem[2])
            self.df_sig[label] = self.RDFrame
            self.df_sig[label].Filter(f'(events % 2 ==0 && tau1_vis_pt >{pT_th} && tau1_vis_eta < 2.5 && tau1_vis_eta > -2.5)',"").Snapshot("T",train_outfile,columns)
            self.df_sig[label].Filter(f'(events % 2 ==1 && tau1_vis_pt >{pT_th} && tau1_vis_eta < 2.5 && tau1_vis_eta > -2.5)',"").Snapshot("T",test_outfile,columns)
            os.remove(outfile)

        if turnOn_plot == True:
            self.CreateControlPlot('tau1_vis_pt','#tau pT',20,0,3000)
            self.CreateControlPlot('met','missing E_{T}',20,0,3500)
            self.CreateControlPlot('CosTheta','cos#theta',20,-1,1)
            self.CreateControlPlot('mT','mT',20,0,5000)
            self.CreateControlPlot('LeadChPtOverTauPt','p_{T}^{#pi}/p_{T}^{#tau}',20,0,1.1)
            self.CreateControlPlot('DeltaPtOverPt','#Delta pT/p_{T}^{#tau}',20,0,1.1)


    def CreateControlPlot(self,variable,title,nbins,xlow,xhigh):
        fill_colors = {"DYsamples":38,"WToTauNusamples":46,"TTbarsamples":9,"Dibosonsamples":7}
        legend_titles = {"DYsamples":"Z #rightarrow #tau #tau","WToTauNusamples":"W #rightarrow #tau_{L} #nu_{#tau}","TTbarsamples":"ttbar","Dibosonsamples":"VV"}
       
        histSignalName = ''
        histSignalNames = []
        HistoSig = []
        for label in self.signal_samples.keys():
            histSignalName = label+'_'+variable
            histSignalNames.append(histSignalName)
            HistoSig.append(self.df_sig[label].Filter(f'(tau1_vis_pt >{pT_th} && tau1_vis_eta < 2.5 && tau1_vis_eta > -2.5)',"").Histo1D((histSignalName,histSignalName,nbins,xlow,xhigh),variable,"weight"))
        

        HistoBkg_set = dict()
        for key in background_dict.keys():
            HistoBkg_set[key] = []
            for label in background_dict[key].keys():
                histBkgName = key+'_'+label+'_'+variable
                HistoBkg_set[key].append(self.df_bkg[label].Filter(f'(tau1_vis_pt >{pT_th} && tau1_vis_eta < 2.5 && tau1_vis_eta > -2.5)',"").Histo1D((histBkgName,histSignalName,nbins,xlow,xhigh),variable,"weight"))
        
        # Adding Signal histograms
        HistoSigLeft = HistoSig[0].Clone()
        HistoSigRightN0 = HistoSig[1].Clone()
        HistoSigRightN1 = HistoSig[2].Clone()

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

        HistoSigLeft.SetLineColor(397)
        HistoSigLeft.SetLineWidth(3)
        HistoSigLeft.SetLineStyle(9)

        HistoSigRightN0.SetLineColor(410)
        HistoSigRightN0.SetLineWidth(3)
        HistoSigRightN0.SetLineStyle(9)

        HistoSigRightN1.SetLineColor(1)
        HistoSigRightN1.SetLineWidth(3)
        HistoSigRightN1.SetLineStyle(9)

        hreff.Draw("histo")
        stack.Draw("histosame")
        HistoSigLeft.Draw("histosame")
        HistoSigRightN0.Draw("histosame")
        HistoSigRightN1.Draw("histosame")

        maxbin = -1
        if stack.GetMaximum() > HistoSigLeft.GetMaximum():
            maxbin = stack.GetMaximum()*stack.GetMaximum()
        else:
            maxbin = HistoSigLeft.GetMaximum()*HistoSigLeft.GetMaximum()

        hreff.GetYaxis().SetRangeUser(0.01,maxbin)
        hreff.SetStats(0)
        
        hreff.SetTitle("")
        hreff.GetXaxis().SetTitle(title)

        plotname = self.sig_type+f'_SM_pT{pT_th}'+variable+'.pdf'
        legend = ROOT.TLegend(0.45,0.55,0.85,0.75)
        legend.SetNColumns(2)
        for key in background_dict.keys():
            legend.AddEntry(histBkgNames[key],legend_titles[key],"f")
        legend.AddEntry(histSignalNames[0],"pp #rightarrow W'_{L} #rightarrow #tau_{L} #nu","l")
        legend.AddEntry(histSignalNames[1],"pp #rightarrow W'_{R} #rightarrow #tau_{R} #nu","l")
        legend.AddEntry(histSignalNames[2],"pp #rightarrow W'_{R} #rightarrow #tau_{R} N (M = 1TeV)","l")
        legend.SetLineWidth(0)
        legend.Draw("same")

        c1.SaveAs(work_dir+"/plots/"+plotname)

    



import argparse
parser = argparse.ArgumentParser()

parser.add_argument("th", help="The thereshold ",type=int)
parser.add_argument("-s","--sig_type", help="Left or Right")
parser.add_argument("-d","--plot",  help="produce the variable plot",action="store_true")
parser.add_argument("-t","--train", help="boolean for BDT training",action="store_true")
parser.add_argument("--tune", help="Tune the BDT",action="store_true")
parser.add_argument("-p","--predict", help="boolean for prediction",action="store_true")

args = parser.parse_args()

polarisation = args.sig_type
pT_th = args.th


signal_dir = work_dir+'/data/mva_ntuples/signal/'
bkg_dir = work_dir+'/data/mva_ntuples/bkg/'

if args.predict != True:
    skiming = SkimAnalyzer(pT_th,polarisation,args.plot)

if args.train == True and args.plot == False:
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
    
    os.chdir(signal_dir)
    os.system(f'hadd train_{polarisation}pT{pT_th}_sig.root train_pT{pT_th}*.root')
    os.system(f'hadd test_{polarisation}pT{pT_th}_sig.root test_pT{pT_th}*.root')
    os.system(f'rm train_pT{pT_th}*.root')
    os.system(f'rm test_pT{pT_th}*.root')

    # Prepare background samples 

    os.chdir(bkg_dir)
    os.system(f'hadd train_{polarisation}pT{pT_th}_bkg.root train_pT{pT_th}*.root')
    os.system(f'hadd test_{polarisation}pT{pT_th}_bkg.root test_pT{pT_th}*.root')
    os.system(f'rm train_pT{pT_th}*.root')
    os.system(f'rm test_pT{pT_th}*.root')

    print("Training inputs are, ")
    print(f'train_{polarisation}pT{pT_th}_sig.root','  and   ',f'train_{polarisation}pT{pT_th}_bkg.root')

    os.chdir('/Users/vinaykrishnan/Documents/tau_polarization/MVA/training')
    xtrain, ytrain, wtrain = load_data(signal_dir+f'train_{polarisation}pT{pT_th}_sig.root', bkg_dir+f'train_{polarisation}pT{pT_th}_bkg.root')
    if args.tune == True:
        estimator = XGBClassifier(
            objective= 'binary:logistic',
            nthread=4,
            seed=42
        )
        parameters = {
            'max_depth': range (2, 10, 2),
            'n_estimators': [50,100,150,200,250,300,400,500],
            'learning_rate': [0.1, 0.01, 0.05]
        }
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=parameters,
            scoring = 'roc_auc',
            n_jobs = 10,
            cv = 10,
            verbose=True
        )
        grid_search.fit(xtrain, ytrain)
        best_params = grid_search.best_estimator_
        print(best_params)
        
    else:
        best_params = {
            'max_depth': 10,
            'n_estimators': 500,
            'learning_rate': 0.01
        }
    bdt = XGBClassifier(max_depth= best_params['max_depth'],min_split_loss=0.001,n_estimators= best_params['n_estimators'],learning_rate = best_params['learning_rate'])
    bdt.fit(xtrain, ytrain, wtrain)
    ROOT.TMVA.Experimental.SaveXGBoost(bdt, "myBDT", f'tmvapT{pT_th}_{polarisation}.root',6)
    feature_importance = list(bdt.get_booster().get_score(importance_type='gain').values())
    print(feature_importance)
    importance = pd.DataFrame(data=feature_importance,index=mva_variables,columns=["score"]).sort_values(by = "score", ascending=False)
    fig = importance.nlargest(6, columns="score").plot(kind='barh', figsize = (20,10)).get_figure()
    fig.savefig(work_dir+"/plots/"+f'feature_{polarisation}_importance.png')

    # Save model in TMVA format


if args.predict == True and args.plot == False:
    xtest ,y_true, w = load_data(signal_dir+f'test_{polarisation}pT{pT_th}_sig.root', bkg_dir+f'test_{polarisation}pT{pT_th}_bkg.root')

    File = f'tmvapT{pT_th}_{polarisation}.root'
    if (ROOT.gSystem.AccessPathName(File)) :
        ROOT.Info(File+"does not exist")
        exit()

    bdt = ROOT.TMVA.Experimental.RBDT[""]("myBDT", File)

    # Make  prediction
    y_pred = bdt.Compute(xtest)
    roc_plotname = work_dir+'/plots/'+f'ROC_pT{pT_th}of{polarisation}.png'
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
        outfile = work_dir+'/DataCards/'+'DataCard'+outfilename
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

    
        









        

