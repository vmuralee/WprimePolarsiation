import ROOT
import numpy as np
import pickle


from samplesAndVariables import mva_variables





def load_data(signal_filename, background_filename):
    # Read data from ROOT files
    data_sig = ROOT.RDataFrame("T", signal_filename).AsNumpy()
    data_bkg = ROOT.RDataFrame("T", background_filename).AsNumpy()
 
    # Convert inputs to format readable by machine learning tools
    x_sig = np.vstack([data_sig[var] for var in mva_variables]).T
    x_bkg = np.vstack([data_bkg[var] for var in mva_variables]).T
    x = np.vstack([x_sig, x_bkg])
 
    # Create labels
    num_sig = x_sig.shape[0]
    num_bkg = x_bkg.shape[0]
    y = np.hstack([np.ones(num_sig), np.zeros(num_bkg)])
 
    # Compute weights balancing both classes
    num_all = num_sig + num_bkg
    w = np.hstack([np.ones(num_sig) * num_all / num_sig, np.ones(num_bkg) * num_all / num_bkg])
    w = np.hstack([data_sig['weight'],data_bkg['weight']])
    return x, y, w




# Compute ROC using sklearn
from sklearn.metrics import roc_curve, auc

def CreateROC(y_true,y_pred,w,plotname):
    fpr, tpr, _ = roc_curve(y_true, y_pred, sample_weight=w)
    score = auc(fpr, tpr)
    # Plot ROC
    c = ROOT.TCanvas("roc", "", 600, 600)
    g = ROOT.TGraph(len(fpr), fpr, tpr)
    g.SetTitle("AUC = {:.2f}".format(score))
    g.SetLineWidth(3)
    g.SetLineColor(ROOT.kRed)
    g.Draw("AC")
    g.GetXaxis().SetRangeUser(0, 1)
    g.GetYaxis().SetRangeUser(0, 1.2)
    g.GetXaxis().SetTitle("False-positive rate")
    g.GetYaxis().SetTitle("True-positive rate")
    c.SaveAs(plotname)
    c.Draw()


## params 
mva_params = {
    'Left_MW3TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Left_MW3.5TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Left_MW4TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Left_MW4.5TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Left_MW5TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Left_MW5.5TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Left_MW6TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Left_MW6.5TeV':{'max_depth':6,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.01},
    'Left_MW7TeV':{'max_depth': 4,'n_estimators': 500, 'learning_rate': 0.1,'min_split_loss':0.01},
    'Right_N0_MW3.0TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Right_N0_MW3.5TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Right_N0_MW4TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Right_N0_MW4.5TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Right_N0_MW5TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Right_N0_MW5.5TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Right_N0_MW6TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Right_N0_MW6.5TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Right_N0_MW7TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Right_N1_MW3.0TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Right_N1_MW3.5TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Right_N1_MW4TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Right_N1_MW4.5TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Right_N1_MW5TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.001},
    'Right_N1_MW5.5TeV':{'max_depth': 5,'n_estimators': 500, 'learning_rate': 0.1,'min_split_loss':0.01},
    'Right_N1_MW6TeV':{'max_depth': 5,'n_estimators': 500, 'learning_rate': 0.1,'min_split_loss':0.01},
    'Right_N1_MW6.5TeV':{'max_depth':2,'n_estimators': 500, 'learning_rate': 0.1,'min_split_loss':0.01},
    'Right_N1_MW7TeV':{'max_depth': 10,'n_estimators': 500, 'learning_rate': 0.01,'min_split_loss':0.01}

}   


