import pyhf
from pyhf.contrib.viz import brazil
import matplotlib.pyplot as plt
import numpy as np
import ROOT
import json
import sys 

class POIEstimator:
    def __init__(self,sig_list,bkg_list,bkg_uncert):
        self.sig_l = sig_list
        self.bkg_l = bkg_list
        self.bkg_un = bkg_uncert
        model = self.GetModel()
        self.obs = self.bkg_l + model.config.auxdata
    
    def GetModel(self):
        model = pyhf.simplemodels.hepdata_like(signal_data=self.sig_l,bkg_data=self.bkg_l,bkg_uncerts=self.bkg_un)
        return model

    def POIrange(self,val_lo,val_hi,nvals):
        return np.linspace(val_lo,val_hi,nvals)
    
    def hypo_test(self,poi_values):
        model = self.GetModel()
        results = [
        pyhf.infer.hypotest(
            poi_value,
            self.obs,
            model,
            test_stat="qtilde",
            return_expected_set=True,
        )
        for poi_value in poi_values
        ]
        return results

    def compute_limits(self,poi_values,alpha):
        model = self.GetModel()
        obs_limit, exp_limits, (scan, results) = pyhf.infer.intervals.upperlimit(
            self.obs, model, poi_values, level=alpha, return_results=True
        )
        return exp_limits

    def plot_CLsPOI(self,poi_values):
        results = self.hypo_test(poi_values)
        fig, ax = plt.subplots()
        fig.set_size_inches(10.5, 7)
        ax.set_title("Hypothesis Tests")
        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel(r"$\mathrm{CL}_{s}$")

        brazil.plot_results(ax, poi_values, results)
        

json_file = sys.argv[1]

variable_name = ''
labels = [3000,4000,5000,6000,7000]
xsec = [0.01661*1000,0.003286*1000,0.0006527*1000,0.0001339*1000,0.0000328*1000]
title_name = "M_{W'}"
if 'mT' in json_file:
    variable_name = 'mT'
    # labels = [500,750,1000,1250,1500,1750,2000,2250]
    #title_name = "MT_{min}"
elif 'mva_score' in json_file:
    variable_name = 'mva_score'
    # labels = [0.1,0.3,0.4,0.5,0.6,0.7,0.8]
    #title_name = "MVA_"
else:
    print('The vaiable is not found')

f = open(json_file)
datacard = json.load(f)

limit_list = []
for sample in datacard.keys():
    print(sample)

    sig_bins = datacard[sample]['singal_bins']
    bkg_bins = datacard[sample]['background_bins']
    bkg_unc_bins = datacard[sample]['background_bins']

    est =  POIEstimator(sig_bins,bkg_bins,bkg_unc_bins)

    poi_val = est.POIrange(0.00001,2,100)

    exp_limits = est.compute_limits(poi_val,0.05)

    limit_list.append(exp_limits)

print(limit_list)
N = len(labels)


yellow = ROOT.TGraph(2*N)
green  = ROOT.TGraph(2*N)
median = ROOT.TGraph(N)
theory = ROOT.TGraph(N)

up2s = [ ]
for i in range(N):
    up2s.append(limit_list[i][4]*xsec[i])
    print(i,'  ',limit_list[i][2]*xsec[i])
    yellow.SetPoint(i,  labels[i] , limit_list[i][4]*xsec[i])
    green.SetPoint( i,  labels[i] , limit_list[i][3]*xsec[i])
    median.SetPoint(i,  labels[i] , limit_list[i][2]*xsec[i])
    theory.SetPoint(i,  labels[i] , xsec[i])
    green.SetPoint(  2*N-1-i, labels[i], limit_list[i][1]*xsec[i]) 
    yellow.SetPoint( 2*N-1-i, labels[i], limit_list[i][0]*xsec[i])




W = 800
H  = 600
T = 0.08*H
B = 0.12*H
L = 0.12*W
R = 0.04*W
c = ROOT.TCanvas("c","c",100,100,W,H)

c.SetFillColor(0)
c.SetBorderMode(0)
c.SetFrameFillStyle(0)
c.SetFrameBorderMode(0)
c.SetLeftMargin( L/W )
c.SetRightMargin( R/W )
c.SetTopMargin( T/H )
c.SetBottomMargin( B/H )
c.SetTickx(0)
c.SetTicky(0)
#c.SetGrid()
c.SetLogy()
c.cd()
frame = c.DrawFrame(1.4,0.001, 4.1, 10)
frame.GetYaxis().CenterTitle()
frame.GetYaxis().SetTitleSize(0.05)
frame.GetXaxis().SetTitleSize(0.05)
frame.GetXaxis().SetLabelSize(0.04)
frame.GetYaxis().SetLabelSize(0.04)
frame.GetYaxis().SetTitleOffset(0.9)
frame.GetXaxis().SetNdivisions(508)
frame.GetYaxis().CenterTitle(True)
frame.GetYaxis().SetTitle("95% upper limit on #sigma #times B(W'#rightarrow #tau#nu) [fb] ")
#    frame.GetYaxis().SetTitle("95% upper limit on #sigma #times BR / (#sigma #times BR)_{SM}")
frame.GetXaxis().SetTitle(title_name)
frame.SetMinimum(0.001)
frame.SetMaximum(max(up2s)*10.5)
if variable_name == 'mva_score':
    frame.GetXaxis().SetLimits(labels[0],labels[-1]+100)
else:
    frame.GetXaxis().SetLimits(labels[0],labels[-1]+100)

yellow.SetFillColor(ROOT.kOrange)
yellow.SetLineColor(ROOT.kOrange)
yellow.SetFillStyle(1001)
yellow.Draw('F')
 
green.SetFillColor(ROOT.kGreen+1)
green.SetLineColor(ROOT.kGreen+1)
green.SetFillStyle(1001)
green.Draw('Fsame')
 
median.SetLineColor(1)
median.SetLineWidth(4)
median.SetLineStyle(2)
median.Draw('lsame')

theory.SetLineColor(1)
theory.SetLineWidth(2)
theory.SetLineStyle(4)
theory.Draw('lsame')



x1 = 0.5
x2 = x1 + 0.24
y2 = 0.90
y1 = 0.75
legend = ROOT.TLegend(x1,y1,x2,y2)
legend.SetFillStyle(0)
legend.SetBorderSize(0)
legend.SetTextSize(0.041)
legend.SetTextFont(42)
legend.AddEntry(median, "Asymptotic CL_{s} expected",'L')
legend.AddEntry(green, "#pm 1 std. deviation",'f')
#    legend.AddEntry(green, "Asymptotic CL_{s} #pm 1 std. deviation",'f')
legend.AddEntry(yellow,"#pm 2 std. deviation",'f')
#    legend.AddEntry(green, "Asymptotic CL_{s} #pm 2 std. deviation",'f')
legend.Draw()
 
print(" ")
c.SaveAs(f'UpperLimit_Right_{variable_name}.png')
c.Close()



