from tkinter import DoubleVar, Variable
import ROOT
import json
from matplotlib.pyplot import xlabel

from pandas import array

work_dir ='/Users/vinaykrishnan/Documents/tau_polarization/MVA/'

class CreateDatacard:
    def __init__(self,pTth,polarisation,variable,nbins,xlow,xhigh):
        sig_sample = work_dir+f'DataCards/DataCard{polarisation}pT{pTth}_sig.root'
        bkg_sample = work_dir+f'DataCards/DataCard{polarisation}pT{pTth}_bkg.root'

        self.df_sig = ROOT.RDataFrame("tree",sig_sample)
        self.df_bkg = ROOT.RDataFrame("tree",bkg_sample)
        self.PlotHisto(polarisation,variable,nbins,xlow,xhigh)

        sighistname = f'sig_{variable}'
        bkghistname = f'bkg_{variable}'
        
        datacard_sig_histo = self.df_sig.Histo1D((sighistname,sighistname,nbins,xlow,xhigh),variable,"weight")
        datacard_bkg_histo = self.df_bkg.Histo1D((bkghistname,bkghistname,nbins,xlow,xhigh),variable,"weight")

        samplename = f'samples_{polarisation}_{variable}'
        datacard_json = dict()
        datacard_json[samplename] = dict()
        datacard_json[samplename]['singal_bins'] = []
        datacard_json[samplename]['background_bins'] = []
        datacard_json[samplename]['bkgUncertainity_bins'] = []

        for ib in range(1,datacard_sig_histo.GetNbinsX()+1):
            nan_val = 4e-02
            if datacard_sig_histo.GetBinContent(ib) == 0.0:
                datacard_json[samplename]['singal_bins'].append(datacard_sig_histo.GetBinContent(ib-1))
            else:
                datacard_json[samplename]['singal_bins'].append(datacard_sig_histo.GetBinContent(ib))

        for ib in range(1,datacard_bkg_histo.GetNbinsX()+1):
            nan_val = 4e-02
            if datacard_bkg_histo.GetBinContent(ib) == 0.0:
                datacard_json[samplename]['background_bins'].append(nan_val)
                datacard_json[samplename]['bkgUncertainity_bins'].append(0.2)
                #datacard_json[samplename]['background_bins'].append(datacard_json[samplename]['background_bins'][-1])
                #datacard_json[samplename]['bkgUncertainity_bins'].append(datacard_json[samplename]['bkgUncertainity_bins'][-1])
            else:
                datacard_json[samplename]['background_bins'].append(datacard_bkg_histo.GetBinContent(ib))
                datacard_json[samplename]['bkgUncertainity_bins'].append(datacard_bkg_histo.GetBinError(ib))

        json_object = json.dumps(datacard_json, indent = 4)
        outfilename = work_dir+f'DataCards/DataCard{variable}_{polarisation}pT{pTth}.json'
        with open(outfilename, "w") as outfile:
            outfile.write(json_object)
        

    def PlotHisto(self,polarsiation,variable,nbins,xlow,xhigh):
        sig_hist = self.df_sig.Histo1D(("sighistname","sighistname",nbins,xlow,xhigh),variable,"weight")
        bkg_hist = self.df_bkg.Histo1D(("bkghistname","bkghistname",nbins,xlow,xhigh),variable,"weight")
        for ib in range(1,bkg_hist.GetNbinsX()+1):
            if bkg_hist.GetBinContent(ib) == 0:
                bkg_hist.SetBinContent(ib,0.04)
                #bkg_hist.SetBinContent(ib,bkg_hist.GetBinContent(ib-1))
            else:
                bkg_hist.SetBinContent(ib,bkg_hist.GetBinContent(ib))
        hreff = ROOT.TH1F("hreff","",nbins,xlow,xhigh)
        c1 = ROOT.TCanvas('c1','',500,500)
        c1.SetLogy()
        c1.SetTickx()
        c1.SetTicky()

        hreff.Draw("histo")
        sig_hist.Draw("histosame")
        bkg_hist.Draw("histosame")

        sig_hist.SetFillColor(2)
        #bkg_hist.SetFillColor(3)
        bkg_hist.SetFillColorAlpha(3,0.5)
        maxbin = -1
        if sig_hist.GetMaximum() > bkg_hist.GetMaximum():
            maxbin = sig_hist.GetMaximum()*sig_hist.GetMaximum()
        else:
            maxbin = bkg_hist.GetMaximum()*bkg_hist.GetMaximum()

        hreff.GetYaxis().SetRangeUser(0.01,maxbin)
        hreff.SetStats(0)

        hreff.SetTitle("")
        hreff.GetXaxis().SetTitle(variable)
        legend = ROOT.TLegend(0.55,0.65,0.85,0.75)
        
        legend.AddEntry("sighistname","signal","f")
        legend.AddEntry("bkghistname","background","f")

        legend.SetLineWidth(0)
        legend.Draw("same")

        plotname = f'Datacard{polarsiation}_{variable}.png'
        c1.SaveAs(work_dir+"DataCards/"+plotname)


def combine_datacard(cards):
    list_of_cards = []
    for card in cards:
        f = open(work_dir +"DataCards/"+str(card))
        datacard = json.load(f)
        list_of_cards.append(datacard)
    combined_dict = list_of_cards[0]
    for i in range(len(list_of_cards)-1):
        combined_dict = combined_dict | list_of_cards[i+1]
    json_object = json.dumps(combined_dict, indent = 4)
    outfilename = work_dir+f'DataCards/DataCard.json'
    with open(outfilename, "w") as outfile:
        outfile.write(json_object)
    return combined_dict

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("th", help="The thereshold ",type=int)
parser.add_argument("-s","--sig_type", help="Left or Right",type=str,required=False)
parser.add_argument("-v","--variable", help="mT or mva_score",type=str,required=False)
parser.add_argument("--hbins",nargs='+',help='bins for the histogram, e.g. 20 100 3500',required=False)
parser.add_argument("-c","--combine",action="store_true",required=False)
parser.add_argument("--list",nargs='+',help='list of data cards ',required=False)

args = parser.parse_args()

variable = args.variable
polarisation = args.sig_type
pT_th = args.th
nbins = int(args.hbins[0])
xlow  = float(args.hbins[1])
xhigh = float(args.hbins[2])

if args.combine == False:
    
    datacard = CreateDatacard(pT_th,polarisation,variable,nbins,xlow,xhigh)
else:
    combine_datacard(args.list)

# mTdatacard = CreateDatacard(500,'Right_MW3TeV','mT',20,100,3500)
# scoredatacard = CreateDatacard(500,'Right_MW3TeV','mva_score',20,0.1,1.0)
