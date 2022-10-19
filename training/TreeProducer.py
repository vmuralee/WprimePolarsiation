import ROOT
import numpy as np
import uproot3

## Please check the follwing Variables ##### 
work_dir ='/home/vinay/private/WprimeAnalysisPart2/WprimePolarsiation/data/'
#
Lumi  = 300  #fb-1    


########


class CreateRDataFrame:
    def __init__(self,filename,outfile,xsec,N,misID):
        filepath = work_dir+'/'+filename
        up3_file = uproot3.open(filepath)
        up3_events = up3_file["T"]
        
        boson_p4 = up3_events.array("Wp_p4")[:10000]
        tau1_p4  = up3_events.array("tau1_p4")[:10000]
        tau1_charge = up3_events.array("tau1_charge")[:10000]
        tau1_vis_p4 = up3_events.array("tau1_vis_p4")[:10000]
        tau2_p4  = up3_events.array("tau2_p4")[:10000]
        tau2_charge = up3_events.array("tau2_charge")[:10000]
        tau2_vis_p4 = up3_events.array("tau2_vis_p4")[:10000]
        met1_p4  = up3_events.array("missing1_p4")[:10000]
        met2_p4  = up3_events.array("missing2_p4")[:10000]
        if misID:
            nJets = up3_events.array("nJets")[:10000]
            jetPx = up3_events.array("jetPx")[:10000]
            jetPy = up3_events.array("jetPy")[:10000]
            jetPz = up3_events.array("jetPz")[:10000]
            jetEn = up3_events.array("jetEn")[:10000]
            
        

        met_p4   = up3_events.array("missing_p4")[:10000]
        LeadChargePion_tau1_p4 = up3_events.array("tau1_lead_Ch_p4")[:10000]
        LeadChargePion_tau2_p4 = up3_events.array("tau2_lead_Ch_p4")[:10000]
        NeutralPion_tau1_p4 = up3_events.array("tau1_neutral_p4")[:10000]
        NeutralPion_tau2_p4 = up3_events.array("tau2_neutral_p4")[:10000]
        
        tau1DecayMode = up3_events.array("tau1_decayMode")[:10000]
        tau2DecayMode = up3_events.array("tau2_decayMode")[:10000]
        
        #Create new Branchs
        newBranchs = dict()
        newBranchs['events']  = np.arange(0,tau1_p4.shape[0])
        newBranchs['xsec_weight']  = np.repeat(xsec*1000*Lumi/N,tau1_p4.shape[0])
        newBranchs['tau1_px'] = tau1_p4[:,1]
        newBranchs['tau1_py'] = tau1_p4[:,2]
        newBranchs['tau1_pz'] = tau1_p4[:,3]
        newBranchs['tau1_e'] = tau1_p4[:,0]
        newBranchs['tau1_charge'] = tau1_charge
        
        newBranchs['tau2_px'] = tau2_p4[:,1]
        newBranchs['tau2_py'] = tau2_p4[:,2]
        newBranchs['tau2_pz'] = tau2_p4[:,3]
        newBranchs['tau2_e'] = tau2_p4[:,0]
        newBranchs['tau2_charge'] = tau2_charge
        
        
        newBranchs['boson_mass'] = np.sqrt(boson_p4[:,0]**2 - (boson_p4[:,1]**2 + boson_p4[:,1]**2 + boson_p4[:,2]**2 + boson_p4[:,3]**2))
        
        
            
        
        
        newBranchs['tau1_vis_px'] = tau1_vis_p4[:,1]
        newBranchs['tau1_vis_py'] = tau1_vis_p4[:,2]
        newBranchs['tau1_vis_pz'] = tau1_vis_p4[:,3]
        newBranchs['tau1_vis_e']  = tau1_vis_p4[:,0]
        newBranchs['tau1_vis_pt']  = np.array([self.CalcPtEta(tau1_vis_p4,i)[0] for i in range(0,tau1_vis_p4.shape[0])],dtype=np.float32)
        newBranchs['tau1_vis_eta'] = np.array([self.CalcPtEta(tau1_vis_p4,i)[1] for i in range(0,tau1_vis_p4.shape[0])],dtype=np.float32)
        
        newBranchs['tau2_vis_px'] = tau2_vis_p4[:,1]
        newBranchs['tau2_vis_py'] = tau2_vis_p4[:,2]
        newBranchs['tau2_vis_pz'] = tau2_vis_p4[:,3]
        newBranchs['tau2_vis_e']  = tau2_vis_p4[:,0]
        newBranchs['tau2_vis_pt']  = np.array([self.CalcPtEta(tau2_vis_p4,i)[0] for i in range(0,tau2_vis_p4.shape[0])],dtype=np.float32)
        newBranchs['tau2_vis_eta'] = np.array([self.CalcPtEta(tau2_vis_p4,i)[1] for i in range(0,tau2_vis_p4.shape[0])],dtype=np.float32)
        if misID:
            newBranchs['nJets'] = nJets
            newBranchs['jetPx'] = jetPx
            newBranchs['jetPy'] = jetPy
            newBranchs['jetPz'] = jetPz
            newBranchs['jetEn'] = jetEn

        
            jet2tau_vis_p4   = [self.JetsToFakeTau(tau1_vis_p4,nJets,jetPx,jetPy,jetPz,jetEn,i)[0] for i in range(0,tau1_vis_p4.shape[0])]
            fake_rate_weight = np.array([self.JetsToFakeTau(tau1_vis_p4,nJets,jetPx,jetPy,jetPz,jetEn,i)[1] for i in range(0,tau1_vis_p4.shape[0])],dtype=np.float32)
        
        ## tau1 +tau2 
        tau_vis_p4 = tau1_vis_p4 + tau2_vis_p4
        newBranchs['weight'] = newBranchs['xsec_weight']
        newBranchs['m_vis'] = np.array([p4.M() for p4 in self.TLorentzVector(tau_vis_p4)],dtype=np.float32)
        if misID:
            newBranchs['weight'] = newBranchs['xsec_weight']
            newBranchs['tau2_vis_px'] = np.array([self.JetsToFakeTau(tau1_vis_p4,nJets,jetPx,jetPy,jetPz,jetEn,i)[0].Px() for i in range(0,tau1_vis_p4.shape[0])],dtype=np.float32)
            newBranchs['tau2_vis_py'] = np.array([self.JetsToFakeTau(tau1_vis_p4,nJets,jetPx,jetPy,jetPz,jetEn,i)[0].Py() for i in range(0,tau1_vis_p4.shape[0])],dtype=np.float32)
            newBranchs['tau2_vis_pz'] = np.array([self.JetsToFakeTau(tau1_vis_p4,nJets,jetPx,jetPy,jetPz,jetEn,i)[0].Pz() for i in range(0,tau1_vis_p4.shape[0])],dtype=np.float32)
            newBranchs['tau2_vis_e']  = np.array([self.JetsToFakeTau(tau1_vis_p4,nJets,jetPx,jetPy,jetPz,jetEn,i)[0].E() for i in range(0,tau1_vis_p4.shape[0])],dtype=np.float32)
            newBranchs['tau2_vis_pt']  = np.array([self.JetsToFakeTau(tau1_vis_p4,nJets,jetPx,jetPy,jetPz,jetEn,i)[0].Pt() for i in range(0,tau1_vis_p4.shape[0])],dtype=np.float32)
            newBranchs['tau2_vis_eta'] = np.array([self.JetsToFakeTau(tau1_vis_p4,nJets,jetPx,jetPy,jetPz,jetEn,i)[0].Eta() for i in range(0,tau1_vis_p4.shape[0])],dtype=np.float32)

            newBranchs['m_vis'] = np.array([self.CalcInvMass(tau1_vis_p4,jet2tau_vis_p4,i) for i in range(0,tau1_vis_p4.shape[0])],dtype=np.float32)

        newBranchs['met_px'] = met_p4[:,1]
        newBranchs['met_py'] = met_p4[:,2]
        newBranchs['met_pz'] = met_p4[:,3]
        newBranchs['met_e']  = met_p4[:,0]
        newBranchs['met_pt'] = np.sqrt(met_p4[:,1]**2 + met_p4[:,2]**2)
        
        newBranchs['LeadChargePion_tau1_px'] = LeadChargePion_tau1_p4[:,1]
        newBranchs['LeadChargePion_tau1_py'] = LeadChargePion_tau1_p4[:,2]
        newBranchs['LeadChargePion_tau1_pz'] = LeadChargePion_tau1_p4[:,3]
        newBranchs['LeadChargePion_tau1_e']  = LeadChargePion_tau1_p4[:,0]
        newBranchs['LeadChargePion_tau1_pt'] = np.sqrt(LeadChargePion_tau1_p4[:,1]**2 + LeadChargePion_tau1_p4[:,2]**2)
        
        newBranchs['LeadChargePion_tau2_px'] = LeadChargePion_tau2_p4[:,1]
        newBranchs['LeadChargePion_tau2_py'] = LeadChargePion_tau2_p4[:,2]
        newBranchs['LeadChargePion_tau2_pz'] = LeadChargePion_tau2_p4[:,3]
        newBranchs['LeadChargePion_tau2_e']  = LeadChargePion_tau2_p4[:,0]
        newBranchs['LeadChargePion_tau2_pt'] = np.sqrt(LeadChargePion_tau2_p4[:,1]**2 + LeadChargePion_tau2_p4[:,2]**2)
        
        newBranchs['NeutralPion_tau1_px'] = NeutralPion_tau1_p4[:,1]
        newBranchs['NeutralPion_tau1_py'] = NeutralPion_tau1_p4[:,2]
        newBranchs['NeutralPion_tau1_pz'] = NeutralPion_tau1_p4[:,3]
        newBranchs['NeutralPion_tau1_e']  = NeutralPion_tau1_p4[:,0]
        newBranchs['NeutralPion_tau1_pt'] = np.sqrt(NeutralPion_tau1_p4[:,1]**2 + NeutralPion_tau1_p4[:,2]**2)
        newBranchs['tau1DecayMode'] = tau1DecayMode
        
        newBranchs['NeutralPion_tau2_px'] = NeutralPion_tau2_p4[:,1]
        newBranchs['NeutralPion_tau2_py'] = NeutralPion_tau2_p4[:,2]
        newBranchs['NeutralPion_tau2_pz'] = NeutralPion_tau2_p4[:,3]
        newBranchs['NeutralPion_tau2_e']  = NeutralPion_tau2_p4[:,0]
        newBranchs['NeutralPion_tau2_pt'] = np.sqrt(NeutralPion_tau2_p4[:,1]**2 + NeutralPion_tau2_p4[:,2]**2)
        newBranchs['tau2DecayMode'] = tau2DecayMode

        tau1_vis_et = np.array([self.CalcET(tau1_vis_p4,i) for i in range(0,tau1_vis_p4.shape[0])],dtype=np.float32)
        tau2_vis_et = np.array([self.CalcET(tau2_vis_p4,i) for i in range(0,tau2_vis_p4.shape[0])],dtype=np.float32)
        tau_vis_et  = np.array([p4.Et() for p4 in self.TLorentzVector(tau_vis_p4)],dtype=np.float32)

        newBranchs['met'] = np.array([self.CalcET(met_p4,i) for i in range(0,met_p4.shape[0])],dtype=np.float32)
        ScalarSumET = tau_vis_et + newBranchs['met']
        VectorSumPx = newBranchs['tau1_vis_px']+newBranchs['tau2_vis_px'] + newBranchs['met_px']
        VectorSumPy = newBranchs['tau1_vis_py']+newBranchs['tau2_vis_py'] + newBranchs['met_py']
        tau1_mag = np.array([p4.Mag() for p4 in self.TLorentzVector(tau1_vis_p4)],dtype=np.float32)
        tau2_mag = np.array([p4.Mag() for p4 in self.TLorentzVector(tau2_vis_p4)],dtype=np.float32)

        # newBranchs["CosTheta"] = np.array([self.Cosine(tau1_p4,tau2_p4,i) for i in range(0,tau1_p4.shape[0])],dtype=np.float32)
        
        # newBranchs["CosTheta"] = np.array([0 if i == 1 else i for i in newBranchs["CosTheta"]],dtype=np.float32)
        # Cosdphi = (newBranchs['tau1_vis_px']*newBranchs['met_px']+newBranchs['tau1_vis_py']*newBranchs['met_py'])/(newBranchs['tau1_vis_pt']*newBranchs['met']) 
        

        # newBranchs["mT"] = np.sqrt(ScalarSumET*ScalarSumET - VectorSumPx*VectorSumPx - VectorSumPy*VectorSumPy)
        # newBranchs["LeadChPtOverTau1Pt"] = newBranchs['LeadChargePion_tau1_pt']/newBranchs['tau1_vis_pt']
        # newBranchs["LeadChPtOverTau2Pt"] = newBranchs['LeadChargePion_tau2_pt']/newBranchs['tau2_vis_pt']
        # newBranchs["DeltaPtOverTau1Pt"] = abs(newBranchs['LeadChargePion_tau1_pt']-newBranchs['NeutralPion_tau1_pt'])/newBranchs['tau1_vis_pt']
        # newBranchs["DeltaPtOverTau2Pt"] = abs(newBranchs['LeadChargePion_tau2_pt']-newBranchs['NeutralPion_tau2_pt'])/newBranchs['tau2_vis_pt']

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
        self.RDFrame = ROOT.RDataFrame("tree",outfile)
        
        
        
    def CalcPtEta(self,vec_p4,i):
        E_ar  = vec_p4[:,0][i]
        px_ar = vec_p4[:,1][i]
        py_ar = vec_p4[:,2][i]
        pz_ar = vec_p4[:,3][i]

        p4_ = ROOT.TLorentzVector(px_ar,py_ar,pz_ar,E_ar)
        return [p4_.Pt(),p4_.Eta()]

    def CalcET(self,vec_p4,i):
        E_ar  = vec_p4[:,0][i]
        px_ar = vec_p4[:,1][i]
        py_ar = vec_p4[:,2][i]
        pz_ar = vec_p4[:,3][i]

        p4_ = ROOT.TLorentzVector(px_ar,py_ar,pz_ar,E_ar)
        return p4_.Et()
        
    def TLorentzVector(self,vec_p4):
        E_ar  = vec_p4[:,0]
        px_ar = vec_p4[:,1]
        py_ar = vec_p4[:,2]
        pz_ar = vec_p4[:,3]

        p4_ar = [ROOT.TLorentzVector(px_ar[i],py_ar[i],pz_ar[i],E_ar[i]) for i in range(0,vec_p4.shape[0])]
        return p4_ar


    def CalcInvMass(self,tau_p4,jet_p4,i):
        tau_p4_ = self.TLorentzVector(tau_p4)
        inv = [(tau_p4_[i]+jet_p4[i]).M() for i in range(0,tau_p4.shape[0])]
        return inv

    def Cosine(self,tau1_p4,tau2_p4,i):
        E1_ar  = tau1_p4[:,0][i]
        px1_ar = tau1_p4[:,1][i]
        py1_ar = tau1_p4[:,2][i]
        pz1_ar = tau1_p4[:,3][i]

        E2_ar  = tau2_p4[:,0][i]
        px2_ar = tau2_p4[:,1][i]
        py2_ar = tau2_p4[:,2][i]
        pz2_ar = tau2_p4[:,3][i]

        tau1 = ROOT.TVector3(px1_ar,py1_ar,pz1_ar)
        tau2 = ROOT.TVector3(px2_ar,py2_ar,pz2_ar)

        cos_theta = tau1.Dot(tau2)/(tau1.Mag()*tau2.Mag())
        return cos_theta


    def JetsToFakeTau(self,vec_p4,nJets,jetPx,jetPy,jetPz,jetEn,i):
        E_ar  = vec_p4[:,0][i]
        px_ar = vec_p4[:,1][i]
        py_ar = vec_p4[:,2][i]
        pz_ar = vec_p4[:,3][i]
        
        
        p4_ = ROOT.TLorentzVector(px_ar,py_ar,pz_ar,E_ar)
        fake_p4 = ROOT.TLorentzVector(0,0,0,0)
        fakeRateWeight = 0.005
        if p4_.Pt() > 100 and abs(p4_.Eta()) < 2.5:
            for j in range(0,nJets[i]):
              jet_p4 = ROOT.TLorentzVector(jetPx[i][j],jetPx[i][j],jetPy[i][j],jetEn[i][j])
              if jet_p4.Pt() < 100 or abs(jet_p4.Eta()) > 2.5:continue
              if jet_p4.DeltaR(p4_) < 0.5:continue
              fake_p4 = fake_p4 + jet_p4
              fakeRateWeight = fakeRateWeight * 0.005
        jetlist = [fake_p4,fakeRateWeight]
        return jetlist

#filename = 'WpToTauTauJJ_rr_new.root'
#CreateRDataFrame(filename,'out.root',0.005,10000)
