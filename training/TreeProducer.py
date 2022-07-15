import ROOT
import numpy as np
import uproot3

## Please check the follwing Variables ##### 
work_dir ='/Users/vinaykrishnan/Documents/tau_polarization/MVA/data/'
Lumi  = 300  #fb-1    


########


class CreateRDataFrame:
    def __init__(self,filename,outfile,xsec,N):
        filepath = work_dir+'/'+filename
        up3_file = uproot3.open(filepath)
        up3_events = up3_file["T"]
        
        boson_p4 = up3_events.array("Wp_p4")
        tau1_p4  = up3_events.array("tau1_p4")
        tau1_vis_p4 = up3_events.array("tau1_vis_p4")
        met1_p4  = up3_events.array("missing1_p4")
        met2_p4  = up3_events.array("missing2_p4")
        met_p4   = up3_events.array("missing_p4")
        LeadChargePion_p4 = up3_events.array("tau1_lead_Ch_p4")
        NeutralPion_p4 = up3_events.array("tau1_neutral_p4")
        tauDecayMode = up3_events.array("tau1_decayMode")

        #Create new Branchs
        newBranchs = dict()
        newBranchs['events']  = np.arange(0,tau1_p4.shape[0])
        newBranchs['weight']  = np.repeat(xsec*1000*Lumi/N,tau1_p4.shape[0])
        newBranchs['tau1_px'] = tau1_p4[:,1]
        newBranchs['tau1_py'] = tau1_p4[:,2]
        newBranchs['tau1_pz'] = tau1_p4[:,3]
        newBranchs['tau1_e'] = tau1_p4[:,0]
        
        newBranchs['boson_mass'] = np.sqrt(boson_p4[:,0]**2 - (boson_p4[:,1]**2 + boson_p4[:,1]**2 + boson_p4[:,2]**2 + boson_p4[:,3]**2))

        newBranchs['tau1_vis_px'] = tau1_vis_p4[:,1]
        newBranchs['tau1_vis_py'] = tau1_vis_p4[:,2]
        newBranchs['tau1_vis_pz'] = tau1_vis_p4[:,3]
        newBranchs['tau1_vis_e']  = tau1_vis_p4[:,0]
        newBranchs['tau1_vis_pt']  = np.array([self.CalcPtEta(tau1_vis_p4,i)[0] for i in range(0,tau1_vis_p4.shape[0])],dtype=np.float32)
        newBranchs['tau1_vis_eta'] = np.array([self.CalcPtEta(tau1_vis_p4,i)[1] for i in range(0,tau1_vis_p4.shape[0])],dtype=np.float32)
        
        newBranchs['met1_px'] = met1_p4[:,1]
        newBranchs['met1_py'] = met1_p4[:,2]
        newBranchs['met1_pz'] = met1_p4[:,3]
        newBranchs['met1_e']  = met1_p4[:,0]
        newBranchs['met1_pt'] = np.sqrt(met1_p4[:,1]**2 + met1_p4[:,2]**2)

        newBranchs['met2_px'] = met2_p4[:,1]
        newBranchs['met2_py'] = met2_p4[:,2]
        newBranchs['met2_pz'] = met2_p4[:,3]
        newBranchs['met2_e']  = met2_p4[:,0]

        newBranchs['met_px'] = met_p4[:,1]
        newBranchs['met_py'] = met_p4[:,2]
        newBranchs['met_pz'] = met_p4[:,3]
        newBranchs['met_e']  = met_p4[:,0]
        newBranchs['met_pt'] = np.sqrt(met_p4[:,1]**2 + met_p4[:,2]**2)
        
        newBranchs['LeadChargePion_px'] = LeadChargePion_p4[:,1]
        newBranchs['LeadChargePion_py'] = LeadChargePion_p4[:,2]
        newBranchs['LeadChargePion_pz'] = LeadChargePion_p4[:,3]
        newBranchs['LeadChargePion_e']  = LeadChargePion_p4[:,0]
        newBranchs['LeadChargePion_pt'] = np.sqrt(LeadChargePion_p4[:,1]**2 + LeadChargePion_p4[:,2]**2)
        
        newBranchs['NeutralPion_px'] = NeutralPion_p4[:,1]
        newBranchs['NeutralPion_py'] = NeutralPion_p4[:,2]
        newBranchs['NeutralPion_pz'] = NeutralPion_p4[:,3]
        newBranchs['NeutralPion_e']  = NeutralPion_p4[:,0]
        newBranchs['NeutralPion_pt'] = np.sqrt(NeutralPion_p4[:,1]**2 + NeutralPion_p4[:,2]**2)
        newBranchs['tauDecayMode'] = tauDecayMode

        tau1_vis_et = np.array([self.CalcET(tau1_vis_p4,i) for i in range(0,tau1_vis_p4.shape[0])],dtype=np.float32)
        newBranchs['met'] = np.array([self.CalcET(met_p4,i) for i in range(0,met_p4.shape[0])],dtype=np.float32)
        ScalarSumET = tau1_vis_et + newBranchs['met']
        VectorSumPx = newBranchs['tau1_vis_px'] + newBranchs['met_px']
        VectorSumPy = newBranchs['tau1_vis_py'] + newBranchs['met_py']

        newBranchs["CosTheta"] = newBranchs["tau1_vis_pz"]/np.sqrt(newBranchs['tau1_vis_px']**2 + newBranchs['tau1_vis_py']**2 +newBranchs['tau1_vis_pz']**2)

        newBranchs["CosTheta"] = np.array([0 if i == 1 else i for i in newBranchs["CosTheta"]],dtype=np.float32)

        Cosdphi = (newBranchs['tau1_vis_px']*newBranchs['met_px']+newBranchs['tau1_vis_py']*newBranchs['met_py'])/(newBranchs['tau1_vis_pt']*newBranchs['met']) 
        

        newBranchs["mT"] = np.sqrt(ScalarSumET*ScalarSumET - VectorSumPx*VectorSumPx - VectorSumPy*VectorSumPy)
        newBranchs["LeadChPtOverTauPt"] = newBranchs['LeadChargePion_pt']/newBranchs['tau1_vis_pt']
        newBranchs["DeltaPtOverPt"] = abs(newBranchs['LeadChargePion_pt']-newBranchs['NeutralPion_pt'])/newBranchs['tau1_vis_pt']

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
        
        
        
    def CalcPtEta(self,vec_p4):
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
        

