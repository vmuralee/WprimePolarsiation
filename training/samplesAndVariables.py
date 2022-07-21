variables     = ['weight','m_vis','met','CosTheta','LeadChPtOverTau1Pt','DeltaPtOverTau1Pt','LeadChPtOverTau2Pt','DeltaPtOverTau2Pt']
mva_variables = ['m_vis','CosTheta','met','LeadChPtOverTau1Pt','DeltaPtOverTau1Pt','LeadChPtOverTau2Pt','DeltaPtOverTau2Pt']

signal_dict  = {
    
    "Right_SS_MW4TeV" : {
        'right_rr_signal' : ['WpToTauTauJJ_rr_new.root',4.127E-08*1619,10000],
        'right_rl_signal' : ['WpToTauTauJJ_rl_new.root',3.802E-05*11.5,10000],
    },
    "Right_RR_MW4TeV" : {
        'right_rr_signal' : ['WpToTauTauJJ_rr_new.root',4.127E-08*1619,10000],
    }
}

Background_dict = {
    "TTbarsamples" : {
         'sm_bkg_TTbar_1' :['pptt_1.root', 6.726,1000000],
         'sm_bkg_TTbar_2' :['pptt_2.root', 6.726,1000000],              # TTbar samples
     },

     "Dibosonsamples" : {
         'sm_bkg_WW' : ['ppww_tavt.root',0.8394,252000],
         'sm_bkg_WZ' : ['ppwz_tavt.root',0.09931,30000],
         'sm_bkg_ZZ' : ['ppzz_tata.root',0.01152,3500],
    },

   "DYsamples" :{
        #'sm_bkg_DY100to200':['WpToTauTauJJ_ppztata_100-200.root',23.33,1000000],
        'sm_bkg_DY100to200_1':['ppztata_100-200_1.root',23.33/100,1000000],
        'sm_bkg_DY100to200_2':['ppztata_100-200_2.root',23.33/100,1000000],
        'sm_bkg_DY100to200_3':['ppztata_100-200_3.root',23.33/100,1000000],
        'sm_bkg_DY100to200_4':['ppztata_100-200_4.root',23.33/100,1000000],
        'sm_bkg_DY100to200_5':['ppztata_100-200_5.root',23.33/100,1000000],
        'sm_bkg_DY100to200_6':['ppztata_100-200_6.root',23.33/100,1000000],
        'sm_bkg_DY100to200_7':['ppztata_100-200_7.root',23.33/100,1000000],
        'sm_bkg_DY200to400':['ppztata_200-400.root',0.5316/100,160000],
        'sm_bkg_DY400to500':['ppztata_400-500.root',0.02907/100,9000],  # Drell-Yan samples
        'sm_bkg_DY500to700':['ppztata_500-700.root',0.01733/100,5200],
        'sm_bkg_DY700to800':['ppztata_700-800.root',0.002861/100,100],
        'sm_bkg_DY800to1000':['ppztata_800-1000.root',0.002492/100,100],
        'sm_bkg_DY1000to1500':['ppztata_1000-1500.root',0.001392/100,100],
        'sm_bkg_DY1500to2000':['ppztata_1500-2000.root',0.0002012/100,100],
        'sm_bkg_DY2000to3000':['ppztata_2000-3000.root',0.00005268/100,100],

    },

   # "WToTauNusamples" : {
   #      'sm_bkg_WToTauNuM600' : ['ppwtavt_600.root',0.1101,33000],
   #      'sm_bkg_WToTauNuM800' : ['ppwtavt_800.root',0.03553,10000],
   #      'sm_bkg_WToTauNuM1000' : ['ppwtavt_1000.root',0.01407,4000],
   #      'sm_bkg_WToTauNuM2000' : ['ppwtavt_2000.root',0.0004796,150],
   #      'sm_bkg_WToTauNuM3000' : ['ppwtavt_3000.root',0.00003819,10],
   #      'sm_bkg_WToTauNuM4000' : ['ppwtavt_4000.root',0.000004067,5],
   #  },

}

RL_Background_dict = {
    "Right_RL_MW4TeVsamples" :{'right_rl_bkg' : ['WpToTauTauJJ_rl_new.root',3.802E-05*11.5,10000]}
    
}






