variables = ['weight','tau1_vis_pt','met','CosTheta','LeadChPtOverTauPt','DeltaPtOverPt','mT']
mva_variables = ['tau1_vis_pt','met','CosTheta','LeadChPtOverTauPt','DeltaPtOverPt','mT']

signal_dict  = {
    "Left" : {
        'left_signal'  : ['WpToTauTauJJ_pp_w2l.root',0.004165,10000]
    },
    "Right" : {
        'right_N0_signal' :['WpToTauTauJJ_w2r_N_0_new.root',0.004152,10000],
        'right_N1_signal' :['WpToTauTauJJ_w2r_N_1_new.root',0.003286,10000],
    },
    "Right_MW3TeV" : {
        'right_N0_3_signal' :['WpToTauTauJJ_w2R_3_mN_0.root',0.02163,10000],
        'right_N1_3_signal' :['WpToTauTauJJ_w2R_3_mN_1.root',0.01661,10000],
    },
    "Right_MW4TeV" : {
        'right_N0_4_signal' :['WpToTauTauJJ_w2R_4_mN_0.root',0.004152,10000],
        'right_N1_4_signal' :['WpToTauTauJJ_w2R_4_mN_1.root',0.003286,10000],
    },
    "Right_MW5TeV" : {
        'right_N0_5_signal' :['WpToTauTauJJ_w2R_5_mN_0.root',0.000907,10000],
        'right_N1_5_signal' :['WpToTauTauJJ_w2R_5_mN_1.root',0.0006527,10000],
    },
    "Right_MW6TeV" : {
        'right_N0_6_signal' :['WpToTauTauJJ_w2R_6_mN_0.root',0.0002386,1000],
        'right_N1_6_signal' :['WpToTauTauJJ_w2R_6_mN_1.root',0.0001339,1000],
    },
    "Right_MW7TeV" : {
        'right_N0_7_signal' :['WpToTauTauJJ_w2R_7_mN_0.root',0.0000867,100],
        'right_N1_7_signal' :['WpToTauTauJJ_w2R_7_mN_1.root',0.0000328,100],
    },
    "control" :{
        'left_signal'  : ['WpToTauTauJJ_pp_w2l.root',0.004165,10000],
        'right_N0_signal' :['WpToTauTauJJ_w2r_N_0_new.root',0.004152,10000],
        'right_N1_signal' :['WpToTauTauJJ_w2r_N_1_new.root',0.003286,10000],
    }
}
background_dict = {
     "TTbarsamples" : {
        'sm_bkg_TTbar' :['WpToTauTauJJ_pptbwta.root', 63.21,10000],               # TTbar samples
    },

     "Dibosonsamples" : {
        'sm_bkg_WW' : ['WpToTauTauJJ_ppww.root',0.8383,10000],
        'sm_bkg_WZ' : ['WpToTauTauJJ_ppwz.root',0.09931,10000],
        'sm_bkg_ZZ' : ['WpToTauTauJJ_ppzz.root',0.01152,10000],
    },

   "DYsamples" :{
        'sm_bkg_DY100to200':['WpToTauTauJJ_ppztata_100-200.root',23.33,1000000],
        'sm_bkg_DY200to400':['WpToTauTauJJ_ppztata_200-400.root',0.532,20000],
        'sm_bkg_DY400to500':['WpToTauTauJJ_ppztata_400-500.root',0.0292,1000],  # Drell-Yan samples
        'sm_bkg_DY500to700':['WpToTauTauJJ_ppztata_500-700.root',0.0173,1000],
        'sm_bkg_DY700to800':['WpToTauTauJJ_ppztata_700-800.root',0.0028,100],
        'sm_bkg_DY800to1000':['WpToTauTauJJ_ppztata_800-1000.root',0.0025,100],
        'sm_bkg_DY1000to1500':['WpToTauTauJJ_ppztata_1000-1500.root',0.0014,100],
        'sm_bkg_DY1500to2000':['WpToTauTauJJ_ppztata_1500-2000.root',0.0002,10],
        'sm_bkg_DY2000to3000':['WpToTauTauJJ_ppztata_2000-3000.root',0.00005,10],

    },
   
    "WToTauNusamples" : {
        # 'sm_bkg_pT200' :['WpToTauTauJJ_pp_sm_200.root',245,10000],
        # 'sm_bkg_pT300' :['WpToTauTauJJ_pp_sm_300.root',53.39,10000],
        # 'sm_bkg_pT400' :['WpToTauTauJJ_pp_sm_400.root',16.64,10000],
        # 'sm_bkg_pT500' :['WpToTauTauJJ_pp_sm_500.root',6.35,10000],
        # 'sm_bkg_pT600' :['WpToTauTauJJ_pp_sm_600.root',2.749,10000],
        # 'sm_bkg_pT700' :['WpToTauTauJJ_pp_sm_700.root',1.303,10000],
        # 'sm_bkg_pT800' :['WpToTauTauJJ_pp_sm_800.root',0.635,10000],       # W+jets sample
        # 'sm_bkg_pT900' :['WpToTauTauJJ_pp_sm_900.root',0.3486,10000],
        # 'sm_bkg_pT1000':['WpToTauTauJJ_pp_sm_1000.root',0.1914,10000],
        # 'sm_bkg_pT1100':['WpToTauTauJJ_pp_sm_1100.root',0.1083,10000],
        # 'sm_bkg_pT1200':['WpToTauTauJJ_pp_sm_1200.root',0.06284,10000],
        # 'sm_bkg_pT1300':['WpToTauTauJJ_pp_sm_1300.root',0.03713,10000],
        # 'sm_bkg_pT1400':['WpToTauTauJJ_pp_sm_1400.root',0.02245,10000],
        # 'sm_bkg_pT1500':['WpToTauTauJJ_pp_sm_1500.root',0.01357,10000],
        # 'sm_bkg_pT1600':['WpToTauTauJJ_pp_sm_1600.root',0.008407,10000],
        # 'sm_bkg_pT1700':['WpToTauTauJJ_pp_sm_1700.root',0.004021,10000],
        # 'sm_bkg_pT1800':['WpToTauTauJJ_pp_sm_1800.root',0.003241,10000],
        # 'sm_bkg_pT1900':['WpToTauTauJJ_pp_sm_1900.root',0.002005,10000],
        # 'sm_bkg_pT2000':['WpToTauTauJJ_pp_sm_2000.root',0.0002823,10000],
        'sm_bkg_WToTauNupT200' : ['WToTauNu_pT200.root',0.2047,10000],
        'sm_bkg_WToTauNupT400' : ['WToTauNu_pT400.root',0.01197,500],
        'sm_bkg_WToTauNupT600' : ['WToTauNu_pT600.root',0.00226,100],
        'sm_bkg_WToTauNuM200'  : ['WpToTauTauJJ_WToTauNuM_200.root',6.236,220000],
        'sm_bkg_WToTauNuM500'  : ['WpToTauTauJJ_WToTauNuM_500.root',0.214,10000],
        'sm_bkg_WToTauNuM1000' : ['WpToTauTauJJ_WToTauNuM_1000.root',0.01281,500],
        'sm_bkg_WToTauNuM3000' : ['WpToTauTauJJ_WToTauNuM_3000.root',0.00002904,100]

    },
     
}

