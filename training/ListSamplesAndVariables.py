variables     = ['weight','m_vis','met','tau1_vis_pt','tau2_vis_pt']#,'CosTheta','LeadChPtOverTau1Pt','DeltaPtOverTau1Pt','LeadChPtOverTau2Pt','DeltaPtOverTau2Pt','tau1_vis_p4','nJets','jetPx','jetPy','jetPz','jetEn']#["xsec_weight","tau1_vis_p4"]#
#mva_variables = ['m_vis','CosTheta','met','LeadChPtOverTau1Pt','DeltaPtOverTau1Pt','LeadChPtOverTau2Pt','DeltaPtOverTau2Pt']
mva_variables = ['tau1_vis_pt','tau2_vis_pt']

signal_dict  = {
    
    "Right_SS_MW3TeV" : {
        'right_rr_signal' : ['pp_rr_3TeV.root',3.367E-04,10000],
        'right_rl_signal' : ['pp_rl_3TeV.root',2.156E-03,10000],
    },
    "Right_SS_MW3.5TeV" : {
        'right_rr_signal' : ['pp_rr_3.5TeV.root',1.493E-04,10000],
        'right_rl_signal' : ['pp_rl_3.5TeV.root',9.709E-04,10000],
    },
    "Right_SS_MW4TeV" : {
        'right_rr_signal' : ['pp_rr_4TeV.root',6.671E-05,10000],
        'right_rl_signal' : ['pp_rl_4TeV.root',4.371E-04,10000],
    },
    "Right_SS_MW4.5TeV" : {
        'right_rr_signal' : ['pp_rr_4.5TeV.root',2.972E-05,10000],
        'right_rl_signal' : ['pp_rl_4.5TeV.root',1.955E-04,10000],
    },
    "Right_SS_MW5TeV" : {
        'right_rr_signal' : ['pp_rr_5TeV.root',1.3242E-05,10000],
        'right_rl_signal' : ['pp_rl_5TeV.root',8.709E-05,10000],
    },
    "Right_SS_MW5.5TeV" : {
        'right_rr_signal' : ['pp_rr_5.5TeV.root',5.951E-06,10000],
        'right_rl_signal' : ['pp_rl_5.5TeV.root',3.914E-05,10000],
    },
    "Right_SS_MW6TeV" : {
        'right_rr_signal' : ['pp_rr_6TeV.root',2.707E-06,10000],
        'right_rl_signal' : ['pp_rl_6TeV.root',1.787E-05,10000],
    },
    "Right_SS_MW6.5TeV" : {
        'right_rr_signal' : ['pp_rr_6.5TeV.root',1.289E-06,10000],
        'right_rl_signal' : ['pp_rl_6.5TeV.root',8.512E-06,10000],
    },
    "Right_SS_MW7TeV" : {
        'right_rr_signal' : ['pp_rr_7TeV.root',6.584E-07,10000],
        'right_rl_signal' : ['pp_rl_7TeV.root',4.343E-06,10000],
    },

     "Right_RR_MW3TeV" : {
        'right_rr_signal' : ['pp_rr_3TeV.root',3.367E-04,10000],
    },
    "Right_RR_MW3.5TeV" : {
        'right_rr_signal' : ['pp_rr_3.5TeV.root',1.493E-04,10000],
    },
    "Right_RR_MW4TeV" : {
       'right_rr_signal' : ['pp_rr_4TeV.root',6.671E-05,10000],
    },
    "Right_RR_MW4.5TeV" : {
        'right_rr_signal' : ['pp_rr_4.5TeV.root',2.972E-05,10000],
    },
    "Right_RR_MW5TeV" : {
        'right_rr_signal' : ['pp_rr_5TeV.root',1.3242E-05,10000],
    },
    "Right_RR_MW5.5TeV" : {
        'right_rr_signal' : ['pp_rr_5.5TeV.root',5.951E-06,10000],
    },
    "Right_RR_MW6TeV" : {
        'right_rr_signal' : ['pp_rr_6TeV.root',2.707E-06,10000],
    },
    "Right_RR_MW6.5TeV" : {
        'right_rr_signal' : ['pp_rr_6.5TeV.root',1.289E-06,10000],
    },
    "Right_RR_MW7TeV" : {
        'right_rr_signal' : ['pp_rr_7TeV.root',6.584E-07,10000],
    },
  
    "Right_RL_MW3TeV" : {
        'right_rl_signal' : ['pp_rl_3TeV.root',2.156E-03,10000],
    },
    "Right_RL_MW3.5TeV" : {
        'right_rl_signal' : ['pp_rl_3.5TeV.root',9.709E-04,10000],
    },
    "Right_RL_MW4TeV" : {
       'right_rl_signal' : ['pp_rl_4TeV.root',4.371E-04,10000],
    },
    "Right_RL_MW4.5TeV" : {
        'right_rl_signal' : ['pp_rl_4.5TeV.root',1.955E-04,10000],
    },
    "Right_RL_MW5TeV" : {
        'right_rl_signal' : ['pp_rl_5TeV.root',8.709E-05,10000],
    },
    "Right_RL_MW5.5TeV" : {
        'right_rl_signal' : ['pp_rl_5.5TeV.root',3.914E-05,10000],
    },
    "Right_RL_MW6TeV" : {
        'right_rl_signal' : ['pp_rl_6TeV.root',1.787E-05,10000],
    },
    "Right_RL_MW6.5TeV" : {
        'right_rl_signal' : ['pp_rl_6.5TeV.root',8.512E-06,10000],
    },
    "Right_RL_MW7TeV" : {
        'right_rl_signal' : ['pp_rl_7TeV.root',4.343E-06,10000],
    }


}

Background_dict = {
    "TTbarsamples" : {
        'sm_bkg_TTbar_1' :['pp_ttbar_1.root', 61.9,1000000],
        'sm_bkg_TTbar_2' :['pp_ttbar_2.root', 61.9,1000000],              # TTbar samples
        'sm_bkg_TTbar_3' :['pp_ttbar_3.root', 61.9,1000000],
        'sm_bkg_TTbar_4' :['pp_ttbar_4.root', 61.9,1000000],
        'sm_bkg_TTbar_5' :['pp_ttbar_5.root', 61.9,1000000],
        'sm_bkg_TTbar_6' :['pp_ttbar_6.root', 61.9,1000000],
        'sm_bkg_TTbar_7' :['pp_ttbar_7.root', 61.9,1000000],
        'sm_bkg_TTbar_8' :['pp_ttbar_8.root', 61.9,1000000],
        'sm_bkg_TTbar_9' :['pp_ttbar_9.root', 61.9,1000000],
        'sm_bkg_TTbar_10' :['pp_ttbar_10.root', 61.9,1000000],
        'sm_bkg_TTbar_11' :['pp_ttbar_11.root', 61.9,1000000],
        'sm_bkg_TTbar_12' :['pp_ttbar_12.root', 61.9,1000000],
        'sm_bkg_TTbar_13' :['pp_ttbar_13.root', 61.9,1000000],
        'sm_bkg_TTbar_14' :['pp_ttbar_14.root', 61.9,1000000],
        'sm_bkg_TTbar_15' :['pp_ttbar_15.root', 61.9,1000000],
        'sm_bkg_TTbar_16' :['pp_ttbar_16.root', 61.9,1000000],
        'sm_bkg_TTbar_17' :['pp_ttbar_17.root', 61.9,1000000],
        'sm_bkg_TTbar_18' :['pp_ttbar_18.root', 61.9,1000000],
        'sm_bkg_TTbar_19' :['pp_ttbar_19.root', 61.9,1000000],
    },
    "TTbar1Jetsamples":{
        'sm_bkg_TTbar1Jet_1' :['pp_ttbar_j_1.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_2' :['pp_ttbar_j_2.root', 72.41,1000000],              # TTbar samples
        'sm_bkg_TTbar1Jet_3' :['pp_ttbar_j_3.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_4' :['pp_ttbar_j_4.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_5' :['pp_ttbar_j_5.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_6' :['pp_ttbar_j_6.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_7' :['pp_ttbar_j_7.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_8' :['pp_ttbar_j_8.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_9' :['pp_ttbar_j_9.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_10' :['pp_ttbar_j_10.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_11' :['pp_ttbar_j_11.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_12' :['pp_ttbar_j_12.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_13' :['pp_ttbar_j_13.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_14' :['pp_ttbar_j_14.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_15' :['pp_ttbar_j_15.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_16' :['pp_ttbar_j_16.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_17' :['pp_ttbar_j_17.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_18' :['pp_ttbar_j_18.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_19' :['pp_ttbar_j_19.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_20' :['pp_ttbar_j_20.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_21' :['pp_ttbar_j_21.root', 72.41,1000000],
        'sm_bkg_TTbar1Jet_22' :['pp_ttbar_j_22.root', 72.41,1000000],

       
    },
    "TTbar2Jetsamples":{
        'sm_bkg_TTbar2Jet_1' :['pp_ttbar_jj_1.root', 54.64,1000000],
        'sm_bkg_TTbar2Jet_2' :['pp_ttbar_jj_2.root', 54.64,1000000],              # TTbar samples
        'sm_bkg_TTbar2Jet_3' :['pp_ttbar_jj_3.root', 54.64,1000000],
        'sm_bkg_TTbar2Jet_4' :['pp_ttbar_jj_4.root', 54.64,1000000],
        'sm_bkg_TTbar2Jet_5' :['pp_ttbar_jj_5.root', 54.64,1000000],
        'sm_bkg_TTbar2Jet_6' :['pp_ttbar_jj_6.root', 54.64,1000000],
        'sm_bkg_TTbar2Jet_7' :['pp_ttbar_jj_7.root', 54.64,1000000],
        'sm_bkg_TTbar2Jet_8' :['pp_ttbar_jj_8.root', 54.64,1000000],
        'sm_bkg_TTbar2Jet_9' :['pp_ttbar_jj_9.root', 54.64,1000000],
        'sm_bkg_TTbar2Jet_10' :['pp_ttbar_jj_10.root', 54.64,1000000],
        'sm_bkg_TTbar2Jet_11' :['pp_ttbar_jj_11.root', 54.64,1000000],
        'sm_bkg_TTbar2Jet_12' :['pp_ttbar_jj_12.root', 54.64,1000000],
        'sm_bkg_TTbar2Jet_13' :['pp_ttbar_jj_13.root', 54.64,1000000],
        'sm_bkg_TTbar2Jet_14' :['pp_ttbar_jj_14.root', 54.64,1000000],
        'sm_bkg_TTbar2Jet_15' :['pp_ttbar_jj_15.root', 54.64,1000000],
        'sm_bkg_TTbar2Jet_16' :['pp_ttbar_jj_16.root', 54.64,1000000],
        
       
    },

     "Dibosonsamples" : {
         'sm_bkg_WZ' : ['pp_wz.root',0.06169,18507],
         'sm_bkg_ZH' : ['pp_zh.root',0.01353,4059],
         'sm_bkg_ZZ' : ['pp_zz.root',0.3652,109560],
    },
    "Wplus1JetToTauNusamples" : {
        # 'sm_bkg_WJ_1': ['pp_wj_tavt_1.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_2': ['pp_wj_tavt_2.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_3': ['pp_wj_tavt_3.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_4': ['pp_wj_tavt_4.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_5': ['pp_wj_tavt_5.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_6': ['pp_wj_tavt_6.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_7': ['pp_wj_tavt_7.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_8': ['pp_wj_tavt_8.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_9': ['pp_wj_tavt_9.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_10':['pp_wj_tavt_10.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_11':['pp_wj_tavt_11.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_12':['pp_wj_tavt_12.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_13':['pp_wj_tavt_13.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_14':['pp_wj_tavt_14.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_15':['pp_wj_tavt_15.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_16':['pp_wj_tavt_16.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_17':['pp_wj_tavt_17.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_18':['pp_wj_tavt_18.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_19':['pp_wj_tavt_19.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_20':['pp_wj_tavt_20.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_21':['pp_wj_tavt_21.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_22':['pp_wj_tavt_22.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_23':['pp_wj_tavt_23.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_24':['pp_wj_tavt_24.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_25':['pp_wj_tavt_25.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_26':['pp_wj_tavt_26.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_27':['pp_wj_tavt_27.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_28':['pp_wj_tavt_28.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_29':['pp_wj_tavt_29.root', 106.6/100, 1000000],
        # 'sm_bkg_WJ_30':['pp_wj_tavt_30.root', 106.6/100, 1000000],
        'sm_bkg_WJ_31':['pp_wj_tavt_31.root', 106.6/100, 1000000],
        'sm_bkg_WJ_32':['pp_wj_tavt_32.root', 106.6/100, 1000000],
    },

   "Wplus2JetToTauNusamples" : {
        'sm_bkg_WJJ_1':['pp_wjj_tavt_1.root', 38.75/100, 1000000],
        'sm_bkg_WJJ_2':['pp_wjj_tavt_2.root', 38.75/100, 1000000],
        'sm_bkg_WJJ_3':['pp_wjj_tavt_3.root', 38.75/100, 1000000],
        'sm_bkg_WJJ_4':['pp_wjj_tavt_4.root', 38.75/100, 1000000],
        'sm_bkg_WJJ_5':['pp_wjj_tavt_5.root', 38.75/100, 1000000],
        'sm_bkg_WJJ_6':['pp_wjj_tavt_6.root', 38.75/100, 1000000],
        'sm_bkg_WJJ_7':['pp_wjj_tavt_7.root', 38.75/100, 1000000],
        'sm_bkg_WJJ_8':['pp_wjj_tavt_8.root', 38.75/100, 1000000],
        'sm_bkg_WJJ_9':['pp_wjj_tavt_9.root', 38.75/100, 1000000],
        'sm_bkg_WJJ_10':['pp_wjj_tavt_10.root', 38.75/100, 1000000],
        'sm_bkg_WJJ_11':['pp_wjj_tavt_11.root', 38.75/100, 1000000],
        'sm_bkg_WJJ_12':['pp_wjj_tavt_12.root', 38.75/100, 1000000],
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


}

RL_Background_dict = {
    # "Right_RR_MW3TeV" :{'right_rl_bkg' : ['pp_rl_3TeV.root',2.156E-03,10000]},
    # "Right_RR_MW3.5TeV" :{'right_rl_bkg' : ['pp_rl_3.5TeV.root',9.709E-04,10000]},
    # "Right_RR_MW4TeV" :{'right_rl_bkg' : ['pp_rl_4TeV.root',4.370E-04,10000]},
    # "Right_RR_MW4.5TeV" :{'right_rl_bkg' : ['pp_rl_4.5TeV.root',1.955E-04,10000]},
    "Right_RR_MW5TeV" :{'right_rl_bkg' : ['pp_rl_5TeV.root',8.709E-05,10000]},
    # "Right_RR_MW5.5TeV" :{'right_rl_bkg' : ['pp_rl_5.5TeV.root',3.914E-05,10000]},
    # "Right_RR_MW6TeV" :{'right_rl_bkg' : ['pp_rl_6TeV.root',1.787E-05,10000]},
    # "Right_RR_MW6.5TeV" :{'right_rl_bkg' : ['pp_rl_6.5TeV.root',8.512E-06,10000]},
    # "Right_RR_MW7TeV" :{'right_rl_bkg' : ['pp_rl_7TeV.root',4.343E-06,10000]},
    
}






