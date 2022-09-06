variables = ['weight','tau1_vis_pt','met','CosTheta','LeadChPtOverTauPt','DeltaPtOverPt','mT']
mva_variables = ['tau1_vis_pt','met','CosTheta','LeadChPtOverTauPt','DeltaPtOverPt','mT']

signal_dict  = {
    "Left" : {
        'left_signal'  : ['pp_w2l_w2_4.root',0.004165,10000]
    },
    "Left_MW3TeV" : {
        'left_3_signal' :['pp_w2l_w2_3.root',0.02166,10000],
        
    },
    "Left_MW3.5TeV" : {
        'left_3.5_signal' :['pp_w2l_w2_3.5.root',0.009345,10000],
    },
    "Left_MW4TeV" : {
        'left_4_signal' :['pp_w2l_w2_4.root',0.004172,10000],
    },
    "Left_MW4.5TeV" : {
        'left_4.5_signal' :['pp_w2l_w2_4.5.root',0.00191,10000],
    },
    "Left_MW5TeV" : {
        'left_5_signal' :['pp_w2l_w2_5.root',0.000905,10000],
    },
    "Left_MW5.5TeV" : {
        'left_5.5_signal' :['pp_w2l_w2_5.5.root',0.0004502,10000],
    },
    "Left_MW6TeV" : {
        'left_6_signal' :['pp_w2l_w2_6.root',0.0002387,10000],
    },
    "Left_MW6.5TeV" : {
        'left_6.5_signal' :['pp_w2l_w2_6.5.root',0.000138,10000],
    },
    "Left_MW7TeV" : {
        'left_7_signal' :['pp_w2l_w2_7.root',0.00008707,10000],
    },
    "Left_MW7.5TeV" : {
        'left_7.5_signal' :['pp_w2l_w2_7.5.root',0.000059,10000],
    },
     "Left_MW8TeV" : {
        'left_8_signal' :['pp_w2l_w2_8.root',0.00004249,10000],
    },
     "Left_MW8.5TeV" : {
        'left_8.5_signal' :['pp_w2l_w2_8.5.root',0.0000319,10000],
    },
     "Left_MW9TeV" : {
        'left_9_signal' :['pp_w2l_w2_9.root',0.00002464,10000],
    },
    "Left_MW9.5TeV" : {
        'left_9.5_signal' :['pp_w2l_w2_9.5.root',0.00001947,10000],
    },
     "Left_MW10TeV" : {
        'left_10_signal' :['pp_w2l_w2_10.root',0.00001581,10000],
    },

    "Right" : {

        'right_N0_signal' :['pp_w2r_N_0.root',0.004152,10000],
        'right_N1_signal' :['pp_w2r_N_1.root',0.003286,10000],
    },
    "Right_N0" : {

        'right_N0_signal' :['pp_w2r_N_0.root',0.004152,10000],
    },
    "Right_N1" : {
        'right_N1_signal' :['pp_w2r_N_1.root',0.003286,10000],
    },

    "Right_N0_MW3.0TeV" : {
        'right_N0_3_signal' :['pp_w2r_N_0_w2_3.root',0.02166,10000],
    },
    "Right_N1_MW3.0TeV" : {
        'right_N1_3_signal' :['pp_w2r_N_1_w2_3.root',0.01661,10000],
    },
    "Right_N0_MW3.5TeV" : {
        'right_N0_3.5_signal' :['pp_w2r_N_0_w2_3.5.root',0.009345,10000],
    },
    "Right_N1_MW3.5TeV" : {
        'right_N1_3.5_signal' :['pp_w2r_N_1_w2_3.5.root',0.007414,10000],
    },
    "Right_N0_MW4TeV" : {
        'right_N0_4_signal' :['pp_w2r_N_0_w2_4.root',0.004172,10000],
    },
    "Right_N1_MW4TeV" : {
        'right_N1_4_signal' :['pp_w2r_N_1_w2_4.root',0.003296,10000],
    },
    "Right_N0_MW4.5TeV" : {
        'right_N0_4.5_signal' :['pp_w2r_N_0_w2_4.5.root',0.00191,10000],
    },
    "Right_N1_MW4.5TeV" : {
        'right_N1_4.5_signal' :['pp_w2r_N_1_w2_4.5.root',0.00147,10000],
    },
    "Right_N0_MW5TeV" : {
        'right_N0_5_signal' :['pp_w2r_N_0_w2_5.root',0.000905,10000],
    },
    "Right_N1_MW5TeV" : {
        'right_N1_5_signal' :['pp_w2r_N_1_w2_5.root',0.0006515,10000],
    },
    "Right_N0_MW5.5TeV" : {
        'right_N0_5.5_signal' :['pp_w2r_N_0_w2_5.5.root',0.0004502,10000],
    },
    "Right_N1_MW5.5TeV" : {
        'right_N1_5.5_signal' :['pp_w2r_N_1_w2_5.5.root',0.0002934,10000],
    },
    "Right_N0_MW6TeV" : {
        'right_N0_6_signal' :['pp_w2r_N_0_w2_6.root',0.0002387,10000],
    },
    "Right_N1_MW6TeV" : {
        'right_N1_6_signal' :['pp_w2r_N_1_w2_6.root',0.0001347,10000],
    },
    "Right_N0_MW6.5TeV" : {
        'right_N0_6.5_signal' :['pp_w2r_N_0_w2_6.5.root',0.000138,10000],
    },
    "Right_N1_MW6.5TeV" : {
        'right_N1_6.5_signal' :['pp_w2r_N_1_w2_6.5.root',0.00006383,10000],
    },
    "Right_N0_MW7TeV" : {
        'right_N0_7_signal' :['pp_w2r_N_0_w2_7.root',0.00008707,10000],
    },
    "Right_N1_MW7TeV" : {
        'right_N1_7_signal' :['pp_w2r_N_1_w2_7.root',0.00003297,10000],
    },

 

    "control" :{
        'left_signal'  : ['pp_w2l_w2_4.root',0.004165,10000],
        'right_N0_signal' :['pp_w2r_N_0_w2_4.root',0.004172,10000],
        'right_N1_signal' :['pp_w2r_N_1_w2_4.root',0.003296,10000],
    }
}

background_dict = {
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
        'sm_bkg_DY100to200_1':['ppztata_100-200_1.root',23.33,1000000],
        'sm_bkg_DY100to200_2':['ppztata_100-200_2.root',23.33,1000000],
        'sm_bkg_DY100to200_3':['ppztata_100-200_3.root',23.33,1000000],
        'sm_bkg_DY100to200_4':['ppztata_100-200_4.root',23.33,1000000],
        'sm_bkg_DY100to200_5':['ppztata_100-200_5.root',23.33,1000000],
        'sm_bkg_DY100to200_6':['ppztata_100-200_6.root',23.33,1000000],
        'sm_bkg_DY100to200_7':['ppztata_100-200_7.root',23.33,1000000],
        'sm_bkg_DY200to400':['ppztata_200-400.root',0.5316,160000],
        'sm_bkg_DY400to500':['ppztata_400-500.root',0.02907,9000],  # Drell-Yan samples
        'sm_bkg_DY500to700':['ppztata_500-700.root',0.01733,5200],
        'sm_bkg_DY700to800':['ppztata_700-800.root',0.002861,100],
        'sm_bkg_DY800to1000':['ppztata_800-1000.root',0.002492,100],
        'sm_bkg_DY1000to1500':['ppztata_1000-1500.root',0.001392,100],
        'sm_bkg_DY1500to2000':['ppztata_1500-2000.root',0.0002012,100],
        'sm_bkg_DY2000to3000':['ppztata_2000-3000.root',0.00005268,100],

    },

    "WToTauNusamples" : {
        'sm_bkg_WToTauNuM600' : ['ppwtavt_600.root',0.1101,33000],
        'sm_bkg_WToTauNuM800' : ['ppwtavt_800.root',0.03553,10000],
        'sm_bkg_WToTauNuM1000' : ['ppwtavt_1000.root',0.01407,4000],
        'sm_bkg_WToTauNuM2000' : ['ppwtavt_2000.root',0.0004796,150],
        'sm_bkg_WToTauNuM3000' : ['ppwtavt_3000.root',0.00003819,10],
        'sm_bkg_WToTauNuM4000' : ['ppwtavt_4000.root',0.000004067,5],
    },
    "MISIDsamples" : {
        'sm_bkg_WToTauNuM600' : ['ppwtavt_600.root',0.1101*0.622,33000],
        'sm_bkg_WToTauNuM800' : ['ppwtavt_800.root',0.03553*0.622,10000],
        'sm_bkg_WToTauNuM1000' : ['ppwtavt_1000.root',0.01407*0.622,4000],
        'sm_bkg_WToTauNuM2000' : ['ppwtavt_2000.root',0.0004796*0.622,150],
        'sm_bkg_WToTauNuM3000' : ['ppwtavt_3000.root',0.00003819*0.622,10],
        'sm_bkg_WToTauNuM4000' : ['ppwtavt_4000.root',0.000004067*0.622,5],
    },

}








