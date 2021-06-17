import sys, os
sys.path.append(os.path.join(sys.path[0], 'paper_version', 'psi'))

from poly_multi import Poly_Multi_HSIC
from hsic_lasso_psi import Split_HSIC_Lasso
import numpy as np
import pandas as pd

"""Evaluation of the 'Communities and Crime' dataset"""

# Column names for data (taken from UCI Repository)
column_names = ['population', 'householdsize', 'racepctblack',
                'racePctWhite', 'racePctAsian', 'racePctHisp',
                'agePct12t21', 'agePct12t29', 'agePct16t24',
                'agePct65up', 'numbUrban', 'pctUrban', 'medIncome',
                'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec',
                'pctWPubAsst', 'pctWRetire', 'medFamInc', 'perCapInc',
                'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap',
                'OtherPerCap', 'HispPerCap', 'NumUnderPov', 'PctPopUnderPov',
                'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore',
                'PctUnemployed', 'PctEmploy', 'PctEmplManu', 'PctEmplProfServ',
                'PctOccupManu', 'PctOccupMgmtProf', 'MalePctDivorce',
                'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv', 'PersPerFam',
                'PctFam2Par', 'PctKids2Par', 'PctYoungKids2Par', 'PctTeen2Par',
                'PctWorkMomYoungKids', 'PctWorkMom', 'NumIlleg', 'PctIlleg',
                'NumImmig', 'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8',
                'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5', 'PctRecImmig8',
                'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell',
                'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous',
                'PersPerOwnOccHous', 'PersPerRentOccHous', 'PctPersOwnOccup',
                'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant',
                'PctHousOccup', 'PctHousOwnOcc', 'PctVacantBoarded',
                'PctVacMore6Mos', 'MedYrHousBuilt', 'PctHousNoPhone',
                'PctWOFullPlumb', 'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart',
                'RentLowQ', 'RentMedian', 'RentHighQ', 'MedRent',
                'MedRentPctHousInc', 'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg',
                'NumInShelters', 'NumStreet', 'PctForeignBorn', 'PctBornSameState',
                'PctSameHouse85', 'PctSameCity85', 'PctSameState85',
                'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps',
                'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop',
                'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol',
                'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian',
                'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz',
                'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans',
                'PolicCars', 'PolicOperBudg', 'LemasPctPolicOnPatr',
                'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop',
                'ViolentCrimesPerPop']

# Reading in data
df = pd.read_csv('Datasets/communities.csv', sep = ',',
                     header = None)
df = df.iloc[:, 5:] # first 5 columns non-predictive
df = df.replace('?', np.NaN) # changing '?' to nan
df.columns = column_names # adding column names
n, p = df.shape # n = 1994, p = 128


# Dataset with complete data points only: n = 319, p = 122
df_comp = df.dropna(axis = 'index')
df_comp = df_comp.sample(frac=1) # shuffling data
Y_comp = df_comp.loc[:,'ViolentCrimesPerPop'].values
Y_comp = Y_comp.astype(float)
X_comp = df_comp.iloc[:,:-1].values
X_comp = X_comp.astype(float)


# Dataset where columns with missing values are deleted:
# n = 1993, p = 100
# (As exception, we keep one column with one missing value
# and delete the corresponding data point.)
df_miss = (df.dropna(axis = 1, thresh = 1950)).dropna(axis = 0)
df_miss = df_miss.sample(frac=1) # shuffling data
Y_miss = df_miss.loc[:,'ViolentCrimesPerPop'].values
Y_miss = Y_miss.astype(float)
X_miss = df_miss.iloc[:,:-1].values
X_miss = X_miss.astype(float)


# Models - complete dataset
targets = ['partial', 'H']
alpha = 0.05
split_comp_block10 = Split_HSIC_Lasso(targets, split_ratio = 0.25,
                                      n_screen = 122, adaptive_lasso = False,
                                      cv = True, cov_mode = 'oas',
                                      M_estimator = 'unbiased', H_estimator = 'block',
                                      H_B = 10, discrete_output = False)
multi_comp_block10 = Poly_Multi_HSIC(n_select = 10, poly = False, estimator = 'block',
                                     B = 10, discrete_output = False)
split_comp_block5 = Split_HSIC_Lasso(targets, split_ratio = 0.25,
                                     n_screen = 122, adaptive_lasso = False,
                                     cv = True, cov_mode = 'oas',
                                     M_estimator = 'unbiased', H_estimator = 'block',
                                     H_B = 5, discrete_output = False)
multi_comp_block5 = Poly_Multi_HSIC(n_select = 10, poly = False, estimator = 'block',
                                    B = 5, discrete_output = False)

split_comp_inc1 = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 122,
                                   adaptive_lasso = False, cv = True, cov_mode = 'oas',
                                   M_estimator = 'unbiased', H_estimator = 'inc',
                                   H_l = 1, discrete_output = False)
multi_comp_inc1 = Poly_Multi_HSIC(n_select = 10, poly = False, estimator = 'block',
                                  l = 1, discrete_output = False)
split_comp_inc5 = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 122,
                                   adaptive_lasso = False, cv = True, cov_mode = 'oas',
                                   M_estimator = 'unbiased', H_estimator = 'inc',
                                   H_l = 5, discrete_output = False)
multi_comp_inc5 = Poly_Multi_HSIC(n_select = 10, poly = False, estimator = 'block',
                                  l = 5, discrete_output = False)
split_comp_inc10 = Split_HSIC_Lasso(targets, split_ratio = 0.25, n_screen = 122,
                                    adaptive_lasso = False, cv = True, cov_mode = 'oas',
                                    M_estimator = 'unbiased', H_estimator = 'inc',
                                    H_l = 10, discrete_output = False)
multi_comp_inc10 = Poly_Multi_HSIC(n_select = 10, poly = False, estimator = 'block',
                                   l = 10, discrete_output = False)
models_comp = [split_comp_block10, multi_comp_block10, split_comp_block5,
               multi_comp_block5, split_comp_inc1, multi_comp_inc1,
               split_comp_inc5, multi_comp_inc5, split_comp_inc10, multi_comp_inc10]
model_names_comp = ['split_comp_block10', 'multi_comp_block10', 'split_comp_block5',
                    'multi_comp_block5', 'split_comp_inc1', 'multi_comp_inc1',
                    'split_comp_inc5', 'multi_comp_inc5', 'split_comp_inc10', 'multi_comp_inc10']


# Inference complete dataset
H0 = np.ones(122) # dummy variable
M0 = np.eye(122) # dummy variable
results_comp = []
for m in models_comp:
    res = m.sel_inf(X_comp, Y_comp, 'test', alpha, None, H0, M0,
                    unbiased_parallel = True, n_jobs = 30)
    results_comp.append(res)

# Printing results for complete dataset
for name, r in zip(model_names_comp, results_comp):
    print(name)
    r.print_summary()
    print('##########################################')


# Models - missing dataset
split_miss_block10 = Split_HSIC_Lasso(targets, split_ratio = 0.2,
                                      n_screen = 100, adaptive_lasso = False,
                                      cv = True, cov_mode = 'oas',
                                      M_estimator = 'unbiased', H_estimator = 'block',
                                      H_B = 10, discrete_output = False)
multi_miss_block10 = Poly_Multi_HSIC(n_select = 10, poly = False, estimator = 'block',
                                     B = 10, discrete_output = False)
split_miss_block5 = Split_HSIC_Lasso(targets, split_ratio = 0.2,
                                     n_screen = 100, adaptive_lasso = False,
                                     cv = True, cov_mode = 'oas',
                                     M_estimator = 'unbiased', H_estimator = 'block',
                                     H_B = 5, discrete_output = False)
multi_miss_block5 = Poly_Multi_HSIC(n_select = 10, poly = False, estimator = 'block',
                                    B = 5, discrete_output = False)
split_miss_block20 = Split_HSIC_Lasso(targets, split_ratio = 0.2,
                                      n_screen = 100, adaptive_lasso = False,
                                      cv = True, cov_mode = 'oas',
                                      M_estimator = 'unbiased', H_estimator = 'block',
                                      H_B = 20, discrete_output = False)
multi_miss_block20 = Poly_Multi_HSIC(n_select = 10, poly = False, estimator = 'block',
                                     B = 20, discrete_output = False)


split_miss_inc1 = Split_HSIC_Lasso(targets, split_ratio = 0.2,
                                   n_screen = 100, adaptive_lasso = False,
                                   cv = True, cov_mode = 'oas',
                                   M_estimator = 'unbiased', H_estimator = 'inc',
                                   H_l = 1, discrete_output = False)
multi_miss_inc1 = Poly_Multi_HSIC(n_select = 10, poly = False, estimator = 'block',
                                  l = 1, discrete_output = False)
split_miss_inc5 = Split_HSIC_Lasso(targets, split_ratio = 0.2,
                                   n_screen = 100, adaptive_lasso = False,
                                   cv = True, cov_mode = 'oas',
                                   M_estimator = 'unbiased', H_estimator = 'inc',
                                   H_l = 5, discrete_output = False)
multi_miss_inc5 = Poly_Multi_HSIC(n_select = 10, poly = False, estimator = 'block',
                                  l = 5, discrete_output = False)
split_miss_inc10 = Split_HSIC_Lasso(targets, split_ratio = 0.2,
                                    n_screen = 100, adaptive_lasso = False,
                                    cv = True, cov_mode = 'oas',
                                    M_estimator = 'unbiased', H_estimator = 'inc',
                                    H_l = 10, discrete_output = False)
multi_miss_inc10 = Poly_Multi_HSIC(n_select = 10, poly = False, estimator = 'block',
                                   l = 10, discrete_output = False)
split_miss_inc20 = Split_HSIC_Lasso(targets, split_ratio = 0.2,
                                    n_screen = 100, adaptive_lasso = False,
                                    cv = True, cov_mode = 'oas',
                                    M_estimator = 'unbiased', H_estimator = 'inc',
                                    H_l = 20, discrete_output = False)
multi_miss_inc20 = Poly_Multi_HSIC(n_select = 10, poly = False, estimator = 'block',
                                   l = 20, discrete_output = False)

models_miss = [split_miss_block10, multi_miss_block10, split_miss_block5,
               multi_miss_block5, split_miss_inc1, multi_miss_inc1,
               split_miss_inc5, multi_miss_inc5, split_miss_inc10, multi_miss_inc10]
model_names_miss = ['split_miss_block10', 'multi_miss_block10', 'split_miss_block5',
                    'multi_miss_block5', 'split_miss_inc1', 'multi_miss_inc1',
                    'split_miss_inc5', 'multi_miss_inc5', 'split_miss_inc10', 'multi_miss_inc10']


# Inference missing dataset
H0 = np.ones(100) # dummy variable
M0 = np.eye(100) # dummy variable
results_miss = []
for m in models_miss:
    res = m.sel_inf(X_miss, Y_miss, 'test', alpha, None, H0, M0,
                    unbiased_parallel = True, n_jobs = 30)
    results_miss.append(res)

# Printing results for missing dataset
for name, r in zip(model_names_miss, results_miss):
    print(name)
    r.print_summary()
    print('##########################################')
