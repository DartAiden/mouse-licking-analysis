import pandas as pd
lickdfs = [
'T28_250429_lick_Video_licks.csv',
'T29_250429_lick_Video_licks.csv',
'T30_250429_lick_Video_licks.csv',
'T31_250429_lick_Video_licks.csv',
'T33_250429_lick_Video_licks.csv',
'T34_250429_lick_Video_licks.csv',
'T35_250429_lick_Video_licks.csv',
'T36_250429_lick_Video_licks.csv',
'T37_250429_lick_Video_licks.csv',
'T38_250429_lick_Video_licks.csv',
'T51_250429_lick_Video_licks.csv',
'T55_250429_lick_Video_licks.csv',
'T58_250429_lick_Video_licks.csv',
'T62_250429_lick_Video_licks.csv',
'T63_250429_lick_Video_licks.csv',
'T64_250429_lick_Video_CROP_licks.csv',
'T65_250429_lick_Video_licks.csv',

]
stimdfs = [
'T28_250429_lick_Video_stims.csv',
'T29_250429_lick_Video_stims.csv',
'T30_250429_lick_Video_stims.csv',
'T31_250429_lick_Video_stims.csv',
'T33_250429_lick_Video_stims.csv',
'T34_250429_lick_Video_stims.csv',
'T35_250429_lick_Video_stims.csv',
'T36_250429_lick_Video_stims.csv',
'T37_250429_lick_Video_stims.csv',
'T38_250429_lick_Video_stims.csv',
'T51_250429_lick_Video_stims.csv',
'T55_250429_lick_Video_stims.csv',
'T58_250429_lick_Video_stims.csv',
'T62_250429_lick_Video_stims.csv',
'T63_250429_lick_Video_stims.csv',
'T64_250429_lick_Video_CROP_stims.csv',
'T65_250429_lick_Video_stims.csv',
]

for i in range(len(lickdfs)):
    temptitle = lickdfs[i].replace('licks','complete')
    lickdf = pd.read_csv(lickdfs[i])
    stimdf = pd.read_csv(stimdfs[i])
    temp = stimdf['Signal']
    lickdf.rename(columns={'Signal' : 'Lick_Signal'})
    temp = stimdf['Signal']
    lickdf['Stim_Signal'] = temp
    lickdf.to_csv(temptitle, index = False)
