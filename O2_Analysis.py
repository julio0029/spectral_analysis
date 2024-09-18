import os
import pandas as pd
import numpy as np

CURRENT_PATH=os.path.dirname(os.path.abspath(__file__))

# PARAMETERS
flow_rate = 3.5 #mL.min-1
HEART_MASS={
        "PW3":174,
        "PW9":570,
        "PW10":665,
        "PW18":400,
        "PW20":407,
        "PW22":473,
        "PW23":310,
        "PW28":220,
        '74':375.7,
        '76':262.1,
        '77':374.3,
        '78':223.7,
        '79':593.3,
        '80':550,
        '81':273.8,
        '85':350
    }

def O2_pct_to_M(O2_pct, temperature):
    '''
    O2_pct and temperature are expected to be same length arrays.

    Vapour pressure of water: https://www.wiredchemist.com/chemistry/data/vapor-pressure
    Absorption coefficient slope: https://www.microelectrodes.com/_files/ugd/f659cb_f2718895f6c54576a2a436831377f0d6.pdf
    Formula for conversion of percent oxygen to solubility in moles/liter:
    S = (a/22.414) x (760-P)/760) x (r%/100)
    S = solubility of gas in moles per liter
    a = absorption coefficient of gas at temperature
    P = vapor pressure of water at temperature
    r% = actual reading in percent Oxygen
    '''
    vap_pressure_slope=1.825 # vap_pres_coef = 1.825 * TºC - 22.509
    vap_pressure_inter=-22.509

    abs_coef_slope=-0.000404791  # abs_coef = -0.000404791 * TºC + 0.038585149
    abs_coef_inter=0.038585149

    return (((abs_coef_slope*temperature)+abs_coef_inter)/22.414)*((760-((vap_pressure_slope*temperature)+vap_pressure_inter))/760)*(O2_pct)


# Caluclate 100% O2 content to mol/L
Temp=np.arange(20,41,1)
Y=np.array([1]*len(Temp))
air_cal=O2_pct_to_M(Y, Temp)

# Read O2 values recorded from LabChart !
df=pd.read_csv(f"{CURRENT_PATH}/O2.csv").dropna(how='all', axis=0)

# Reverses the current (i.e. low V = low O2)
df.loc[:,['O2']]=abs(df.loc[:,['O2']].astype('float64').values)

fdf=[] # List to append each analysis for latter concat

for wrasse, wdf in df.groupby(df['Wrasse']):
    # normalise 0-100%
    # 100% = first value (air_sat); 0% = 0.360V sa determined in background expe
    _100_pct = wdf['O2'].max()+0.0036
    _zero_pct = _100_pct - 0.36 # -0.36 mV being the difference between air_sat - 0%
    wdf['O2_calib'] = (wdf.loc[:,'O2'] - _zero_pct) / (_100_pct-_zero_pct)
    wdf=wdf.set_index('Temp').sort_index()


    # Resample for each temperature using the datetime resampling fct from pandas. Much simpler.
    wdf.index=pd.to_datetime(wdf.index, unit='s')
    wdf=wdf.resample('1s').mean(numeric_only=True).interpolate(method='linear', limit_direction='forward', axis=0)
    wdf.index=wdf.index.second
    

    # Convert O2 pct to millimol per litre:
    wdf['O2_M']=O2_pct_to_M(wdf['O2_calib'].values, wdf.index.values)

    # Calculate O2 consumption rate in pmolO2 /(s*mg)
    wdf['JO2'] = ((flow_rate/1000)*abs(wdf['O2_M'].diff())*1000000000000)/(HEART_MASS[wrasse]*60)
    fdf.append(wdf['JO2'].rename(wrasse))

# Combine into one df and save
fdf=pd.concat(fdf, axis=1, ignore_index=False).fillna(method='bfill')
fdf.to_csv(f'{CURRENT_PATH}/Heart_JO2.csv')
