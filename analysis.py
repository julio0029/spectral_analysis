#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'''
-------------------------------------------------------------------------------
Copyright© 2021 Jules Devaux, Alice Harford, Tony Hickey. All Rights Reserved
Open Source script under Apache License 2.0
-------------------------------------------------------------------------------
'''
import os
current_path = os.path.dirname(os.path.abspath(__file__))
from global_variables import CYCLES
import pandas as pd
import numpy as np
import datetime
import json
from matplotlib import pyplot as plt


#========= PARAMETERS =========
DATAPATH=f"{current_path}"
FISHES=['PW1', 'PW3', 'PW6', 'PW9', 'PW10', 'PW18', 'PW20', 'PW22', 'PW23', 'PW28',
       'W76', 'W77', 'W78', 'W79', 'W80', 'W81', 'W85']
FILENAME=None
EXCS=[0,385,400,457,650]
EMM=[450,500,550,575,600,650]
# Define main Exc-Emm for each cytochrome 
CYTOCHROMES_ABS={
	'c450':[457, 450],
	'c_1':[400, 450],
	'b':[385,450],
	'aa3_1':[650, 600],
	'Hb_deox':[650,650]
}
FLUORESCENCE={
	'nadh':[385,450],
	'fad':[385, 550],
	'fmn':[457,[500, 550]], #525 is the peak for FMN, so get the average between 500-550.
}
#=============================


def exctract_file(_path):
	_fish=_path.split('/')[-2]
	with open(_path) as f:
		_str=f.read()

	_l=[f"{i}"+'}'.replace("'","") for i in _str.split('}')]
	_f=[]
	for i in _l:
		_dict={}
		i=i.replace('{','').replace('}','')
		for j in i.split(','):
			try:_dict.update({str(j.split(':')[0]): j.split(':')[1]})
			except:pass
		_f.append(_dict)

	df=pd.DataFrame(_f).dropna().astype('float64')
	# rename columns, expecting EMM - 'exc' as columns
	df.columns=EMM+['exc']
	df['cycle']=df.index
	df['fish']=_fish

	# Add temperature based on 'global.CYCLE'
	_idxs=[k for k,_ in CYCLES[_fish].items()]
	_temps=[v for _,v in CYCLES[_fish].items()]
	df['temperature']=float('nan')
	df.loc[_idxs,'temperature']=_temps
	df['temperature']=df['temperature'].interpolate(method='linear', limit_direction='forward', axis=0)

	# Select only experimental portion
	df=df.loc[(df['cycle']>=_idxs[0]) & (df['cycle']<=_idxs[-1])]

	return df


def seperate_exc_df(df, bckg_correction=True):
	'''
	Seperates the main df to emmission df for each EXCS
	If bckg_correction==True: removes background signal
	Returns a dict as {excs: emmission df}
	'''
	# Group df by excitation
	excs={nm: d.set_index('temperature') for nm, d in df.groupby(df['exc'])}

	# Remove background
	# only selecting emmission columns
	if bckg_correction:
		dark = excs[0].loc[:, EMM]
		for nm, _df in excs.items():
			bckg=dark.iloc[:len(excs[nm])].values
			excs[nm].loc[:,EMM]=(excs[nm].loc[:,EMM]-bckg)#/bckg

	return excs


def graph(dfs):
	'''
	wrapper function to plot all spectra
	EXCS in rows, EMM in columnsç
	dfs as {nm_exc: emmission df}
		- df cycle index and Emm columns
	'''

	_EXCS=EXCS[1:]
	_DATA={}

	fig, ax= plt.subplots(len(_EXCS), len(EMM))

	# Title as wrasse number
	fig.suptitle(f"{dfs[0]['fish'].iloc[1]}", fontsize=12, fontweight='bold')
	fig.supxlabel('Temperature ºC', fontsize=11, fontweight='bold')
	fig.supylabel('Excitation', fontsize=11, fontweight='bold')
	plt.text(.5, 1, 'Emmission', fontsize=11, transform=fig.transFigure, horizontalalignment='center', fontweight='bold')

	# treat each row excitation
	for i in range(len(_EXCS)):
		exc=_EXCS[i]
		_df=dfs[exc].loc[:,EMM]

		# each column emmision
		for j in range(len(_df.columns)):
			emm=_df.columns[j]

			#_df[emm]=_df[emm].ewm(span=20).mean()
			
			ax[i,j].plot(_df[emm])
			# add heart_failure
			ax[i,j].vlines(36, ymin=_df[emm].min(), ymax=_df[emm].max(), linestyles='--', label='heart failure',color='red')


			if i==0:
				ax[i,j].set_title(f"{emm} nm", fontsize=11, fontweight='bold')
			if j==0:
				ax[i,j].set_ylabel(f"{exc} nm", fontsize=11, fontweight='bold')
			elif j!=0:
				ax[i,j].set_yticklabels([])


	plt.subplots_adjust(wspace=0, hspace=0)
	plt.show()


def absorption(excs, exc_emm):
	# Calculate the absorption
	denom=(excs[exc_emm[0]][exc_emm[1]].values[1]-excs[0][exc_emm[1]].iloc[:20].mean())
	if denom==0: denom=1
	_abs=-np.log10((excs[exc_emm[0]][exc_emm[1]].values)/denom)

	# Assign temperature index
	_abs=pd.DataFrame(_abs, index=excs[exc_emm[0]].index).sort_index()

	# Resample for each temperature using the datetime resampling fct from pandas. Much simpler.
	_abs.index=pd.to_datetime(_abs.index, unit='s') #
	_abs=_abs.resample('1s').mean(numeric_only=True)
	_abs.index=_abs.index.second
	
	return _abs



def main():

	files=[]
	for fish in FISHES:
		for file in os.listdir(f"{DATAPATH}/{fish}/"):
			if 'Rasp' in file:
				files.append(f"{DATAPATH}/{fish}/{file}")

	# final df with keys being each cytochrome,
	# values being df with row as temperatures and col as individual
	fdf={i:[] for i, _ in CYTOCHROMES_ABS.items()}
	fdf.update({i:[] for i, _ in FLUORESCENCE.items()})

	for file in files:
		try:
			df=exctract_file(file)
			excs=seperate_exc_df(df, bckg_correction=False)

			# Calculate absorption for each cytochrom
			for cytochrome, emms in CYTOCHROMES_ABS.items():
				_data=absorption(excs, emms)
				_data.columns=[df['fish'].values[0]]
				fdf[cytochrome].append(_data)

			for cytochrome, exc_emm in FLUORESCENCE.items():

				if type(exc_emm[1]) is not list:
					_data=excs[exc_emm[0]][exc_emm[1]].values-excs[0][exc_emm[1]].iloc[1:].values

				else:
					_data=(excs[exc_emm[0]][exc_emm[1][0]]+excs[exc_emm[0]][exc_emm[1][1]]).mean()-excs[0][exc_emm[1][0]].iloc[1:].values
					# Assign temperature index
				_data=pd.DataFrame(_data, index=excs[exc_emm[0]].index).sort_index()
				# Resample for each temperature using the datetime resampling fct from pandas. Much simpler.
				_data.index=pd.to_datetime(_data.index, unit='s') #
				_data=_data.resample('1s').mean(numeric_only=True)
				_data.index=_data.index.second

				_data.columns=[df['fish'].values[0]]
				fdf[cytochrome].append(_data)

		except Exception as e:print(e)

	# Save all data into an excel spreadsheet
	# Tabs as cytochromes, row= temperatures, columns = fish number.
	# Create an Excel writer
	writer = pd.ExcelWriter(f"{current_path}/cytochromes_data.xlsx", engine='xlsxwriter')

	# Iterate over cytochromes and add data to corresponding sheets
	for cyto, fish_data in fdf.items():
		#print(fish_data)
		try:
			df=pd.concat(fish_data, axis=1, ignore_index=False)
			df.interpolate(method='linear', limit_direction='backward', inplace=True)
			df.to_excel(writer, sheet_name=cyto, index=False)
		except:pass

	writer.save()


if __name__=='__main__':
	main()
