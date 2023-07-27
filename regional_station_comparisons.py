import gc
import glob
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Define the directory containing the CSV files
data_dir = '../data/supermag/'

# Define the degree grid
mlt_min = 0
mlt_max = 24
mlt_step = (1/60)

# defining the twins era start and end times
twins_start = pd.to_datetime('2010-01-01')
twins_end = pd.to_datetime('2017-12-31')
twins_time_period = pd.date_range(start=twins_start, end=twins_end, freq='min')


def load_feather_file(filepath):
	'''
	Load a feather file into a pandas data frame.
	'''
	df = pd.read_feather(filepath)

	return df


def filter_data(df, mlt_min_bin, mlt_max_bin):
	'''
	Filter a pandas data frame to MLT bins.
	'''

	df = df[(df['MLT'] >= mlt_min_bin) & (df['MLT'] < mlt_max_bin)]

	return df


def process_file(df, mlt_min_bin, mlt_max_bin):
	'''
	Process a single feather file and return a filtered data frame for the degree bins.
	'''

	df_filtered = filter_data(df, mlt_min_bin, mlt_max_bin)
	df_filtered.reset_index(inplace=True, drop=True)

	return df_filtered

def calculate_max_rsd(dbdt_df):
	'''
	takes in the dbdt values df for the stations in the region and calculates
	the max RSD values. Includes a column in the final dataframe that labels
	which station was responsible for the max rsd at each time step.

	Args:
		dbdt_df (pd.dataframe): contains the dbdt information for each time step for each station in the region

	Returns:
		pd.dataframe: column containing the max rsd values for each time step and another
						column labeling the station responsible for the max value
	'''

	rsd_df = pd.DataFrame(index=dbdt_df.index)
	max_df = pd.DataFrame(index=dbdt_df.index)

	for station in dbdt_df.columns:
		ss = dbdt_df[station]
		temp_df = dbdt_df.drop(station,axis=1)
		ra = temp_df.mean(axis=1)
		rsd_df[f'{station}'] = ss-ra

	max_df['max_rsd'] = rsd_df.max(axis=1)
	max_df['max_rsd_station'] = rsd_df.idxmax(axis=1)

	return max_df


def process_directory(stations_dict, data_dir, mlt_min, mlt_max, mlt_step):
	'''
	Process all feather files in a directory and return a list
	of filtered data frames for each degree bins.
	'''
	stats_df = {}
	for region in stations_dict.keys():
		print(f'Region: {region}')
		stats_df[region] = {}
		dbdt_df = pd.DataFrame(index=twins_time_period)
		for stats in stations_dict[region]['station']:
			temp_df = pd.DataFrame()
			filepath = os.path.join(data_dir, f'{stats}.feather')
			df = load_feather_file(filepath)
			df.set_index('Date_UTC', inplace=True, drop=False)
			df = df[twins_start:twins_end]
			dbdt_df = pd.concat([dbdt_df, df.copy()['dbht']], axis=1, ignore_index=False)
			dbdt_df.rename(columns={'dbht':stats}, inplace=True)
			stats_df[region][f'{stats}_dates'] = df.copy()['Date_UTC'].dropna()
			for mlt in np.arange(mlt_min, mlt_max, mlt_step):
				mlt_min_bin = mlt
				mlt_max_bin = mlt + mlt_step
				df_filtered = process_file(df, mlt_min_bin, mlt_max_bin)
				if not df_filtered.empty:
					stat = compute_statistics(df_filtered, mlt, twins=True)
					temp_df = pd.concat([temp_df, stat], axis=0, ignore_index=True)

			if temp_df.empty:
				stats_df[region][stats] = pd.DataFrame({'count':np.nan,
														'mean':np.nan,
														'median':np.nan,
														'std':np.nan,
														'max':np.nan},
														index=np.arange(mlt_min, mlt_max, mlt_step))
			else:
				temp_df.set_index("MLT", inplace=True)
				stats_df[region][stats] = temp_df

		rsd_df = calculate_max_rsd(dbdt_df)
		stats_df[region]['max_rsd'] = rsd_df

	return stats_df


def compute_statistics(df_combined, mlt, twins=False):
	'''
	Compute the statistics of the 'dbht' parameter for each degree bins.
	'''

	df_combined = df_combined[df_combined['dbht'].notna()]
	stats_df = pd.DataFrame({'MLT': mlt,
							'count':len(df_combined),
							'mean': df_combined['dbht'].mean(),
							'median':df_combined['dbht'].median(),
							'std': df_combined['dbht'].std(),
							'max':df_combined['dbht'].max()},
							index=[0])

	return stats_df


def extracting_dates(data_dir, stations):

	dates = pd.DataFrame()
	dates['Date_UTC'] = twins_time_period
	dates.set_index('Date_UTC', inplace=True, drop=True)
 	# time_period = pd.Series(range(len(time_period)), index=time_period)
	for i, stat in enumerate(stations):
		df = load_feather_file(data_dir+f'{stat}.feather')
		df.dropna(subset=['dbht'], inplace=True)
		df.reset_index(inplace=True, drop=True)
		date = pd.DataFrame(index=df['Date_UTC'])
		date[f'{stat}_top'] = int(1)*(i+1)
		date[f'{stat}_bottom'] = int(1)*(i+0.1)
		dates = pd.concat([dates, date], ignore_index=False, axis=1)

	dates.fillna(0, inplace=True)

	return dates

def getting_solar_cycle():

	solar = pd.read_json('../data/observed-solar-cycle-indices.json')
	solar['time-tag'] = pd.to_datetime(solar['time-tag'])
	solar.set_index(solar['time-tag'], inplace=True)

	return solar

def getting_geo_location(stat, geo_df):

	temp_df = geo_df[geo_df['IAGA'] == stat]
	lat = temp_df['GEOLAT']
	lon = temp_df['GEOLON']

	return lat, lon


def plotting(stations, stats, data_dir, solar, geo_df, region):
	'''
	plots a heatmap of a particular parameter using imshow. First transforms the data frame into a 2d array for plotting

	Args:
		stats (pd.df): dataframe containing the locations and values
	'''
	params = ['mean', 'median', 'max', 'std']
	dates = extracting_dates(data_dir, stations)

	# xticks = [0, 24, 48, 72, 95]
	# xtick_labels = [0, 6, 12, 18, 24]

	colors = sns.color_palette('tab20', len(stations))
	color_map = dict(zip(stations, colors))
	color_map.update({np.nan:(0,0,0)})

	# fig = plt.figure(figsize=(20,15))
	fig, axs = plt.subplots(5, 2, figsize=(20,15))
	fig.suptitle(f'{region} - Stations: {str(stations)[1:-1]}', fontsize=25)
	for i, param in enumerate(params):

		ax = plt.subplot(5,2,i+1)
		plt.ylabel(param, fontsize=15)
		for col, stat in zip(colors, stations):
			if i ==0:
				plt.plot(stats[stat][param], label=f'{stat} {np.round(np.log10(stats[stat]["count"].sum()), 1)}', color=col)
			else:
				plt.plot(stats[stat][param], label=stat, color=col)
		plt.xlabel('MLT')
		# plt.xticks(xticks, labels=xtick_labels)
		plt.legend()
		plt.margins(x=0)

	ax = plt.subplot(5,2,5)
	plt.title('Station locations')
	# plt.xlim(geo_df['GEOLON'].min()-5, geo_df['GEOLON'].max()+5)
	# plt.ylim(geo_df['GEOLAT'].min()-5, geo_df['GEOLAT'].max()+5)
	plt.xlabel('geolon')
	plt.ylabel('geolat')
	for col, stat in zip(colors, stations):
		lat, lon = getting_geo_location(stat, geo_df)
		plt.scatter(lon, lat, color=col, s=70)


	ax = plt.subplot(5,2,6)
	plt.title('Max RSD Stations')
	value_counts = stats['max_rsd']['max_rsd_station'].value_counts()
	order = []
	for stat in stations:
		order.append(value_counts.loc[stat])
	plt.pie(order, labels=stations, colors=colors)


	ax = plt.subplot(5,1,4)

	plt.xlim(twins_start, twins_end)
	stats['max_rsd']['colors'] = stats['max_rsd']['max_rsd_station'].map(color_map)
	stats['max_rsd'].index = pd.to_datetime(stats['max_rsd'].index)
	plt.plot(stats['max_rsd']['max_rsd'])
	plt.legend()
	plt.title('Max RSD')
	plt.ylabel('nT/min')
	plt.margins(x=0, y=0)

	ax = plt.subplot(5,1,5)

	plt.xlim(twins_start, twins_end)

	for col, stat in zip(colors, stations):
		plt.fill_between(dates.index, dates[f'{stat}_bottom'], dates[f'{stat}_top'], color=col, alpha=1, label=stat,
							where=np.array(dates[f'{stat}_top'])>np.array(dates[f'{stat}_bottom']))
		plt.yticks([])

	plt.title('Data Availability')
	ax2 = ax.twinx()
	# plt.fill_between(twins_period.index, 0, solar['smoothed_ssn'].max(), color='black', alpha=0.2, label='TWINS period')
	ax2.plot(solar['smoothed_ssn'], color='black', label='Solar Cycle')
	plt.margins(y=0)
	plt.yticks([])
	plt.legend()

	plt.savefig(f'plots/station_comparisons/{region}.png')
	plt.close()
	gc.collect()


def main():

	# Process the directory of feather files and compute the statistics for each 5 degree bin
	with open(f'outputs/twins_era_identified_regions_min_2.pkl', 'rb') as f:
		stations_dict = pickle.load(f)

	if not os.path.exists(f'outputs/twins_era_stats_dict_radius_regions_min_2.pkl'):
		stats_dict = process_directory(stations_dict, data_dir, mlt_min, mlt_max, mlt_step)

		# stats = compute_statistics(data_frames)
		print('Sys size of stats_dict: '+str(sys.getsizeof(stats_dict)))

		with open(f'outputs/twins_era_stats_dict_radius_regions_min_2.pkl', 'wb') as s:
			pickle.dump(stats_dict, s)

	else:
		with open(f'outputs/twins_era_stats_dict_radius_regions_min_2.pkl', 'rb') as o:
			stats_dict = pickle.load(o)

	solar = getting_solar_cycle()
	start_date = pd.to_datetime('1995-01-01')
	end_date = pd.to_datetime('2019-12-31')
	solar = solar[(solar.index > start_date) & (solar.index < end_date)]

	geo_df = pd.read_csv('supermag-stations-info.csv')

	regions = [key for key in stats_dict.keys()]
	for region in tqdm(regions):
		# Plot the results
		plotting(stations_dict[region]['station'], stats_dict[region], data_dir, solar, geo_df, region)
		plt.close()
		gc.collect()



if __name__ == '__main__':
	main()
