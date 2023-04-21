import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define the directory containing the CSV files
data_dir = '../data/supermag/'

# Define the 5 degree grid
mlat_min = -90
mlat_max = 90
mlt_min = 0
mlt_max = 24
mlat_step = 5
mlt_step = .25

def creating_dict_of_stations(data_dir, mlat_min, mlat_max, mlt_min, mlt_max, mlat_step, mlt_step):
	'''
	Creates a dictonary of which stations are in each grid. Used so we don't have to loop over each file every time only once.
	'''
	stations_dict = {}
	for mlat in tqdm(np.arange(mlat_min, mlat_max, mlat_step)):
		stations_dict[f'mlat_{mlat}'] = {}
		stats = []
		for mlt in tqdm(np.arange(mlt_min, mlt_max, mlt_step)):
			mlat_min_bin = mlat
			mlat_max_bin = mlat + mlat_step
			mlt_min_bin = mlt
			mlt_max_bin = mlt + mlt_step
			for filename in sorted(glob.glob(data_dir+'*.feather', recursive=True)):
				df = pd.read_feather(filename)
				if (df['MLAT'].any() >= mlat_min_bin) & (df['MLAT'].any() < mlat_max_bin):
					file_name = os.path.basename(filename)
					station = file_name.split('.')[0]
					stat.append(station)

			stations_dict[f'mlat_{mlat}'][mlt] = stats

	with open(f'outputs/stations_dict_{mlat_step}MLAT_x_{int(1/mlt_step)}mlt.pkl', 'wb') as f:
		pickle.dump(f, stations_dict)

	return stations_dict


def load_data(filepath):
	'''
	Load a feather file into a pandas data frame.
	'''
	df = pd.read_feather(filepath)

	return df


def filter_data(df, mlat_min_bin, mlat_max_bin, mlt_min_bin, mlt_max_bin):
	'''
	Filter a pandas data frame to a 5 degree bin.
	'''
	df = df[(df['MLAT'] >= mlat_min_bin) & (df['MLAT'] < mlat_max_bin) & (df['MLT'] >= mlt_min_bin) & (df['MLT'] < mlt_max_bin)]

	return df


def process_file(filepath, mlat_min_bin, mlat_max_bin, mlt_min_bin, mlt_max_bin):
	'''
	Process a single feather file and return a filtered data frame for a 5 degree bin.
	'''
	df = load_data(filepath)
	df_filtered = filter_data(df, mlat_min_bin, mlat_max_bin, mlt_min_bin, mlt_max_bin)

	return df_filtered


def process_directory(data_dir, mlat_min, mlat_max, mlt_min, mlt_max, mlat_step, mlt_step, stations_dict):
	'''
	Process all feather files in a directory and return a list of filtered data frames for each 5 degree bin.
	'''
	data_frames = []
	for mlat in np.arange(mlat_min, mlat_max, mlat_step):
		for mlt in np.arange(mlt_min, mlt_max, mlt_step):
			print(f'MLAT: {mlat}' + f' MLT: {mlt}')
			mlat_min_bin = mlat
			mlat_max_bin = mlat + mlat_step
			mlt_min_bin = mlt
			mlt_max_bin = mlt + mlt_step
			for stats in stations_dict[f'mlat_{mlat}'][mlt]:
				filepath = os.path.join(data_dir, f'{stats}.feather')
				df_filtered = process_file(filepath, mlat_min_bin, mlat_max_bin, mlt_min_bin, mlt_max_bin)
				if len(df_filtered) > 0:
					data_frames.append(df_filtered)

	return data_frames


def compute_statistics(data_frames):
	'''
	Compute the statistics of the 'dbht' parameter for each 5 degree bin.
	'''
	df_combined = pd.concat(data_frames)
	stats = df_combined.groupby([pd.cut(df_combined['MLAT'], np.arange(mlat_min, mlat_max + mlat_step, mlat_step)),
									pd.cut(df_combined['MLT'], np.arange(mlt_min, mlt_max + mlt_step, mlt_step))])['dbht'].agg(['mean', 'std', 'count']).reset_index()

	return stats


def plot_results(stats):
	'''
	Plot the results.
	'''
	plt.scatter(stats['MLT'], stats['MLAT'], s=stats['count'], c=stats['mean'])
	plt.colorbar()
	plt.xlabel('Longitude')
	plt.ylabel('Latitude')
	plt.title('Mean DBHT in 5 Degree Bins')
	plt.savefig('plots/mlt_bins_count_and_mean.png')


def main():
	# Process the directory of feather files and compute the statistics for each 5 degree bin
	if not os.path.exists(f'outputs/stations_dict_{mlat_step}MLAT_x_{int(1/mlt_step)}mlt.pkl'):
		stations_dict = creating_dict_of_stations(data_dir, mlat_min, mlat_max, mlt_min, mlt_max, mlat_step, mlt_step)
	else:
		with open(f'outputs/stations_dict_{mlat_step}MLAT_x_{int(1/mlt_step)}mlt.pkl', 'rb') as f:
			stations_dict = pickle.load(f)

	data_frames = process_directory(data_dir, mlat_min, mlat_max, mlt_min, mlt_max, mlat_step, mlt_step, stations_dict)
	stats = compute_statistics(data_frames)

	# Plot the results
	plot_results(stats)


if __name__ == '__main__':
	main()
