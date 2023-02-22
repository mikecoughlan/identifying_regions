import glob
import math
import os
import pickle

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point, Polygon


def loading_dict():

	with open('outputs/identified_regions_geo_grid.pkl', 'rb') as f:
		regions = pickle.load(f)

	return regions

def calculating_stats(region):

	dbht = pd.Series(dtype='float32')
	col_dbht = pd.DataFrame()
	rsd = pd.DataFrame()

	for mag_station in region['stations']:
		station = pd.read_feather(f'../data/supermag/{mag_station}.feather')
		station.set_index(station['Date_UTC'], inplace=True, drop=True)
		station = station['2009-01-01 00:00:00':'2017-12-31 12:59:00']
		if station.empty:
			continue
		dbht_column = pd.Series(station['dbht'], name='dbht')
		col_dbht[f'{mag_station}'] = station['dbht']
		dbht = pd.concat([dbht,dbht_column], axis=0)
		# col_dbht = pd.concat([col_dbht,dbht_column], axis=1)
	if dbht.empty:
		region['amount'] = np.nan
		region['mean_dbht'] = np.nan
		region['median_dbht'] = np.nan
		region['max_dbht'] = np.nan
		region['std_dbht'] = np.nan
		region['99th_dbht'] = np.nan

		region['mean_rsd'] = np.nan
		region['median_rsd'] = np.nan
		region['max_rsd'] = np.nan
		region['std_rsd'] = np.nan
		region['99th_rsd'] = np.nan


	else:
		for col in col_dbht.columns:
			ss = col_dbht[col]
			temp_df = col_dbht.drop(col,axis=1)
			ra = temp_df.mean(axis=1)
			rsd[f'{col}_rsd'] = ss-ra
		max_rsd = rsd.max(axis=1)

		region['amount'] = len(dbht)
		region['mean_dbht'] = dbht.mean(axis=0)
		region['median_dbht'] = dbht.median(axis=0)
		region['max_dbht'] = dbht.max(axis=0)
		region['std_dbht'] = dbht.std(axis=0)
		region['99th_dbht'] = dbht.quantile(0.99)

		region['mean_rsd'] = max_rsd.mean(axis=0)
		region['median_rsd'] = max_rsd.median(axis=0)
		region['max_rsd'] = max_rsd.max(axis=0)
		region['std_rsd'] = max_rsd.std(axis=0)
		region['99th_rsd'] = max_rsd.quantile(0.99)

		region['max_rsd_df'] = max_rsd

	return region


def adding_statistics(regions):

	for region in regions.keys():
		print(region)
		val = calculating_stats(regions[region])

	return regions


def creating_plotting_dataframe(regions):

	polys, num, amount = [], [], []
	mean_dbht, median_dbht, max_dbht, std_dbht, perc_dbht = [], [], [], [], []
	mean_rsd, median_rsd, max_rsd, std_rsd, perc_rsd = [], [], [], [], []

	for region in regions.keys():
		polys.append(regions[region]['shape'])
		num.append(regions[region]['num_stations_in_region'])
		amount.append(regions[region]['amount'])

		mean_dbht.append(regions[region]['mean_dbht'])
		median_dbht.append(regions[region]['median_dbht'])
		max_dbht.append(regions[region]['max_dbht'])
		std_dbht.append(regions[region]['std_dbht'])
		perc_dbht.append(regions[region]['99th_dbht'])

		mean_rsd.append(regions[region]['mean_rsd'])
		median_rsd.append(regions[region]['median_rsd'])
		max_rsd.append(regions[region]['max_rsd'])
		std_rsd.append(regions[region]['std_rsd'])
		perc_rsd.append(regions[region]['99th_rsd'])


	print(regions)
	gdf = pd.DataFrame({'geometry':polys,
						'num_stations_in_region': num,
						'amount_dbht': amount,
						'mean_dbht':mean_dbht,
						'median_dbht':median_dbht,
						'max_dbht':max_dbht,
						'std_dbht':std_dbht,
						'99th_dbht':perc_dbht,
						'mean_rsd':mean_rsd,
						'median_rsd':median_rsd,
						'max_rsd':max_rsd,
						'std_rsd':std_rsd,
						'99th_rsd':perc_rsd})
	gdf = gpd.GeoDataFrame(gdf, geometry=gdf.geometry, crs='epsg:4326')
	regions['plotting_gdf'] = gdf.reset_index(drop=True)

	print(gdf)

	with open('outputs/identified_regions_geo_grid_statistics.pkl', 'wb') as f:
		pickle.dump(regions, f)

	return regions


def plotting_regions(regions, var, statistic):

	df = regions['plotting_gdf']
	fig = plt.figure(figsize=(10,7))
	ax = plt.subplot(111)
	ax.set_title(f'{statistic} {var} in 5$^\circ$ x 5$^\circ$ Bins')
	world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
	world.set_crs('epsg:4326')
	world.boundary.plot(ax=ax, zorder=0)

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.1)

	df.plot(column=f'{statistic}_{var}', ax=ax, legend=True, cax=cax, zorder=1)
	# world.plot(ax=ax)
	# df.plot(ax=ax, column='num_stations_in_region', legend=True)
	plt.savefig(f'plots/5x5_{statistic}_{var}.png')



def main():

	print(os.getcwd())
	variables = ['dbht', 'rsd']
	statistics = ['mean', 'median', 'max' ,'std', 'amount', '99th']

	# if not os.path.exists('outputs/identified_regions_geo_grid.pkl'):
	regions = loading_dict()
	regions.pop('plotting_gdf', None)
	print(regions.keys())

	regions = adding_statistics(regions)


	# if os.path.exists('outputs/identified_regions_geo_grid.pkl'):
	# 	with open('outputs/identified_regions_geo_grid.pkl', 'rb') as f:
	# 		regions = pickle.load(f)
	# 		regions.pop('plotting_gdf', None)

	regions = creating_plotting_dataframe(regions)

	for var in variables:
		for stat in statistics:
			if var =='rsd' and stat == 'amount':
				continue
			plotting_regions(regions, var, stat)


if __name__ == '__main__':
	main()
