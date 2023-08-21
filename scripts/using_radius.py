import glob
import math
import os
import pickle

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

# defining the twins era start and end times
twins_start = pd.to_datetime('2010-01-01')
twins_end = pd.to_datetime('2017-12-31')
twins_time_period = pd.date_range(start=twins_start, end=twins_end, freq='min')

def getting_geo_coordinates(stations, station):

	df = pd.read_feather(f'../data/supermag/{station}.feather')
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index)
	df = df[twins_start:twins_end]
	df.dropna(inplace=True)

	if not df.empty:

		df = df[['GEOLAT', 'GEOLON']][:1]

		df['station'] = station

		stations = pd.concat([stations, df], axis=0)


	return stations

def converting_from_degrees_to_km(lat_1, lon_1, lat_2, lon_2):

	mean_lat = (lat_1 + lat_2)/2
	x = lon_2 - lon_1
	y = lat_2 - lat_1
	dist_x = x*(111.320*math.cos(math.radians(mean_lat)))
	dist_y = y*110.574

	distance = math.sqrt((dist_x**2)+(dist_y**2))

	return distance


def converting_regions_to_polygons(df):

	lon_point_list = df['GEOLON'].tolist()
	lat_point_list = df['GEOLAT'].tolist()
	polygon = Polygon(zip(lon_point_list, lat_point_list))
	# polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=geometry)

	return polygon

def finding_regions(stations):

	regions = {}
	polys, num = [], []
	for i in range(len(stations)):
		print('\nStation examined: '+str(stations['station'][i]))
		lat_1 = stations['GEOLAT'][i]
		lon_1 = stations['GEOLON'][i]
		stations_in_region = []
		for j in range(len(stations)):
			dist = converting_from_degrees_to_km(lat_1, lon_1, stations['GEOLAT'][j], stations['GEOLON'][j])
			print('Station: '+str(stations['station'][j])+' Distance: '+str(dist))
			if dist<250:
				print('I\'m here')
				stations_in_region.append(stations['station'][j])
		df = stations[stations['station'].isin(stations_in_region)]
		print(df)
		if len(df)>2:
			poly = converting_regions_to_polygons(df)
			regions['region_{0}'.format(i)] = {}
			regions['region_{0}'.format(i)]['shape'] = poly
			regions['region_{0}'.format(i)]['station'] = stations_in_region
			regions['region_{0}'.format(i)]['num_stations_in_region'] = len(stations_in_region)
			polys.append(poly)
			num.append(len(stations_in_region))

	with open('outputs/identified_regions_min_2.pkl', 'wb') as f:
		pickle.dump(regions, f)
	print(regions)
	gdf = pd.DataFrame({'geometry':polys,
						'num_stations_in_region': num})
	gdf = gpd.GeoDataFrame(gdf, geometry=gdf.geometry)
	regions['plotting_gdf'] = gdf

	return regions


def plotting_regions(regions):

	df = regions['plotting_gdf']
	ax = plt.figure(figsize=(60,55))
	world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
	newdf = world.overlay(df, how='union')
	# world.plot(ax=ax)
	# df.plot(ax=ax, column='num_stations_in_region', legend=True)
	newdf.plot(column='num_stations_in_region', legend=True)
	plt.savefig('plots/finding_regions_min_2.png')



def main():

	if not os.path.isfile('outputs/twins_era_station_geo_locations.csv'):
		print(os.getcwd())

		all_stations = [os.path.splitext(file_name)[0] for file_name in os.listdir('../data/supermag/')]
		stations = pd.DataFrame()
		for station in all_stations:
			print(station)
			stations = getting_geo_coordinates(stations, station)
		stations.reset_index(drop=True, inplace=True)
		stations.to_csv('outputs/twins_era_station_geo_locations.csv', index=False)

	else:
		stations = pd.read_csv('outputs/twins_era_station_geo_locations.csv')
	regions = finding_regions(stations)
	plotting_regions(regions)


if __name__ == '__main__':
	main()
