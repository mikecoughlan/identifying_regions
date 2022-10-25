import glob
import math
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


def getting_geo_coordinates(stations, station):

	list_of_dfs = sorted(glob.glob('../../../../supermag/baseline/{0}/*.csv'.format(station), recursive=True))
	df = pd.read_csv(list_of_dfs[0])
	df = df[['GEOLAT', 'GEOLON']][:1]

	df['station'] = station

	print(df)

	stations = pd.concat([stations, df], axis=0)

	print(stations)

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
	geometry = Polygon(zip(lon_point_list, lat_point_list))
	polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=geometry)

	return polygon

def finding_regions(stations):

	regions = {}
	print(stations)
	for i in range(len(stations)):
		df, polys, num = pd.DataFrame(), [], []
		print(stations.iloc[i])
		lat_1 = stations['GEOLAT'][i]
		lon_1 = stations['GEOLON'][i]
		for j in range(len(stations)):
			stations_in_region = []
			dist = converting_from_degrees_to_km(lat_1, lon_1, station['GEOLAT'][j], station['GEOLON'][j])
			if dist<250:
				df = pd.concat([df,df.iloc[j]], axis=0)
				stations_in_region.append(stations['stations'][j])

		if not df.empty:
			poly = converting_regions_to_polygons(df)
			regions['region_{0}'.format(i)] = {}
			regions['region_{0}'.format(i)]['shape'] = poly
			regions['region_{0}'.format(i)]['stations'] = stations_in_region
			regions['region_{0}'.format(i)]['num_stations_in_region'] = len(stations_in_region)
			polys.append(poly)
			num.append(len(stations_in_region))

		gdf = pd.DataFrame({'geometry':polys,
							'num_stations_in_region': num})
		gdf = gpd.GeoDataFrame(gdf, geometry=gdf.geometry)
		regions['plotting_gdf'] = gdf

	return regions


def plotting_regions(regions):

	df = regions['plotting_df']
	fig = plt.figure(figsize=(15,10))

	df.plot(column='num_stations_in_region', legend=True)

	plt.savefig('../plots/finding_regions_ver1.png')



def main():

	if not os.path.isfile('../outputs/station_geo_locations.csv'):

		all_stations = [name for name in os.listdir('../../../../supermag/baseline/')]
		stations = pd.DataFrame()
		for station in all_stations:
			print(station)
			stations = getting_geo_coordinates(stations, station)
		stations.reset_index(drop=True, inplace=True)
		stations.to_csv('../outputs/station_geo_locations.csv', index=False)

	else:
		stations = pd.read_csv('../outputs/station_geo_locations.csv')
	regions = finding_regions(stations)
	plotting_regions(regions)


if __name__ == '__main__':
	main()
