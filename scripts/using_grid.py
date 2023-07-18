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


def getting_geo_coordinates(stations, station):

	list_of_dfs = sorted(glob.glob('../../../../supermag/baseline/{0}/*.csv'.format(station), recursive=True))
	df = pd.read_csv(list_of_dfs[0])
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


def converting_grid_to_polygons(lat, lon, lat_deg, lon_deg):

	lat_point_list = list([lat, lat, lat+lat_deg, lat+lat_deg])
	lon_point_list = list([lon, lon+lon_deg, lon+lon_deg, lon])
	polygon = Polygon(zip(lon_point_list, lat_point_list))
	# polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=geometry)

	return polygon

def finding_stations_in_grid(stations, lat_deg, lon_deg):

	lat_grid = np.arange(-90, 90, lat_deg)
	lon_grid = np.arange(-180, 180, lon_deg)

	regions = {}
	polys, num = [], []
	i=0
	for lat in lat_grid:
		for lon in lon_grid:
			stations_in_region = stations[(stations['GEOLAT']>=lat)&(stations['GEOLAT']<lat+lat_deg)
											&(stations['GEOLON']>=lon)&(stations['GEOLON']<lon+lon_deg)]
			if (lat >55)&(lon>0)&(len(stations_in_region)>0):
				print(f'lat: {lat} - lon: {lon}')
				print(stations_in_region)
			stations_in_region = stations_in_region['station'].tolist()
			if (lat >55)&(lon>0)&(len(stations_in_region)>0):
				print(stations_in_region)
			if len(stations_in_region) > 0:

				poly = converting_grid_to_polygons(lat, lon, lat_deg, lon_deg)
				regions['region_{0}'.format(i)] = {}
				regions['region_{0}'.format(i)]['shape'] = poly
				regions['region_{0}'.format(i)]['stations'] = stations_in_region
				regions['region_{0}'.format(i)]['num_stations_in_region'] = len(stations_in_region)
				polys.append(poly)
				num.append(len(stations_in_region))
				i+=1

	with open('outputs/identified_regions_geo_grid.pkl', 'wb') as f:
		pickle.dump(regions, f)
	print(regions)
	gdf = pd.DataFrame({'geometry':polys,
						'num_stations_in_region': num})
	gdf = gpd.GeoDataFrame(gdf, geometry=gdf.geometry, crs='epsg:4326')
	regions['plotting_gdf'] = gdf

	return regions


def plotting_regions(regions):

	df = regions['plotting_gdf']
	fig = plt.figure(figsize=(10,7))
	ax = plt.subplot(111)
	ax.set_title('Num Stations in 5$^\circ$ x 5$^\circ$ Bins')
	world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
	world.set_crs('epsg:4326')
	world.boundary.plot(ax=ax, zorder=0)

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.1)

	df.plot(column='num_stations_in_region', ax=ax, legend=True, cax=cax, zorder=1)
	# world.plot(ax=ax)
	# df.plot(ax=ax, column='num_stations_in_region', legend=True)
	plt.savefig('../plots/finding_regions_grid.png')



def main():
	print(os.getcwd())
	if not os.path.isfile('outputs/station_geo_locations.csv'):

		all_stations = [name for name in os.listdir('../../../../supermag/baseline/')]
		stations = pd.DataFrame()
		for station in all_stations:
			stations = getting_geo_coordinates(stations, station)
		stations.reset_index(drop=True, inplace=True)
		stations.to_csv('outputs/station_geo_locations.csv', index=False)

	else:
		stations = pd.read_csv('outputs/station_geo_locations.csv')
	stations['GEOLON'] = (stations['GEOLON'] + 180) % 360 - 180 # redefining the (0,360) geolon to a (-180,180) coordinate system

	regions = finding_stations_in_grid(stations, 5, 5)
	plotting_regions(regions)


if __name__ == '__main__':
	main()
