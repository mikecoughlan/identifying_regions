import math
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


def getting_geo_coordinates(stations, station):

	df = pd.read_csv('../../../../supermag/baseline/{0}'.format(station))
	df = df[['GEOLAT', 'GEOLON']][0]

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
	geometry = Polygon(zip(lon_point_list, lat_point_list))
	polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=geometry)

	return polygon

def finding_regions(stations):

	regions = {}
	for station in stations:
		df = pd.DataFrame()
		lat_1 = station['GEOLAT']
		lon_1 = station['GEOLON']
		for other_stations in stations:
			dist = converting_from_degrees_to_km(lat_1, lon_1, other_stations['GEOLAT'], other_stations['GEOLON'])
			if dist<250:
				df = pd.concat([df,other_stations], axis=0)

		# df = converting_regions_to_polygons(df)
		regions['station'] = df


def main():

	all_stations = [name for name in os.listdir('../../../../supermag/baseline/') if os.path.isdir(name)]
	stations = pd.DataFrame()
	for station in all_stations:
		print(station)
		stations = getting_geo_coordinates(stations, station)


if __name__ == '__main__':
	main()
