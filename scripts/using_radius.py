import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def getting_geo_coordinates(stations, station):

	df = pd.read_csv('../../data/supermag/{0}'.format(station))
	df = df[['geolat', 'geolon']][0]

	df['station'] = station

	stations = pd.concat([stations, df], axis=0)

	return stations


def main():

	all_stations = []
	stations = pd.DataFrame()
	for station in all_stations:
		stations = getting_geo_coordinates(stations, station)
