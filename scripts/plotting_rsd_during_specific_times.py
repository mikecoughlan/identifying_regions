import glob
import math
import os
import pickle

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import moviepy
import moviepy.video.io.ImageSequenceClip
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point, Polygon


def loading_dict():

	with open('outputs/identified_regions_geo_grid_statistics.pkl', 'rb') as f:
		regions = pickle.load(f)

	return regions


def segmenting_rsd_to_timestamps(regions, stime, etime):

	storm_plotting_dataframe = []
	geometry = []

	for region in regions:

		try:
			df = regions[region]['max_rsd_df']
		except KeyError:
			continue

		storm_plotting_dataframe.append(df[stime:etime])

		geometry.append(regions[region]['shape'])

	storm_plotting_dataframe = pd.concat(storm_plotting_dataframe, axis=1, ignore_index=False)
	storm_plotting_dataframe = storm_plotting_dataframe.T
	storm_plotting_dataframe['geometry'] = geometry
	storm_plotting_dataframe = gpd.GeoDataFrame(storm_plotting_dataframe, geometry=storm_plotting_dataframe.geometry, crs='epsg:4326')

	return storm_plotting_dataframe


def plotting_minute_resoultion(df, day):

	temp_df = df.drop('geometry', axis=1).to_numpy()
	vmin = np.nanmin(temp_df)
	vmax = np.nanmax(temp_df)
	for col in df.columns:
		if col == 'geometry':
			continue

		fig = plt.figure(figsize=(10,7))
		ax = plt.subplot(111)
		ax.set_title(col)
		world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
		world.set_crs('epsg:4326')
		world.boundary.plot(ax=ax, zorder=0)

		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.1)

		df.plot(column=col, ax=ax, legend=True, cax=cax, zorder=1, norm=plt.Normalize(vmin=vmin, vmax=vmax))

		plt.savefig(f'plots/{day}/{col}.png')


def making_video(day):

	fps=10
	image_folder = f'plots/{day}'
	image_files = [os.path.join(image_folder,img)
				for img in os.listdir(image_folder)
				if img.endswith('.png')]
	clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(sorted(image_files), fps=fps)
	clip.write_videofile(f'plots/{day}/video.mp4', logger=None)


def main():

	start_times = ['2012-03-12 00:00:00']
	end_times = ['2012-03-13 00:00:00']

	regions = loading_dict()

	for stime, etime in zip(start_times, end_times):
		day = stime.split()[0]

		if not os.path.exists(f'plots/{day}'):
			os.makedirs(f'plots/{day}')

		df = segmenting_rsd_to_timestamps(regions, stime, etime)
		plotting_minute_resoultion(df, day)
		making_video(day)







if __name__ == '__main__':
	main()