from shapely.geometry import LineString
import geopandas
from shapely.geometry import Polygon, Point, LineString
from pprint import pprint

# Load the shapefile
gdf = geopandas.read_file("ll-a.shp")

# Access geometries
# for index, row in gdf.iterrows():
#     geometry = row['geometry']
#     if isinstance(geometry, Polygon):
#         print(f"Polygon: {geometry}")
#     elif isinstance(geometry, Point):
#         print(f"Point: {geometry}")
#     elif isinstance(geometry, LineString):
#         print(f"LineString: {geometry} {len(geometry.coords)}")

ls1 = gdf['geometry'][0]
ls2 = gdf['geometry'][1]


def sample_polyline(polyline, num_samples):
    """Samples points along a Shapely LineString at regular intervals.

    Args:
        polyline: A list of (x, y) tuples representing the polyline vertices.
        num_samples: The desired number of sample points.

    Returns:
        A list of (x, y) tuples representing the sampled points.
    """
    if isinstance(polyline, LineString):
        line = polyline
    else:
        line = LineString(polyline)
    if num_samples <= 1:
        return list(line.coords)

    total_length = line.length
    sample_interval = total_length / (num_samples - 1)
    # sampled_points = []
    for i in range(num_samples):
        point = line.interpolate(i * sample_interval)
        # sampled_points.append((point.x, point.y))
        yield point


def average_distance(line1, line2):
    s = 0
    numpoints = 1000

    for point in sample_polyline(line1, numpoints):
        s += line2.distance(point)

    return s/numpoints

# Example usage:
# polyline = [(0, 0), (1, 2), (3, 1), (4, 3), (6, 2), (8, 4)]
# num_samples = 5
# sampled_points = sample_polyline_shapely(polyline, num_samples)
# print(sampled_points)


print('distance: ', average_distance(ls1, ls2))
print('distance: ', average_distance(ls2, ls1))

# average distance:  13.892792787889356
# average distance:  13.252893177143987
