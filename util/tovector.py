import rasterio
from rasterio import features
from shapely.geometry import shape
from shapely import LineString
import geopandas as gpd
from pprint import pprint
import math


def lower_line(poly, tolerance):
    """Process points starting from lower,
    if from the previous one a dot is upper
    then tolerance, then we consider it tervunal
    """
    points = list(poly.exterior.coords)
    (minx, miny, maxx, maxy) = bounds = poly.bounds

    idx = 0
    pd  = +math.inf
    direc = 1

    def dist(p1, p2):
        (x1, y1) = p1
        (x2, y2) = p2
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    llp = (maxx, miny)
    for i, p in enumerate(points):
        d = dist(p, llp)
        if d < pd:
            pd = d
            idx = i

    print("ll index =", idx, pd)

    vd = 0

    ii = idx
    pp = points[ii]
    left = []

    while (vd < tolerance):
        ii = ii+1
        if ii>=len(points):
            ii=0
        cp = points[ii]
        vd = abs(cp[1] - pp[1]) # vertical distance
        left.append(pp)
        pp = cp

    ii = idx
    pp = points[ii]
    right = []
    vd = 0

    while (vd < tolerance):
        if ii==0:
            ii=len(points)
        ii = ii-1
        cp = points[ii]
        vd = abs(cp[1] - pp[1]) # vertical distance
        right.append(pp)
        pp = cp

    left.reverse()
    right = left + right[1:]

    pprint(right)

    return LineString(right)


def raster_to_shapefile(raster_path, output_path, reference=None):
    if reference is not None:
        sourceCRS = rasterio.open(reference).crs
    else:
        sourceCRS = None

    with rasterio.open(raster_path) as src:
        image = src.read(1)
        transform = src.transform
        mask = (image != 0)
        # pprint(mask)
        image[mask]=255
        shapes = features.shapes(image, transform=transform, mask=mask)
        geoms = [shape(s) for s, v in shapes]

        gdf = gpd.GeoDataFrame({'geometry': geoms}, crs=sourceCRS)
        gdf.to_file(output_path)

        simps = [s.simplify(3, preserve_topology=False) for s in geoms]  # 1.4
        gdf = gpd.GeoDataFrame({'geometry': simps}, crs=sourceCRS)
        gdf.to_file('sim-'+output_path)

        ll = lower_line(simps[0], tolerance=100)
        gdf = gpd.GeoDataFrame({'geometry': [ll]}, crs=sourceCRS)
        gdf.to_file('lln-'+output_path)

raster_to_shapefile("./a.tiff", "a.shp", reference='./q2008.tif')
