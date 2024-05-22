from rdflib import Graph, URIRef, BNode, Literal
from rdflib.namespace import RDF, RDFS, FOAF, XSD, Namespace
from pprint import pprint

SH = Namespace("https://irnok.net/ontology/shores/t/1.0/")
SD = Namespace("https://irnok.net/ontology/shores/a/1.0/")
SCHEMA = Namespace("https://schema.org/")


def fe_proc(g, image, masks, uuid, name):
    p1 = True
    g.bind('shape', SH)
    g.bind('sd', SD)
    tp = SD[uuid]  # URIRef("http://example.org/{}".format(uuid))
    a = g.add
    t = RDF.type
    l = RDFS.label
    a((tp, t, FOAF.Image))
    a((tp, l, Literal(name, lang="en")))

    def add_shape(arr, owner):
        bn = SD[uuid + "-shape"]
        a((owner, SH["shape"], bn))
        a((bn, t, SH['Shape']))
        # (970, 1570, 3)
        shape = arr.shape
        sh2 = shape[:2]
        rows, cols = sh2
        a((bn, SH["rows"], Literal(rows)))
        a((bn, SH["cols"], Literal(cols)))
        if len(shape) == 3:
            a((bn, SH["colors"], Literal(shape[2])))

    add_shape(image, tp)
    pd = []
    for i, mask in enumerate(masks):

        pdm = {}
        pdm.update(mask)
        del pdm["segmentation"]
        pd.append(pdm)

        muuid = uuid + '-' + str(i)
        ms = SD[muuid + '-mask']
        a((ms, t, SH["Mask"]))
        a((tp, SH["mask"], ms))
        a((ms, SCHEMA.sku, Literal(i, datatype=XSD.integer)))
        segm = mask["segmentation"]
        suuid = muuid + '-segmentation'
        mss = SD[suuid]
        a((ms, SH["segmentation"], mss))
        a((mss, t, SH["Segmentation"]))
        add_shape(segm, mss)
        for k, v in mask.items():
            if k == 'segmentation':
                pass
            elif k == 'bbox':
                bbox = SD[suuid + "-bbox"]
                a((mss, SH.bbox, bbox))
                a((bbox, t, SH.Bbox))
                for bn, bv in zip(["left", "top", "width", "height"], v):
                    a((bbox, SH[bn], Literal(int(bv), datatype=XSD.integer)))
                # 'bbox': array([   0.,    0., 1569.,  795.]),
            elif k == 'crop_box':
                bb = SD[suuid + "-crop-box"]
                a((mss, SH["crop-box"], bb))
                a((bb, t, SH["Crop-box"]))
                for bn, bv in zip(["left", "top", "width", "height"], v):
                    a((bb, SH[bn], Literal(int(bv), datatype=XSD.integer)))
                # 'crop_box': array([   0,    0, 1570,  970]),
            elif k == 'point_coords':
                pc = SD[suuid + "-point-coords"]
                a((mss, SH["point-coords"], pc))
                a((pc, t, SH["Point-coords"]))
                for bn, bv in zip(["x", "y"], v[0]):
                    a((pc, SH[bn], Literal(float(bv)))) #, datatype=XSD.float)))
                # 'point_coords': array([[1152.96875,  197.03125]]),
            else:
                 a((ms, SH[k], Literal(v)))
    # pprint(pd)
