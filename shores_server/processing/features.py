from rdflib import Graph, URIRef, BNode, Literal
from rdflib.namespace import RDF, RDFS, FOAF, Namespace


SH = Namespace("http://irnok.net/ontology/shores/1.0/")


def fe_proc(g, image, masks, uuid, name):
    p1 = True
    g.bind('shape', SH)
    tp=URIRef("http://example.org/{}".format(uuid))
    a = g.add
    t = RDF.type
    l = RDFS.label
    a((tp, t, FOAF.Image))
    a((tp, l, Literal(name, lang="en")))
    def add_shape(arr, owner):
        bn = BNode(uuid+"-shape")
        a((owner, SH["shape"], bn))
        a((bn, t, SH['Shape']))
        # (970, 1570, 3)
        shape = arr.shape
        sh2 = shape[:2]
        rows, cols = sh2
        a((bn, SH["rows"], Literal(rows)))
        a((bn, SH["cols"], Literal(cols)))
        if len(shape)==3:
            a((bn, SH["colors"], Literal(shape[2])))
    add_shape(image, tp)
    for i, mask in enumerate(masks):
        muuid = uuid+'-'+str(i)
        ms = SH[muuid+'-mask']
        a((ms, t, SH["Mask"]))
        a((tp, SH["mask"], ms))
        segm = mask["segmentation"]
        mss = SH[muuid+'-segmentation']
        a((ms, SH["segmentation"], mss))
        a((mss, t, SH["Segmentation"]))
        add_shape(segm, mss)
        # if p1:
        #     q = {}
        #     q.update(mask)
        #     del q["segmentation"]
        #     from pprint import pprint
        #     pprint(q)
        #     p1 = False
        for k, v in mask.items():
            if k == 'segmentation':
                pass
            elif k == 'bbox':
                # 'bbox': array([   0.,    0., 1569.,  795.]),
                pass
            elif k == 'crop_box':
                # 'crop_box': array([   0,    0, 1570,  970]),
                pass
            elif k == 'point_coords':
                # 'point_coords': array([[1152.96875,  197.03125]]),
                pass
            else:
                a((ms, SH[k], Literal(v)))
