from pyramid_celery import celery_app as app

import h5py
import cv2
import logging
import redis
import json
import time

from .views.rest import (storage_begin, storage_end, STORAGE, UUIDGRP, INGRP)

from rdflib import Graph
from shores_server.processing.features import fe_proc
import torch

log = logging.getLogger(__name__)

RUNNING = {}

ANSWERDB = 6
ANSWERS = redis.Redis(db=ANSWERDB)
ANS = 'processes'


def rc_get(uuid, field=None, f=None):
    rc = ANSWERS.get(uuid)
    # print("GET rc:", rc)
    if rc is not None:
        # print("Read rc:", rc)
        rc = json.loads(rc)
        if field is None:
            return rc
        if field in rc:
            f(rc[field], rc)
        return rc
    else:
        return rc


def rc_set(uuid, dict_data):
    if dict_data is None:
        raise ValueError("setting None")
    js = json.dumps(dict_data)
    # print("rc_set:", uuid, dict_data, js)
    ANSWERS.set(uuid, js)
    # print("rc_set ctrl:", ANSWERS.get(uuid))


def rc_delete(uuid):
    return ANSWERS.delete(uuid)


def gs(str_ds):
    return str_ds.asstr()[()]


def rc_update(uuid, f):
    rc = rc_get(uuid)
    rc_ = f(rc)
    if rc_ is not None:
        rc = rc_
    rc_set(uuid, rc)


from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

import os.path as op
import os
import cv2
import numpy as np
import numpy.ma as ma
from pprint import pprint
from PIL import Image

CWD = os.getcwd()

ROOTDIR = "/home/eugeneai/projects/code/shores-server"
# ROOTDIR = CWD
CPDIR = op.join(ROOTDIR, "tmp", "sa-data")
CP_default = op.join(CPDIR, "sam_vit_h_4b8939.pth")
CP_VIT_L = op.join(CPDIR, "sam_vit_l_0b3195.pth")
CP_VIT_B = op.join(CPDIR, "sam_vit_b_01ec64.pth")

MODEL = 'vit_b'

IDIR = op.join(ROOTDIR, "images")
ODIR = op.join(ROOTDIR, "out")

SAM = None
SAM_NAME = None
mask_generator = None


def loadModel(name="default"):
    global SAM, mask_generator, SAM_NAME
    logging.info("SAM starts loading")
    if name == "default":
        SAM = sam_model_registry["default"](checkpoint=CP_default)
    elif name == "vit_l":
        SAM = sam_model_registry["vit_l"](checkpoint=CP_VIT_L)
    elif name == "vit_b":
        SAM = sam_model_registry["vit_b"](checkpoint=CP_VIT_B)
    else:
        raise ValueError("Wrong parameter for SAM model")
    # SAM.to(device='cuda')
    SAM.to(device='cpu')
    mask_generator = SamAutomaticMaskGenerator(
        model=SAM,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    logging.info("SAM loaded '{}'".format(name))
    SAM_NAME = name


def segment(image, model=MODEL):
    global SAM
    if SAM is None:
        loadModel(name=model)
    #predictor = SamPredictor(SAM)
    #predictor = SamPredictor(SAM)
    #predictor.set_image(image)
    #masks, v1, v2 = predictor.predict("borders")
    #print(type(masks), type(v1), type(v2))

    logging.info("Start Recognition/Segmentation")
    masks = mask_generator.generate(image)
    logging.info("Finish Recognition/Segmentation")

    return masks


def testLoadAndSaveMasks(image, masks, outFN, gen=False):
    print("Test with load")
    if isinstance(image, str):
        imagename = op.join(IDIR, image)
        image = imagename
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if isinstance(masks, str):
        masks = op.join(ODIR, masks)
        import pickle
        try:
            i = open(masks, "rb")
            print("Loading from pickle {}".format(masks))
            masks = pickle.load(i)
            i.close()
        except FileNotFoundError:
            logging.info("The Cache is not found, recognizing!")
            print("The Cache is not found, recognizing!")
            masks = segment(image)
    rows, cols, chans = image.shape
    # maskt = np.full((rows, cols), 255, dtype=int)
    maskt = np.full((rows, cols), 0, dtype=image.dtype)
    cnt = {}
    tifLayers = []
    tifLayers.append(Image.fromarray(image))
    for mask in masks:
        # m = cv2.
        area = mask["area"]
        c = cnt.setdefault(area, 1)
        cnt[area] += 1
        name = op.join(ODIR, "{}-{}-{}-".format(SAM_NAME, area, c) + outFN)
        logging.info("Writing file {}".format(name))
        mm = np.copy(maskt)
        # msk = ma.masked_equal(mask["segmentation"], True)
        msk = mask["segmentation"]
        # print("Mask from SAnything")
        # pprint(msk)
        # print("OUR template")
        # pprint(mm)
        mm[msk] = 255
        print("MASK")
        pprint(mm)
        # quit()
        # mm = ma.masked_array(mask, mask["segmentation"])
        # img = cv2.bitwise_and(image, image, mask = mm)
        img = cv2.bitwise_and(image, image, mask=mm)
        tifLayers.append(Image.fromarray(img))
        print("writing "+name)
        cv2.imwrite(name, img)
        if gen:
            yield (imagename, name, img)

    name = op.join(ODIR, "join-{}-".format(SAM_NAME) + outFN + '.tif')
    tifLayers[0].save(name,
                      save_all=True,
                      append_images=tifLayers[1:],
                      compression='tiff_lzw')
    if gen:
        yield (imagename, name, tifLayers[1:])


# if __name__=="__main__":

#     fn = "Uley2.JPG"
#     # TI = op.join(IDIR, fn)
#     # testRecognize(fn, "vit_l")
#     for objName in "default-Uley2.JPG.obj vit_b-Uley2.JPG.obj vit_l-Uley2.JPG.obj".split():
#         testLoadAndSaveMasks(fn, objName, objName+".tif")


@app.task
def sa_start(uuids):

    def f(js):
        js["ready"] = True
        js["result"] = "a Good result"

    def fd(js):
        js["ready"] = False

    rc_update(uuids, fd)
    log.info(
        'creating task processing image identified by UUID {}'.format(uuids))
    # Load Image
    storage, ingrp, uuidgrp = storage_begin()
    name = gs(uuidgrp[uuids])
    logging.info("Image name is '{}'".format(name))
    imgg = ingrp[name]
    image = imgg["content"]
    image = image[()]
    logging.info("Image shape is {}".format(image.shape))
    try:
        req = imgg["masks"]
        logging.info("Already recognized {}".format(req))
        rc_update(uuids, f)
        storage, ingrp, uuidgrp = storage_end()
        return
    except KeyError:
        logging.info("New recognition starting.")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logging.info("Torch engine is {}.".format(device))
        storage, ingrp, uuidgrp = storage_end()

    del storage
    del ingrp
    del uuidgrp
    del imgg

    masks = segment(image)
    # masks = []

    logging.info("Recognition finished, saving into storage.")
    storage, ingrp, uuidgrp = storage_begin()

    imgg = ingrp[name]

    mgrp = imgg.create_group("masks")
    for num, mask in enumerate(masks):
        d = {}
        mname = str(num)
        d.update(mask)
        mg = mgrp.create_group(mname)
        del d["area"]
        del d["predicted_iou"]
        del d["stability_score"]
        # print(list[d.items()])
        for k, v in d.items():
            mg.create_dataset(
                k, data=v, compression="lzf" if k == "segmentation" else None)
        mg.attrs["area"] = mask["area"]
        mg.attrs["predicted_iou"] = mask["predicted_iou"]
        mg.attrs["stability_score"] = mask["stability_score"]
        # pprint(d)
    storage, ingrp, uuidgrp = storage_end()

    # from pprint import pprint

    logging.info("Found {} masks".format(len(masks)))
    #pprint(masks)

    rc_update(uuids, f)
    # task = TaskItem(task=task)
    # DBSession.add(task)
    # transaction.commit


STORE_FORMAT = "turtle"


@app.task
def feature_recognition(uuid):

    def f(js):
        js["ready"] = True
        js["result"] = True

    def bf(js):
        js["ready"] = True
        js["result"] = False

    def fd(js):
        js["ready"] = False

    rc_update(uuid, fd)

    log.info("Creating task processing masks for image identified by UUID {}".
             format(uuid))
    # Make a copy of mask data
    storage, ingrp, uuidgrp = storage_begin()
    name = gs(uuidgrp[uuid])
    igrp = ingrp[name]
    try:
        mgrp = igrp["masks"]
    except KeyError:
        mgrp = None
    dd = []
    if mgrp is not None:
        for num in mgrp.keys():
            number = str(num)
            try:
                mg = mgrp[number]
            except KeyError:
                break
            d = {}
            d["area"] = int(mg.attrs["area"])
            d["predicted_iou"] = float(mg.attrs["predicted_iou"])
            d["stability_score"] = float(mg.attrs["stability_score"])
            for k, v in mg.items():
                d[k] = v[()]
            # pprint((number, mg, d))
            dd.append(d)
        image = igrp["content"][()]
    # release database
    storage, ingrp, uuidgrp = storage_end()
    if mgrp is None:
        rc_update(uuid, bf)
        logging.info("Cannot start FE, no masks found")
        return
    logging.info("Starting FE on {} shape image and {} its masks".format(
        image.shape, len(dd)))
    # pprint(dd)
    g = Graph(bind_namespaces="rdflib")
    fe_proc(g, image, dd, uuid, name)
    ser = g.serialize(format=STORE_FORMAT)
    print(ser)
    storage, ingrp, uuidgrp = storage_begin()
    name = gs(uuidgrp[uuid])
    igrp = ingrp[name]
    if "features" in igrp:
        logging.warning("Removing old feature graph")
        del igrp["features"]
    igrp.create_dataset("features", data=ser)
    logging.info("Storing feature graph in {} format".format(STORE_FORMAT))
    storage, ingrp, uuidgrp = storage_end()
    rc_update(uuid, f)
