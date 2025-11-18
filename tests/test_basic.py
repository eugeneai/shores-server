from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

import os.path as op
import os
import cv2
import numpy as np
import numpy.ma as ma
import pprint
from PIL import Image

CWD = os.getcwd()

# ROOTDIR="/home/eugeneai/projects/code/shores/"
ROOTDIR = CWD
CPDIR = op.join(ROOTDIR, "checkpoints")
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
    print("SAM starts loading")
    if name == "default":
        SAM = sam_model_registry["default"](checkpoint=CP_default)
    elif name == "vit_l":
        SAM = sam_model_registry["vit_l"](checkpoint=CP_VIT_L)
    elif name == "vit_b":
        SAM = sam_model_registry["vit_b"](checkpoint=CP_VIT_B)
    else:
        raise ValueError("Wrong parameter for SAM model")
    mask_generator = SamAutomaticMaskGenerator(SAM)
    print("SAM loaded '{}'".format(name))
    SAM_NAME = name


def imRead(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (0,0), fx=0.08, fy=0.08) # TODO: Calculate sizes from the original size
    print("INFO: Image '{}' loaded".format(filename))
    return image


def segment(imgName, model=MODEL):
    global SAM
    if SAM is None:
        loadModel(name=model)
    image = imRead(imgName)
    #predictor = SamPredictor(SAM)
    #predictor.set_image(image)
    #masks, v1, v2 = predictor.predict("borders")
    #print(type(masks), type(v1), type(v2))

    print("Start Recognition/Segmentation")
    masks = mask_generator.generate(image)
    print("Finish Recognition/Segmentation")

    return masks


# cv2.imwrite(TO, masks)


def testRecognize(filename, model=MODEL):
    # global TI, PO
    ti = op.join(IDIR, filename)
    masks = segment(ti, model=model)
    print(masks)
    import pickle
    pprint.pprint(masks)
    po = op.join(ODIR, "{}-".format(SAM_NAME) + filename + ".obj")
    of = open(po, "wb")
    pickle.dump(masks, of)
    of.close()
    return masks


def testLoadAndSaveMasks(image, masks, outFN):
    print("Test with load")
    if isinstance(masks, str):
        masks = op.join(ODIR, masks)
        import pickle
        print("Loading from pickle {}".format(masks))
        try:
            i = open(masks, "rb")
            masks = pickle.load(i)
            i.close()
        except FileNotFoundError:
            print("The Cache is not found, recognizing!")
            masks = testRecognize()
    if isinstance(image, str):
        image = op.join(IDIR, image)
        image = imRead(image)
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
        print("INFO: Writing file {}".format(name))
        mm = np.copy(maskt)
        # msk = ma.masked_equal(mask["segmentation"], True)
        msk = mask["segmentation"]
        print("Mask from SAnything")
        pprint.pprint(msk)
        print("OUR template")
        pprint.pprint(mm)
        mm[msk] = 255
        print("MASK")
        pprint.pprint(mm)
        # quit()
        # mm = ma.masked_array(mask, mask["segmentation"])
        # img = cv2.bitwise_and(image, image, mask = mm)
        img = cv2.bitwise_and(image, image, mask=mm)
        tifLayers.append(Image.fromarray(img))
        cv2.imwrite(name, img)
    name = op.join(ODIR, "join-{}-".format(SAM_NAME) + outFN + '.tif')
    tifLayers[0].save(name,
                      save_all=True,
                      append_images=tifLayers[1:],
                      compression='tiff_lzw')


if __name__ == "__main__":

    fn = "Uley2.JPG"
    # TI = op.join(IDIR, fn)
    # testRecognize(fn, "vit_l")
    for objName in "default-Uley2.JPG.obj vit_b-Uley2.JPG.obj vit_l-Uley2.JPG.obj".split(
    ):
        testLoadAndSaveMasks(fn, objName, objName + ".tif")
