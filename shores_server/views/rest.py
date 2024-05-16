from cornice import Service
from uuid import uuid1 as uuidgen
from uuid import UUID
import mmh3
import h5py
import cv2
import numpy as np
import redis
import time

from celery.result import AsyncResult

import logging

# from ..tasks import (sa_start)

log = logging.getLogger(__name__)

STORAGE = INGRP = UUIDGRP = None

LOCKDB = 5
WAITSEC = 3
MAXUNLOCKCOUNT = 3
LOCKS = redis.Redis(db=LOCKDB)

LOCKS.flushdb()


def storage_begin():
    global STORAGE, INGRP, UUIDGRP
    if STORAGE is not None:
        log.warning('database is open')
        STORAGE.flush()
        # LOCKS.delete("data.hdf5")
    cnt = MAXUNLOCKCOUNT
    while LOCKS.get('data.hdf5') is not None:
        log.warning('database is locked, wait for {} sec'.format(WAITSEC))
        time.sleep(WAITSEC)
        cnt -= 1
        if cnt <= 0:
            raise RuntimeError("cannot lock database")
    try:
        STORAGE = h5py.File('data.hdf5', 'a')
        log.info('Successfully opened the database')
    except OSError:
        log.warning('Killed the previous database')
        STORAGE = h5py.File('data.hdf5', 'w')
    LOCKS.set('data.hdf5', 'rest')

    INGRP = STORAGE.require_group("input")
    UUIDGRP = STORAGE.require_group("uuid")

    STORAGE.flush()
    return STORAGE, INGRP, UUIDGRP


def storage_end():
    global STORAGE, INGRP, UUIDGRP
    if STORAGE is None:
        log.warning('database is already closed')
        return STORAGE, INGRP, UUIDGRP
    STORAGE.close()
    if LOCKS.delete('data.hdf5') == 0:
        log.warning('Lock has been lost')
    STORAGE = INGRP = UUIDGRP = None
    return STORAGE, INGRP, UUIDGRP


def mmh(content):
    return mmh3.hash128(content)


def gs(str_ds):
    return str_ds.asstr()[()]


imgs = Service(name='imgstore-list',
               path='/sa-1.0/images/{limit}',
               description="Image collection, operating lists")


@imgs.get()
def get_list(request):
    """Returns collection of <name>:<uuid> pairs of images loaded
    limited by {limit} value
    """
    STORAGE, INGRP, UUIDGRP = storage_begin()
    try:
        limit = request.matchdict['limit']
    except KeyError:
        STORAGE, INGRP, UUIDGRP = storage_end()
        return {"list": [], "error": "wrong limit arg", "ok": False}
    try:
        limit = int(limit)
    except ValueError:
        return {
            "list": [],
            "error": "limit value cannot convert to integer",
            "ok": False
        }
    ds = []
    c = 0
    for key, val in UUIDGRP.items():
        ds.append((key, gs(val)))
        c += 1
        if c > limit: break
    STORAGE, INGRP, UUIDGRP = storage_end()
    return {"list": ds, "error": None, "ok": True, "limit": limit}


img = Service(name='imgstore',
              path='/sa-1.0/image/{img_name}',
              description="Image collection")


@img.get()
def get_id(request):
    """Returns the UUID of the image.
    """
    STORAGE, INGRP, UUIDGRP = storage_begin()
    try:
        name = request.matchdict['img_name']
    except KeyError:
        STORAGE, INGRP, UUIDGRP = storage_end()
        return {"uuid": None, "error": "not found", "ok": False}
    STORAGE, INGRP, UUIDGRP = storage_end()
    return {"uuid": gs(UUIDGRP[name]), "error": None, "ok": True}


def add_image(name, content, replace=True):
    # content = fileobj.read()
    # oimg = open("tmp/"+name, "wb")
    # oimg.write(content)
    # oimg.close()
    nparr = np.frombuffer(content, np.uint8)
    #log.info("Imported data: {}".format(nparr))
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # log.info("Imported image: {}".format(image))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    STORAGE, INGRP, UUIDGRP = storage_begin()
    if name in INGRP:
        del INGRP[name]
    imgg = INGRP.create_group(name)
    ds = imgg.create_dataset('content', data=image, compression="lzf")

    # image = cv2.resize(image, (0,0), fx=0.08, fy=0.08) # TODO: Calculate sizes from the original size
    log.info("Image '{}' loaded".format(name))
    STORAGE, INGRP, UUIDGRP = storage_end()
    return (imgg, ds)


@img.put()
def put_image(request):
    """Receive image

    Returns UUID of the image
    """
    name = request.matchdict['img_name']

    imgg, ds = add_image(name, request.body)

    uui = uuidgen()
    uuis = str(uui)
    pth = ds.name
    STORAGE, INGRP, UUIDGRP = storage_begin()
    if name in UUIDGRP:
        ouuis = gs(UUIDGRP[name])
        del UUIDGRP[name]
        del UUIDGRP[ouuis]
    UUIDGRP.create_dataset(name, data=uuis)
    UUIDGRP.create_dataset(uuis, data=name)
    #STORAGE.flush()
    STORAGE, INGRP, UUIDGRP = storage_end()
    return {
        "error": None,
        "ok": True,
        "uuid": uuis,
        "content": pth,
        "name": name,
        "namepath": imgg.name
    }


imgctrl = Service(name='image-control-by-uuid',
                  path='/sa-1.0/image-uuid/{img_uuid}',
                  description="Control of image collection")


@imgctrl.delete()
def del_image(request):

    uuids = request.matchdict['img_uuid']
    STORAGE, INGRP, UUIDGRP = storage_begin()
    if uuids not in UUIDGRP:
        return {"error": "not found", "ok": False, "uuid": uuids}
    name = gs(UUIDGRP[uuids])
    try:
        imgg = INGRP[name]
        path = imgg.name
        del INGRP[name]
        del UUIDGRP[name]
        del UUIDGRP[uuids]
    except KeyError:
        STORAGE, INGRP, UUIDGRP = storage_end()
        return {"error": "data not found", "ok": False, "uuid": uuids}
    STORAGE.flush()
    STORAGE, INGRP, UUIDGRP = storage_end()
    return {
        "error": None,
        "ok": True,
        "uuid": uuids,
        "name": name,
        "namepath": path
    }


imgctrln = Service(name='image-control-by-name',
                   path='/sa-1.0/image-name/{img_name}',
                   description="Control of image collection")


@imgctrln.delete()
def del_image_by_name(request):

    name = request.matchdict['img_name']
    STORAGE, INGRP, UUIDGRP = storage_begin()
    if name not in UUIDGRP:
        STORAGE, INGRP, UUIDGRP = storage_end()
        return {"error": "not found", "ok": False, "name": name}
    uuids = gs(UUIDGRP[name])
    try:
        imgg = INGRP[name]
        path = imgg.name
        del INGRP[name]
        del UUIDGRP[name]
        del UUIDGRP[uuids]
    except KeyError:
        STORAGE, INGRP, UUIDGRP = storage_end()
        return {
            "error": "data not found",
            "ok": False,
            "uuid": uuids,
            "name": name
        }
    STORAGE.flush()
    STORAGE, INGRP, UUIDGRP = storage_end()
    return {
        "error": None,
        "ok": True,
        "uuid": uuids,
        "name": name,
        "namepath": path
    }


sactrl = Service(name='segment-any-control',
                 path='/sa-1.0/sa/{img_uuid}/{cmd}',
                 description="Functions of SA on a image identified by uuid")

# cf6649e6-f7d4-11ee-865a-704d7b84fd9f


@sactrl.post()
def start_recognition(request):

    from ..tasks import (sa_start, ANSWERS, rc_set, rc_get, rc_delete,
                         rc_update)

    uuids = request.matchdict['img_uuid']
    cmd = request.matchdict['cmd']
    # cmd = 'start'

    STORAGE, INGRP, UUIDGRP = storage_begin()
    isimg = uuids in UUIDGRP
    STORAGE, INGRP, UUIDGRP = storage_end()

    rd = {"error": False, "ok": True, "cmd": cmd}

    if cmd == "flush":
        ANSWERS.flushdb()
        return rd

    if not isimg:
        return {
            "error": "not found",
            "ok": False,
            "uuid": uuids,
            "cmd": cmd,
            "processuuid": None
        }
    if cmd == "start":
        prevrc = rc_get(uuids)
        if prevrc is not None:
            return {
                "error": "already running",
                "ok": False,
                "uuid": uuids,
                "cmd": cmd,
                "processuuid": prevrc.get("processuuid", None),
                "ready": prevrc.get("ready", False)
            }
        del prevrc
        rc = {"uuid": uuids, "ready": False}
        rc_set(uuids, rc)
        arc = sa_start.delay(uuids)
        puuid = str(arc.id)

        def _u(r):
            r["processuuid"] = puuid

        rc = rc_update(uuids, _u)
        print(rc_get(uuids))
        puuid = rd["processuuid"] = puuid
    elif cmd == "status":
        rd["ready"] = False

        def _a(v, rr):
            rd["ready"] = v
            rd["result"] = rr.get("result", None)

        rc = rc_get(uuids, "ready", _a)
        if rc is None:
            return {
                "error": "no process",
                "ok": False,
                "uuid": uuids,
                "cmd": cmd,
                "ready": None
            }

        rd["processuuid"] = rc["processuuid"]
    elif cmd == "finalize":
        rcg = rc_get(uuids)
        if rcg is None:
            return {
                "error": "not running",
                "ok": False,
                "uuid": uuids,
                "cmd": cmd,
                "ready": None
            }
        rc_delete(uuids)
        rd.update({"ready": rcg["ready"], "processuuid": rcg["processuuid"]})
    elif cmd == "discard":
        rcg = rc_get(uuids)
        if rcg is not None:
            return {
                "error": "still running",
                "description":
                "cannot stop SA, wait its finishing. Use status command.",
                "ok": False,
                "uuid": uuids,
                "cmd": cmd,
                "ready": None
            }
        STORAGE, INGRP, UUIDGRP = storage_begin()
        name = gs(UUIDGRP[uuids])
        imgg = INGRP[name]
        if "masks" in imgg:
            del imgg["masks"]
            rc = "removed"
        else:
            rc = "no mask"
        STORAGE, INGRP, UUIDGRP = storage_end()
        rd.update({
            "ready": None,
            "processuuid": None,
            "description": rc
        })

    return rd


masks = Service(
    name='masks-operation',
    path='/sa-1.0/mask/{img_uuid}/{number}',
    description="Mask operation by image uuid and number, starting from 0.")


@masks.get()
def get_mask(request):
    uuids = request.matchdict['img_uuid']
    number = request.matchdict['number']
    STORAGE, INGRP, UUIDGRP = storage_begin()
    isimg = uuids in UUIDGRP
    STORAGE, INGRP, UUIDGRP = storage_end()
    if not isimg:
        return {
            "error": "not found",
            "ok": False,
            "uuid": uuids,
        }
    STORAGE, INGRP, UUIDGRP = storage_begin()
    name = gs(UUIDGRP[uuids])
    imggrp = INGRP[name]
    try:
        mgrp = imggrp["masks"]
    except KeyError:
        mgrp = None
    d = {}
    if mgrp is not None:
        mg = mgrp[number]
        d["area"] = int(mg.attrs["area"])
        d["predicted_iou"] = float(mg.attrs["predicted_iou"])
        d["stability_score"] = float(mg.attrs["stability_score"])
        for k, v in mg.items():
            if k == "segmentation":
                d[k] = v[()].astype(int).tolist()
            else:
                d[k] = v[()].tolist()
    STORAGE, INGRP, UUIDGRP = storage_end()
    if mgrp is None:
        return {
            "error": "masks not found",
            "description": "Masks not found, run recognition first",
            "ok": False,
            "uuid": uuids,
            "name": name
        }
    rc = {"error": False, "ok": True, "result": d}
    # from pprint import pprint
    # pprint(d.keys())
    return rc


masks_cnt = Service(
    name='masks-operation-return number of them',
    path='/sa-1.0/masks/{img_uuid}',
    description="Mask operation by image uuid. Return number of masks")


@masks_cnt.get()
def get_mask_number(request):
    uuids = request.matchdict['img_uuid']
    STORAGE, INGRP, UUIDGRP = storage_begin()
    isimg = uuids in UUIDGRP
    STORAGE, INGRP, UUIDGRP = storage_end()
    if not isimg:
        return {
            "error": "not found",
            "ok": False,
            "uuid": uuids,
        }
    STORAGE, INGRP, UUIDGRP = storage_begin()
    name = gs(UUIDGRP[uuids])
    imggrp = INGRP[name]
    try:
        mgrp = imggrp["masks"]
    except KeyError:
        mgrp = None
    d = {}
    if mgrp is not None:
        d["number"] = len(mgrp.keys())
    STORAGE, INGRP, UUIDGRP = storage_end()
    if mgrp is None:
        return {
            "error": "masks not found",
            "description": "Masks not found, run recognition first",
            "ok": False,
            "uuid": uuids,
            "name": name
        }
    rc = {"error": False, "ok": True, "result": d}
    return rc


# If number is not supplied, return number of masks

# sparql = Service(name='SPARQL graph operations',
#                  path='/sa-1.0/sparql/{img_uuid}/query/',
#                  description="Sparql Endpoint in the context of an image")

# @sparql.post()
# def get_sparql(request):
# from pyramid.config import add_view
from pyramid.response import Response
import json


def post_sparql(request):

    uuids = request.matchdict['img_uuid']
    STORAGE, INGRP, UUIDGRP = storage_begin()
    isimg = uuids in UUIDGRP
    if isimg:
        name = gs(UUIDGRP[uuids])
        igrp = INGRP[name]
        if "features" in igrp:
            kb = gs(igrp["features"])
        else:
            kb = None
    STORAGE, INGRP, UUIDGRP = storage_end()
    if not isimg:
        return json.dumps({
            "error": "not found",
            "ok": False,
            "uuid": uuids,
        })
    if kb is None:
        return json.dumps({
            "error": "not features",
            "ok": False,
            "uuid": uuids,
            "description": "Perform feature extraction first"
        })

    from rdflib import Graph

    from rdflib.namespace import FOAF, RDF, RDFS, DC
    # import json

    # from SPARQLWrapper import SPARQLWrapper, JSON

    g = Graph(bind_namespaces="rdflib")
    # g = Graph(bind_namespaces="rdflib", store="Oxygraph")
    # g.parse("http://www.w3.org/People/Berners-Lee/card")
    g.parse(data=kb)
    # print(request.body)
    # print(request.POST)
    answer = g.query(request.POST["query"])
    # from pprint import pprint
    ser = answer.serialize(format='json')
    #d = json.loads(ser)
    #return d
    return Response(ser)


featctrl = Service(
    name='feature-recognition-control',
    path='/sa-1.0/fe/{img_uuid}/{cmd}',
    description="Functions of Feature Extraction on a image identified by uuid"
)


@featctrl.post()
def start_fe(request):

    from ..tasks import (feature_recognition, ANSWERS, rc_set, rc_get,
                         rc_delete, rc_update)

    uuids = request.matchdict['img_uuid']
    cmd = request.matchdict['cmd']
    # cmd = 'start'

    STORAGE, INGRP, UUIDGRP = storage_begin()
    isimg = uuids in UUIDGRP
    STORAGE, INGRP, UUIDGRP = storage_end()

    rd = {"error": False, "ok": True, "cmd": cmd}

    if cmd == "flush":
        ANSWERS.flushdb()
        return rd

    if not isimg:
        return {
            "error": "not found",
            "ok": False,
            "uuid": uuids,
            "cmd": cmd,
            "processuuid": None
        }
    if cmd == "start":
        STORAGE, INGRP, UUIDGRP = storage_begin()
        name = gs(UUIDGRP[uuids])
        imgg = INGRP[name]
        mok = "masks" in imgg
        STORAGE, INGRP, UUIDGRP = storage_end()
        if not mok:
            return {
                "error": "masks not found",
                "description": "masks not found: apply SA first to the image",
                "ok": False,
                "uuid": uuids,
                "cmd": cmd,
                "processuuid": None
            }

        prevrc = rc_get(uuids)
        if prevrc is not None:
            return {
                "error": "already running",
                "ok": False,
                "uuid": uuids,
                "cmd": cmd,
                "processuuid": prevrc.get("processuuid", None),
                "ready": prevrc.get("ready", False)
            }
        del prevrc
        rc = {"uuid": uuids, "ready": False}
        rc_set(uuids, rc)
        arc = feature_recognition.delay(uuids)
        puuid = str(arc.id)

        def _u(r):
            r["processuuid"] = puuid

        rc = rc_update(uuids, _u)
        print(rc_get(uuids))
        puuid = rd["processuuid"] = puuid
    elif cmd == "status":
        rd["ready"] = False

        def _a(v, rr):
            rd["ready"] = v
            rd["result"] = rr.get("result", None)

        rc = rc_get(uuids, "ready", _a)
        if rc is None:
            return {
                "error": "no process",
                "ok": False,
                "uuid": uuids,
                "cmd": cmd,
                "ready": None
            }

        rd["processuuid"] = rc["processuuid"]
    elif cmd == "finalize":
        rcg = rc_get(uuids)
        if rcg is None:
            return {
                "error": "not running",
                "ok": False,
                "uuid": uuids,
                "cmd": cmd,
                "ready": None
            }
        rc_delete(uuids)
        rd.update({"ready": rcg["ready"], "processuuid": rcg["processuuid"]})

    return rd


"""
Определение операций API с точки зрения методов HTTP
Протокол HTTP определяет несколько методов, назначающих запросу семантическое значение. Ниже приведены наиболее распространенные методы HTTP, используемые большинством веб-API RESTful:

- GET. Возвращает представление ресурса по указанному универсальному коду ресурса (URI). Текст ответного сообщения содержит сведения о запрашиваемом ресурсе.

- POST. Создает новый ресурс по указанному URI. Текст запроса содержит сведения о новом ресурсе. Обратите внимание, что метод POST также можно использовать для запуска операций, не относящихся непосредственно к созданию ресурсов.

- PUT. Создает или заменяет ресурсы по указанному URI. В тексте сообщения запроса указан создаваемый или обновляемый ресурс.

- PATCH. Выполняет частичное обновление ресурса. Текст запроса определяет набор изменений, применяемых к ресурсу.

- DELETE. Удаляет ресурс по указанному URI.
"""
