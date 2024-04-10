from cornice import Service
from uuid import uuid1 as uuidgen
import mmh3
import h5py
import cv2
import numpy as np

import logging

log = logging.getLogger(__name__)

img = Service(name='imgstore',
              path='/sa-1.0/image/{img_name}',
              description="Segment Any processor image collection")

try:
    STORAGE = h5py.File('data.hdf5', 'a')
    log.info('Successfully opened the database')
except OSError:
    log.warning('Killed the previous database')
    STORAGE = h5py.File('data.hdf5', 'w')

INGRP = STORAGE.require_group("input")
UUIDGRP = STORAGE.require_group("uuid")

STORAGE.flush()


def mmh(content):
    return mmh3.hash128(content)


def gs(str_ds):
    return str_ds.asstr()[()]


@img.get()
def get_id(request):
    """Returns the UUID of the image.
    """
    try:
        name = request.matchdict['img_name']
    except KeyError:
        return {"uuid": None, "error": "not found", "ok": True}
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

    if name in INGRP:
        del INGRP[name]
    imgg = INGRP.create_group(name)
    ds = imgg.create_dataset('content', data=image, compression="lzf")

    # image = cv2.resize(image, (0,0), fx=0.08, fy=0.08) # TODO: Calculate sizes from the original size
    log.info("Image '{}' loaded".format(name))
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
    if name in UUIDGRP:
        ouuis = gs(UUIDGRP[name])
        del UUIDGRP[name]
        del UUIDGRP[ouuis]
    UUIDGRP.create_dataset(name, data=uuis)
    UUIDGRP.create_dataset(uuis, data=name)
    STORAGE.flush()
    return {
        "error": None,
        "ok": True,
        "uuid": uuis,
        "content": pth,
        "name": name,
        "namepath": imgg.name
    }


imgctrl = Service(name='imgcontrol',
                  path='/sa-1.0/image-uuid/{img_uuid}',
                  description="Control of image collection")

@imgctrl.delete()
def put_image(request):
    """Receive image

    Returns UUID of the image
    """
    uuids = request.matchdict['img_uuid']
    if uuids not in UUIDGRP:
        return {"error":"not found", "ok":False, "uuid": uuids}
    name = gs(UUIDGRP[uuids])
    try:
        imgg = INGRP[name]
        path = imgg.name
        del INGRP[name]
        del UUIDGRP[name]
        del UUIDGRP[uuids]
    except KeyError:
        return {"error":"data not found", "ok":False, "uuid": uuids}
    STORAGE.flush()
    return {
        "error": None,
        "ok": True,
        "uuid": uuids,
        "name": name,
        "namepath": path
    }



"""
Определение операций API с точки зрения методов HTTP
Протокол HTTP определяет несколько методов, назначающих запросу семантическое значение. Ниже приведены наиболее распространенные методы HTTP, используемые большинством веб-API RESTful:

- GET. Возвращает представление ресурса по указанному универсальному коду ресурса (URI). Текст ответного сообщения содержит сведения о запрашиваемом ресурсе.

- POST. Создает новый ресурс по указанному URI. Текст запроса содержит сведения о новом ресурсе. Обратите внимание, что метод POST также можно использовать для запуска операций, не относящихся непосредственно к созданию ресурсов.

- PUT. Создает или заменяет ресурсы по указанному URI. В тексте сообщения запроса указан создаваемый или обновляемый ресурс.

- PATCH. Выполняет частичное обновление ресурса. Текст запроса определяет набор изменений, применяемых к ресурсу.

- DELETE. Удаляет ресурс по указанному URI.
"""
