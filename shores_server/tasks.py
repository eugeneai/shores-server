from pyramid_celery import celery_app as app

import h5py
import cv2
import logging
import redis
import json
import time

from .views.rest import (storage_begin, storage_end, STORAGE, UUIDGRP, INGRP)

log = logging.getLogger(__name__)

RUNNING = {}

ANSWERDB = 6
ANSWERS = redis.Redis(db=ANSWERDB)
ANS = 'processes'

def rc_get(uuid, field=None, f=None):
    rc = ANSWERS.get(uuid)
    print("GET rc:", rc)
    if rc is not None:
        print("Read rc:", rc)
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
    print("rc_set:", uuid, dict_data, js)
    ANSWERS.set(uuid, js)
    print("rc_set ctrl:", ANSWERS.get(uuid))

def rc_remove(uuid):
    return ANSWERS.remove(uuid)


def rc_update(uuid, f):
    rc = rc_get(uuid)
    rc_ = f(rc)
    if rc_ is not None:
        rc = rc_
    rc_set(uuid, rc)

@app.task
def sa_start(uuids):
    def fd(js):
        js["ready"] = False
    rc_update(uuids, fd)
    log.info('creating task processing image identified by UUID {}'.format(uuids))
    time.sleep(2)
    def f(js):
        js["ready"] = True
        js["result"] = "a Good result"
    rc_update(uuids, f)
    # task = TaskItem(task=task)
    # DBSession.add(task)
    # transaction.commit()
