from pyramid_celery import celery_app as app

import h5py
import cv2
import logging
import redis

from .views.rest import (storage_begin, storage_end, STORAGE, UUIDGRP, INGRP)

log = logging.getLogger(__name__)

RUNNING = {}

ANSWERDB = 6
ANSWERS = redis.Redis(db=ANSWERDB)

@app.task
def sa_start(uuids):
    ANSWERS.delete(uuids)
    log.info('creating task processing image identified by UUID {}'.format(uuids))
    time.delay(10)
    ANSWERS.set(uuids, "ready")
    # task = TaskItem(task=task)
    # DBSession.add(task)
    # transaction.commit()
