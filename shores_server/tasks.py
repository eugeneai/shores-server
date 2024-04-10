from pyramid_celery import celery_app as app

import h5py
import cv2
import logging
from .views.rest import (storage_begin, storage_end, STORAGE, UUIDGRP, INGRP)

log = logging.getLogger(__name__)

@app.task
def sa_start(uuids):
    log.info('creating task processing image identified by UUID {}'.format(uuids))
    # task = TaskItem(task=task)
    # DBSession.add(task)
    # transaction.commit()
