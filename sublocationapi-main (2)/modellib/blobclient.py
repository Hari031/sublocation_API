# decompyle3 version 3.9.0
# Python bytecode version base 3.7.0 (3394)
# Decompiled from: Python 3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: C:\work\GM\code\sublocation\sublocation_api\modellib\blobclient.py
# Compiled at: 2022-03-09 23:26:16
# Size of source mod 2**32: 1139 bytes
import urllib3
from azure.storage.blob import BlockBlobService
account_name = 'ocuityappimages'
account_key = 'csrp5zAhcZ+LjDJ5EAa+sAoeNRWUej3VqMDGQ3RkwS20UTKnvMNeGzAWa4Cr/XcdXWPk7LiNBci2DKnt7g9i5g=='
CONTAINER_NAME = 'suboutput'
con_str = 'DefaultEndpointsProtocol=https;AccountName=ocuityappimages;AccountKey=csrp5zAhcZ+LjDJ5EAa+sAoeNRWUej3VqMDGQ3RkwS20UTKnvMNeGzAWa4Cr/XcdXWPk7LiNBci2DKnt7g9i5g==;EndpointSuffix=core.windows.net'
CONTAINER_URL = 'https://' + account_name + '.blob.core.windows.net/' + CONTAINER_NAME + '/'
http = urllib3.PoolManager()
blob_service_client = BlockBlobService(account_name, account_key)

def upload_to_blob(blobName, img_byte):
    blob_service_client.create_blob_from_bytes(CONTAINER_NAME, blobName, img_byte)
    image_url = blob_service_client.make_blob_url(CONTAINER_NAME, blobName)
    return image_url