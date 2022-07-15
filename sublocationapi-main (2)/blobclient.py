# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import urllib3
from azure.storage.blob import BlockBlobService




account_name='ocuityappimages'
account_key='csrp5zAhcZ+LjDJ5EAa+sAoeNRWUej3VqMDGQ3RkwS20UTKnvMNeGzAWa4Cr/XcdXWPk7LiNBci2DKnt7g9i5g=='
CONTAINER_NAME='suboutput'
con_str='DefaultEndpointsProtocol=https;AccountName=ocuityappimages;AccountKey=csrp5zAhcZ+LjDJ5EAa+sAoeNRWUej3VqMDGQ3RkwS20UTKnvMNeGzAWa4Cr/XcdXWPk7LiNBci2DKnt7g9i5g==;EndpointSuffix=core.windows.net'
CONTAINER_URL = 'https://' + account_name + '.blob.core.windows.net/' + CONTAINER_NAME + '/'

http = urllib3.PoolManager()
blob_service_client = BlockBlobService(account_name, account_key)

def upload_to_blob(blobName, img_byte):
    # Upload to blob storage
    
    blob_service_client.create_blob_from_bytes(CONTAINER_NAME, blobName, img_byte)  
    
    # Get the url for uploaded images
    image_url = blob_service_client.make_blob_url(CONTAINER_NAME, blobName)
    
    return image_url
    