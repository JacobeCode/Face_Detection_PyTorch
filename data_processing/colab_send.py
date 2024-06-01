import os
import pkg_resources
import sys

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
from datetime import date

# from google.oauth2 import service_account
# from googleapiclient.discovery import build

# credentials = service_account.Credentials.service_account_file(ACCOUNT_FILE, scopes=SERVICE)
# drive_service = build('drive', 'v3', credentials=credentials)

# Simple function for uploading files from dataset to Google Drive (for small examples)
def gdrive_upload():
    # Credentials and scope
    CREDS = "credentials_directory"
    SCOPE = ['https://www.googleapis.com/auth/drive']

    # Auth for gdrive
    google_auth = GoogleAuth()
    google_auth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
        pkg_resources.resource_filename(__name__, CREDS), scopes=SCOPE 
    )

    # Init for gdrive with authorization
    gdrive = GoogleDrive(google_auth)

    # Parent dir for dataset
    parent_dir_id = '1n3Tg1-PT9PrrTyr3gIB9VCBMt6OtBm4R'
    new_folder_name = 'img_data'

    # Meta for creating folders 
    folder_meta = {
        'title': new_folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [{'id': parent_dir_id}]
    }

    # Checking possible doubles in cloud
    folders = gdrive.ListFile(
        {'q': "'"+parent_dir_id+"' in parents and trashed=false"}).GetList()

    # Chceking titles of folds
    new_folder_id = None
    for files in folders:
        if files['title'] == folder_meta['title']:
            new_folder_id = files['id']

    if new_folder_id == None:
        new_folder = gdrive.CreateFile(folder_meta)
        new_folder.Upload()
        new_folder_id = new_folder.get('id')

    # Listing db files
    db_dir = "dataset_directory"
    img_list = os.listdir(db_dir)

    # Uploading whole dataset
    for iter, img in enumerate(img_list):
        img_file = gdrive.CreateFile(
            {'parents': [{'id': new_folder_id}], 'title': img}
        )

        img_file.SetContentFile(db_dir + '\\' + img)
        img_file.Upload()
        print(f"---------- File {iter} / {len(img_list)} ----------")

gdrive_upload()