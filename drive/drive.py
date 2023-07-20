import os
import io
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account

class DriveService:
    def __init__(self, tokenjson):
        self.tokenjson = tokenjson
        credentials = service_account.Credentials.from_service_account_file(
        tokenjson,
        scopes=['https://www.googleapis.com/auth/drive'],
        )
        self.drive_service = build('drive', 'v3', credentials=credentials)
    
    def show(self):
        results = self.drive_service.files().list().execute()
        files = results.get('files', [])
        return files

    def download(self, fileID, saveto='./'):
        request = self.drive_service.files().get_media(fileId=fileID)
        bytesio = io.BytesIO()
        downloader = MediaIoBaseDownload(bytesio, request)

        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(F'Download {int(status.progress() * 100)}.')

        results = self.drive_service.files().list().execute()
        files = results.get('files', [])
        for f in files:
            if f['id'] == fileID:
                filename = f['name']
                break
        path = os.path.join(saveto, filename)

        with open(path, "wb") as f:
            f.write(bytesio.getbuffer())
    
    def mkdir(self, folder_name):
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        created_folder = self.drive_service.files().create(body=folder_metadata).execute()
        return created_folder
    
    def upload(self, file_path, parentID=None):
        file_name = os.path.basename(file_path)
        file_metadata = {
            'name': file_name, 
            'parent':[parentID]
            }
        media = MediaFileUpload(file_path, resumable=True)
        uploaded_file = self.drive_service.files().create(body=file_metadata, media_body=media).execute()
        return uploaded_file

    def delete(self, fileID):
        self.drive_service.files().delete(fileId=fileID).execute()

    def deleteall(self):
        results = self.drive_service.files().list().execute()
        files = results.get('files', [])
        for f in files:
            self.drive_service.files().delete(fileId=f['id']).execute()

