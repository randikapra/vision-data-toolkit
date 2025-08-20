"""Gdrive Manager module."""

import os
import io
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError

from loguru import logger
from tqdm import tqdm

class GDriveManager:
    """Google Drive manager for dataset operations"""
    
    SCOPES = ['https://www.googleapis.com/auth/drive']
    
    def __init__(self, credentials_path: str, token_path: str):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None
        self._lock = threading.Lock()
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Drive API"""
        creds = None
        
        # Load existing token
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, self.SCOPES)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(f"Credentials file not found: {self.credentials_path}")
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('drive', 'v3', credentials=creds)
        logger.info("Google Drive authentication successful")
    
    def create_folder(self, name: str, parent_id: Optional[str] = None) -> str:
        """Create a folder in Google Drive"""
        file_metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        if parent_id:
            file_metadata['parents'] = [parent_id]
        
        try:
            folder = self.service.files().create(body=file_metadata, fields='id').execute()
            folder_id = folder.get('id')
            logger.info(f"Created folder '{name}' with ID: {folder_id}")
            return folder_id
        except HttpError as e:
            logger.error(f"Failed to create folder '{name}': {e}")
            raise
    
    def list_files(self, folder_id: str, mime_type: Optional[str] = None) -> List[Dict]:
        """List files in a Google Drive folder"""
        query = f"'{folder_id}' in parents and trashed=false"
        if mime_type:
            query += f" and mimeType='{mime_type}'"
        
        try:
            results = self.service.files().list(
                q=query,
                pageSize=1000,
                fields="nextPageToken, files(id, name, size, modifiedTime, md5Checksum)"
            ).execute()
            
            files = results.get('files', [])
            logger.info(f"Found {len(files)} files in folder {folder_id}")
            return files
        except HttpError as e:
            logger.error(f"Failed to list files in folder {folder_id}: {e}")
            return []
    
    def get_file_info(self, file_id: str) -> Optional[Dict]:
        """Get file information"""
        try:
            file_info = self.service.files().get(
                fileId=file_id,
                fields="id, name, size, modifiedTime, md5Checksum, parents"
            ).execute()
            return file_info
        except HttpError as e:
            logger.error(f"Failed to get file info for {file_id}: {e}")
            return None
    
    def download_file(self, file_id: str, local_path: str, 
                     chunk_size: int = 8192, show_progress: bool = True) -> bool:
        """Download a file from Google Drive"""
        try:
            with self._lock:
                request = self.service.files().get_media(fileId=file_id)
            
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(local_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request, chunksize=chunk_size)
                done = False
                
                if show_progress:
                    pbar = tqdm(desc=f"Downloading {Path(local_path).name}", unit='B', unit_scale=True)
                
                while not done:
                    status, done = downloader.next_chunk()
                    if show_progress and status:
                        pbar.total = status.total_size
                        pbar.update(status.resumable_progress - pbar.n)
                
                if show_progress:
                    pbar.close()
            
            logger.info(f"Downloaded file to {local_path}")
            return True
            
        except HttpError as e:
            logger.error(f"Failed to download file {file_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {file_id}: {e}")
            return False
    
    def upload_file(self, local_path: str, gdrive_folder_id: str, 
                   filename: Optional[str] = None) -> Optional[str]:
        """Upload a file to Google Drive"""
        if not os.path.exists(local_path):
            logger.error(f"Local file not found: {local_path}")
            return None
        
        filename = filename or Path(local_path).name
        
        file_metadata = {
            'name': filename,
            'parents': [gdrive_folder_id]
        }
        
        media = MediaFileUpload(local_path, resumable=True)
        
        try:
            with self._lock:
                file = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
            
            file_id = file.get('id')
            logger.info(f"Uploaded {filename} with ID: {file_id}")
            return file_id
            
        except HttpError as e:
            logger.error(f"Failed to upload {filename}: {e}")
            return None
    
    def batch_download(self, file_list: List[Dict], local_dir: str, 
                      max_workers: int = 3) -> Dict[str, bool]:
        """Download multiple files concurrently"""
        results = {}
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        def download_single(file_info):
            file_id = file_info['id']
            filename = file_info['name']
            local_path = local_dir / filename
            success = self.download_file(file_id, str(local_path), show_progress=False)
            return filename, success
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_single, file_info): file_info['name'] 
                      for file_info in file_list}
            
            with tqdm(total=len(file_list), desc="Batch downloading") as pbar:
                for future in as_completed(futures):
                    filename, success = future.result()
                    results[filename] = success
                    pbar.update(1)
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Batch download completed: {successful}/{len(file_list)} successful")
        return results
    
    def stream_file(self, file_id: str, chunk_size: int = 8192) -> Generator[bytes, None, None]:
        """Stream file content in chunks"""
        try:
            with self._lock:
                request = self.service.files().get_media(fileId=file_id)
            
            with io.BytesIO() as f:
                downloader = MediaIoBaseDownload(f, request, chunksize=chunk_size)
                done = False
                
                while not done:
                    status, done = downloader.next_chunk()
                    f.seek(0)
                    chunk = f.read()
                    if chunk:
                        yield chunk
                    f.seek(0)
                    f.truncate(0)
                    
        except HttpError as e:
            logger.error(f"Failed to stream file {file_id}: {e}")
            raise
    
    def get_folder_size(self, folder_id: str) -> float:
        """Get total size of folder in GB"""
        files = self.list_files(folder_id)
        total_size = 0
        
        for file in files:
            if 'size' in file:
                total_size += int(file['size'])
        
        return total_size / (1024**3)  # Convert to GB
    
    def verify_file_integrity(self, file_id: str, local_path: str) -> bool:
        """Verify file integrity using MD5 checksum"""
        try:
            file_info = self.get_file_info(file_id)
            if not file_info or 'md5Checksum' not in file_info:
                logger.warning(f"No MD5 checksum available for file {file_id}")
                return True  # Assume valid if no checksum
            
            # Calculate local file MD5
            hasher = hashlib.md5()
            with open(local_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            
            local_md5 = hasher.hexdigest()
            remote_md5 = file_info['md5Checksum']
            
            is_valid = local_md5 == remote_md5
            if not is_valid:
                logger.error(f"File integrity check failed for {local_path}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error verifying file integrity: {e}")
            return False
    
    def search_files(self, query: str, folder_id: Optional[str] = None) -> List[Dict]:
        """Search for files in Google Drive"""
        search_query = query
        if folder_id:
            search_query = f"'{folder_id}' in parents and ({query})"
        
        try:
            results = self.service.files().list(
                q=search_query,
                pageSize=100,
                fields="files(id, name, size, modifiedTime)"
            ).execute()
            
            return results.get('files', [])
        except HttpError as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete_file(self, file_id: str) -> bool:
        """Delete a file from Google Drive"""
        try:
            self.service.files().delete(fileId=file_id).execute()
            logger.info(f"Deleted file {file_id}")
            return True
        except HttpError as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            return False
    
    def create_dataset_structure(self, dataset_name: str, 
                               subdirs: Optional[List[str]] = None) -> str:
        """Create folder structure for a dataset"""
        # Create main dataset folder
        main_folder_id = self.create_folder(dataset_name)
        
        # Create subdirectories if specified
        if subdirs:
            for subdir in subdirs:
                self.create_folder(subdir, main_folder_id)
        
        return main_folder_id
    
    def sync_folder(self, gdrive_folder_id: str, local_dir: str, 
                   download_missing: bool = True) -> Dict[str, str]:
        """Synchronize Google Drive folder with local directory"""
        results = {"downloaded": [], "skipped": [], "failed": []}
        
        # Get remote files
        remote_files = self.list_files(gdrive_folder_id)
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Get local files
        local_files = {f.name: f for f in local_dir.iterdir() if f.is_file()}
        
        for remote_file in remote_files:
            filename = remote_file['name']
            local_path = local_dir / filename
            
            should_download = False
            
            if filename not in local_files:
                should_download = download_missing
                reason = "missing locally"
            elif not self.verify_file_integrity(remote_file['id'], str(local_path)):
                should_download = True
                reason = "integrity check failed"
            else:
                results["skipped"].append(filename)
                continue
            
            if should_download:
                if self.download_file(remote_file['id'], str(local_path)):
                    results["downloaded"].append(filename)
                    logger.info(f"Downloaded {filename} ({reason})")
                else:
                    results["failed"].append(filename)
        
        return results