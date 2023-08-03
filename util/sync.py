import os
import paramiko
import yaml
import argparse

def load_credentials(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

def sync_folders(sftp, remote_dir, local_dir):
    # get list of files in remote directory
    remote_files = set(sftp.listdir(remote_dir))
    
    # get list of files in local directory
    local_files = set(os.listdir(local_dir))

    # find files present in remote directory but not in local
    files_to_download = remote_files - local_files

    for file in files_to_download:
        remote_file = os.path.join(remote_dir, file)
        local_file = os.path.join(local_dir, file)
        print(remote_file, local_file)
        sftp.get(remote_file, local_file)

    # find files present in local directory but not in remote
    files_to_upload = local_files - remote_files

    for file in files_to_upload:
        local_file = os.path.join(local_dir, file)
        remote_file = os.path.join(remote_dir, file)
        sftp.put(local_file, remote_file)

parser = argparse.ArgumentParser(description='Load credentials from YAML file.')
parser.add_argument('-c', '--credentials', type=str, help='Path to the YAML file with credentials')
parser.add_argument('-d', '--directory', type=str, help='Path to the directory to sync')

args = parser.parse_args()

credentials = load_credentials(args.credentials)

hostname = credentials['hostname']
username = credentials['username']
password = credentials['password']

ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(hostname, username=username, password=password)
sftp = ssh_client.open_sftp()

remote_dir = os.path.join(credentials['remotehome'], args.directory)
local_dir = os.path.join(credentials['localhome'], args.directory)

sync_folders(sftp, remote_dir, local_dir)

sftp.close()
ssh_client.close()
