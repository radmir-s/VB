import os
import paramiko
import yaml
import argparse

def load_credentials(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

parser = argparse.ArgumentParser(description='Send a python script to a server and submit it as sbatch job.')
parser.add_argument('-c', '--credentials', type=str, help='Path to the YAML file with credentials.')
parser.add_argument('-s', '--script', type=str, help='Path to the script file to transfer and run.')
parser.add_argument('-a', '--args', type=str, help='Path to the script file to transfer and run.', default="")
args = parser.parse_args()

credentials = load_credentials(args.credentials)
hostname = credentials['hostname']
username = credentials['username']
password = credentials['password']
remotehome = credentials['remotehome']
localhome = credentials['localhome']

ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(hostname, username=username, password=password)
# sftp = ssh_client.open_sftp()

# remote_script = os.path.join(remotehome, args.script)
# local_script = os.path.join(localhome, args.script)
# sftp.put(local_script, remote_script)

stdin, stdout, stderr = ssh_client.exec_command(f'cd {remotehome}; git pull; sbatch tensorjob.sh {remote_script} {args.args};')
print(stdout.read().decode())

sftp.close()
ssh_client.close()