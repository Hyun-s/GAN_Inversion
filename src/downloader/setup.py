from downloader.g_down import g_down
from downloader import file_id
import os, subprocess

use_pydrive = False
use_colab = False
downloader = g_down(use_pydrive=False)

ffhq_f = file_id.style_gan['ffhq_f']
e4e = file_id.e4e["ffhq_encode"]

downloader.download_file(file_id=ffhq_f['id'],file_name=ffhq_f['name'])
downloader.download_file(file_id=e4e['id'],file_name=e4e['name'])



def set_ninja():
    command_list = ['wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip',
                    'sudo unzip ninja-linux.zip -d /usr/local/bin/',
                    'sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force']
    results = []
    for command in command_list:
        result = subprocess.call(command, shell=True)

if use_colab:
    set_ninja()