from downloader.g_down import g_down
from downloader import file_id

downloader = g_down(use_pydrive=False)

ffhq_f = file_id.style_gan['ffhq_f']
e4e = file_id.e4e["ffhq_encode"]

downloader.download_file(file_id=ffhq_f['id'],file_name=ffhq_f['name'])
downloader.download_file(file_id=e4e['id'],file_name=e4e['name'])

