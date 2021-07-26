from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
gauth = GoogleAuth()
gdrive = GoogleDrive(gauth)
gfile = gdrive.CreateFile({'parents': [{'id': '1FDVkMC80Wncc24DPwoDIFzwJaTG_OMoH'}]})
# Read file and set it as the content of this instance.
gfile.SetContentFile("./ckpt/slot_crf/model.ckpt")
gfile.Upload() # Upload the file.
