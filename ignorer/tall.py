from pycaret.classification import *
import pycaret
#loaded_model = pycaret.load_model('code')
file_path1   = 'code'

loaded_model= load_model(file_path1)
model_version = loaded_model.__version__
print(f"PyCaret version of the saved model: {model_version}")
