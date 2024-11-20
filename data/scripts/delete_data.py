import shutil
from data.constants import Constants

def delete_data():
    shutil.rmtree(Constants.IMAGE_DIR)
    shutil.rmtree(Constants.MASK_DIR)

if __name__ == '__main__':
    delete_data()