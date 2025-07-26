import os
import shutil
import requests

# Download RARE-UNet (best model)

def download_model():
    url = '...'

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
        
    response = requests.get(url)
    with open('checkpoints/rare_unet_best_model.pth', 'wb') as f:
        f.write(response.content)

if __name__ == '__main__':
    download_model()