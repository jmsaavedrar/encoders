import torch
from torchvision import transforms, models
import numpy as np

from PIL import Image
import argparse

if __name__ == '__main__':
    # Comprobamos si hay GPU disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Definimos las transformaciones para la imagen
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]),
        ])
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    #torch.load('models/dinov2_vits14_pretrain.pth', weights_only = False)
    print(model)
    image_path = "images/example_2.jpg"
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)  # add an extra dimension for batch
    # Pasamos la imagen por el modelo
    with torch.no_grad():
        features = model(image)
        dim = features.shape[1]
        
        import glob
        imagedir = "/hd_data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/"
        files = glob.glob(imagedir + "*.jpg")
        n_images = len(files)
        features = np.zeros((n_images, dim), dtype = np.float32)
        with open("data/voc.txt", "w+") as output:            
            for i, file in enumerate(files) :
                output.write(file + "\n")                
                image = Image.open(file).convert('RGB')
                image = preprocess(image).unsqueeze(0).to(device)
                features[i,:] = model(image).cpu()[0,:]
                if i%100 == 0 :
                    print('{}/{}'.format(i, n_images))
            print('saving data')
        
        print(features)
             
        np.save("data/dinov2_voc_feats.npy", features)
        print('saving data ok')