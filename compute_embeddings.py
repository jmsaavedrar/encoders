import torch                                                                                                                                                                                                                                                                                                                                                                                
from torchvision import transforms, models
import numpy as np
import os
from PIL import Image
import pickle                                                                                                                                                                                                              


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
    
def compute_embeddings(lfiles) :
    n = len(lfiles)
    dim = 384
    features = np.zeros((n, dim), dtype = np.float32)
    with torch.no_grad():
        for i, ifile in enumerate(lfiles) :
            image = Image.open(ifile).convert('RGB')
            image = preprocess(image).unsqueeze(0).to(device)
            features[i,:] = model(image).cpu()[0,:]
            if i%10 == 0 :
                print('{}/{}'.format(i, n))
    return features

data_dir = '/home/data/Homy'
fdata = '/home/data/Homy/eval.txt'
lfiles = []
with open(fdata) as f:
    for ifile in f :
        ifile = os.path.join(data_dir, ifile.strip())
        lfiles.append(ifile)
features = compute_embeddings(lfiles)
print(features.shape)
with open('data/dinov2_homy.pk', 'wb') as f:
    pickle.dump(lfiles, f)
np.save("data/dinov2_homy_feats.npy", features)
print('saving data ok')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             