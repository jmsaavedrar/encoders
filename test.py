import numpy as np
import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt

if __name__ == '__main__' :
    
    with open("data/voc.txt", "r+") as output:        
        files = [f for f in output]

    feats = np.load('data/dinov2_voc_feats.npy')
    print(files)
    print(feats.shape)
    norm2 = np.linalg.norm(feats, ord = 2, keepdims = True)
    feats_n = feats / norm2
    sim = np.matmul(feats_n, np.transpose(feats_n))
    sim_idx = np.argsort(-sim, axis = 1)

    query = 10
    best_idx = sim_idx[query, :11]
    print(best_idx)
    print(sim[query, best_idx])

    collage = np.zeros((64,64*12, 3))
    w = 0
    for idx in best_idx :        
        im = io.imread(files[idx].strip())
        im = transform.resize(im, (64,64)) 
        collage[:,w:w+64,:] = im
        w = w + 64

    plt.imshow(collage)
    plt.show()