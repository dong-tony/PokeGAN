#%%
import glob, os, shutil
from PIL import Image
from tqdm import tqdm
import pandas as pd
from itertools import chain
import math

#%% Image resizing
# size = 224, 224

# for image in tqdm(glob.glob("*.png")):
#     im = Image.open(image)
#     im.thumbnail(size)
#     im.save("resized/" + image, "PNG")

#%% Pokemon image sorting by type
df = pd.read_csv("pokemon types.csv")
id = 592
#%%
for id in tqdm(range(1, 7)):
        for image in chain(glob.glob("{}.png".format(id)), glob.glob("{}-*.png".format(id)), glob.glob("{}f.png".format(id))):
                ptype = df.loc[df['#'] == id, ['Type 1']].iloc[0]['Type 1']
                stype = df.loc[df['#'] == id, ['Type 2']].iloc[0]['Type 2']
                if not os.path.exists(ptype):
                        os.makedirs(ptype)
                if isinstance(stype, str):
                        if not os.path.exists(stype):
                                os.makedirs(stype)
                        shutil.copy(image, stype + '/' + image)
                shutil.move(image, ptype + '/' + image)
#%%
