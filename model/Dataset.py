import torch
from torch.utils.data import Dataset
import imageio
import numpy as np
import torch.nn.functional as F
from PIL import Image

class NewsDataset(Dataset):
    def __init__(
        self,
        dct_data
        ):
        super(NewsDataset, self).__init__()
        self.data = dct_data
        #print(type(graph_data))    

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        instance = self.data[index]
        label = instance["label_binary"]
        image_clip = instance["image_clip"]
        post_clip = instance["post_clip"]
        text_bert = instance["text_bert"]
        image_ori = instance["image_ori"].resize((256,256))
        image_ori = np.array(image_ori)
        #image_ori  = imageio.imread(image_ori)
        if image_ori.shape[-1] == 4:
            image_ori = self.rgba2rgb(image_ori)
        image_ori = torch.from_numpy(image_ori.astype(np.float32) / 255).permute(2, 0, 1)
        #print(image_ori.shape)
        event = instance["event"]
        text_ent = instance["text_ent"]
        image_ent = instance["image_ent"]
        reverse = instance["reverse"]
        timeinfo = instance["timeinfo"]
        Id = instance["Id"]
        

        return  (label, image_clip, post_clip, text_bert, image_ori, event, text_ent, image_ent, reverse, timeinfo, Id)
       
    
        
    def collate_fn(self, samples) :   
        batch={}
        label_list=[]
            
        for s in samples:
            label_list.append(int(s[0]))
        label_list=torch.tensor(label_list)
        batch["label_list"]=label_list

        # ========== news ==============
        
        ids=[s[1] for s in samples]
        batch['image_clip'] = torch.Tensor(np.array([item.cpu().numpy() for item in ids]))
        
        ids=[s[2] for s in samples]
        batch['post_clip'] = torch.Tensor(np.array([item.cpu().numpy() for item in ids]))
        ids=[s[3] for s in samples]
        batch['text_bert'] = torch.Tensor(np.array([item.cpu().numpy() for item in ids]))
        ids=[s[4] for s in samples]
        batch['image_ori'] = torch.Tensor(np.array([item.cpu().numpy() for item in ids]))
        ids=[s[5] for s in samples]
        batch['event'] = torch.Tensor(np.array([item.cpu().numpy() for item in ids]))
        ids=[s[6] for s in samples]
        batch['text_ent'] = torch.Tensor(np.array([item.cpu().numpy() for item in ids]))
        ids=[s[7] for s in samples]
        batch['image_ent'] = torch.Tensor(np.array([item.cpu().numpy() for item in ids]))
        ids=[s[8] for s in samples]
        batch['reverse'] = torch.Tensor(np.array([item.cpu().numpy() for item in ids]))
        ids=[s[9] for s in samples]
        batch['time_info'] = torch.Tensor(np.array([item.cpu().numpy() for item in ids]))
        ids=[s[10] for s in samples]
        batch['Id'] = ids
        return batch
    
    def rgba2rgb(self, rgba, background=(255, 255, 255)):
        row, col, ch = rgba.shape

        rgb = np.zeros((row, col, 3), dtype='float32')
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

        a = np.asarray(a, dtype='float32') / 255.0

        R, G, B = background

        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B

        return np.asarray(rgb, dtype='uint8')
    
   
        
        
        

    