import torch
import csv
import json
import os
from sklearn.metrics import classification_report,auc,precision_recall_curve
from tqdm import tqdm
from torch.nn import functional as F
import pickle
class Tester:
    def __init__(self, args, Classifier, test_set, test_size):
        self.args = args
        self.classifier = Classifier
        self.test_set = test_set
        self.test_size = test_size

        
    def test(self):
        classdict = self.classifier.state_dict()
        pre_weights = torch.load(os.path.join(self.args.test_path ,f"{self.args.name}ckpt.classifier"), map_location= self.args.device).state_dict()
        pretrained_dict = {k: v for k, v in pre_weights.items() if k in classdict}
        classdict.update(pretrained_dict)

        self.classifier.load_state_dict(classdict)
        
        self.classifier.eval()
        preds = []
        ans_list = []
        id_list = []
 
        with tqdm(self.test_set) as pbar:
            for batches in pbar:
                
                ans = batches["label_list"].to(self.args.device)
                    #######Load Data#########
                post_clip = batches["post_clip"].to(self.args.device)
                img_clip = batches["image_clip"].to(self.args.device)
                text_bert = batches["text_bert"].to(self.args.device)
                event = batches["event"].to(self.args.device)
                textent = batches["text_ent"].to(self.args.device)
                imageent = batches["image_ent"].to(self.args.device)
                reverse = batches["reverse"].to(self.args.device)
                timeinfo = batches["time_info"].to(self.args.device)
                ori_img = batches["image_ori"].to(self.args.device)
                Id = batches["Id"]

                
                for ans_label in ans:
                    ans_label = int(ans_label)
                    ans_list.append(ans_label)
                
                for index in Id:
                    id_list.append(index)
                    
                with torch.no_grad():
                    
                    pred,fake_score, entity_score, event_score, reverse_score, noevent_score = self.classifier(
                        post_clip, img_clip, text_bert, imageent, textent, reverse, event, timeinfo, ori_img
                        )
                    
                    _, label= torch.max(pred,dim=1)
                    
                    for i, y_em in enumerate(pred):
                        thisdict = {
                            "embedding":y_em,
                            "label":int(ans[i].cpu())
                        }
          
                        
                prob = F.softmax(pred, dim=-1)[:, 0]
          
                    
        print(classification_report(ans_list, preds, digits=4))

        with open(os.path.join(self.args.output_dir, f"{self.args.name}report.txt"), mode="w") as f:
            f.write(classification_report(ans_list,preds,digits=4))
            
