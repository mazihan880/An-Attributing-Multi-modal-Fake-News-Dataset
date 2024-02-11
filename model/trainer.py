import torch
from torch import nn
import os
import json
import torch.optim as optim
from tqdm import trange, tqdm
from sklearn.metrics import classification_report
from torch.nn import functional as F
from numpy import mean


torch.autograd.set_detect_anomaly(True)




def evaluation(outputs, labels):
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct




class Trainer:
    def __init__(self, 
                 args,
                 DetectionModel,
                 tr_set,
                 tr_size, 
                 dev_set, 
                 dev_size):
        self.args = args 
        self.tr_set = tr_set
        self.tr_size = tr_size
        self.dev_set = dev_set
        self.dev_size = dev_size
        self.total_loss = nn.CrossEntropyLoss()
        self.fakeloss = nn.BCELoss()
        self.entityloss = nn.BCELoss()
        self.eventloss = nn.BCELoss()
        self.spaceloss = nn.BCELoss()
        self.noeventloss = nn.BCELoss()
        self.classifier = DetectionModel
    
    def train(self):
        
        
        NET_Classifier = optim.Adam(self.classifier.parameters(), lr = self.args.lr,  weight_decay = 1e-6, eps = 1e-4)
        
        epoch_pbar = trange(self.args.num_epoch, desc="Epoch")
        best_f1=0
        for epoch in epoch_pbar:
            self.classifier.train()
            total_train_acc = []  
            total_loss_list = []
            fake_loss_list = []
            entity_loss_list = []
            event_loss_list = []
            space_loss_list = []
            noevent_loss_list = []

            with tqdm(self.tr_set) as pbar:
                for batches in pbar:
                    pbar.set_description(desc = f"Epoch{epoch}")
                    #######Load Data#########
                    y = batches["label_list"].to(self.args.device)

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
                    
                   

                    pred,fake_score, entity_score, event_score, reverse_score, noevent_score = self.classifier(
                        post_clip, img_clip, text_bert, imageent, textent, reverse, event, timeinfo, ori_img
                        )
                    
                    
                    _, label = torch.max(pred,dim=-1)
                    
                    
                    
                    y = y.to(torch.long)
                    y1 = y.unsqueeze(-1).to(torch.float)
                    fake_loss = self.fakeloss(fake_score, y1)
                    entity_loss = self.entityloss(entity_score, y1)
                    event_loss = self.eventloss(event_score, y1)
                    reverse_loss = self.spaceloss(reverse_score, y1)
                    noevent_loss = self.noeventloss(noevent_score, y1)   
                    total_loss = self.total_loss(pred, y) 
                    #cos_loss = self.cos_loss(cos, y)
                    all_loss = total_loss+0.2*(fake_loss+event_loss+entity_loss+reverse_loss+noevent_loss)

                    
                    correct = evaluation(label, y)/len(label)
                    
                    
                    
                    class_loss = all_loss# + self.args.beta *cos_loss
                    
                    NET_Classifier.zero_grad()
                    class_loss.backward()
                    NET_Classifier.step()
                    
                    
                   
                    total_train_acc.append(correct)
                    total_loss_list.append(total_loss.item())
                    fake_loss_list.append(fake_loss.item())
                    
    
                    pbar.set_postfix(loss = class_loss.item())
                    
                    
                    
                    

          
            train_acc_info_json = {"epoch": epoch,"train Acc": mean(total_train_acc), "total_loss": mean(total_loss_list), "fake_loss": mean(fake_loss_list)} 

            print(f"{'#' * 10} TRAIN ACCURACY: {str(train_acc_info_json)} {'#' * 10}")
            
            
            with open(os.path.join(self.args.output_dir, f"log{self.args.name}.txt"), mode="a") as fout:
                fout.write(json.dumps(train_acc_info_json) + "\n")
            
            
                
            
            self.classifier.eval()  
            valid_acc = []
            ans_list = []
            preds = []
            test_precision_values = []
            test_recall_values = []
            test_f1_values = []
            test_acc_values = []
            
            
            with torch.no_grad():
                for batches in self.dev_set:
                    y = batches["label_list"].to(self.args.device)
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

                    
                    for ans_label in y:
                        ans_label = int(ans_label)
                        ans_list.append(ans_label)

                    pred,fake_score, entity_score, event_score, reverse_score, noevent_score = self.classifier(
                        post_clip, img_clip, text_bert, imageent, textent, reverse, event, timeinfo, ori_img
                        )
                   

                    
                    
                    
                    
                    
                    
                    
                    _, label= torch.max(pred,-1)
                    
                    y=y.to(torch.long)
                    correct = evaluation(label, y)/len(label)
                
                    valid_acc.append(correct)
                    
                    for p in label.cpu().numpy():
                        preds.append(p)
                        
                
                report = classification_report(ans_list, preds, digits=4, output_dict = True)
                
                #print(report)
                
                test_precision_values.append(float(report["macro avg"]["precision"]))
                test_recall_values.append(float(report["macro avg"]["recall"]))
                test_f1_values.append(float(report["macro avg"]["f1-score"]))
                testf1 = report["macro avg"]["f1-score"]
                            
                
                with open(os.path.join(self.args.output_dir, f"log{self.args.name}.txt"), mode="a") as f:
                    f.write(classification_report(ans_list,preds, digits=4))
     
                valid_info_json = {"epoch": epoch,"valid_Acc":mean(valid_acc),"f1":report["macro avg"]["f1-score"]}
                test_acc_values.append(mean(valid_acc))
                print(f"{'#' * 10} VALID: {str(valid_info_json)} {'#' * 10}")
                
                with open(os.path.join(self.args.output_dir, f"log{self.args.name}.txt"), mode="a") as fout:
                    fout.write(json.dumps(valid_info_json) + "\n")
            
                if testf1 > best_f1:
                    best_f1 = testf1
                    torch.save(self.classifier, f"{self.args.ckpt_dir}/{self.args.name}ckpt.classifier")
                    print('saving model with f1 {:.3f}\n'.format(testf1))
                    with open(os.path.join(self.args.output_dir, f"{self.args.name}best_valid_log.txt"), mode="a") as fout:
                        fout.write(json.dumps(valid_info_json) + "\n")
                        
                        

        

                
                
                
         
         
 
        



