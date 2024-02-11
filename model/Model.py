import torch
import torch.nn as nn
from torch.nn import functional as F

class ImageFakeDetection(nn.Module):
    def __init__(self, hrnet, cls_net):
        super(ImageFakeDetection, self).__init__()
        self.hrnet = hrnet
        self.clsnet =cls_net
        self.get_feature = nn.Linear(128, 16)
        self.bn = nn.BatchNorm1d(16)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        
        self.activation = nn.Sequential(
            nn.Linear(2,1),
            nn.Sigmoid()
        )
        
    def forward(self, image):
        out = self.hrnet(image)
        out = self.clsnet(out)
        out = self.get_feature(out)
        out = self.bn(out)
        out = F.dropout(out ,p = 0.5)
        out = self.classifier(out)
        out_score = self.activation(out)
        
        return out, out_score
    
class EntityDetection(nn.Module):
    def __init__(self, model_dim, bert_dim):
        super(EntityDetection, self).__init__()
        self.linear_text=nn.Sequential(
            nn.Linear(bert_dim, bert_dim//2),
            nn.ReLU(),
            nn.Linear(bert_dim//2,model_dim)
        )
        self.bn1 = nn.BatchNorm1d(model_dim)
        self.bn2 = nn.BatchNorm1d(model_dim)
        self.linear_image=nn.Sequential(
            nn.Linear(bert_dim, bert_dim//2),
            nn.ReLU(),
            nn.Linear(bert_dim//2,model_dim)
        )

        self.linear_compare=nn.Sequential(
            nn.Linear(model_dim*4,model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 2)
        )
        
        self.simiactivation = nn.Sequential(
            nn.Linear(3,1),
            nn.Sigmoid()
        )
    def forward(self, text, image):
        image = self.linear_image(image)
        image = self.bn1(image)
        text = self.linear_text(text)
        text = self.bn2(text)
        combine_feature = torch.cat([image, text, text-image, text*image],  dim=-1)
        out = self.linear_compare(combine_feature)
        image_norm = F.normalize(image, p=2, dim=1)  
        text_norm = F.normalize(text, p=2, dim=1)  
        simi = torch.bmm(image_norm.unsqueeze(-1).permute(0,2,1), text_norm.unsqueeze(-1))
        #print(image_norm.unsqueeze(-1).shape)
        #print(out.shape)
        #print(simi.shape)
        out_score = self.simiactivation(torch.cat([out, simi.squeeze(-1)],dim=1))
        return out, out_score

class EventDetection(nn.Module):
    def __init__(self, model_dim, bert_dim):
        super(EventDetection, self).__init__()
        self.linear_text=nn.Sequential(
            nn.Linear(bert_dim, bert_dim//2),
            nn.ReLU(),
            nn.Linear(bert_dim//2,model_dim)
        )
        self.bn1 = nn.BatchNorm1d(model_dim)
        self.bn2 = nn.BatchNorm1d(model_dim)
        self.linear_event=nn.Sequential(
            nn.Linear(bert_dim, bert_dim//2),
            nn.ReLU(),
            nn.Linear(bert_dim//2,model_dim)
        )

        self.linear_compare=nn.Sequential(
            nn.Linear(model_dim*4,model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 2)
        )
        
        self.simiactivation = nn.Sequential(
            nn.Linear(3,1),
            nn.Sigmoid()
        )
    def forward(self, text, event):
        event = self.bn1(self.linear_event(event))
        text = self.bn2(self.linear_text(text))

        combine_feature = torch.cat([event, text, text-event, text*event],  dim=-1)
        out = self.linear_compare(combine_feature)
        event_norm = F.normalize(event, p=2, dim=1)  
        text_norm = F.normalize(text, p=2, dim=1)  
        simi = torch.bmm(event_norm.unsqueeze(-1).permute(0,2,1), text_norm.unsqueeze(-1))
        out_score = self.simiactivation(torch.cat([out, simi.squeeze(-1)],dim=1))
        return out, out_score

class SpaceDetection(nn.Module):
    def __init__(self, model_dim, bert_dim):
        super(SpaceDetection, self).__init__()
        self.linear_text=nn.Sequential(
            nn.Linear(bert_dim, bert_dim//2),
            nn.ReLU(),
            nn.Linear(bert_dim//2,model_dim)
        )
        self.bn1 = nn.BatchNorm1d(model_dim)
        self.bn2 = nn.BatchNorm1d(model_dim)
        self.bn3 = nn.BatchNorm1d(3)
        self.linear_evidence=nn.Sequential(
            nn.Linear(bert_dim, bert_dim//2),
            nn.ReLU(),
            nn.Linear(bert_dim//2,model_dim)
        )

        self.linear_compare=nn.Sequential(
            nn.Linear(model_dim*4,model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 2)
        )
        self.lineartime = nn.Linear(1,1)
        self.activation = nn.Sequential(
            nn.Linear(3,1),
            nn.Sigmoid()
        )
    def forward(self, text, evidence, time_info):
        evidence = self.bn1(self.linear_evidence(evidence))
        text = self.bn2(self.linear_text(text))
        time_info = self.lineartime(time_info.unsqueeze(-1))
        combine_feature = torch.cat([evidence, text, text-evidence, text*evidence],  dim=-1)
        out = self.linear_compare(combine_feature)
        out = torch.cat([out, time_info],dim=1)
        out = self.bn3(out)
        out_score = self.activation(out)
        return out, out_score


class NoeventDetection(nn.Module):
    def __init__(self, model_dim, img_dim):
        super(NoeventDetection, self).__init__()
        self.linear_img=nn.Sequential(
            nn.Linear(img_dim, img_dim//2),
            nn.ReLU(),
            nn.Linear(img_dim//2, model_dim)
        )

        self.bn1 = nn.BatchNorm1d(model_dim)
        
        self.linear_activate=nn.Sequential(
            nn.Linear(model_dim,16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        
        self.activation = nn.Sequential(
            nn.Linear(2,1),
            nn.Sigmoid()
        )
    def forward(self, image):
        image = self.bn1(self.linear_img(image))


        out = self.linear_activate(image)
 
        out_score = self.activation(out)
        return out, out_score

class Detection_Module(nn.Module):
    def __init__(self, hrnet, cls_net, model_dim, bert_dim, clip_dim):
        super(Detection_Module, self).__init__()
        self.Fakeimage = ImageFakeDetection(hrnet, cls_net)
        self.Entity = EntityDetection(model_dim, bert_dim)
        self.Event = EventDetection(model_dim, bert_dim)
        self.Space = SpaceDetection(model_dim, bert_dim)
        self.Noevent = NoeventDetection(model_dim, clip_dim)
        self.TEXT = nn.Sequential(
            nn.Linear(512, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 8)
        )
        self.IMAGE = nn.Sequential(
            nn.Linear(512, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 8)
        )
        self.classifier = nn.Sequential(
            nn.Linear(8+8+2+2+2+3+2, 16),
            nn.ReLU(),
            nn.Linear(16,2) 
        )
        self.bn = nn.BatchNorm1d(8+8+2+2+2+3+2)
    def forward(self, image_clip, text_clip, text_bert, imgentity, textentity, reverse, event, timeinfo, ori_image):
        fake_out, fake_score = self.Fakeimage(ori_image)
        entity_out, entity_score = self.Entity(textentity, imgentity)
        event_out ,event_score = self.Event(text_bert, event)
        reverse_out, reverse_score = self.Space(text_bert, reverse, timeinfo)        
        noevent_out , noevent_score =self.Noevent(image_clip)
        text_feature = self.TEXT(text_clip)
        image_feature = self.IMAGE(image_clip)
        out_feature = torch.cat([text_feature, image_feature, fake_out * fake_score, entity_out * entity_score, event_out * event_score, reverse_out * reverse_score, noevent_out*noevent_score], dim=-1)
        out_feature = self.bn(out_feature)
        out = self.classifier(out_feature)
        return out ,fake_score, entity_score, event_score, reverse_score, noevent_score
        
        

