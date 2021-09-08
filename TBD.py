import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.optim import optimizer
from torch.utils.data import Dataset, DataLoader
from RGT import RGTLayer
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from os import listdir

def get_metrics(probs, labels):
    probs = torch.argmax(probs, dim = 1)
    correct = 0
    for i in range(len(probs)):
        if probs[i] == labels[i]:
            correct += 1
    return correct / len(probs)

def F1_score(pred, truth):
    pred_v=torch.argmax(pred, dim = 1)
    tp,tn,fp,fn = 0,0,0,0

    tp = (pred_v * truth).sum()
    tn = ((1 - pred_v) * (1 - truth)).sum()
    fp = (pred_v * (1 - truth)).sum()
    fn = ((1 - pred_v) * truth).sum()

    try:
        precision = tp / (tp+fp)
    except:
        precision = tp/(tp + fp + 1)
    try:
        recall = tp / (tp+fn)
    except:
        recall = tp/ (tp + fn + 1)

    try:
        f1 = 2*(precision * recall) / (precision + recall)
    except:
        f1 = 2*(precision * recall) / (precision + recall + 1)

    return f1

class BotDataset(Dataset):
    def __init__(self, name, batch_size, user_num):
        # path = "/new_temp/fsb/TZX/Twibot-20/processed/"
        path = "/content/drive/MyDrive/"
        self.name = name
        self.user_num = user_num
        
        self.label = torch.load(path + 'label_list.pt')

        ########### NEED TO CHANGE ###############################
        
        follower_edge = torch.load(path + "follower_edge.pt")
        following_edge = torch.load(path + "following_edge.pt")

        self.edge_index_list = [follower_edge, following_edge]
        ##########################################################
        
        # Bidirectional Edges?
        # self.follower_edge = torch.cat((self.follower_edge ,torch.stack([self.follower_edge[1],self.follower_edge[0]])), dim = 1)
        # self.following_edge = torch.cat((self.follower_edge ,torch.stack([self.following_edge[1],self.following_edge[0]])), dim = 1)

        self.batch_size = batch_size
        
        # different features dataset
        self.user_features_numeric = torch.load(path + "My_user_feature_ZS.pt")
        self.user_features_bool = torch.load(path + "My_user_feature_bool.pt")
        self.user_features_tweet = torch.load(path + "user_features2.pt")
        self.user_feature_des = torch.load(path + "user_features3.pt")
        #train 
        if self.name == "train":
            self.length = int(0.7 * user_num / self.batch_size)
        else:
            self.length = 1

    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        return self.user_features_numeric, self.user_features_bool, self.edge_index_list, \
            self.label, self.user_features_tweet, self.user_feature_des

class RGTBotDetector(pl.LightningModule):
    def __init__(self, HAN_hid_out, tweet_in_channel,numeric_in_channels, bool_in_channels,\
         des_in_channels, num_heads, semantic_heads, linear_out_channels, dropout_rate, user_num, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.numeric_in_channels = numeric_in_channels
        self.bool_in_channels = bool_in_channels
        self.linear_out_channels = linear_out_channels
        self.tweet_in_channel = tweet_in_channel
        self.dropout_rate = dropout_rate
        self.user_num = user_num
        self.num_heads = num_heads
        self.des_in_channel = des_in_channels
        self.semantic_heads = semantic_heads

        self.in_linear_numeric = nn.Linear(self.numeric_in_channels, int(self.linear_out_channels/4), bias=True)
        self.in_linear_bool = nn.Linear(self.bool_in_channels, int(self.linear_out_channels/4), bias=True)
        self.in_linear_tweet = nn.Linear(self.tweet_in_channel, int(self.linear_out_channels/4), bias=True)
        self.in_linear_des = nn.Linear(self.des_in_channel, int(self.linear_out_channels/4), bias=True)

        self.linear1 = nn.Linear(linear_out_channels, linear_out_channels)
        
        self.RGT_layers = nn.ModuleList()
        self.RGT_layers.append(RGTLayer(num_edge_type=2,in_size=linear_out_channels,out_size = HAN_hid_out,\
                    layer_num_heads=num_heads[0], semantic_head=semantic_heads[0], dropout=DROPOUT_RATE))

        for l in range(1,len(num_heads)):
            self.RGT_layers.append(RGTLayer(num_edge_type=2, in_size=HAN_hid_out,\
                out_size = HAN_hid_out, layer_num_heads=num_heads[l], semantic_head=semantic_heads[l], dropout= DROPOUT_RATE ))

        self.output1 = nn.Linear(HAN_hid_out, 64)
        self.output2 = nn.Linear(64, 2)

        torch.nn.init.kaiming_normal_(self.in_linear_bool.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.in_linear_numeric.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.in_linear_tweet.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.in_linear_des.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.output1.weight, nonlinearity="leaky_relu")

        self.dropout = nn.Dropout(self.dropout_rate)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()


    def training_step(self, train_batch, batch_idx):
        user_features_numeric = train_batch[0].squeeze(0)
        user_features_bool = train_batch[1].squeeze(0)
        edge_index_list = train_batch[2]
        label = train_batch[3].squeeze(0).tolist()
        user_features_tweet = train_batch[4].squeeze(0).squeeze(1)
        user_features_des = train_batch[5].squeeze(0).squeeze(1)

        # Train data labels
        label = torch.LongTensor(label[0:int(0.7*self.user_num)])

        user_features_numeric = self.dropout(self.ReLU(self.in_linear_numeric(user_features_numeric)))
        user_features_bool = self.dropout(self.ReLU(self.in_linear_bool(user_features_bool)))
        user_features_tweet = self.dropout(self.ReLU(self.in_linear_tweet(user_features_tweet)))
        user_features_des = self.dropout(self.ReLU(self.in_linear_des(user_features_des)))

        user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
        user_features = self.dropout(self.ReLU(self.linear1(user_features)))

        for HAN in self.RGT_layers:
            user_features = HAN(user_features, edge_index_list)
        
        user_features = self.dropout(self.ReLU(self.output1(user_features)))
        
        batch_id = torch.randperm(int(0.7 * self.user_num))[0:self.batch_size].tolist()
    
        pred = self.output2(user_features[batch_id])
        
        loss = self.CELoss(pred, label[batch_id].cuda())

        accuracy = get_metrics(pred, label[batch_id].cuda())
        F1 = F1_score(pred, label[batch_id].cuda())

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        self.log("train_F1", F1)

        return loss


    def validation_step(self, val_batch, batch_idx):
        user_features_numeric = val_batch[0].squeeze(0)
        user_features_bool = val_batch[1].squeeze(0)
        edge_index_list = val_batch[2]
        label = val_batch[3].squeeze(0).tolist()
        user_features_tweet = val_batch[4].squeeze(0).squeeze(1)
        user_features_des = val_batch[5].squeeze(0).squeeze(1)

        label = torch.LongTensor(label[int(0.7 * self.user_num) : int(0.9 * self.user_num)])
        # forward pass
        user_features_numeric = self.ReLU(self.in_linear_numeric(user_features_numeric))
        user_features_bool = self.ReLU(self.in_linear_bool(user_features_bool))
        user_features_tweet = self.ReLU(self.in_linear_tweet(user_features_tweet))
        user_features_des = self.ReLU(self.in_linear_des(user_features_des))

        user_features = torch.cat((user_features_numeric, user_features_bool,user_features_tweet,user_features_des), dim = 1)
        user_features = self.ReLU(self.linear1(user_features))

        for RGT in self.RGT_layers:
            user_features = RGT(user_features, edge_index_list)

        user_features = self.ReLU(self.output1(user_features))

        pred = self.output2(user_features[int(0.7 * self.user_num) : int(0.9 * self.user_num)])

        loss = self.CELoss(pred, label.cuda())
        accuracy = get_metrics(pred, label.cuda())
        F1 = F1_score(pred, label.cuda())

        self.log('val_acc', accuracy)
        self.log('val_loss', loss)
        self.log("val_F1", F1)

        print("Your model's accuracy on validation dataset is {}".format(accuracy))
        print("Your model's CrossEntropyLoss on validation dataset is {}".format(loss))
        print("Your model's F1-score on validation dataset is {}".format(F1))

        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=WD, amsgrad=False)
        #optimizer = torch.optim.SGD(self.parameters, lr = LEARNING_RATE, momentum=0.9, weight_decay=WD)
        return optimizer

    def test_step(self, test_batch, batch_idx):
        user_features_numeric = test_batch[0].squeeze(0)
        user_features_bool = test_batch[1].squeeze(0)
        edge_index_list = test_batch[2]
        label = test_batch[3].squeeze(0).tolist()
        user_features_tweet = test_batch[4].squeeze(0).squeeze(1)
        user_features_des = test_batch[5].squeeze(0).squeeze(1)

        label = torch.LongTensor(label[int(0.9 * self.user_num) : (1 * self.user_num)])

        # forward pass   
        user_features_numeric = self.ReLU(self.in_linear_numeric(user_features_numeric))
        user_features_bool = self.ReLU(self.in_linear_bool(user_features_bool))
        user_features_tweet = self.ReLU(self.in_linear_tweet(user_features_tweet))
        user_features_des = self.ReLU(self.in_linear_des(user_features_des))

        user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)

        user_features = self.ReLU(self.linear1(user_features))

        for RGT in self.RGT_layers:
            user_features = RGT(user_features, edge_index_list)
        
        user_features = self.ReLU(self.output1(user_features))

        pred = self.output2(user_features[int(0.9 * self.user_num) : int(1 * self.user_num)])

        loss = self.CELoss(pred, label.cuda())

        accuracy = get_metrics(pred, label.cuda())
        F1 = F1_score(pred, label.cuda())

        self.log("test_accuracy", accuracy, on_epoch=True)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_F1", F1, on_epoch=True)
        
        print("Your model's accuracy on test dataset is {}".format(accuracy))
        print("Your model's CrossEntropyLoss on test dataset is {}".format(loss))
        print("Your model's F1-score on test dataset is {}".format(F1))
        
        return user_features

########### Hyperparameters tuning ################ 

LEARNING_RATE = 1e-3
WD = 3e-5   # Weight Decay
DROPOUT_RATE = 0.5
USER_NUM = 11826
BATCH_SIZE = 256
IN_FEATURES_BOOL = 11
IN_FEATURES_NUMERIC = 7
IN_FEATURES_TWEET = 768
IN_FEATURE_DES = 768
MAX_EPOCHS = 40
NUM_HEAD = [8, 8]
SEMANTIC_HEAD = [8, 8]
#####################################################
if __name__ == "__main__":
    train_dataset = BotDataset(name = "train",batch_size = BATCH_SIZE, user_num = USER_NUM)
    valid_dataset = BotDataset(name = "dev", batch_size = 1, user_num = USER_NUM)

    train_loader = DataLoader(train_dataset, batch_size = 1)
    val_loader = DataLoader(valid_dataset, batch_size = 1)
    
    model = RGTBotDetector(HAN_hid_out=128, tweet_in_channel=IN_FEATURES_TWEET, numeric_in_channels=IN_FEATURES_NUMERIC,\
        bool_in_channels=IN_FEATURES_BOOL, linear_out_channels=128, dropout_rate=DROPOUT_RATE,user_num=USER_NUM, \
        batch_size=BATCH_SIZE, num_heads=NUM_HEAD, des_in_channels=IN_FEATURE_DES, semantic_heads=SEMANTIC_HEAD)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        filename='{val_acc:.4f}',
        save_top_k=3,
        verbose=True
    )

    trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs = MAX_EPOCHS, precision=16, callbacks=[checkpoint_callback],log_every_n_steps=1)
    print("Training begin!")
    trainer.fit(model, train_loader, val_loader)

    dir = './lightning_logs/version_{}/checkpoints/'.format(trainer.logger.version)
    best_path1 = './lightning_logs/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[0])
    best_path2 = './lightning_logs/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[1])
    best_path3 = './lightning_logs/version_{}/checkpoints/{}'.format(trainer.logger.version, listdir(dir)[2])
    
    print('best_path1:', best_path1)
    print('best_path2', best_path2)
    print('best_path3', best_path3)

    best_model1 = RGTBotDetector.load_from_checkpoint(checkpoint_path=best_path1,HAN_hid_out=128, tweet_in_channel=IN_FEATURES_TWEET, numeric_in_channels=IN_FEATURES_NUMERIC,\
        bool_in_channels=IN_FEATURES_BOOL, linear_out_channels=128, dropout_rate=DROPOUT_RATE,user_num=USER_NUM, \
        batch_size=BATCH_SIZE, num_heads=NUM_HEAD, des_in_channels=IN_FEATURE_DES, semantic_heads=SEMANTIC_HEAD)

    best_model2 = RGTBotDetector.load_from_checkpoint(checkpoint_path=best_path2,HAN_hid_out=128, tweet_in_channel=IN_FEATURES_TWEET, numeric_in_channels=IN_FEATURES_NUMERIC,\
        bool_in_channels=IN_FEATURES_BOOL, linear_out_channels=128, dropout_rate=DROPOUT_RATE,user_num=USER_NUM, \
        batch_size=BATCH_SIZE, num_heads=NUM_HEAD, des_in_channels=IN_FEATURE_DES, semantic_heads=SEMANTIC_HEAD)

    best_model3 = RGTBotDetector.load_from_checkpoint(checkpoint_path=best_path3,HAN_hid_out=128, tweet_in_channel=IN_FEATURES_TWEET, numeric_in_channels=IN_FEATURES_NUMERIC,\
        bool_in_channels=IN_FEATURES_BOOL, linear_out_channels=128, dropout_rate=DROPOUT_RATE,user_num=USER_NUM, \
        batch_size=BATCH_SIZE, num_heads=NUM_HEAD, des_in_channels=IN_FEATURE_DES, semantic_heads=SEMANTIC_HEAD)
    
    print('best_model1')
    trainer.test(best_model1, test_dataloaders = val_loader, verbose = True)
    
    print('best_model2')
    trainer.test(best_model2, test_dataloaders = val_loader, verbose = True)

    print('best_model3')
    trainer.test(best_model3, test_dataloaders = val_loader, verbose = True)
    
