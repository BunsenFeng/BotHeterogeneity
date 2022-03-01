# Preprocess the bool data
import torch
import json
import numpy
import datetime
from math import sqrt

file_list = ['train', 'dev', 'test', 'support']
user_idlist = torch.load('user_idlist.pt').tolist()

#process bool value
now = 0
user_features1 = []
for file in file_list:
    print(file)
    f = open(file + '.json')
    users = json.load(f)
    for user in users:
        assert user_idlist[now] == int(user['ID'])
        now += 1
        feature_temp1 = []
        try:
            #verified
            feature_temp1.append(int(user['profile']['verified']=='True '))
            #Geo_enabled
            feature_temp1.append(int(user['profile']['geo_enabled']=='False '))
            #contributed enabled
            feature_temp1.append(int(user['profile']['contributors_enabled']=='True '))
            #profile_use_background_image
            feature_temp1.append(int(user['profile']['profile_use_background_image']=='True '))
            #has_extended_profile
            feature_temp1.append(int(user['profile']['has_extended_profile']=='True '))
            #default_profile
            feature_temp1.append(int(user['profile']['default_profile']=='True '))
            #default_profile_image
            feature_temp1.append(int(user['profile']['default_profile_image']=='True '))

        except:
            feature_temp1 = [0, 0, 0, 0, 0, 0, 0]
            print('Bool missing')

        user_features1.append(feature_temp1)
    f.close()

user_features1 = torch.tensor(user_features1).float()

torch.save(user_features1, 'My_user_feature_bool.pt')
temp = torch.load('My_user_feature_bool.pt')
print(temp.size())
