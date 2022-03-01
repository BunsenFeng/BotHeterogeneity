# generate the follower and following edge file
import torch
import json
file_list = ['train', 'dev', 'test','support']
user_idlist = torch.load('user_idlist.pt').tolist()
user_iddict = {}
for i in range(len(user_idlist)):
    user_iddict[user_idlist[i]] = i
print(len(user_iddict))

follower_list = []
following_list = []

for file in file_list:
    print(file)
    f = open(file + '.json')
    users = json.load(f)
    cnt = 0
    for user in users:
        if cnt % 100 == 0:
            print(cnt)
        cnt += 1
        user_id = int(user['ID'])
        if not user['neighbor']:
            continue
        for following in user['neighbor']['following']:
            try:
                following_list.append([user_iddict[user_id], user_iddict[int(following)]]) #R-GAT so that bidirectional?
            except:
                print('np')
                pass
                #continue
    print('half done')
    cnt = 0
    for user in users:
        if cnt % 100 == 0:
            print(cnt)
        cnt += 1
        user_id = int(user['ID'])
        if not user['neighbor']:
            continue
        for follower in user['neighbor']['follower']:
            try:
                follower_list.append([user_iddict[user_id], user_iddict[int(follower)]]) #R-GAT so that bidirectional?
            except:
                pass
    f.close()

torch.save(torch.tensor(follower_list).t(), 'processed/follower_edge.pt')
torch.save(torch.tensor(following_list).t(), 'processed/following_edge.pt')
temp1 = torch.load('processed/follower_edge.pt')
temp2 = torch.load('processed/following_edge.pt')
print(temp1.size())
print(temp2.size())
