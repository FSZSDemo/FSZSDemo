import json
from tqdm import tqdm
path = 'train.json'
new_datas = []
seen_label_list = []
with open(path, 'r') as src:
    for line in tqdm(src):
        line = json.loads(line)
        new_data = {}
        new_data['text_a'] = line['text']
        new_data['label'] = line['label']
        if line['label'] not in seen_label_list:
            seen_label_list.append(line['label'])
        new_datas.append(new_data)
with open("seen_data.json","w") as f:
    for output in new_datas:
        line = json.dumps(output, ensure_ascii=False)
        f.write(line+'\n')
for i in range(0,len(seen_label_list)):
    print(i.__str__() + ":'" + seen_label_list[i] + '\',')