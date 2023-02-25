import json
from tqdm import tqdm
file_path = "label_map"
samplelist = []
class_dict = {}
label_name = []
for line in open(file_path,'rb'):
    sample = {}
    arr1 = str(line.strip(), 'utf-8')
    arr = arr1.split(' ')
    class_map = ''
    for i in range(0,len(arr)):
        if i == 0:
            continue
        else:
            if class_map != '':
                class_map = class_map + ' '
            class_map = class_map + arr[i]
    # class_map = arr[1]
    class_dict[arr[0]] = class_map
print("here")
result = []
path = 'test-old.json'
with open(path, 'r') as src:
    for line in tqdm(src):
        line = json.loads(line)
        new_data = {}
        new_data['text'] = line['text']
        new_data['label'] = class_dict[line['label']]
        result.append(new_data)
with open("test.json","w") as f:
    for output in result:
        line = json.dumps(output, ensure_ascii=False)
        f.write(line+'\n')