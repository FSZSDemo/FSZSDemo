import json
file_path = "labels.test.txt"
samplelist = []
class_dict = []
label_name = []
for line in open(file_path,'rb'):
    sample = {}
    arr1 = str(line.strip(), 'utf-8')
    arr = arr1.split('_')
    label = ""
    for i in arr:
        label += i + " "
    label = label[:-1]
    label_name.append(arr1)
    class_dict.append(label)
print("here")
output = {}
output["task_name"] = "Liu"
output["labels"]={}
# output["labels"]['distance'] = []
for idx in range(0,len(class_dict)):
    output["labels"][label_name[idx]] = {}
    text = "If I want "+class_dict[idx]+", I will say: \""
    output["labels"][label_name[idx]]["instruction"] = text
    output["labels"][label_name[idx]]["counter_labels"] = []
print("here")
with open("../Liu.json","w") as f:
    # for i in output['labels']:
    line = json.dumps(output, ensure_ascii=False)
    f.write(line+'\n')
labellist = []
for idx in range(0,len(class_dict)):
    label = {}
    label["text_a"] = class_dict[idx]
    label["text_b"] = None
    label["label"] = label_name[idx]
    labellist.append(label)
    print(idx.__str__()+":'"+label_name[idx]+'\',')
with open("../Liulabel_list.json","w") as f:
    for output in labellist:
        line = json.dumps(output, ensure_ascii=False)
        f.write(line+'\n')