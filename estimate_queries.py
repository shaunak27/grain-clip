import os
import json

src = '/#/bboxes.json'
with open(src, 'r') as f:
    data = json.load(f)

#file is arranged as {image_path: [bbox1, bbox2, ..., bboxN]}
#count average number of items in list for each key
    
#count average number of items in list for each key
#also compute min and max number of items in list for each key
min_count = float('inf')
max_count = 0
total = 0
for key in data:
    total += len(data[key])
    if len(data[key]) < min_count:
        min_count = len(data[key])
    if len(data[key]) > max_count:
        max_count = len(data[key])
        max_key = key
print(total/len(data)) #average number of bounding boxes per image
print(len(data)) #number of images
print(min_count) #min number of bounding boxes per image
print(max_count) #max number of bounding boxes per image
print(max_key) #image with max number of bounding boxes
tot = 0
# anns_data = json.load(open(anns, 'r'))
# max_anns = 0
# for key in anns_data:
#     tot += len(anns_data[key])
#     if len(anns_data[key]) > max_anns:
#         max_anns = len(anns_data[key])
#         max_key = key
# print(tot/len(anns_data)) #average number of annotations per image
# print(len(anns_data)) #number of images

# print(max_anns) #max number of annotations per image
# print(anns_data[max_key]) #image with max number of annotations