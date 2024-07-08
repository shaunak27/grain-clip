from transformers import Owlv2Processor, Owlv2ForObjectDetection
import os
import json
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
import numpy as np
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from tqdm import tqdm
from dataloaders.cc_desc import CC_Desc
from tqdm import tqdm
import argparse

def transforms_bbox(n_px):
    #return train_transform
    return transforms.Compose([
        transforms.Resize((n_px,n_px), interpolation=Image.BICUBIC),
        #transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def get_preprocessed_image(pixel_values):
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    #unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image

def collate(batch):
    images, desc, img_paths, avail_idx = zip(*batch)
    images = torch.stack(images, 0)
    #len_desc = torch.tensor(len_desc, dtype=torch.int32)
    desc = list(desc)
    img_paths = list(img_paths)
    return images, desc, img_paths, avail_idx

def main(args):
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble",do_pad=False,do_rescale=False)
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").cuda()
    
    tfms = transforms_bbox(224)
    dataset = CC_Desc(root=args.root, n_chunks=args.n_chunks, chunk_idx=args.chunk_idx, transform=tfms)
    batch_size = args.batch_size
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8,collate_fn=collate)

    bounding_boxes = json.load(open(args.output)) if os.path.exists(args.output) else dict()

    for idx, (image,desc,img_paths,avail_idx) in enumerate(tqdm(loader)):
            #desc is array of array of strings
            #convert it to list of strings
            if img_paths[0] in bounding_boxes:
                continue
            texts = desc
            image = image.cuda()
            #try:
            inputs = processor(text=texts, images=image, return_tensors="pt",truncation=True)
            inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
            # except :
            #     print(img_paths)
            #     continue
            #print(inputs.pixel_values.shape)
            with torch.no_grad():
                try:
                    outputs = model(**inputs)
                except RuntimeError:
                    print("RuntimeError")
                    with open(args.output, 'w') as f:
                        json.dump(bounding_boxes, f)
                    exit()
            unnormalized_image = get_preprocessed_image(inputs["pixel_values"].cpu())
            #print(unnormalized_image.shape[::-1])
            #unnormalized_image is of the shape [3, 960, 960, batch_size]
            #make it [batch_size, 960, 960, 3]
            unnormalized_image = unnormalized_image.transpose(3, 0, 1, 2)
            target_sizes = torch.tensor(unnormalized_image.shape, dtype=torch.int32)[-2:]
            #target_sizes = [batch_size, 960, 960, 3]
            #make it [[960, 960,3], [960, 960,3] ... batch_size times]
            target_sizes = target_sizes.repeat(image.shape[0], 1)
            #target_sizes = target_sizes[0,:-1]
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.3)
            for index in range(len(image)):
                text = texts[index]
                boxes, scores, labels = results[index]["boxes"], results[index]["scores"], results[index]["labels"]
                
                #filter out boxes with low confidence
                boxes = boxes[scores > 0.3]
                labels = labels[scores > 0.3]
                scores = scores[scores > 0.3]

                # for box, score, label in zip(boxes, scores, labels):
                #     box = [round(i, 2) for i in box.tolist()]
                #     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
                # visualized_image = unnormalized_image[index].copy()
                # visualized_image = visualized_image.transpose(1, 2, 0)
                # visualized_image = Image.fromarray(visualized_image,mode='RGB')
                # draw = ImageDraw.Draw(visualized_image)

                # for box, score, label in zip(boxes, scores, labels):
                #     box = [round(i, 2) for i in box.tolist()]
                #     x1, y1, x2, y2 = tuple(box)
                #     draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red")
                #     draw.text(xy=(x1, y1), text=text[label])
                
                #store bxes, text[labels], scores in COCO format
                
                for box, score, label in zip(boxes, scores, labels):
                    box = [i for i in box.tolist()]
                    x1, y1, x2, y2 = tuple(box)
                    if text[label.item()] == '<pad>':
                        continue
                    if img_paths[index] not in bounding_boxes:
                        bounding_boxes[img_paths[index]] = []
                    bounding_boxes[img_paths[index]].append({'description': text[label.item()], 'bbox': [x1, y1, x2, y2], 'score': score.item()})
                
                # visualized_image.save(os.path.join('./img11/', f"{str(index)}.jpg"))
            if idx%1000==0:
                with open(args.output, 'w') as f:
                    json.dump(bounding_boxes, f)

    #save bounding boxes
    with open(args.output, 'w') as f:
        json.dump(bounding_boxes, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Box Supervision')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--root', type=str, default='#/DownloadConceptualCaptions/desc_cc3m_0.json', help='root directory of dataset')
    parser.add_argument('--output', type=str, default='#/DownloadConceptualCaptions/bbox_cc3m_0.json', help='output file')
    parser.add_argument("--n-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    main(args)