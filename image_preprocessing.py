from multiprocessing import Pool
import os
import glob
from tqdm import tqdm
import h5py
import logging

from PIL import Image
import numpy as np
import pydicom

from transformers import AutoImageProcessor
from transformers import BertTokenizer

FILES_DIR = '../scrap/physionet.org/files/mimic-cxr/2.0.0/files'

def crop_center(img, border_fraction=0.1):
    width, height = img.size

    left = width*border_fraction
    top = height*border_fraction
    right = width*(1-border_fraction)
    bottom = height*(1-border_fraction)

    img_cropped = img.crop((left, top, right, bottom))

    return img_cropped

def process_patient(patient):

    path = os.path.join(FILES_DIR, patient, "s*")
    hdf5_path = "/".join([hdf5_top_path, patient]) + ".h5"

    try:
        hdf5_file = h5py.File(hdf5_path, 'a')
    except:
        print('error on hdf5 path', hdf5_path)
        return

    slist = list(filter(lambda x: ".txt" not in x, glob.glob(path)))

    for s in slist:
        study_id = s.split('/')[-1]

        dicom_paths = glob.glob("/".join([s, '*.dcm']))
        for dicom_path in dicom_paths:
            image_id = dicom_path.split('/')[-1].split('-')[0]

            dataset_name = f"{study_id}_{image_id}"
            if dataset_name in hdf5_file:
                continue
            
            try:
                image = (pydicom.dcmread(dicom_path).pixel_array / 16)
                image = image.astype(np.uint8)
                image = Image.fromarray(image)
                image = crop_center(image).convert('RGB')

                pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
                pixel_values = pixel_values[0, 0]

                hdf5_file.create_dataset(dataset_name, data=pixel_values)
            except ValueError as e:
                logging.warning(f"Skipping image {dicom_path} due to error: {e}")

        text_path = s + ".txt"
        dataset_name = f"{study_id}_txt"
        if dataset_name in hdf5_file:
            continue
        with open(text_path, "r") as f:
            study_txt = f.read()
        encoded_text = tokenizer(study_txt, return_tensors='pt')["input_ids"] #max length trunc pas mis :)
        encoded_text = encoded_text.squeeze()
        hdf5_file.create_dataset(dataset_name, data=encoded_text)
    
    length = len(hdf5_file)
    
    hdf5_file.close()
    if length == 0:
        os.remove(hdf5_path)

plist = list(map(lambda x: x.replace(FILES_DIR+"/", ''), glob.glob(os.path.join(FILES_DIR, "p*", "p*"))))

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

hdf5_top_path = "preprocessed_data/xcr"

if not os.path.exists(hdf5_top_path):
    os.mkdir(hdf5_top_path)
    for idx in range(10):
        os.mkdir(os.path.join(hdf5_top_path, f"p1{idx}"))

# process_patient(plist[1])

with Pool(int(0.65*os.cpu_count())) as p:
    list(tqdm(p.imap(process_patient, plist), total=len(plist)))