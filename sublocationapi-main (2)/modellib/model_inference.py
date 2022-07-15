# decompyle3 version 3.9.0
# Python bytecode version base 3.7.0 (3394)
# Decompiled from: Python 3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: C:\work\GM\code\sublocation\sublocation_api\modellib\model_inference.py
# Compiled at: 2022-03-10 12:12:11
# Size of source mod 2**32: 10559 bytes
import pandas as pd, glob, cv2, torch, numpy as np, time, socket
from sys import argv
import os, io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from modellib.blobclient import upload_to_blob
from collections import OrderedDict
path = argv
import urllib3
http = urllib3.PoolManager()

class Model_predic_reconv:
    hostname = socket.gethostname()
    image_size = 300
    image_mean = np.array([127, 127, 127])
    image_std = 128.0

    def resize(self, image, boxes, labels):
        """
            Resize image to a fixed size
        """
        image = cv2.resize(image, (self.image_size, self.image_size))
        return (
         image, boxes, labels)

    def subtract_means(self, image, boxes, labels):
        """
            subtract mean value from the image to normalize it
        """
        mean = np.array((self.image_mean), dtype=(np.float32))
        image = image.astype(np.float32)
        image -= mean
        return (
         image.astype(np.float32), boxes, labels)

    def misc(self, image, boxes, labels):
        """
            Normalize image
        """
        image = image / self.image_std
        return (
         image, None, None)

    def to_tensor(self, image, boxes, labels):
        """
            convert Image(numpy) to Tensor
        """
        return (
         torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1), boxes, labels)

    def get_infer_transformed(self, image, boxes, labels, infer_transforms):
        for t in infer_transforms:
            image, boxes, labels = t(image, boxes, labels)

        return image

    def read_img(self, img_url):
        """
            read image using cv2 and convert from BGR to RGB
        """
        img = http.request('GET', img_url).data
        img = np.array(Image.open(io.BytesIO(img)).convert('RGB'))
        return img

    def sequential_batch_train(self, img_name, model_var, voc_txt_path, md_name):
        image_name = img_name
        dict_sequential = {}
        if md_name == 'Big Object Model':
            print('\n{} is running for object detection'.format(md_name))
        else:
            if md_name == 'Small Object Model':
                print('\n{} is running for object detection'.format(md_name))
            else:
                if md_name == 'Lable model':
                    print('\n{} is running for object detection'.format(md_name))
        df_list = []
        combined_lst = self.Predict(image_name, model_var, voc_txt_path)
        return combined_lst

    def Predict(self, img_name, model_var, voc_txt_path):
        list_put_all_details = []
        mAP_list = []
        model = model_var
        list_image = []
        s = 0
        v = 0
        temp_store = []
        img_list = []
        model.eval()
        with torch.no_grad():
            orig_image = self.read_img(img_name)
            img_list.append(orig_image)
            image_size = 300
            image_mean = np.array([127, 127, 127])
            image_std = 128.0
            infer_transforms = [self.resize, self.subtract_means, self.misc, self.to_tensor]
            class_names = [name.strip() for name in open(voc_txt_path).readlines()]
            actual_label_list = []
            pred_label = None
            probs = None
            xmin = None
            ymin = None
            xmax = None
            ymax = None
            pred_label_list = []
            image_name_list = []
            height, width, _ = orig_image.shape
            image = self.get_infer_transformed(orig_image, None, None, infer_transforms)
            images = image.unsqueeze(0)
            t1 = time.time()
            all_out = model.forward(images)
            list_image.append(all_out)
            bx = all_out[:, :4]
            lb = all_out[:, 4]
            conf = all_out[:, 5]
            bx[:, 0] *= width
            bx[:, 1] *= height
            bx[:, 2] *= width
            bx[:, 3] *= height
            if bx[(0, 0)].item() <= -999:
                empty_l = ['BACKGROUND',0.0,0,0,0,0]
                pred_label_list.append('NO_DETECTION')
            else:
                v += 1
                for kj in range(bx.shape[0]):
                    box = bx[kj, :]
                    labels = lb[kj]
                    probs = conf[kj]
                    probs = round(probs.detach().item() * 100, 3)
                    xmin, ymin, xmax, ymax = (
                     int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                    pred_label = class_names[int(labels.item())]
                    mAP_list = [
                     img_name,pred_label,probs,xmin,ymin,xmax,ymax]
                    temp_store.append([pred_label,probs,xmin,ymin,xmax,ymax])

            df = pd.DataFrame(temp_store, columns=['location_code','probs','xmin','ymin','xmax','ymax'])
        keys = [
         'location_code','probs','xmin','ymin','xmax','ymax']
        combined_lst = []
        for sub_list in temp_store:
            dict_predict = {}
            if len(sub_list) != 0:
                for key_name, val in zip(keys, sub_list):
                    dict_predict[key_name] = val

                combined_lst.append(dict_predict)
            else:
                combined_lst = []

        test_run = drawBoundingBox(orig_image, df)
        filename = str(img_name).split('/')[-1]
        cv2.imwrite(filename, test_run)
        with Image.open(filename) as img:
            buf = io.BytesIO()
            img.convert('RGB').save(buf, format='png')
            img_link = upload_to_blob(filename, buf.getvalue())
        return combined_lst


def driver(full_img_list, model_path, voc_txt_path, model_path_fr, voc_txt_path_fr, location_model, location_txt):
    full_img_list = full_img_list
    Model_predic_reconv.location_model = location_model
    Model_predic_reconv.location_txt = location_txt
    Model_predic_reconv.model_path = model_path
    Model_predic_reconv.voc_txt_path = voc_txt_path
    Model_predic_reconv.model_path_fr = model_path_fr
    Model_predic_reconv.voc_txt_path_fr = voc_txt_path_fr
    b_m = 'Big Object Model'
    s_m = 'Small Object Model'
    l_m = 'Lable model'
    mpc = Model_predic_reconv()
    t1 = time.perf_counter()
    var_exists = 'big_obj_model' in locals() or 'big_obj_model' in globals()
    t0 = time.time()
    if not var_exists:
        big_obj_model = torch.jit.load(Model_predic_reconv.model_path)
    _t0 = time.time()
    var_exists_2 = 'small_obj_model' in locals() or 'small_obj_model' in globals()
    if not var_exists_2:
        small_obj_model = torch.jit.load(Model_predic_reconv.model_path_fr)
    _t1 = time.time()
    var_exists_3 = 'lable_model' in locals() or 'lable_model' in globals()
    if not var_exists_2:
        lable_model = torch.jit.load(Model_predic_reconv.location_model)
    output_file_name = '{}_Model_predicted_raw_annotated_file.jpg'
    driver_dict = {}
    dict_sequential = {}
    output_lst = []
    ordered_dict = {}
    for img_name in full_img_list:
        print('img_name', img_name)
        image_name = img_name.split('/')[-1]
        final_lst = []
        lable_dict = {}
        sublocation_dict = {}
        tuples_of_model = [
         (
          img_name, big_obj_model, Model_predic_reconv.voc_txt_path, b_m),
         (
          img_name, small_obj_model, Model_predic_reconv.voc_txt_path_fr, s_m)]
        lable_tuple = (
         img_name, lable_model, Model_predic_reconv.location_txt, l_m)
        lable_dict['Location'] = (mpc.sequential_batch_train)(*lable_tuple)[0]
        for double in tuples_of_model:
            combined_lst = (mpc.sequential_batch_train)(*double)
            final_lst.append(combined_lst)

        sublocation_dict['Sublocation'] = [j for i in final_lst for j in iter(i)]
        lable_dict['Location'].update(sublocation_dict)
        driver_dict['fileurl'] = img_name
        driver_dict['img_name'] = image_name
        driver_dict.update(lable_dict)
        myorder = [
         'fileurl', 'img_name', 'Location']
        ordered = OrderedDict()
        for k in myorder:
            ordered[k] = driver_dict[k]

        print('ordered', ordered)
        print('Updated_dict', driver_dict)
        ordered_dict = dict(ordered)
        output_lst.append(ordered_dict)

    print('output_lst', output_lst)
    t2 = time.perf_counter()
    return output_lst


def drawBoundingBox(image, df):
    for box in df.index:
        x1, x2, y1, y2 = (
         df['xmin'][box], df['xmax'][box], df['ymin'][box], df['ymax'][box])
        np_array = np.array(image)
        pil_image = Image.fromarray(np_array)
        crop_box = (x1,y1, x2,y2)
        cropped_img = pil_image.crop((crop_box))
        cropped_img.save()
        label = df['location_code'][box]
        conf = df['probs'][box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 6)
        labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1, 5)
        _x1 = x1
        _y1 = y1
        _x2 = _x1 + labelSize[0][0]
        _y2 = y1 - int(labelSize[0][1])
        cv2.rectangle(image, (_x1, _y1), (_x2, _y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(image, str('%.2f' % conf), (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (255,
                                                                                       0,
                                                                                       0), 2)

    return image