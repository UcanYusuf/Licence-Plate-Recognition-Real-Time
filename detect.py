import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *

import cv2
import numpy as np
import pytesseract
from multiprocessing import Pool, Manager
import functools
import imutils
import ctypes
from imutils import paths
import string

manager_plaka_reader = Manager()
camera_plaka_reader_val = manager_plaka_reader.Value(ctypes.Array, [])
counter_lock_plaka_reader = manager_plaka_reader.Lock()

manager_plaka_reader_text = Manager()
camera_plaka_reader_val_text = manager_plaka_reader_text.Value(ctypes.Array, [])
counter_lock_plaka_reader_text = manager_plaka_reader_text.Lock()

def build_tesseract_options(psm=7):
    # tell Tesseract to only OCR alphanumeric characters
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    # set the PSM mode
    options += " --psm {}".format(psm)
    # return the built options string
    return options

def detect(camera_plaka_reader_val, camera_plaka_reader_val_text):
    out, source, weights, view_img, save_txt, imgsz = "inference/output", "video1.mp4", "weights/best_licplate_small.pt", True, True, 640
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device("0")
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float().eval()  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=True)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.35, 0.4, classes=0, agnostic=True)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        text = camera_plaka_reader_val_text.value[:]

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                
                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        c1,c2 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        try:
                            start_x = c1[0]
                            end_x = c2[0]
                            start_y = c1[1]
                            end_y = c2[1]  
                            im_crop = im0[start_y:end_y,start_x:end_x]
                            #cv2.imshow("crop",im_crop)
                            camera_plaka_reader_val.value = im_crop                   
                        except:
                            pass
                        
            # Print time (inference + NMS)
            #print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                try:
                    if len(text) > 0:
                        cv2.putText(im0, text[:-2], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                except:
                    pass

                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

    #print('Done. (%.3fs)' % (time.time() - t0))

def ProjectMain(camera_plaka_reader_val, camera_plaka_reader_val_text):
    lpText_List = []
    options = build_tesseract_options(psm=7)

    while True:
        try:
            img_plaka = camera_plaka_reader_val.value[:]
            gray = cv2.cvtColor(img_plaka, cv2.COLOR_BGR2GRAY)
            
            rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
            
            img_plaka = cv2.threshold(blackhat, 50, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # apply automatic license plate recognition
            
            lpText = pytesseract.image_to_string(img_plaka, config=options)
            
            if lpText != None and lpText not in lpText_List:
                lpText_List = []
                lpText_List.append(lpText)
            elif lpText != None and lpText in lpText_List:
                lpText_List.append(lpText)
            
            if lpText != None and len(lpText_List)>0 and len(lpText)>5: # This line for eliminate some wrong detect.
                camera_plaka_reader_val_text.value = lpText
                print("[INFO] {}".format(lpText))
            else:
                pass
                camera_plaka_reader_val_text.value = []
        except:
            continue

def smap(f):
    return f()

def main():
    f_ProjectMain = functools.partial(ProjectMain,camera_plaka_reader_val, camera_plaka_reader_val_text)
    f_plakaReader = functools.partial(detect,camera_plaka_reader_val, camera_plaka_reader_val_text)

    with Pool() as pool:
        res = pool.map(smap, [f_ProjectMain, f_plakaReader])

if __name__ == '__main__':
    with torch.no_grad():
        main()

