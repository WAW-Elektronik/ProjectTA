import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import serial
import time
import csv
#from pynput import keyboard
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Penyimpanan root YOLOv5
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # ambil ROOT ke PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relatif

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode

def save_prints_to_csv_realtime(print_sets, file_path):
    # Buka file dalam mode tambahkan
    with open(file_path, 'a', newline='') as file:
        # Buat penulis CSV
        writer = csv.writer(file)

        # Tulis setiap set pernyataan cetak sebagai satu baris
        for print_set in print_sets:
            writer.writerow(print_set)

        file.flush()  # Flush buffer untuk memastikan penulisan

print_set1 = ['Hello', 'World', 'This is set 1']
print_set2 = ['Python', 'OpenAI', 'ChatGPT', 'This is set 2']
print_sets = [print_set1, print_set2]
file_path = 'datamalam_6011try_050623.csv' # Nama file yang tersimpan

# Menambahkan trackbar di jendela openCV2
def on_trackbar1(AdjustableSpeed):
    print("Speed Adjustable: ", AdjustableSpeed)

def on_trackbar2(konfirmasipersetujuan):
    print("Konfirmasi persetujuan: ", konfirmasipersetujuan)

@smart_inference_mode()
def run(
        weights='/usr/local/lib/python3.9/dist-packages/yolov5/best.pt', # penyimpanan path
        source=ROOT / '/home/user/freedomtech',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=None,  # Ukuran size (height, width)
        img=None,  # Ukuran size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # deteksi maksimum per gambar
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # Penampilan hasil
        simpan_txt=False,  # Penyimpanan hasil ke *.txt
        simpan_conf=False,  # Penyimpanan confidence di --save-txt labels
        simpan_crop=False,  # Simpan kotak prediksi yang dipotong
        tidaksimpan=False,  # jangan simpan gambar/video
        kelas=None,  # filter berdasarkan kelas: --class 0, atau --class 0 2 3
        agnostic_nms=False,  # NMS kelas-agnostik
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # memperbarui semua model
        project='/home/user/yoloresult',  # Simpan hasil ke project/name
        name='exp',  # Simpan hasil ke project/name
        exist_ok=False,  # Proyek/nama yang ada ok, jangan bertambah
        tebal_boundingbox=3,  # Ketebalan bounding box (pixels)
        hide_labels=False,  # Sembunyikan label
        hide_conf=False,  # Sembunyikan confidence
        half=False,  # gunakan inferensi setengah presisi FP16
        dnn=False,  # gunakan OpenCV DNN untuk inferensi ONNX
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    simpan_gambar = not tidaksimpan and not source.endswith('.txt')  # menyimpan gambar inferensi
    inputfile = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    inputurl = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    inputwebcam = source.isnumeric() or source.endswith('.streams') or (inputurl and not inputfile)
    inputscreenshot = source.lower().startswith('screen')
    if inputurl and inputfile:
        source = check_file(source)  # download

    if imgsz is None and img is None:
        imgsz = 640
    elif img is not None:
        imgsz = img

    if isinstance(imgsz, int):
        imgsz = [imgsz, imgsz]

    # Direktori
    simpan_direktori = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run (peningkatan berjalan)
    (simpan_direktori / 'labels' if simpan_txt else simpan_direktori).mkdir(parents=True, exist_ok=True)  # membuat direktori

    # Memuat model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # Periksa ukuran gambar

    # Pemuat data
    bs = 1  # Ukuran size
    if inputwebcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif inputscreenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Menjalankan inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup (pemanasan)
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 ke fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # perluas untuk peredupan batch

        # Inference
        with dt[1]:
            visualize = increment_path(simpan_direktori / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, kelas, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Proses prediksi
        for i, det in enumerate(pred):  # per gambar
            seen += 1
            if inputwebcam:  # Ukuran batch >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # ke Path
            save_path = str(simpan_direktori / p.name)  # im.jpg
            txt_path = str(simpan_direktori / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # menampilkan string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if simpan_crop else im0  # untuk simpan_crop
            annotator = Annotator(im0, line_width=tebal_boundingbox, example=str(names))
            if len(det):
                # Skala ulang kotak dari img_size ke ukuran im0
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Menampilkan hasil
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # Deteksi per kelas
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # tambahkan ke string
                    #print(names)

                # Menulis Hasil
                for *xyxy, conf, cls in reversed(det):
                    if simpan_txt:  # Menulis ke file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if simpan_conf else (cls, *xywh)  # Format label
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if simpan_gambar or simpan_crop or view_img:  # Tambahkan bbox to image
                        c = int(cls)  # Ubah kelas ke tipe data integer
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                        print(c,conf)
                        #adjspeed = cv2.getTrackbarPos("AdjSpeed","WebCam/default")
                    if simpan_crop:
                        save_one_box(xyxy, imc, file=simpan_direktori / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Hasil streaming
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # izinkan pengubahan ukuran jendela (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.createTrackbar("AdjSpeed",str(p),0,200,on_trackbar1)
                    cv2.createTrackbar("AktifasiLaneKeeping",str(p),0,1,on_trackbar2)
                adjspeed = cv2.getTrackbarPos("AdjSpeed",str(p))
                aktifasiLaneKeeping = cv2.getTrackbarPos("AktifasiLaneKeeping",str(p))
                new_print_set = [c, conf, adjspeed, 'DataKecepatan']
                print_sets.append(new_print_set) # Tambahkan pernyataan cetak baru ke daftar kumpulan cetak
                save_prints_to_csv_realtime([new_print_set], file_path) # Simpan set cetak ke file CSV secara real-time
                cv2.imshow(str(p), im0)
                
                if cv2.waitKey(1)&0xFF==ord('q'):
                    break
                #cv2.waitKey(1)  # 1 millisecond

            # Simpan hasil (gambar dengan deteksi)
            if simpan_gambar:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' atau 'stream'
                    if vid_path[i] != save_path:  # video baru
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # rilis penulis video sebelumnya
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # memaksa akhiran *.mp4 pada video hasil
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Menampilkan waktu (hanya inferensi)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Menampilkan hasil
    t = tuple(x.t / seen * 1E3 for x in dt)  # kecepatan per gambar
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if simpan_txt or simpan_gambar:
        s = f"\n{len(list(simpan_direktori.glob('labels/*.txt')))} labels saved to {simpan_direktori / 'labels'}" if simpan_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', simpan_direktori)}{s}")
    if update:
        strip_optimizer(weights[0])  # Perbarui model (untuk memperbaiki SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # Memperluas
    print_args(vars(opt))
    return opt


def main():
    opt = parse_opt()
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    main()
