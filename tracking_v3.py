import argparse
import pyrealsense2 as rs
import numpy as np
import cv2

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets_v3 import *
from utils.utils import *
from utils.tugas_akhir_v3_5 import *

def detect(save_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadRealSense2(width = 640, height = 480, fps = 30, img_size = img_size)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    matching = matching_object(match_model = opt.match_model, max_obj = opt.n_object) #Matching Object Initialization
    # test_col = collision_test(max_obj = opt.n_object)

    t0 = time.time()
    img = torch.zeros((1, 3, img_size, img_size), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once

    # Process Variable
    id = None
    prev_n = 0
    frame = 0
    warning = None

    # Area
    w = 2000
    h = 2000

    # Time Variable
    total_time = 0
    t_obs_pred = 0  # mark observer's predict step

    # Frame Time
    frame_time = 2000  # frame_time

    # YOLO + NMS Time
    yolo_nms = 2000

    # Match + Update Time
    match_time = 2000
    t_match_1 = 0  # mark the start of matching
    t_match_2 = 0  # mark the end of matching

    # Predict Time
    pred_time = 2000
    start_pred = 0  # mark the start of predict
    stop_pred = 0  # mark the end of predict

    # Update Time
    upd_time = 2000


    for path, depth, distance, depth_scale, img, im0s, vid_cap in dataset:

        # Untuk mengukur waktu
        t_start = time.time()

        # Sebelum loop objek, hapus memory untuk objek yang sudah lama hilang (lebih dari 50 frame)
        matching.clear_memory(frame = frame, max_miss = 10)

        # Observer prediction step + save to memory
        start_pred = time.time()
        t_pred = time.time() - t_obs_pred
        matching.predict_and_save(t_pred, frame)
        t_obs_pred = time.time()
        stop_pred = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=[0, 2], agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                # Update maximum allowable id
                new_n = int(len(det))
                matching.update_max_id(prev_n, new_n)

                # Reset id memory
                matching.reset_id_memory()

                # print(str(len(det)) + " objects detected")
                for *xyxy, conf, cls in det:
                    if save_img or view_img:  # Add bbox to image
                        label = '%s' % (names[int(cls)])
                        
                        # Calculate depth
                        # Splitting xyxy* (measurement)
                        xmin = int(xyxy[0])
                        ymin = int(xyxy[1])
                        xmax = int(xyxy[2])
                        ymax = int(xyxy[3])

                        # Calculate width and height
                        w = xmax - xmin
                        h = ymax - ymin

                        # Calculating measured centroid of the object (in Pixel)
                        xc = int(round(((xmax + xmin) / 2), 0))
                        yc = int(round(((ymax + ymin) / 2), 0))
                        depth_pixel = [xc, yc]
                        xc_msr = float((xyxy[2] + xyxy[0])/2)
                        yc_msr = float((xyxy[3] + xyxy[1])/2)
                        meas_pixel = [xc_msr, yc_msr]

                        # Calculating depth using CV2.Mean
                        jarak = calcdepth(xmin, ymin, xmax, ymax, distance, depth_scale)

                        # Cropping newly detected object
                        object = im0[ymin:ymax, xmin:xmax]

                        if depth_pixel is None:
                            print("depth_pixel is None")
                            print(depth_pixel)
                            continue

                        else:
                            t_match_1 = time.time()
                            id, upd_time = matching.main(object, meas_pixel, frame, label, jarak)
                            t_match_2 = time.time()
                            if id is not None:
                                # Collision Test
                                # collision, final_time = test_col.collision_time(1, matching.kalman_array[id, frame, 0],
                                #                                        matching.kalman_array[id, frame, 6],
                                #                                        matching.kalman_array[id, frame, 1],
                                #                                        matching.kalman_array[id, frame, 7],
                                #                                        matching.kalman_array[id, frame, 2],
                                #                                        matching.kalman_array[id, frame, 8])
                                # warning = test_col.warning(collision, final_time, id)

                                # Visualization
                                plot_one_box_gilbert(xyxy, im0, id=str(id), color=colors[int(id)], dist = round(matching.kalman_array[id, frame, 1], 1))

                    # Save trajectory to results.txt
                    if save_txt and id is not None:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 9 + '\n') % (
                                total_time, id,
                                matching.kalman_array[id, frame, 0], matching.kalman_array[id, frame, 1],
                                matching.kalman_array[id, frame, 2], matching.kalman_array[id, frame, 3],
                                matching.kalman_array[id, frame, 4], matching.kalman_array[id, frame, 5],
                                frame))

                # Update prev_n
                prev_n = int(new_n)

            matching.write_missing_objects(frame, total_time, save_path, save_txt)

            # Stream results
            if view_img:
                # matching.plot_missing_objects(frame, im0, colors)
                if opt.put_text:
                    try:
                        fps = str(int(round(1 / frame_time, 0))) + " fps"
                    except:
                        fps = "Div by 0"
                    match = os.path.split(opt.match_model)
                    put_txt(im0, str1=match[-1], fps=fps, warning = warning)
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

        # Frame Count
        frame += 1
        area = w * h
        yolo_nms = t2 - t1
        match_time = t_match_2 - t_match_1
        pred_time = stop_pred - start_pred
        frame_time = time.time() - t_start
        total_time = total_time + frame_time
        print("frame = ", frame)
        print("fps = ", (1 / frame_time))
        # if save_txt and id is not None:  # Write to file
        #     with open("C:/Users/HP/Desktop/1 object/r" + '.txt', 'a') as file:
        #         file.write(('%g ' * 9 + '\n') % (
        #             id,
        #             total_time,
        #             frame_time,
        #             yolo_nms,
        #             match_time,
        #             pred_time,
        #             upd_time,
        #             matching.vector_array[id, frame-1, 2],
        #             area))


    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--n-object', type=int, default=2, help='number of objects to be tracked')
    parser.add_argument('--match-model', type=str, default='NN_GA', help='Path to model for object matching')
    parser.add_argument('--put-text', action='store_true', help='display text')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
