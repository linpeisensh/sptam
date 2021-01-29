import numpy as np
import cv2 as cv
import time

from dynaseg import DynaSeg
from msptam import SPTAM, stereoCamera

from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo


def maskofkp(kp,l_mask):
  n = len(kp)
  l_mask = l_mask.squeeze()
  mok = []
  for i in range(n):
    x,y = map(int,kp[i].pt)
    mok.append(l_mask[y,x]==0)
  return np.array(mok)

def save_trajectory(trajectory, filename):
    with open(filename, 'w') as traj_file:
        traj_file.writelines('{r00} {r01} {r02} {t0} {r10} {r11} {r12} {t1} {r20} {r21} {r22} {t2}\n'.format(
            r00=repr(r00),
            r01=repr(r01),
            r02=repr(r02),
            t0=repr(t0),
            r10=repr(r10),
            r11=repr(r11),
            r12=repr(r12),
            t1=repr(t1),
            r20=repr(r20),
            r21=repr(r21),
            r22=repr(r22),
            t2=repr(t2)
        ) for r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 in trajectory)



if __name__ == '__main__':
    import g2o

    import argparse

    from threading import Thread

    from components import Camera
    from components import StereoFrame
    from feature import ImageFeature
    from params import ParamsKITTI, ParamsEuroc
    from dataset import KITTIOdometry, EuRoCDataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-viz', action='store_true', help='do not visualize')
    parser.add_argument('--cocopath', type=str, help='coco path',
                        default='../maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml')
    parser.add_argument('--device', type=str, help='device (cpu/cuda)',
                        default='cpu')
    parser.add_argument('--dataset', type=str, help='dataset (KITTI/EuRoC)',
                        default='KITTI')
    parser.add_argument('--path', type=str, help='dataset path',
                        default='path/to/your/KITTI_odometry/sequences/00')
    args = parser.parse_args()

    if args.dataset.lower() == 'kitti':
        params = ParamsKITTI('ORB-ORB')
        dataset = KITTIOdometry(args.path)
    elif args.dataset.lower() == 'euroc':
        params = ParamsEuroc()
        dataset = EuRoCDataset(args.path)

    disp_path = '/usr/stud/linp/storage/user/linp/disparity/' + args.path[-2:] + '/'

    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.1,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))



    sptam0 = SPTAM(params)
    sptam1 = SPTAM(params)

    config = stereoCamera()
    mtx = np.array([[707.0912, 0, 601.8873], [0, 707.0912, 183.1104], [0, 0, 1]])
    dist = np.array([[0] * 4]).reshape(1, 4).astype(np.float32)

    dilation = 5
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))


    visualize = not args.no_viz
    if visualize:
        from viewer import MapViewer
        viewer = MapViewer(sptam1, params)

    cam = Camera(
        dataset.cam.fx, dataset.cam.fy, dataset.cam.cx, dataset.cam.cy,
        dataset.cam.width, dataset.cam.height,
        params.frustum_near, params.frustum_far,
        dataset.cam.baseline)
    print(dataset.cam.fx)

    otrajectory = []
    atrajectory = []
    n = len(dataset)
    print('sequence {}: {} images'.format(args.path[-2:],n))


    config_file = args.cocopath
    # "configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", args.device])
    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
    )

    paraml = {'minDisparity': 1,
              'numDisparities': 64,
              'blockSize': 10,
              'P1': 4 * 3 * 9 ** 2,
              'P2': 4 * 3 * 9 ** 2,
              'disp12MaxDiff': 1,
              'preFilterCap': 10,
              'uniquenessRatio': 15,
              'speckleWindowSize': 100,
              'speckleRange': 1,
              'mode': cv.STEREO_SGBM_MODE_SGBM_3WAY
              }

    if n:
        iml = cv.imread(dataset.left[0], cv.IMREAD_UNCHANGED)
        dseg = DynaSeg(iml, coco_demo, feature_params, disp_path, config, paraml, lk_params, mtx, dist, dilation)
        for i in range(n):
            iml = cv.imread(dataset.left[i], cv.IMREAD_UNCHANGED)
            imr = cv.imread(dataset.right[i], cv.IMREAD_UNCHANGED)
            featurel = ImageFeature(iml, params)
            featurer = ImageFeature(imr, params)
            timestamp = dataset.timestamps[i]

            time_start = time.time()

            t = Thread(target=featurer.extract)
            t.start()
            featurel.extract()
            t.join()


            print('{}. frame'.format(i))
            frame = StereoFrame(i, g2o.Isometry3d(), featurel, featurer, cam, timestamp=timestamp)

            if not sptam0.is_initialized():
                sptam0.initialize(frame)
            else:
                sptam0.track(frame)

            R = frame.pose.orientation().matrix()
            t = frame.pose.position()
            cur_tra = list(R[0]) + [t[0]] + list(R[1]) + [t[1]] + list(R[2]) + [t[2]]
            otrajectory.append((cur_tra))
            if i % 5 == 0:
                if i:
                    c = dseg.dyn_seg(frame,iml)
                dseg.updata(iml,imr,i,frame)
            else:
                c = dseg.dyn_seg(frame,iml)


            featurel = ImageFeature(iml, params)
            featurer = ImageFeature(imr, params)

            t = Thread(target=featurer.extract)
            t.start()
            featurel.extract()
            t.join()


            if i:
                lm = c
                rm = c
                ofl = np.array(featurel.keypoints)
                ofr = np.array(featurer.keypoints)
                flm = maskofkp(ofl, lm)
                frm = maskofkp(ofr, rm)
                featurel.keypoints = list(ofl[flm])
                featurer.keypoints = list(ofr[frm])
                featurel.descriptors = featurel.descriptors[flm]
                featurer.descriptors = featurer.descriptors[frm]
                featurel.unmatched = featurel.unmatched[flm]
                featurer.unmatched = featurer.unmatched[frm]

            frame = StereoFrame(i, g2o.Isometry3d(), featurel, featurer, cam, timestamp=timestamp)

            if not sptam1.is_initialized():
                sptam1.initialize(frame)
            else:
                sptam1.track(frame)


            R = frame.pose.orientation().matrix()
            t = frame.pose.position()
            cur_tra = list(R[0]) + [t[0]] + list(R[1]) + [t[1]] + list(R[2]) + [t[2]]
            atrajectory.append((cur_tra))


            if visualize:
                viewer.update()


        # print('average time', np.mean(durations))
        save_trajectory(otrajectory,'o{}.txt'.format(args.path[-2:]))
        save_trajectory(atrajectory,'a{}.txt'.format(args.path[-2:]))
        print('save a{}.txt successfully'.format(args.path[-2:]))
        sptam0.stop()
        sptam1.stop()
        if visualize:
            viewer.stop()
    else:
        print('path is wrong')