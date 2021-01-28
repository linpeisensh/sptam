import numpy as np
import cv2 as cv

import time
from itertools import chain
from collections import defaultdict

from covisibility import CovisibilityGraph
from optimization import BundleAdjustment
from mapping import Mapping
from mapping import MappingThread
from components import Measurement
from motion import MotionModel
from loopclosing import LoopClosing
from dynaseg import DynaSeg

from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo

from copy import deepcopy as dp


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

class Tracking(object):
    def __init__(self, params):
        self.optimizer = BundleAdjustment()
        self.min_measurements = params.pnp_min_measurements
        self.max_iterations = params.pnp_max_iterations

    def refine_pose(self, pose, cam, measurements):
        assert len(measurements) >= self.min_measurements, (
            'Not enough points')

        self.optimizer.clear()
        self.optimizer.add_pose(0, pose, cam, fixed=False)

        for i, m in enumerate(measurements):
            self.optimizer.add_point(i, m.mappoint.position, fixed=True)
            self.optimizer.add_edge(0, i, 0, m)

        self.optimizer.optimize(self.max_iterations)
        return self.optimizer.get_pose(0)


class SPTAM(object):
    def __init__(self, params):
        self.params = params

        self.tracker = Tracking(params)
        self.motion_model = MotionModel()

        self.graph = CovisibilityGraph()
        self.mapping = MappingThread(self.graph, params)

        self.loop_closing = LoopClosing(self, params)
        self.loop_correction = None

        self.reference = None  # reference keyframe
        self.preceding = None  # last keyframe
        self.current = None  # current frame
        self.status = defaultdict(bool)

    def stop(self):
        self.mapping.stop()
        if self.loop_closing is not None:
            self.loop_closing.stop()

    def initialize(self, frame):
        mappoints, measurements = frame.triangulate()
        assert len(mappoints) >= self.params.init_min_points, (
            'Not enough points to initialize map.')

        keyframe = frame.to_keyframe()
        keyframe.set_fixed(True)
        self.graph.add_keyframe(keyframe)
        self.mapping.add_measurements(keyframe, mappoints, measurements)
        if self.loop_closing is not None:
            self.loop_closing.add_keyframe(keyframe)

        self.reference = keyframe
        self.preceding = keyframe
        self.current = keyframe
        self.status['initialized'] = True

        self.motion_model.update_pose(
            frame.timestamp, frame.position, frame.orientation)

    def track(self, frame):
        while self.is_paused():
            time.sleep(1e-4)
        self.set_tracking(True)

        self.current = frame
        # print('Tracking:', frame.idx, ' <- ', self.reference.id, self.reference.idx)

        predicted_pose, _ = self.motion_model.predict_pose(frame.timestamp)
        frame.update_pose(predicted_pose)

        if self.loop_closing is not None:
            if self.loop_correction is not None:
                estimated_pose = g2o.Isometry3d(
                    frame.orientation,
                    frame.position)
                estimated_pose = estimated_pose * self.loop_correction
                frame.update_pose(estimated_pose)
                self.motion_model.apply_correction(self.loop_correction)
                self.loop_correction = None

        local_mappoints = self.filter_points(frame)
        measurements = frame.match_mappoints(
            local_mappoints, Measurement.Source.TRACKING)

        # print('measurements:', len(measurements), '   ', len(local_mappoints))

        tracked_map = set()
        for m in measurements:
            mappoint = m.mappoint
            mappoint.update_descriptor(m.get_descriptor())
            mappoint.increase_measurement_count()
            tracked_map.add(mappoint)

        try:
            self.reference = self.graph.get_reference_frame(tracked_map)

            pose = self.tracker.refine_pose(frame.pose, frame.cam, measurements)
            frame.update_pose(pose)
            self.motion_model.update_pose(
                frame.timestamp, pose.position(), pose.orientation())
            tracking_is_ok = True
        except:
            tracking_is_ok = False
            print('tracking failed!!!')

        if tracking_is_ok and self.should_be_keyframe(frame, measurements):
            # print('new keyframe', frame.idx)
            keyframe = frame.to_keyframe()
            keyframe.update_reference(self.reference)
            keyframe.update_preceding(self.preceding)

            self.mapping.add_keyframe(keyframe, measurements)
            if self.loop_closing is not None:
                self.loop_closing.add_keyframe(keyframe)
            self.preceding = keyframe

        self.set_tracking(False)

    def filter_points(self, frame):
        local_mappoints = self.graph.get_local_map_v2(
            [self.preceding, self.reference])[0]

        can_view = frame.can_view(local_mappoints)
        # print('filter points:', len(local_mappoints), can_view.sum(),
        #       len(self.preceding.mappoints()),
        #       len(self.reference.mappoints()))

        checked = set()
        filtered = []
        for i in np.where(can_view)[0]:
            pt = local_mappoints[i]
            if pt.is_bad():
                continue
            pt.increase_projection_count()
            filtered.append(pt)
            checked.add(pt)

        for reference in set([self.preceding, self.reference]):
            for pt in reference.mappoints():  # neglect can_view test
                if pt in checked or pt.is_bad():
                    continue
                pt.increase_projection_count()
                filtered.append(pt)

        return filtered

    def should_be_keyframe(self, frame, measurements):
        if self.adding_keyframes_stopped():
            return False

        n_matches = len(measurements)
        n_matches_ref = len(self.reference.measurements())

        # print('keyframe check:', n_matches, '   ', n_matches_ref)

        return ((n_matches / n_matches_ref) <
                self.params.min_tracked_points_ratio) or n_matches < 20

    def set_loop_correction(self, T):
        self.loop_correction = T

    def is_initialized(self):
        return self.status['initialized']

    def pause(self):
        self.status['paused'] = True

    def unpause(self):
        self.status['paused'] = False

    def is_paused(self):
        return self.status['paused']

    def is_tracking(self):
        return self.status['tracking']

    def set_tracking(self, status):
        self.status['tracking'] = status

    def stop_adding_keyframes(self):
        self.status['adding_keyframes_stopped'] = True

    def resume_adding_keyframes(self):
        self.status['adding_keyframes_stopped'] = False

    def adding_keyframes_stopped(self):
        return self.status['adding_keyframes_stopped']


class stereoCamera(object):
    def __init__(self):
        self.cam_matrix_left = np.array([[707.0912, 0, 601.8873], [0, 707.0912, 183.1104], [0, 0, 1]])
        self.cam_matrix_right = np.array([[707.0912, 0, 601.8873], [0, 707.0912, 183.1104], [0, 0, 1]])

        self.distortion_l = np.array([[0] * 5])
        self.distortion_r = np.array([[0] * 5])

        self.R = np.eye(3)

        self.T = np.array([[0.53715], [0], [0]])

        self.focal_length = 707.0912

        self.baseline = 0.53715

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
        params = ParamsKITTI()
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

            sptam0 = dp(sptam1)


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