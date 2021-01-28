import numpy as np
import cv2 as cv
import os


class DynaSeg():
    def __init__(self,data_path, coco_demo, feature_params,dis_path,config,paraml,lk_params,mtx,dist,dilation):
        self.data_path = data_path
        self.coco = coco_demo
        self.feature_params = feature_params
        self.dis_path = dis_path
        self.config = config
        self.Q = self.getRectifyTransform()
        self.paraml = paraml

        self.lk_params = lk_params

        self.mtx = mtx
        self.dist = dist
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))


    def updata(self,iml, imr, i,k_frame):
        self.iml = iml
        self.imr = cv.imread(self.right[i], cv.IMREAD_UNCHANGED)
        self.h, self.w = self.iml.shape[:2]
        self.old_gray = cv.cvtColor(self.iml, cv.COLOR_BGR2GRAY)
        self.p = cv.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
        self.ast = np.ones((self.p.shape[0], 1))
        self.points = self.get_points(i)
        self.otfm = np.linalg.inv(Rt_to_tran(k_frame.transform_matrix))


    def preprocess(self):
        im1 = cv.cvtColor(self.iml, cv.COLOR_BGR2GRAY)
        im2 = cv.cvtColor(self.imr, cv.COLOR_BGR2GRAY)

        im1 = cv.equalizeHist(im1)
        im2 = cv.equalizeHist(im2)

        return im1, im2

    def get_instance_mask(self):
        image = self.iml.astype(np.uint8)
        prediction = self.coco.compute_prediction(image)
        top = self.coco.select_top_predictions(prediction)
        masks = top.get_field("mask").numpy()
        h, w, c = image.shape
        rmask = np.zeros((h, w))
        n = len(masks)
        i = 0
        for i in range(n):
            mask = masks[i].squeeze()
            rmask[mask] = i + 1
        return rmask, i + 1

    def getRectifyTransform(self):
        left_K = self.config.cam_matrix_left
        right_K = self.config.cam_matrix_right
        left_distortion = self.config.distortion_l
        right_distortion = self.config.distortion_r
        R = self.config.R
        T = self.config.T

        R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                         (self.w, self.h), R, T, alpha=0)
        return Q

    def stereoMatchSGBM(self, down_scale=False):

        left_matcher = cv.StereoSGBM_create(**self.paraml)
        paramr = self.paraml
        paramr['minDisparity'] = -self.paraml['numDisparities']
        right_matcher = cv.StereoSGBM_create(**paramr)

        # 计算视差图
        size = (self.iml.shape[1], self.iml.shape[0])
        if down_scale == False:
            disparity_left = left_matcher.compute(self.iml, self.imr)
            disparity_right = right_matcher.compute(self.imr, self.iml)

        else:
            self.iml_down = cv.pyrDown(self.iml)
            self.imr_down = cv.pyrDown(self.imr)
            factor = self.iml.shape[1] / self.iml_down.shape[1]

            disparity_left_half = left_matcher.compute(self.iml_down, self.imr_down)
            disparity_right_half = right_matcher.compute(self.imr_down, self.iml_down)
            disparity_left = cv.resize(disparity_left_half, size, interpolation=cv.INTER_AREA)
            disparity_right = cv.resize(disparity_right_half, size, interpolation=cv.INTER_AREA)
            disparity_left = factor * disparity_left
            disparity_right = factor * disparity_right

        trueDisp_left = disparity_left.astype(np.float32) / 16.
        trueDisp_right = disparity_right.astype(np.float32) / 16.

        return trueDisp_left, trueDisp_right

    def get_points(self, i):
        iml_, imr_ = self.preprocess(self.iml, self.imr)
        disp, _ = self.stereoMatchSGBM(iml_, imr_, False)
        dis = np.load(self.disp_path + str(i).zfill(6) + '.npy')
        disp[disp == 0] = dis[disp == 0]
        points = cv.reprojectImageTo3D(disp, self.Q)
        return points

    def dyn_seg(self, frame, iml):
        frame_gray = cv.cvtColor(iml, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p, None, **self.lk_params)
        self.ast *= st
        self.old_gray = frame_gray.copy()

        tfm = Rt_to_tran(frame.transform_matrix)
        tfm = self.otfm.dot(tfm)
        b = cv.Rodrigues(tfm[:3, :3])
        R = b[0]
        t = tfm[:3, 3].reshape((3, 1))

        P = p1[self.ast == 1]
        objpa = np.array([self.points[int(y), int(x)] for x, y in self.p[self.ast == 1].squeeze()])
        imgpts, jac = cv.projectPoints(objpa, R, -t, self.mtx, self.dist)
        imgpts = imgpts.squeeze()
        P = P.squeeze()[~np.isnan(imgpts).any(axis=1)]
        imgpts = imgpts[~np.isnan(imgpts).any(axis=1)]
        P = P[(0 < imgpts[:, 0]) * (imgpts[:, 0] < self.w) * (0 < imgpts[:, 1]) * (imgpts[:, 1] < self.h)]
        imgpts = imgpts[(0 < imgpts[:, 0]) * (imgpts[:, 0] < self.w) * (0 < imgpts[:, 1]) * (imgpts[:, 1] < self.h)]
        error = ((P - imgpts) ** 2).sum(-1)
        P = P[error < 1e6]
        imgpts = imgpts[error < 1e6].astype(np.float32)
        error = error[error < 1e6]
        nl2m, res = self.get_instance_mask(self.iml, self.coco_demo)
        nl2m_dil = cv.dilate(nl2m, self.kernel)[:, :, None]
        merror = np.array(error)

        for i in range(len(error)):
            if imgpts[i][0] < 400:
                merror[i] = max(merror[i] - 15 * 15, 0)
            if imgpts[i][0] > 900:
                merror[i] = max(merror[i] - 325, 0)
        ge = merror > np.median(error)
        nres = set()
        for o in range(1, res + 1):
            ao = 0
            co = 0
            for i in range(len(error)):
                if nl2m_dil[min(round(P[i][1]), self.h - 1), min(round(P[i][0]), self.w - 1)] == o:
                    ao += 1
                    if ge[i]:
                        co += 1
            if ao > 1:
                if co / ao > 0.5:
                    nres.add(o)
        c = np.zeros_like(nl2m_dil)
        for i in nres:
            c[nl2m_dil == i] = 255

        self.p = p1
        return c

def Rt_to_tran(tfm):
  res = np.zeros((4,4))
  res[:3,:] = tfm[:3,:]
  res[3,3] = 1
  return res