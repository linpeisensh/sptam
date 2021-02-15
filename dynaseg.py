import numpy as np
import cv2 as cv
from copy import deepcopy as dp


class DynaSeg():
    def __init__(self, iml, coco_demo, feature_params, disp_path, config, paraml, lk_params, mtx, dist, dilation):
        self.h, self.w = iml.shape[:2]
        self.coco = coco_demo
        self.feature_params = feature_params
        self.disp_path = disp_path
        self.config = config
        self.Q = self.getRectifyTransform()
        self.paraml = paraml

        self.lk_params = lk_params

        self.mtx = mtx
        self.dist = dist
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))

        self.obj = np.array([])
        self.IOU_thd = 0.0
        self.dyn_thd = 0.6

        self.a = 0
        self.t = 0

    def updata(self, iml, imr, i, k_frame):
        self.old_gray = cv.cvtColor(iml, cv.COLOR_BGR2GRAY)
        self.p = cv.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
        self.p1 = dp(self.p)
        self.ast = np.ones((self.p.shape[0], 1))
        self.points = self.get_points(i, iml, imr)
        self.otfm = np.linalg.inv(Rt_to_tran(k_frame.transform_matrix))

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

    def stereoMatchSGBM(self, iml, imr):
        left_matcher = cv.StereoSGBM_create(**self.paraml)

        disparity_left = left_matcher.compute(iml, imr)

        trueDisp_left = disparity_left.astype(np.float32) / 16.

        return trueDisp_left

    def get_points(self, i, iml, imr):
        iml_, imr_ = preprocess(iml, imr)
        disp = self.stereoMatchSGBM(iml_, imr_)
        dis = np.load(self.disp_path + str(i).zfill(6) + '.npy')
        disp[disp == 0] = dis[disp == 0]
        points = cv.reprojectImageTo3D(disp, self.Q)
        return points

    def track_obj(self, mask, idx):
        n = len(self.obj)
        max_IOU = 0
        ci = n
        mIOU = 0
        for i in range(n):
            cIOU = get_IOU(mask, self.obj[i][0])
            if cIOU > self.IOU_thd and cIOU > max_IOU:
                max_IOU = cIOU
                ci = i
            mIOU = max(cIOU, mIOU)
        if ci == n:
            # print(n,mIOU)
            self.obj.append([mask.astype(np.bool), 1, 0, idx])
        else:
            self.obj[ci][1] += 1
            self.obj[ci][0] = mask.astype(np.bool)
            self.obj[ci][3] = idx
        return ci

    def projection(self, frame, frame_gray):
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p1, None, **self.lk_params)
        self.ast *= st
        tfm = Rt_to_tran(np.array(frame.transform_matrix))
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

        if len(imgpts):
            cverror = cv.norm(P, imgpts, cv.NORM_L2) / len(imgpts)
        else:
            cverror = float('inf')
        print(cverror)
        self.p1 = p1
        ge = norm(error,imgpts)
        return ge, P

    def dyn_seg(self, frame, iml):
        frame_gray = cv.cvtColor(iml, cv.COLOR_BGR2GRAY)
        ge, P = self.projection(frame, frame_gray)


        image = iml.astype(np.uint8)
        prediction = self.coco.compute_prediction(image)
        top = self.coco.select_top_predictions(prediction)
        masks = top.get_field("mask").numpy()

        c = np.zeros((self.h, self.w))
        n = len(masks)
        for i in range(n):
            mask = masks[i].squeeze()
            mask = mask.astype(np.uint8)
            mask_dil = cv.dilate(mask, self.kernel)
            ao = 0
            co = 0
            for i in range(len(ge)):
                x, y = round(P[i][1]), round(P[i][0])
                if 0 <= x < self.h and 0 <= y < self.w and mask_dil[x, y]:
                    ao += 1
                    if ge[i]:
                        co += 1
            if ao > 1 and co / ao > 0.5:
                c[mask_dil.astype(np.bool)] = 255
        self.old_gray = frame_gray.copy()
        return c

    def iou(self, iml, idx):
        image = iml.astype(np.uint8)
        prediction = self.coco.compute_prediction(image)
        top = self.coco.select_top_predictions(prediction)
        omasks = top.get_field("mask").numpy()
        masks = []
        for mask in omasks:
            mask = mask.squeeze().astype(np.uint8)
            mask = cv.dilate(mask, self.kernel)
            masks.append(mask)
        res = []
        nc = len(self.obj)
        nm = len(masks)
        for i in range(nm):
            for j in range(nc):
                cIOU = get_IOU(masks[i], self.obj[j][0])
                res.append((cIOU, j, i))
        nu_obj = [True] * nc
        nu_mask = [True] * nm
        res.sort(key=lambda x: -x[0])
        for x in res:
            if nu_obj[x[1]] and nu_mask[x[2]]:
                if x[0] > self.IOU_thd:
                    self.obj[x[1]][0] = masks[x[2]].astype(np.bool)
                    self.obj[x[1]][1] += 1
                    self.obj[x[1]][3] = idx
                    self.obj[x[1]][4] = x[1]
                    nu_obj[x[1]] = False
                    nu_mask[x[2]] = False
                else:
                    break
        for i in range(nm):
            if nu_mask[i]:
                self.obj.append([masks[i].astype(np.bool), 1, 0, idx,-1])
        self.track_rate(idx)
        return

    def track_rate(self,idx):
        if idx == 1:
            self.oo = list(self.obj)
        else:
            no = self.obj
            nn = len(no)
            self.a += nn
            for j in range(nn):
                if no[j][4] != -1:
                    o_area = np.sum(self.oo[no[j][4]][0])
                    if o_area:
                        if 0.7 < np.sum(no[j][0]) / o_area < 1.3:
                            self.t += 1
                else:
                    f = 0
                    for obj in self.oo:
                        if np.sum(obj[0]):
                            if 0.7 < np.sum(no[j][0]) / np.sum(obj[0]) < 1.3 and get_IOU(no[j][0], obj[0]):
                                f = 1
                                break
                    if f:
                        self.t += 1
            self.oo = list(self.obj)
        return

    def dyn_seg_rec(self, frame, iml, idx):
        '''
        dynamic segmentation based on projection error and object recording
        :param frame: original sptam frame after tracking
        :param iml: left image
        :return:
        c: dynamic segmentation of iml
        '''
        frame_gray = cv.cvtColor(iml, cv.COLOR_BGR2GRAY)
        ge, P = self.projection(frame, frame_gray)

        nobj = len(self.obj)
        for i in range(nobj):
            cm = np.where(self.obj[i][0] == True)
            cmps = np.array(list(zip(cm[1], cm[0]))).astype(np.float32)
            nmps, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, cmps, None, **self.lk_params)
            nm = np.zeros_like(self.obj[i][0], dtype=np.uint8)
            for nmp in nmps:
                x, y = round(nmp[1]), round(nmp[0])
                if 0 <= x < self.h and 0 <= y < self.w:
                    nm[x, y] = 1
            nm = cv.erode(cv.dilate(nm, self.kernel), self.kernel)
            self.obj[i][0] = nm.astype(np.bool)

        self.obj = list(self.obj)
        self.iou(iml, idx)
        nobj = len(self.obj)
        cnd = [True] * nobj
        for ci in range(nobj):
            if self.obj[ci][3] == idx:
                ao = 0
                co = 0
                for i in range(len(ge)):
                    x, y = round(P[i][1]), round(P[i][0])
                    if 0 <= x < self.h and 0 <= y < self.w and self.obj[ci][0][x, y]:
                        ao += 1
                        if ge[i]:
                            co += 1
                if ao > 1 and co / ao > 0.5:
                    self.obj[ci][2] += 1
                    cnd[ci] = False

        c = np.zeros((self.h, self.w))
        nobj = len(self.obj)
        res = [True] * nobj
        print('num of objs', nobj)
        cc = []
        for i in range(nobj):
            cc.append(list(self.obj[i]))
            if idx - self.obj[i][3] != 0:
                res[i] = False
            elif self.obj[i][2] / self.obj[i][1] >= self.dyn_thd or self.obj[i][2] >= 5:  #
                c[self.obj[i][0]] = 255
            elif cnd[i]:
                self.obj[i][2] = max(0, self.obj[i][2] - 0.5)
        self.obj = np.array(self.obj, dtype=object)
        self.obj = self.obj[res]
        for obj in self.obj:
            print('a: {}, d: {}'.format(obj[1],obj[2]))
        self.old_gray = frame_gray.copy()
        return c


def Rt_to_tran(tfm):
    res = np.zeros((4, 4))
    res[:3, :] = tfm[:3, :]
    res[3, 3] = 1
    return res


def preprocess(img1, img2):
    im1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    im2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    im1 = cv.equalizeHist(im1)
    im2 = cv.equalizeHist(im2)

    return im1, im2


def get_IOU(m1, m2):
    I = np.sum(np.logical_and(m1, m2))
    U = np.sum(np.logical_or(m1, m2))
    s1 = np.sum(m1)
    s2 = np.sum(m2)
    if s1 and s2:
        s = s1 / s2 if s1 > s2 else s2 / s1
        U *= s
    else:
        U = 0
    if U:
        return I / U
    else:
        return 0

def norm(error, imgpts):
    merror = np.array(error)
    lma = imgpts[:, 0] < 400

    rma = imgpts[:, 0] > 840

    mma = np.logical_and((~lma), (~rma))

    ge = np.array([False] * len(merror))
    lm = merror[lma]
    rm = merror[rma]
    mm = merror[mma]
    if len(lm):
        ge[lma] = lm > np.percentile(lm, 89)
    if len(rm):
        ge[rma] = rm > np.percentile(rm, 89)
    if len(mm):
        ge[mma] = mm > np.percentile(mm, 75)
    return ge