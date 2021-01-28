import numpy as np
import cv2 as cv


class DynaSeg():
    def __init__(self,iml,imr,coco_demo, feature_params):
        self.iml = iml
        self.imr = imr
        self.coco = coco_demo
        self.h, self.w = self.iml.shape[:2]
        self.old_gray = cv.cvtColor(self.iml, cv.COLOR_BGR2GRAY)
        self.p = cv.goodFeaturesToTrack(self.old_gray, mask=None, **feature_params)

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

    def dyn_seg(self):
        frame_gray = cv.cvtColor(self.iml, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, p1, None, **lk_params)
        ast *= st
        self.old_gray = frame_gray.copy()

        tfm = Rt_to_tran(frame.transform_matrix)
        tfm = otfm.dot(tfm)
        b = cv.Rodrigues(tfm[:3, :3])
        R = b[0]
        t = tfm[:3, 3].reshape((3, 1))

        P = p1[ast == 1]
        objpa = np.array([points_3d[int(y), int(x)] for x, y in p[ast == 1].squeeze()])
        imgpts, jac = cv.projectPoints(objpa, R, -t, mtx, dist)
        imgpts = imgpts.squeeze()
        P = P.squeeze()[~np.isnan(imgpts).any(axis=1)]
        imgpts = imgpts[~np.isnan(imgpts).any(axis=1)]
        P = P[(0 < imgpts[:, 0]) * (imgpts[:, 0] < width) * (0 < imgpts[:, 1]) * (imgpts[:, 1] < height)]
        imgpts = imgpts[(0 < imgpts[:, 0]) * (imgpts[:, 0] < width) * (0 < imgpts[:, 1]) * (imgpts[:, 1] < height)]
        error = ((P - imgpts) ** 2).sum(-1)
        P = P[error < 1e6]
        imgpts = imgpts[error < 1e6].astype(np.float32)
        error = error[error < 1e6]
        nl2m, res = self.get_instance_mask(l2, coco_demo)
        nl2m_dil = cv.dilate(nl2m, kernel)[:, :, None]
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
                if nl2m_dil[min(round(P[i][1]), height - 1), min(round(P[i][0]), width - 1)] == o:
                    ao += 1
                    if ge[i]:
                        co += 1
            if ao > 1:
                if co / ao > 0.5:
                    nres.add(o)
        c = np.zeros_like(nl2m_dil)
        for i in nres:
            c[nl2m_dil == i] = 255
        return c, p1

def Rt_to_tran(tfm):
  res = np.zeros((4,4))
  res[:3,:] = tfm[:3,:]
  res[3,3] = 1
  return res