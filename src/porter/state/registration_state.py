from math import atan2
import cv2
from .state import State
from src.utils import mediapipe_utils as mpu
import depthai as dai
import numpy as np

class RegistrationState(State):
    def __init__(self, device, manager):
        super().__init__(device, manager)
        self.state_name = "registration"
        self.registration_threshold = 5
        self.counter = 0
        self.targetRegistered = False
        self.sm = manager

        self.q_pd_in = device.getInputQueue(name="pd_in")
        self.q_pd_out = device.getOutputQueue(name="pd_out", maxSize=4, blocking=True)
        self.q_lm_out = device.getOutputQueue(name="lm_out", maxSize=4, blocking=True)
        self.q_lm_in = device.getInputQueue(name="lm_in")

        self.regions = None
        self.nb_active_regions = 0
        self.lm_input_length = 256
        self.nb_kps = 33 # using full body

        self.lm_counter = 0
        self.lm_counter_threshold = 1

        anchor_options = mpu.SSDAnchorOptions(num_layers=4,
                                              min_scale=0.1484375,
                                              max_scale=0.75,
                                              input_size_height=128,
                                              input_size_width=128,
                                              anchor_offset_x=0.5,
                                              anchor_offset_y=0.5,
                                              strides=[8, 16, 16, 16],
                                              aspect_ratios=[1.0],
                                              reduce_boxes_in_lowest_layer=False,
                                              interpolated_scale_aspect_ratio=1.0,
                                              fixed_anchor_size=True)
        self.anchors = mpu.generate_anchors(anchor_options)
        self.nb_anchors = self.anchors.shape[0]
        self.pd_score_thresh = 0.5
        self.pd_nms_thresh = 0.3
        self.lm_score_threshold = 0.7
        print(f"{self.nb_anchors} anchors have been created")

        self.semaphore_flag = {
            (3, 4): 'A', (2, 4): 'B', (1, 4): 'C', (0, 4): 'D',
            (4, 7): 'E', (4, 6): 'F', (4, 5): 'G', (2, 3): 'H',
            (0, 3): 'I', (0, 6): 'J', (3, 0): 'K', (3, 7): 'L',
            (3, 6): 'M', (3, 5): 'N', (2, 1): 'O', (2, 0): 'P',
            (2, 7): 'Q', (2, 6): 'R', (2, 5): 'S', (1, 0): 'T',
            (1, 7): 'U', (0, 5): 'V', (7, 6): 'W', (7, 5): 'X',
            (1, 6): 'Y', (5, 6): 'Z'
        }

    def update(self):

        if self.targetRegistered == True:
            print("target already registered")
            return None

        super().update()

        if len(self.trackletsData) > 1:
            pass
        elif len(self.trackletsData) == 0:
            pass
        else:
            self.counter += 1
            if self.counter > self.registration_threshold:
                frame_nn = dai.ImgFrame()
                frame_nn.setWidth(128)
                frame_nn.setHeight(128)
                frame_nn.setData(self.to_planar(self.frame, (128, 128)))
                self.q_pd_in.send(frame_nn)
                self.counter = 0

                inference = self.q_pd_out.get()
                self.pd_postprocess(inference)

                # get landmarks
                for i, r in enumerate(self.regions):
                    # although we enumerate here, the process only starts if there is only one person in view
                    frame_nn = mpu.warp_rect_img(r.rect_points, self.frame, self.lm_input_length, self.lm_input_length)
                    nn_data = dai.NNData()
                    nn_data.setLayer("input_1", self.to_planar(frame_nn, (self.lm_input_length, self.lm_input_length)))
                    self.lm_counter += 1
                    if self.lm_counter > self.lm_counter_threshold:
                        self.q_lm_in.send(nn_data)

                        # Get landmarks
                        inference = self.q_lm_out.get()
                        gest = self.lm_postprocess(r, inference)
                        #print(gest)
                        if gest is "B" or gest is "C" or gest is "D":
                            print(gest)
                            self.register_target()
                        self.lm_counter = 0

    def register_target(self):
        roi = self.trackletsData[0].roi.denormalize(self.frame.shape[1], self.frame.shape[0])
        x1 = int(roi.topLeft().x)
        y1 = int(roi.topLeft().y)
        x2 = int(roi.bottomRight().x)
        y2 = int(roi.bottomRight().y)

        if self.statusMap[self.trackletsData[0].status] == "TRACKED":
            det_frame = self.frame[y1:y2, x1:x2]
            nn_data = dai.NNData()
            nn_data.setLayer("data", self.to_planar(det_frame, (128, 256)))
            self.device.getInputQueue("reid_in").send(nn_data)

            reid_result = self.device.getOutputQueue("reid_nn").get().getFirstLayerFp16()
            trackedPeople = {}
            trackedPeople[0] = reid_result
            self.sm.SetTargetPerson(trackedPeople)
            self.sm.SetTargetPersonId(0)
            print("Registered target")
            self.targetRegistered = True

    def pd_postprocess(self, inference):
        scores = np.array(inference.getLayerFp16("classificators"), dtype=np.float16)  # 896
        if len(scores) == 0:
            print("no inference for pose")
            return None
        bboxes = np.array(inference.getLayerFp16("regressors"), dtype=np.float16).reshape((self.nb_anchors, 12))  # 896x12

        # Decode bboxes
        self.regions = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors, best_only=True)
        mpu.detections_to_rect(self.regions, kp_pair=[0, 1])
        mpu.rect_transformation(self.regions, self.frame.shape[0], self.frame.shape[0])

    def lm_postprocess(self, region, inference):
        if inference == None:
            return None
        region.lm_score = inference.getLayerFp16("output_poseflag")[0]
        #print("got lm_score: " + str(region.lm_score))
        if region.lm_score > self.lm_score_threshold:
            self.nb_active_regions += 1

            lm_raw = np.array(inference.getLayerFp16("ld_3d")).reshape(-1, 5)
            # Each keypoint have 5 information:
            # - X,Y coordinates are local to the region of
            # interest and range from [0.0, 255.0].
            # - Z coordinate is measured in "image pixels" like
            # the X and Y coordinates and represents the
            # distance relative to the plane of the subject's
            # hips, which is the origin of the Z axis. Negative
            # values are between the hips and the camera;
            # positive values are behind the hips. Z coordinate
            # scale is similar with X, Y scales but has different
            # nature as obtained not via human annotation, by
            # fitting synthetic data (GHUM model) to the 2D
            # annotation.
            # - Visibility, after user-applied sigmoid denotes the
            # probability that a keypoint is located within the
            # frame and not occluded by another bigger body
            # part or another object.
            # - Presence, after user-applied sigmoid denotes the
            # probability that a keypoint is located within the
            # frame.

            # Normalize x,y,z. Scaling in z = scaling in x = 1/self.lm_input_length
            lm_raw[:, :3] /= self.lm_input_length
            # Apply sigmoid on visibility and presence (if used later)
            # lm_raw[:,3:5] = 1 / (1 + np.exp(-lm_raw[:,3:5]))

            # region.landmarks contains the landmarks normalized 3D coordinates in the relative oriented body bounding box
            region.landmarks = lm_raw[:, :3]
            # Calculate the landmark coordinate in square padded image (region.landmarks_padded)
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([(x, y) for x, y in region.rect_points[1:]],
                           dtype=np.float32)  # region.rect_points[0] is left bottom point and points going clockwise!
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(region.landmarks[:self.nb_kps, :2], axis=0)
            lm_xy = np.squeeze(cv2.transform(lm_xy, mat))
            # A segment of length 1 in the coordinates system of body bounding box takes region.rect_w_a pixels in the
            # original image. Then we arbitrarily divide by 4 for a more realistic appearance.
            lm_z = region.landmarks[:self.nb_kps, 2:3] * region.rect_w_a / 4
            lm_xyz = np.hstack((lm_xy, lm_z))
            #if self.smoothing:
                #lm_xyz = self.filter.apply(lm_xyz)
            region.landmarks_padded = lm_xyz.astype(np.int)
            region.landmarks_abs = region.landmarks_padded.copy()
            return self.recognize_gesture(region)

    def recognize_gesture(self, r):

        def angle_with_y(v):
            # v: 2d vector (x,y)
            # Returns angle in degree ofv with y-axis of image plane
            if v[1] == 0:
                return 90
            angle = atan2(v[0], v[1])
            return np.degrees(angle)

        # For the demo, we want to recognize the flag semaphore alphabet
        # For this task, we just need to measure the angles of both arms with vertical
        right_arm_angle = angle_with_y(r.landmarks_abs[14, :2] - r.landmarks_abs[12, :2])
        left_arm_angle = angle_with_y(r.landmarks_abs[13, :2] - r.landmarks_abs[11, :2])
        right_pose = int((right_arm_angle + 202.5) / 45)
        left_pose = int((left_arm_angle + 202.5) / 45)
        r.gesture = self.semaphore_flag.get((right_pose, left_pose), None)
        return r.gesture
