import numpy as np

import tracker
from detector import Detector
import cv2

if __name__ == '__main__':

    # 蓝黄：in
    # 黄蓝：out

    video_file = 'video/test.avi'
    # video_file = 'video/PETS2009.avi'

    video = cv2.VideoCapture(video_file)
    # 分辨率-宽度
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 分辨率-高度
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 检测时的宽度
    detection_width = int(int(width)*0.5)
    # 检测时的高度
    detection_height = int(int(height)*0.5)
    # 根据视频尺寸，填充一个polygon，供撞线计算使用
    mask_image_temp = np.zeros((height, width), dtype=np.uint8)

    # 获取图片第一帧并得到绘制矩形的坐标
    success,frame = video.read()
    img = frame
    list_polygon_bule = []
    list_polygon_yellow = []
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            print(x,y)
            list_polygon_bule.append([x,y])
            cv2.circle(img, (x, y), 2, (255, 0, 0))
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (255,0,0))
            cv2.imshow("image", img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            print(x,y)
            list_polygon_yellow.append([x,y])
            cv2.circle(img, (x, y), 2, (255, 255, 0))
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 255, 255))
            cv2.imshow("image", img)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    while(1):
        cv2.imshow("image", img)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
    print("蓝色polygon坐标：" + str(list_polygon_bule))
    print("黄色polygon坐标：" + str(list_polygon_yellow))
    cv2.destroyAllWindows()

    list_pts_blue = list_polygon_bule
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # 填充第二个polygon
    mask_image_temp = np.zeros((height, width), dtype=np.uint8)
    list_pts_yellow = list_polygon_yellow
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # 缩小尺寸，1920x1080->detection_widthxdetection_height
    polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (detection_width, detection_height))

    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

    # 黄 色盘
    yellow_color_plate = [0, 255, 255]
    # 黄 polygon图片
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image
    # 缩小尺寸，1920x1080->detection_widthxdetection_height
    color_polygons_image = cv2.resize(color_polygons_image, (detection_width, detection_height))

    # list 与蓝色polygon重叠
    list_overlapping_blue_polygon = []

    # list 与黄色polygon重叠
    list_overlapping_yellow_polygon = []

    # 进入数量
    in_count = 0
    # 离开数量
    out_count = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(detection_width * 0.01), int(detection_height * 0.05))

    # 初始化 yolov5
    detector = Detector()

    # 打开视频
    capture = cv2.VideoCapture(video_file)
    # capture = cv2.VideoCapture('TownCentreXVID.avi')

    while True:
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            break

        # 缩小尺寸，1920x1080->detection_widthxdetection_height
        im = cv2.resize(im, (detection_width, detection_height))

        list_bboxs = []
        bboxes = detector.detect(im)

        # 如果画面中 有bbox
        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)

            # 画框
            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)
            pass
        else:
            # 如果画面中 没有bbox
            output_image_frame = im
        pass

        # 输出图片
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)

        if len(list_bboxs) > 0:
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id = item_bbox

                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                y1_offset = int(y1 + ((y2 - y1) * 0.6))

                # 撞线的点
                y = y1_offset
                x = x1

                if polygon_mask_blue_and_yellow[y, x] == 1:
                    # 如果撞 蓝polygon
                    if track_id not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id)
                    pass

                    # 判断 黄polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 外出方向
                    if track_id in list_overlapping_yellow_polygon:
                        # 外出+1
                        out_count += 1

                        print(
                            f'类别: {label} | id: {track_id} | out撞线 | out撞线总数: {out_count} | out id列表: {list_overlapping_yellow_polygon}')

                        # 删除 黄polygon list 中的此id
                        list_overlapping_yellow_polygon.remove(track_id)

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass

                elif polygon_mask_blue_and_yellow[y, x] == 2:
                    # 如果撞 黄polygon
                    if track_id not in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.append(track_id)
                    pass

                    # 判断 蓝polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 进入方向
                    if track_id in list_overlapping_blue_polygon:
                        # 进入+1
                        in_count += 1

                        print(
                            f'类别: {label} | id: {track_id} | in撞线 | in撞线总数: {in_count} | in id列表: {list_overlapping_blue_polygon}')

                        # 删除 蓝polygon list 中的此id
                        list_overlapping_blue_polygon.remove(track_id)

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass
                    pass
                else:
                    pass
                pass

            pass

            # ----------------------清除无用id----------------------
            list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
            for id1 in list_overlapping_all:
                is_found = False
                for _, _, _, _, _, bbox_id in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                        break
                    pass
                pass

                if not is_found:
                    # 如果没找到，删除id
                    if id1 in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.remove(id1)
                    pass
                pass
            list_overlapping_all.clear()
            pass

            # 清空list
            list_bboxs.clear()

            pass
        else:
            # 如果图像中没有任何的bbox，则清空list
            list_overlapping_blue_polygon.clear()
            list_overlapping_yellow_polygon.clear()
            pass
        pass

        # text_draw = 'in: ' + str(in_count) + \
        #             ' , out: ' + str(out_count)
        text_draw = ''
        output_image_frame = cv2.putText(img=output_image_frame, 
                                         text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=1, color=(0, 255, 0), thickness=2)

        cv2.imshow('demo', output_image_frame)
        cv2.waitKey(1)

        pass
    pass

    capture.release()
    cv2.destroyAllWindows()
