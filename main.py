import cv2
from threading import Thread
import time
from det_utils import *
from my_utils import *
from my_utils.sort import *


mot_tracker = Sort(max_age=20,min_hits=3)

class CameraStream(object):
    def __init__(self,streamLink):
        self.streamLink = streamLink
        self.capOpened = False
        self.init_capture()
        _,self.frame = self.cap.read()

        self.thread_cap = Thread(target = self.start_input, daemon=True)
        self.thread_cap.start()


    def init_capture(self):
        self.cap = cv2.VideoCapture(self.streamLink)
        self.capOpened = self.cap.isOpened()
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.fps = max(self.cap.get(cv2.CAP_PROP_FPS)%100, 0) or 30.0
        self.frames = max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')

        cap_width  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        cap_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
    def start_input(self):
	    while True:
		    if self.capOpened is False:
			    self.init_capture()
		    else:
			    self.update_frame()

    def update_frame(self, offset=0):
	    while self.capOpened:
		    ret, frame = self.cap.read()
		    if ret is False:
			    self.capOpened = False
			    break
		    self.frame = frame
		    time.sleep(offset + 1 / self.fps)

    def read(self):
            frame = self.frame.copy()   
            return frame
    
    def frame_release(self):
        self.cap.release()

def filter_out_repeat(objects):
    objects = sorted(objects,key=lambda x: x[1])
    l = len(objects)
    new_objects = []

    if l > 1:
        for i in range(l-1):
            flag = 0
            for j in range(i+1,l):
                x_i, y_i, w_i, h_i = objects[i][3]
                x_j, y_j, w_j, h_j = objects[j][3]
                box1 = [int(x_i - w_i / 2), int(y_i - h_i / 2), int(x_i + w_i / 2), int(y_i + h_i / 2)]
                box2 = [int(x_j - w_j / 2), int(y_j - h_j / 2), int(x_j + w_j / 2), int(y_j + h_j / 2)]
                if cal_iou(box1,box2) >= 0.7:
                    flag = 1
                    break
            #if no repeat
            if not flag:
                new_objects.append(objects[i])
        #add the last one
        new_objects.append(objects[-1])
    else:
        return objects

    return list(tuple(new_objects))

def cal_iou(box1,box2):
    x1 = max(box1[0],box2[0])
    y1 = max(box1[1],box2[1])
    x2 = min(box1[2],box2[2])
    y2 = min(box1[3],box2[3])
    i = max(0,(x2-x1))*max(0,(y2-y1))
    u = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) -  i
    iou = float(i)/float(u)
    return iou

def get_objName(item,objects):
    iou_list = []
    for i,object in enumerate(objects):
        x, y, w, h = object[3]
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        iou_list.append(cal_iou(item[:4],[x1,y1,x2,y2]))
    max_index = iou_list.index(max(iou_list))
    return objects[max_index][0]

if __name__ == "__main__":
    # cam = CameraStream("video2.avi")
    cam = CameraStream("highway2.mp4")
    yolo_dir_path = r'yolov5-v6'
    model_path  = r'model\weights/best.pt'
    det = Detection(yolo_dir_path,model_path)
    history = {}
    removed_id_list = []

    while True:
        try:
            frame = cam.read()
            org_frame = frame.copy()
            objects = det(org_frame)

            objects = filter(lambda x: x[0], objects)
            objects = list(filter(lambda x: x[2] > 0,objects))

            # objects = filter_out_repeat(objects)
            detections = []
            for item in objects:
                detections.append([int(item[3][0] - item[3][2] / 2),
                                    int(item[3][1] - item[3][3] / 2),
                                    int(item[3][0] + item[3][2] / 2),
                                    int(item[3][1] + item[3][3] / 2),
                                    item[1]])
            if detections is not None:
                track_bbs_ids = mot_tracker.update(np.asarray(detections))
            history = {}
            if len(track_bbs_ids) > 0:
                    for bb in track_bbs_ids:    #add all bbox to history
                        id = int(bb[-1])
                        objectName = get_objName(bb, objects)
                        
                        # print("objectName",objectName)
                        
                        if id not in history.keys():  #add new id
                            history[id] = {}
                            history[id]["no_update_count"] = 0
                            history[id]["his"] = []
                            history[id]["his"].append(objectName)
                        else:
                            history[id]["no_update_count"] = 0
                            history[id]["his"].append(objectName)
                    # print("OCC",len(history))
            for i, item in enumerate(track_bbs_ids):
                    # n=random.randint(3,20)
                    bb = list(map(lambda x: int(x), item))
                    id = bb[-1]
                    x1, y1, x2, y2 = bb[:4]

                    his = history[id]["his"]
                    result = {}
                    for i in set(his):
                        result[i] = his.count(i)
                    res = sorted(result.items(), key=lambda d: d[1], reverse=True)
                    #print(res)
                    objectName = res[0][0]

                    cv2.rectangle(org_frame, (x1, y1), (x2, y2), (0,255,0), thickness=2)

                    cv2.putText(org_frame, str(objectName)+'_'+str(id), (x1 - 1, y1 - 3), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,0), thickness=1)
            # del_his = []
            # for j in history.keys():
            #     if history[j]["no_update_count"] > 5:
            #         del_his.append(j)
                
            #     if j not in track_bbs_ids[:,-1]:
            #             history[j]["no_update_count"]=history[j]["no_update_count"]+1
            
            # for del_id in del_his:
            #     del history[del_id]

            cv2.imshow("Out",org_frame)
            if cv2.waitKey(1) & 0XFF==ord('q'):
                break
            

            
        except Exception as e:
            print("Error",e)
    cam.frame_release()
    cv2.destroyAllWindows()
