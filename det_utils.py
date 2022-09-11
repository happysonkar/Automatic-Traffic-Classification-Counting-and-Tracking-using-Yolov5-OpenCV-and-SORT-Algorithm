import torch







class Detection():
    def __init__(self,yolo_dir_path,model_path):
        self.yolo_dir_path = yolo_dir_path
        self.model_path = model_path
        self.indx_to_cls_name = {k:v for k,v in enumerate(["Three_Wheeler","bus","car","lcv","Two_Wheeler","multiaxle","tractor","truck","tractor_with_trailor"])}
        self.load_model()

    def __call__(self,frame):
        dets = self.model(frame).xywh
        dets = dets[0].cpu().numpy()
        objects = []
        for i, det in enumerate(dets):
            item = []
            item.append(self.indx_to_cls_name[int(det[-1])])
            item.append(int(det[-1]))
            item.append(float(det[4]))
            x,y,w,h = det[0:4]
            item.append([x, y, w, h])
            objects.append(item)
        return objects

    def load_model(self):
        self.model = torch.hub.load(self.yolo_dir_path, 'custom', path=self.model_path, source='local', force_reload=True)
        print('Model has been loaded successfully!')


