import cv2
import numpy as np
import torch
import torch.utils.data
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
torch.cuda.empty_cache()

# mira christian el input de la imagen lo he hecho siguiendo esto
# si no te vale me dices y lo cambio
# https://gist.github.com/edumucelli/c2843ed1f6e13ed4706d63be87a0d671


class deteccion:
    def __init__(self, model_path):
        #nombres
        self.clases_nombre = ["bg", "kanguro", "koala"]

        # miramos si hay cpu o gpu
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        # cargamos el modelo
        self.modelo = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)

        # pilamos el head
        in_features = self.modelo.roi_heads.box_predictor.cls_score.in_features

        #creamos un nuevo clasificador con el numero de nuestras clases que son 2 mas el background
        self.modelo.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)

        # cargamos el modelo en el backend
        self.modelo.to(self.device)

        # le metemos nuestros pesos
        self.modelo.load_state_dict(torch.load(
            model_path, map_location=self.device))

        # lo ponemos en modo evaluacion
        self.modelo.eval()

    def predice(self, request):
        '''
        nos llega un request del que pillamos la data... ver ejemplo en link arriba
        '''
        print(request.data)
        nparr = np.fromstring(request.data, np.uint8)

        # numpy array a mat de opencv
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # opencv usar por defecto BGR asi que lo convertimos a RGB
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        # normalizamos entre 0-1
        image_rgb /= 255.0

        # reordenamos
        image_rgb = np.transpose(image_rgb, (2, 0, 1)).astype(np.float)

        # la convertimos en un tensor en la GPU
        # image_rgb = torch.tensor(image_rgb, dtype=torch.float).cuda()
        image_rgb = torch.tensor(image_rgb, dtype=torch.float).cpu()


        # agregamos una dimension
        image_rgb = torch.unsqueeze(image_rgb, 0)

        # la pasamos por el modelo
        outputs = self.modelo(image_rgb)

        # procesamos las predicciones pasandolas a cpu
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        # vamos a mandar todas la predicciones asi si pones luego en el frontend un slider
        # para poner un treshold podemos seleccionar live y filtra las predicciones
        # si hay predicciones
        predicciones = []
        if len(outputs[0]['boxes']) > 0:
            bboxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()

            # solo queremos aquellas que tengan un score mayor que nuestro threshold
            bboxes = bboxes.astype(np.int32)
            clases = [self.clases_nombre[i]
                      for i in outputs[0]['labels'].cpu().numpy()]
            pred_score = outputs[0]['scores'].detach().numpy()

            for i, box in enumerate(bboxes):
                prediccion = {
                    'x': int(box[0]),
                    'y': int(box[1]),
                    'width': int(int(box[2])-int(box[0])),
                    'height': int(int(box[3])-int(box[1])),
                    'score': int(pred_score[i]*100),
                    'clase': clases[i]
                }

                predicciones.append(prediccion)

        return predicciones
