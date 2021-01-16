import json
import os
import warnings

import cv2
import numpy as np
import torch
from tqdm import tqdm

from detector_models import ResNet50_FasterRCNN
from src import paths

warnings.filterwarnings("ignore", category=UserWarning)


class Detector:

    def __init__(self, network, weights_file_path, device):

        self.network = network
        if os.path.isfile(weights_file_path):
            try:
                self.network.set_state_dict(torch.load(weights_file_path))
            except:
                print('Quitting: Cannot load weights to the network')
                return
        else:
            print('Quitting: Network weight\'s file does not exists')
            return

        if torch.cuda.is_available() and device == 'cuda':
            self.device = device
            torch.cuda.init()
        else:
            self.device = 'cpu'

        self.chunk_size = 50
        self.probability_threshold = 0.95

        self.output_folder_path = os.path.join(paths.data_folder_path, 'detected objects')
        if not os.path.isdir(self.output_folder_path):
            os.makedirs(self.output_folder_path)

    def detect_objects(self, video_file_path):
        if not os.path.exists(video_file_path):
            print('Quitting: Video path does not exits')
            return

        video = cv2.VideoCapture(video_file_path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        self.network.eval()
        self.network.to(self.device)

        annotation_file_name = os.path.basename(video_file_path).replace('.mp4', '.json')

        annotations = []

        print('Searching Objects ...')
        progress_bar = tqdm(total=num_frames, unit=' frames')
        for frame_idx in range(num_frames):
            _, frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255

            predicted_annotations = network.predict_batch(
                torch.unsqueeze(
                    torch.as_tensor(np.transpose(frame, [2, 0, 1]), dtype=torch.float32),
                    dim=0
                ).to(self.device)
            )[0]

            predicted_annotations['labels'] = predicted_annotations['labels'].to('cpu').detach().numpy()
            predicted_annotations['boxes'] = predicted_annotations['boxes'].to('cpu').detach().numpy()
            predicted_annotations['scores'] = predicted_annotations['scores'].to('cpu').detach().numpy()

            sorted_indices = np.argsort(predicted_annotations['labels'])
            predicted_annotations['labels'] = predicted_annotations['labels'][sorted_indices]
            predicted_annotations['boxes'] = predicted_annotations['boxes'][sorted_indices]
            predicted_annotations['scores'] = predicted_annotations['scores'][sorted_indices]

            frame_annotations = []
            for i in range(len(predicted_annotations['scores'])):
                if predicted_annotations['scores'][i] > self.probability_threshold:
                    frame_annotations.append(
                        {
                            'label': int(predicted_annotations['labels'][i]),
                            'bbox': predicted_annotations['boxes'][i].tolist(),
                            'coordinates': (
                                float((predicted_annotations['boxes'][i][0] + predicted_annotations['boxes'][i][2]) / 2),
                                float((predicted_annotations['boxes'][i][1] + predicted_annotations['boxes'][i][3]) / 2)
                            ),
                            'probability': float(predicted_annotations['scores'][i])
                        }
                    )

            annotations.append(
                {
                    'frame_idx': frame_idx,
                    'objects': frame_annotations
                }
            )

            progress_bar.update(1)
        progress_bar.close()

        with open(os.path.join(self.output_folder_path, annotation_file_name), 'w') as file:
            json.dump(annotations, file, indent=4)


network = ResNet50_FasterRCNN()
weights_file_path = os.path.join(
    paths.resrc_folder_path,
    'trained models',
    'ResNet50_FasterRCNN',
    'primary.pt'
)
device = 'cuda'

detector = Detector(network, weights_file_path, device)

video_file_path = os.path.join(
    paths.data_folder_path,
    'pre-processed',
    'warped',
    '77_bonus.mp4'
)

detector.detect_objects(video_file_path)
