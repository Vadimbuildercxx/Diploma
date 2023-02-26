import functools
import os

import torch
import time

import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from embedder import Embedder
from enum import Enum
from PIL import Image

backends = [
  'opencv',
  'ssd',
  'dlib',
  'mtcnn',
  'retinaface',
  'mediapipe'
]

models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]
metrics = ["cosine", "euclidean", "euclidean_l2"]

class TakeMod(Enum):
    MEAN = 0
    MIN = 1


class CompareMod(Enum):
    DEFAULT = 0
    COS = 1


def timer(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        runtime = time.perf_counter() - start
        print(f"{func.__name__} took {runtime:.4f} secs")
        return result

    return _wrapper


class FaceComparator(object):
    def __init__(self, device: str = "cuda", dataset_path: str = None,
                 model_name: str = models[0]):
        self.device = torch.device(device)
        self.__dataset_path: str = dataset_path
        self.model_name = model_name
        self.__embedder = Embedder([112, 112], R"weights/backbone_ir50_ms1m_epoch120.pth")

        self.__embeddings_df: pd.DataFrame = self.__load_all_embeddings()
        self.__embeddings_tensor = torch.stack(self.__embeddings_df["embedding"].tolist()).squeeze()

    @timer
    def __load_all_embeddings(self, check_existance: bool = True) -> pd.DataFrame:
        pickle_representations_path = self.__dataset_path + "/" + f"representations_{self.model_name}.pkl"
        pickle_representations_path = pickle_representations_path.replace("-", "_").lower()
        if os.path.exists(pickle_representations_path) and check_existance:
            _embeddings_df = pd.read_pickle(pickle_representations_path)
        else:
            _embeddings_df = self.convert_imgs_to_pickle_format()

        return _embeddings_df

    def convert_imgs_to_pickle_format(self) -> pd.DataFrame:
        file_name = f"representations_{self.model_name}.pkl"
        file_name = file_name.replace("-", "_").lower()

        image_paths = []
        for r, _, f in os.walk(self.__dataset_path):
            for file in f:
                if ((".jpg" in file.lower())
                        or (".jpeg" in file.lower())
                        or (".png" in file.lower())):
                    exact_path = r + "/" + file
                    image_paths.append(exact_path)
        representations = []
        for i_path in image_paths:
            image = Image.open(i_path).convert('RGB')
            row = [i_path, self.extract_embedding(image)]
            representations.append(row)
        embeddings_df = pd.DataFrame(data=representations, columns=["path", "embedding"])

        embeddings_df.to_pickle(f"{self.__dataset_path}/{file_name}")
        return embeddings_df

    def get_embedding_table(self):
        return self.__embeddings_df

    def update_embeddings(self):
        self.__embeddings_df = self.__load_all_embeddings(check_existance=False)

    def find_face(self, face, take_option: TakeMod = TakeMod.MIN,
                  threshold: float = 0.6, encoded: bool = False) -> tuple[float, str]:

        if encoded:
            enc_face = face.copy()
        else:
            enc_face = self.extract_embedding(face)

        b = enc_face.unsqueeze(-2).repeat(self.__embeddings_tensor.shape[0], 1)
        _similarity_matrix = self.cosine_sim(self.__embeddings_tensor, b, dim=1)
        max_score_indx = _similarity_matrix.argmax().to(torch.int32).item()

        return _similarity_matrix[max_score_indx], self.__embeddings_df["path"][max_score_indx]

    def compare_two_faces(self, face_a, face_b: np.ndarray,
                          compare_option: CompareMod = CompareMod.DEFAULT,
                          encoded: bool = False):
        assert isinstance(face_a, list)

        if encoded:
            enc_a = face_a.copy()
            enc_b = face_b.copy()
        else:
            if isinstance(face_a, list):
                enc_a = [self.enc_face(_face) for _face in face_a]
            else:
                enc_a = self.enc_face(face_b)
            enc_b = self.enc_face(face_b)

        if compare_option == CompareMod.DEFAULT:
            if isinstance(face_a, list):
                print(len(enc_a), enc_a[0].shape, enc_b.shape)
                return None # face_recognition.face_distance(enc_a, enc_b)
            else:
                return None #face_recognition.face_distance([enc_a], enc_b)

        elif compare_option == CompareMod.COS:
            return self.cosine_sim(enc_a, enc_b)
        return None

    def extract_embedding(self, image) -> list:
        """
        This function get representations from embedder module.
        :param image_path: path to face image (must be already aligned)
        :return list:  returns image emdedding
        """
        emb = self.__embedder.get_embedding(image)
        return emb

    @staticmethod
    def cosine_sim(vector_a, vector_b, dim=1) -> torch.Tensor:
        cos = torch.nn.CosineSimilarity(dim=dim, eps=1e-6)
        return cos(vector_a, vector_b)


if __name__ == '__main__':

    comp = FaceComparator(dataset_path="person_faces")
    image = Image.open(R"C:\Users\vadim\AI\YOLOV\testData\test_images\bateman\Patrick_Bateman_face.jpg")
    image = Image.open(R"C:\Users\vadim\AI\YOLOV\testData\test_images\bateman\bateman_angry_mirrored.jpg")
    print(comp.find_face(image))


    print()

    # start = time.time()
    # result = DeepFace.verify(img1_path="img1.jpg",
    #                          img2_path="img2.jpg",
    #                          distance_metric=metrics[0]
    #                          )
    # print("Finished at: ", time.time() - start)
    # print(resul))
    # start = time.time()
    # df = DeepFace.find(img_path=R"C:\Users\vadim\AI\YOLOV\testData\test_images\bateman\Patrick_Bateman_face.jpg",
    #                    db_path="person_faces",  enforce_detection=False, model_name=models[0],
    #                    detector_backend="dlib", distance_metric=metrics[0])
    #
    # print("Finished at: ", time.time() - start)
    # print(df, df[0].iloc[0])
    # face_comp = FaceComparator(dataset_path="person_faces")
    # image = face_comp.read_image(R"C:\Users\vadim\AI\YOLOV\testData\test_images\bateman\Patrick_Bateman_face.jpg")
    # print(face_comp.find_face(image))
    # image = face_comp.read_image(R"C:\Users\vadim\AI\YOLOV\testData\test_images\bateman\bateman_2_face\bateman_nervous.jpg")
    # image_1 = face_comp.read_image(R"C:\Users\vadim\AI\YOLOV\testData\test_images\bateman\Patrick_Bateman_face.jpg")
    # print(face_comp.find_face(image_1))

    # df = pd.DataFrame(data=matrix)
    # sns.heatmap(df, annot=True, xticklabels=columns, yticklabels=columns)
    # plt.show()

