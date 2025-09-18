import json
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import PIL.Image
import PIL.ImageOps

_CITATION = """\
@InProceedings{huggingface:dataset,
title = {facial-emotion-recognition-dataset},
author = {TrainingDataPro},
year = {2023}
}
"""

_DESCRIPTION = """\
The dataset consists of images capturing people displaying 7 distinct emotions
(anger, contempt, disgust, fear, happiness, sadness and surprise).
Each image in the dataset represents one of these specific emotions,
enabling researchers and machine learning practitioners to study and develop
models for emotion recognition and analysis.
The images encompass a diverse range of individuals, including different
genders, ethnicities, and age groups*. The dataset aims to provide
a comprehensive representation of human emotions, allowing for a wide range of
use cases.
"""
_NAME = 'facial-emotion-recognition-dataset'

_HOMEPAGE = f"https://huggingface.co/datasets/TrainingDataPro/{_NAME}"

_LICENSE = "cc-by-nc-nd-4.0"

_DATA = f"https://huggingface.co/datasets/TrainingDataPro/{_NAME}/resolve/main/data/"


class FacialEmotionRecognitionDataset(datasets.GeneratorBasedBuilder):

    def _info(self):
        return datasets.DatasetInfo(description=_DESCRIPTION,
                                    features=datasets.Features({
                                        'set_id': datasets.Value('int32'),
                                        'neutral': datasets.Image(),
                                        'anger': datasets.Image(),
                                        'contempt': datasets.Image(),
                                        'disgust': datasets.Image(),
                                        "fear": datasets.Image(),
                                        "happy": datasets.Image(),
                                        "sad": datasets.Image(),
                                        "surprised": datasets.Image(),
                                        "age": datasets.Value('int8'),
                                        "gender": datasets.Value('string'),
                                        "country": datasets.Value('string')
                                    }),
                                    supervised_keys=None,
                                    homepage=_HOMEPAGE,
                                    citation=_CITATION,
                                    license=_LICENSE)

    def _split_generators(self, dl_manager):
        images = dl_manager.download_and_extract(f"{_DATA}images.zip")
        annotations = dl_manager.download(f"{_DATA}{_NAME}.csv")
        images = dl_manager.iter_files(images)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={
                                        "images": images,
                                        'annotations': annotations
                                    }),
        ]

    def _generate_examples(self, images, annotations):
        annotations_df = pd.read_csv(annotations, sep=';')

        images = sorted(images)
        images = [images[i:i + 8] for i in range(0, len(images), 8)]

        for idx, images_set in enumerate(images):
            set_id = int(images_set[0].split('/')[2])
            data = {'set_id': set_id}

            for file in images_set:
                if 'neutral' in file.lower():
                    data['neutral'] = file
                elif 'anger' in file.lower():
                    data['anger'] = file
                elif 'contempt' in file.lower():
                    data['contempt'] = file
                elif 'disgust' in file.lower():
                    data['disgust'] = file
                elif 'fear' in file.lower():
                    data['fear'] = file
                elif 'happy' in file.lower():
                    data['happy'] = file
                elif 'sad' in file.lower():
                    data['sad'] = file
                elif 'surprised' in file.lower():
                    data['surprised'] = file

            data['age'] = annotations_df.loc[annotations_df['set_id'] ==
                                             set_id]['age'].values[0]
            data['gender'] = annotations_df.loc[annotations_df['set_id'] ==
                                                set_id]['gender'].values[0]
            data['country'] = annotations_df.loc[annotations_df['set_id'] ==
                                                 set_id]['country'].values[0]

            yield idx, data
