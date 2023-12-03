import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _test_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.test_generator = test_datagenerator.flow_from_directory(
            directory=self.config.test_data,
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        model = self.load_model(self.config.path_of_model)
        self._test_generator()  # Use the modified method
        self.score = model.evaluate(self.test_generator, return_dict =True)

    def save_score(self, filename="scores.json", directory="."):
        if self.score is not None:
            # scores = {"loss": self.score[0], "accuracy": self.score[1]}
            scores = self.score
            save_json(path=Path(directory) / filename, data=scores)

    

    