import logging
import numpy as np
from sahi.models.base import DetectionModel
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.prediction import ObjectPrediction

logger = logging.getLogger(__name__)


class V0SimOTADetectionModel(DetectionModel):
    def load_model(self):
        from V0_SimOTA.models import build_model
        from V0_SimOTA.train import parse_args
        from V0_SimOTA.config import build_config
        from V0_SimOTA.utils.misc import build_dataset
        args = parse_args()
        cfg = build_config(args)
        _, dataset_info, _ = build_dataset(cfg, args, self.device, is_train=True)
        num_classes = dataset_info[0]
        model = build_model(args=args,
                            cfg=cfg,
                            device=self.device,
                            num_classes=num_classes,
                            trainable=True)
        # update model image size
        if self.image_size is not None:
            cfg['test_size'] = self.image_size
        self.set_model(model)

    def set_model(self, model):
        self.model = model
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping

    def perform_inference(self, image: np.ndarray):

        self.model.trainable = False
        prediction_result = self.model(image)
        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        return self.model.num_classes

    @property
    def category_names(self):
        from V0_SimOTA.dataset.coco import coco_class_labels

        return coco_class_labels

    def _create_object_prediction_list_from_original_predictions(self, shift_amount_list=[[0, 0]], full_shape_list=None):
        original_predictions = self._original_predictions
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)
        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, image_predictions in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]

            object_prediction_list = []

            # process predictions
            for prediction in image_predictions:
                bbox = prediction[0]
                score = prediction[1]
                category_id = int(prediction[2])
                category_name = self.category_mapping[str(category_id)]

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image
