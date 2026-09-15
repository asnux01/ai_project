# 라이브러리
import torch

from pycocotools.cocoeval import COCOeval


class COCOEvaluator:
    
    def __init__(
        self,
        coco_gt,
        class_index_to_category_id,
        image_size=640,
        max_detections=100
    ):
        
        # COCO Ground Truth 저장
        self.coco_gt = coco_gt
        
        # 학습 Class Index를 원래 COCO Category ID로 변환하는 Mapping 저장
        self.class_index_to_category_id = dict(
            class_index_to_category_id
        )
        
        # Validation 설정 저장
        self.image_size = int(image_size)
        self.max_detections = int(max_detections)
        
        # 평가 결과 저장 공간 초기화
        self.reset()
        
    
    def reset(self):
        
        # COCO 형식 Prediction 저장
        self.results = []
        
        # 실제 평가에 사용된 Image ID 저장
        self.image_ids = []
        
    
    def _restore_boxes_to_original(
        self,
        boxes,
        original_shape
    ):
        
         # Prediction Bbox를 원본과 분리해 CPU Tensor로 복사
        boxes = (
            boxes.detach()
            .cpu()
            .to(torch.float32)
            .clone()
        )
        
        # Prediction이 없는 경우 바로 변환
        if boxes.numel() == 0:
            return boxes
        
        # 원본 이미지 높이와 너비
        original_height, original_width = original_shape
        
        # Validation LetterBox에서 사용한 Resize 비율 재계산
        scale = min(
            self.image_size / original_width,
            self.image_size / original_height
        )
        
        # Validation LetterBox에서 사용한 Resize 크기 재계산
        resized_width = round(original_width * scale)
        resized_height = round(original_height * scale)
        
        # Validation LetterBox에서 사용한 Padding 재계산
        pad_left = (self.image_size - resized_width) // 2
        pad_top = (self.image_size - resized_height) // 2
        
        # Padding을 제거해 Resize된 이미지 좌표로 복원
        boxes[:, [0, 2]] -= pad_left
        boxes[:, [1, 3]] -= pad_top
        
        # Resize 비율을 제거해 원본 이미지 좌표로 복원
        boxes[:, [0, 2]] /= scale
        boxes[:, [1, 3]] /= scale
        
        # 원본 이미지 범위를 벗어난 Bbox 제한
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(
            0, original_width
        )
        
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(
            0, original_height
        )
        
        return boxes
    
    
    def update(
        self,
        predictions,
        batch
    ):
        
        # Batch의 Image ID와 원본 이미지 크기 가져오기
        image_ids = batch["image_id"]
        original_shapes = batch["ori_shape"]
        
        # Prediction의 수와 Batch Image 수 검사
        if len(predictions) != len(image_ids):
            raise ValueError(
                "predictions와 image_id의 개수가 같아야 합니다."
            )
        
        # Image 단위 Prediction을 COCO 결과 형식으로 변환
        for prediction, image_id, original_shape in zip(
            predictions,
            image_ids,
            original_shapes
        ):
            
            # 평가 Image ID 저장
            self.image_ids.append(int(image_id))
            
            # LetterBox 좌표를 원본 이미지 좌표로 복원
            boxes = self._restore_boxes_to_original(
                boxes=prediction["boxes"],
                original_shape=original_shape
            )
            
            # Score를 CPU Tensor로 변환
            scores = (
                prediction["scores"]
                .detach()
                .cpu()
                .to(torch.float32)
            )
            
            # Class Index를 CPU Tensor로 변환
            labels = (
                prediction["labels"]
                .detach()
                .cpu()
                .long()
            )
            
            # Detection이 없는 경우 다음 이미지로 이동
            if boxes.numel() == 0:
                continue
            
            # Detection 단위 COCO 결과 생성
            for box, score, label in zip(
                boxes, scores, labels
            ):
                
                # xyxy 좌표 가져오기
                x1, y1, x2, y2 = box.tolist()
                
                # COCO가 사용하는 xywh 형식으로 변환
                width = max(x2 - x1, 0.0)
                height = max(y2 - y1, 0.0)
                
                # 비정상 Bbox 제외
                if width <= 0.0 or height <= 0.0:
                    continue
                
                # Class Index를 원래 COCO Category ID로 복원
                category_id = (
                    self.class_index_to_category_id[
                        int(label.item())
                    ]
                )
                
                # COCO Detection Result 저장
                self.results.append({
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [x1, y1, width, height],
                    "score": float(score.item())
                })
    
    
    def compute(self):
        
        # Prediction이 하나도 없는 경우 0 Metric 반환
        if len(self.results) == 0:
            return {
                "map50_95": 0.0,
                "map50": 0.0,
                "map75": 0.0,
                "mar": 0.0
            }
        
        # Prediction을 COCO Detectin Result 객체로 변환
        coco_dt = self.coco_gt.loadRes(self.results)
        
        # COCO Bbox Evaluator 생성
        coco_eval = COCOeval(
            cocoGt=self.coco_gt,
            cocoDt=coco_dt,
            iouType="bbox"
        )
        
        # 실제 Validation에서 사용한 Image만 평가
        coco_eval.params.imgIds = sorted(
            set(self.image_ids)
        )
        
        # 현재 Postprocessor와 동일한 최대 Detection 수 사용
        coco_eval.params.maxDets = [1, 10, self.max_detections]
        
        # COCO 평가 실행
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # COCOeval 통계값 반환
        return{
            "map50_95": float(
                coco_eval.stats[0]
            ),
            "map50": float(
                coco_eval.stats[1]
            ),
            "map75": float(
                coco_eval.stats[2]
            ),
            "mar": float(
                coco_eval.stats[8]
            )
        }