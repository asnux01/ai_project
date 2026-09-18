# 라이브러리
import torch

from torchvision.ops import batched_nms

class DetectionPostprocessor:
    
    def __init__(
        self,
        num_classes=80,
        reg_max=16,
        strides=(8, 16, 32),
        confidence_threshold=0.001,
        nms_iou_threshold=0.7,
        max_detections=300
    ):
        
        # 입력 검사
        if num_classes <= 0:
            raise ValueError("num_classes는 0보다 커야 합니다.")
        
        if reg_max <= 0:
            raise ValueError("reg_max는 0보다 커야 합니다.")
        
        if len(strides) == 0:
            raise ValueError("strides는 비어 있을 수 없습니다.")
        
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold는 0과 1 사이여야 합니다.")
        
        if not 0.0 <= nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold는 0과 1 사이여야 합니다.")
        
        if max_detections <= 0:
            raise ValueError("max_detections는 0보다 커야 합니다.")
        
        # Detection 설정 저장
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = tuple(strides)
        
        # Postprocess 설정 저장
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections
        
        # Detect Head 출력 채널 수 계산
        self.num_outputs = self.num_classes + 4 * self.reg_max
        
    
    def __call__(self, predictions):
        
        # Prediction 형태 정리
        pred_dist, pred_scores = self._flatten_predictions(predictions)
        
        # Anchor Point와 Stride 생성
        anchor_points, stride_tensor = self._make_anchors(predictions)
        
        # Distribution을 Bbox로 Decode
        pred_bboxes_grid = self._decode_bboxes(
            pred_dist=pred_dist,
            anchor_points=anchor_points
        )
        
        # Grid 좌표를 실제 입력 이미지 좌표로 변환
        pred_bboxes = pred_bboxes_grid * stride_tensor
        
        # Class Logit을 확률로 변환
        pred_scores = pred_scores.sigmoid()
        
        # 모델 입력 이미지 크기 계산
        image_height, image_width = self._infer_image_shape(predictions)
        
        # Bbox를 이미지 범위 내부로 제한
        pred_bboxes = self._clip_boxes(
            boxes=pred_bboxes,
            image_height=image_height,
            image_width=image_width
        )
        
        # Image 단위 Postprocess 결과 저장
        results = []
        
        # Batch Image 순회
        for image_index in range(pred_bboxes.shape[0]):
            
            # 현재 Image의 Bbox Prediction
            boxes = pred_bboxes[image_index]
            
            # 현재 Image의 Class Score
            class_scores = pred_scores[image_index]
            
            # Detection Candidate 선택
            candidate_indices = torch.nonzero(
                class_scores
                > self.confidence_threshold,
                as_tuple=False
            )
            
            # Detection이 없는 경우
            if candidate_indices.numel() == 0:
                
                results.append(
                    self._empty_result(
                        device=pred_bboxes.device,
                        dtype=pred_bboxes.dtype
                    )
                )
                
                continue
            
            # Anchor Index
            anchor_indices = candidate_indices[:, 0]
            
            # Class Label
            labels = candidate_indices[:, 1]
            
            # Bbox 선택
            boxes = boxes[anchor_indices]
            
            # Score 선택
            scores = class_scores[anchor_points, labels]
            
            # NMS Candidate 제한
            max_nms = 30000
            
            if scores.numel() > max_nms:
                
                top_indices = torch.argsort(
                    scores,
                    descending=True
                )[:max_nms]
                
                boxes = boxes[top_indices]
                scores = scores[top_indices]
                labels = labels[top_indices]
                
            # 유효한 크기의 Bbox만 유지
            if boxes.numel() > 0:
                
                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]
                valid_mask = (widths > 0) & (heights > 0)
                boxes = boxes[valid_mask]
                scores = scores[valid_mask]
                labels = labels[valid_mask]
                
            # Detection이 없는 경우
            if boxes.numel() == 0:
                
                results.append(self._empty_result(
                    device=pred_bboxes.device,
                    dtype=pred_bboxes.dtype
                ))
                
                continue
            
            # Class-aware NMS 수행
            keep_indices = batched_nms(
                boxes=boxes,
                scores=scores,
                idxs=labels,
                iou_threshold=self.nms_iou_threshold
            )
            
            # 최대 Detection 수 제한
            keep_indices = keep_indices[:self.max_detections]
            
            # 최종 Prediction 저장
            results.append({
                "boxes": boxes[keep_indices],
                "scores": scores[keep_indices],
                "labels": labels[keep_indices]
            })
            
        return results
    
    
    def _flatten_predictions(self, predictions):
        
        # Prediction Level 수 검사
        if len(predictions) != len(self.strides):
            raise ValueError("Prediction level 수와 strides 수가 같아야 합니다.")
        
        # Feature Map Prediction 저장
        flattened_predictions = []
        
        # Feature Level 순회
        for prediction in predictions:
            
            # Prediction 차원 검사
            if prediction.ndim != 4:
                raise ValueError("각 prediction은 [B, C, H, W] 형태여야 합니다.")
            
            # 출력 채널 수 검사
            if prediction.shape[1] != self.num_outputs:
                raise ValueError(
                    "Detect Head 출력 채널 수가 예상값과 다릅니다. "
                    f"expected={self.num_outputs}, "
                    f"actual={prediction.shape[1]}"
                )
                
            # Batch 크기 가져오기
            batch_size = prediction.shape[0]
            
            # Prediction 형태 변환
            prediction = prediction.view(batch_size, self.num_outputs, -1)
            
            # 현재 Feature Level 저장
            flattened_predictions.append(prediction)
            
        # P3, P4, P5 Prediction 결합
        predictions = torch.cat(flattened_predictions, dim=2)
        
        # Bbox Distribution과 Class Logit 분리
        pred_dist, pred_scores = predictions.split(
            [4 * self.reg_max, self.num_classes],
            dim=1
        )
        
        # [B, C, N] -> [B, N, C]
        pred_dist = pred_dist.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        
        return pred_dist, pred_scores
    
    
    def _make_anchors(self, predictions):
        
        # Anchor Point 저장 공간 생성
        anchor_points = []
        
        # Stride 저장 공간 생성
        stride_values = []
        
        # Feature Level 순회
        for prediction, stride in zip(predictions, self.strides):
            
            # Feature Map 크기 가져오기
            height = prediction.shape[2]
            width = prediction.shape[3]
            
            # X 좌표 생성
            x_coordinates = (
                torch.arange(
                    width, 
                    device=prediction.device, 
                    dtype=prediction.dtype)
                + 0.5
            )
            
            # Y 좌표 생성
            y_coordinates = (
                torch.arange(
                    height, 
                    device=prediction.device, 
                    dtype=prediction.dtype)
                + 0.5
            )
            
            # Grid 좌표 생성
            grid_y, grid_x = torch.meshgrid(
                y_coordinates,
                x_coordinates,
                indexing="ij"
            )
            
            # Anchor Point 생성
            points = torch.stack(
                (grid_x, grid_y),
                dim=-1
            ).reshape(-1, 2)
            
            # 각 Anchor의 Stride 생성
            strides = torch.full(
                (height * width, 1),
                float(stride),
                device=prediction.device,
                dtype=prediction.dtype
            )
            
            # 현재 Feature Level Anchor 저장
            anchor_points.append(points)
            
            # 현재 Feature Level Stride 저장
            stride_values.append(strides)
            
        # 모든 Feature Map Anchor 결합
        anchor_points = torch.cat(anchor_points, dim=0)
        
        # 모든 Stride 결합
        stride_tensor = torch.cat(stride_values, dim=0)
        
        return anchor_points, stride_tensor
    
    
    def _decode_bboxes(
        self,
        pred_dist,
        anchor_points
    ):
        
        # Distribution 형태 변환
        pred_dist = pred_dist.view(
            pred_dist.shape[0],
            pred_dist.shape[1],
            4,
            self.reg_max
        )
        
        # 각 Bin을 확률로 변환
        pred_prob = torch.softmax(pred_dist, dim=-1)
        
        # DFL Bin 값 생성
        projection = torch.arange(
            self.reg_max,
            device=pred_prob.device,
            dtype=pred_prob.dtype
        )
        
        # Distribution의 기대값 계산
        distances = (pred_prob * projection).sum(dim=-1)
        
        # Left-Top 거리 가져오기
        left_top = distances[..., 0:2]
        
        # Right-Bottom 거리 가져오기
        right_bottom = distances[..., 2:4]
        
        # Bbox 왼쪽 위 좌표 계산
        x1y1 = anchor_points - left_top
        
        # Bbox 오른쪽 아래 좌표 계산
        x2y2 = anchor_points + right_bottom
        
        # xyxy Bbox 생성
        pred_bboxes = torch.cat((x1y1, x2y2), dim=-1)
        
        return pred_bboxes
    
    
    def _infer_image_shape(self, predictions):
        
        # 첫 번째 Feature Map 사용
        first_prediction = predictions[0]
        
        # 첫 번째 Feature Level Stride
        first_stride = self.strides[0]
        
        # 모델 입력 이미지 높이 계산
        image_height = first_prediction.shape[2] * first_stride
        
        # 모델 입력 이미지 너비 계산
        image_width = first_prediction.shape[3] * first_stride
        
        return image_height, image_width
    
    
    def _clip_boxes(
        self,
        boxes,
        image_height,
        image_width
    ):
        
        # 원본 Prediction Tensor 보호
        boxes = boxes.clone()
        
        # X1 좌표 제한
        boxes[..., 0] = boxes[..., 0].clamp(min=0, max=image_width)
        
        # Y1 좌표 제한
        boxes[..., 1] = boxes[..., 1].clamp(min=0, max=image_height)
        
        # X2 좌표 제한
        boxes[..., 2] = boxes[..., 2].clamp(min=0, max=image_width)
                
        # Y2 좌표 제한
        boxes[..., 3] = boxes[..., 3].clamp(min=0, max=image_height)
        
        return boxes
    
    
    def _empty_result(self, device, dtype):
        
        # 빈 Prediction 결과 생성
        return {
            "boxes": torch.empty(
                (0, 4),
                device=device,
                dtype=dtype
            ),
            "scores": torch.empty(
                (0,),
                device=device,
                dtype=dtype
            ),
            "labels": torch.empty(
                (0,),
                device=device,
                dtype=torch.long
            )
        }