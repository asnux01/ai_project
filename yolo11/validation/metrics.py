# 라이브러리
import torch


class DetectionMetrics:
    
    def __init__(
        self,
        num_classes=80,
        max_detections=100
    ):
        
        # 입력 검사
        if num_classes <= 0:
            raise ValueError("num_classes는 0보다 커야 합니다.")
        if max_detections <= 0:
            raise ValueError("max_detections는 0보다 커야 합니다.")
        
        # 파라미터
        self.num_classes = num_classes
        self.max_detections = max_detections
        
        # IoU 임계값
        self.iou_thresholds = torch.linspace(0.50, 0.95, 10)
        
        # Metric 저장 공간 초기화
        self.reset()
        
    
    def reset(self):
        
        # Prediction 저장
        self.predictions = []
        
        # Ground Truth 저장
        self.targets = []
        
    
    def update(
        self,
        predictions,
        targets
    ):
        
        # Batch 크기 검사
        if len(predictions) != len(targets):
            raise ValueError("predictions와 targets의 길이가 같아야 합니다.")
        
        # Image 단위 처리
        for prediction, target in zip(predictions, targets):
        
            # Prediction 확인
            pred_boxes = prediction["boxes"].detach().cpu()
            pred_scores = prediction["scores"].detach().cpu()
            pred_labels = prediction["labels"].detach().cpu().long()
            
            # Ground Truth 확인
            target_boxes = target["boxes"].detach().cpu()
            target_labels = target["labels"].detach().cpu().long()
            
            # Prediction Score 정렬
            if pred_scores.numel() > 0:
                order = torch.argsort(pred_scores, descending=True)
                
                # 최대 Detection 수 제한
                order = order[:self.max_detections]
                pred_boxes = pred_boxes[order]
                pred_scores = pred_scores[order]
                pred_labels = pred_labels[order]
                
            # Prediction 저장
            self.predictions.append({
                "boxes": pred_boxes,
                "scores": pred_scores,
                "labels": pred_labels
            })
            
            # Ground Truth 저장
            self.targets.append({
                "boxes": target_boxes,
                "labels": target_labels
            })
            
    
    def compute(self):
        
        # IoU 임계값 수
        num_thresholds = len(self.iou_thresholds)
        
        # AP 저장
        ap_values = torch.full(
            (self.num_classes, num_thresholds),
            float("nan")
        )
        
        # Recall 저장
        recall_values = torch.full(
            (self.num_classes, num_thresholds),
            float("nan")
        )
        
        # Class 단위 Metric 계산
        for class_index in range(self.num_classes):
            
            # 현재 Class의 GT 수
            num_gt = self._count_ground_truths(class_index)
            
            # GT가 없는 Class 제외
            if num_gt == 0:
                continue
            
            # IoU 임계값 순회
            for threshold_index, threshold in enumerate(self.iou_thresholds):
                
                # Prediction Matching
                scores, true_positives = self._collect_matches(
                    class_index=class_index,
                    iou_threshold=float(threshold)
                )
                
                # Prediction이 없는 경우
                if scores.numel() == 0:
                    ap_values[class_index, threshold_index] = 0
                    recall_values[class_index, threshold_index] = 0
                    continue
                
                # Score 기준 정렬
                order = torch.argsort(scores, descending=True)
                
                # True Positive
                true_positives = true_positives[order].to(torch.float32)
                
                # False Positive
                false_positives = (1.0 - true_positives)
                
                # 누적 TP
                cumulative_tp = torch.cumsum(true_positives, dim=0)
                
                # 누적 FP
                cumulative_fp = torch.cumsum(false_positives, dim=0)
                
                # Recall
                recall = cumulative_tp / float(num_gt)
                
                # Precision
                precision = (cumulative_tp 
                             / (cumulative_tp + cumulative_fp).clamp(min=1e-9))
                
                # AP 계산
                ap = self._calculate_ap(
                    precision=precision,
                    recall=recall
                )
                
                # AP 저장
                ap_values[class_index, threshold_index] = ap
                
                # 최대 Recall 저장
                recall_values[class_index, threshold_index] = recall[-1]
                
        # Metric 계산
        map50_95 = self._nanmean(ap_values)
            
        # IoU 0.50 위치
        index_50 = self._find_threshold_index(0.50)
            
        # IoU 0.75 위치
        index_75 = self._find_threshold_index(0.75)
            
        # mAP50
        map50 = self._nanmean(ap_values[:, index_50])
            
        # mAP75
        map75 = self._nanmean(ap_values[:, index_75])
            
        # mAR100
        mar100 = self._nanmean(recall_values)
            
        # 결과 반환
        metrics = {
            "map50_95": map50_95,
            "map50": map50,
            "map75": map75,
            "mar100": mar100
        }
            
        return metrics
        
    
    def _count_ground_truths(
        self,
        class_index
    ):
        
        # GT Counter
        count = 0
        
        # Image 단위 GT 계산
        for target in self.targets:
            count += int((target["labels"] == class_index).sum())
            
        return count
    
    
    def _collect_matches(
        self,
        class_index,
        iou_threshold
    ):
        
        # Score 저장
        all_scores = []
        
        # TP 저장
        all_true_positives = []
        
        # Image 단위 Matching
        for prediction, target in zip(self.predictions, self.targets):
            
            # 현재 Class Prediction
            pred_mask = prediction["labels"] == class_index
            pred_boxes = prediction["boxes"][pred_mask]
            pred_scores = prediction["scores"][pred_mask]
            
            # 현재 Class GT
            target_mask = target["labels"] == class_index
            target_boxes = target["boxes"][target_mask]
            
            # Prediction이 없는 경우
            if pred_scores.numel() == 0:
                continue
            
            # Score 기준 정렬
            order = torch.argsort(
                pred_scores,
                descending=True
            )
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]
            
            # 현재 Image의 TP 상태
            true_positives = torch.zeros(
                pred_scores.shape[0],
                dtype=torch.bool
            )
            
            # GT가 없는 경우
            if target_boxes.shape[0] == 0:
                all_scores.append(pred_scores)
                all_true_positives.append(true_positives)
                continue
            
            # Prediction과 GT IoU
            ious = self._box_iou(
                pred_boxes,
                target_boxes
            )
            
            # GT Matching 상태
            matched_gt = torch.zeros(
                target_boxes.shape[0],
                dtype=torch.bool
            )
            
            # Prediction 순회
            for prediction_index in range(pred_boxes.shape[0]):
                
                # 현재 Prediction의 IoU
                current_ious = ious[prediction_index].clone()
                
                # 이미 Matching된 GT 제외
                current_ious[matched_gt] = -1.0
                
                # 가장 높은 IoU
                best_iou, best_gt_index = torch.max(
                    current_ious,
                    dim=0
                )
                
                # TP 판정
                if best_iou >= iou_threshold:
                    true_positives[prediction_index] = True
                    matched_gt[best_gt_index] = True
            
            # 결과 저장
            all_scores.append(pred_scores)
            all_true_positives.append(true_positives)
            
        # Prediction이 없는 경우
        if len(all_scores) == 0:
            return torch.empty(0), torch.empty(0, dtype=torch.bool)
        
        # Image 결과 결합
        scores = torch.cat(all_scores, dim=0)
        true_positives = torch.cat(all_true_positives, dim=0)
        
        return scores, true_positives
    
    
    def _calculate_ap(
        self,
        precision,
        recall
    ):
        
        # Recall 기준점
        recall_points = torch.linspace(0.0, 1.0, 101)
        
        # Precision 저장
        interpolated_precision = []
        
        # 101개 Recall 기준 순회
        for recall_point in recall_points:
            
            # 현재 Recall 이상 영역
            mask = recall >= recall_point
            
            # Precision 계산
            if mask.any():
                precision_value = precision[mask].max()
            else:
                precision_value = torch.tensor(0.0)
                
            interpolated_precision.append(precision_value)
        
        # 평균 Precision
        ap = torch.stack(interpolated_precision).mean()
        
        return float(ap)
    
    
    def _find_threshold_index(self, threshold):
        
        # 가장 가까운 IoU Threshold 위치
        index = torch.argmin(torch.abs(self.iou_thresholds - threshold))
        
        return int(index)
    
    
    def _nanmean(self, values):
        
        # NaN 제외
        valid_values = values[~torch.isnan(values)]
        
        # 유효한 값이 없는 경우
        if valid_values.numel() == 0:
            return 0.0
        
        return float(valid_values.mean())
    
    def _box_iou(self, boxes1, boxes2):
        
        # Box 넓이
        area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
                * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
        area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
                * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
        
        # Intersection 좌상단
        intersection_lt = torch.maximum(
            boxes1[:, None, :2],
            boxes2[None, :, :2]
        )
        
        # Intersection 우하단
        intersection_rb = torch.minimum(
            boxes1[:, None, 2:],
            boxes2[None, :, 2:]
        )
        
        # Intersection 크기
        intersection_wh = (intersection_rb - intersection_lt).clamp(min=0)
        
        # Intersection 넓이
        intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
        
        # Union
        union = area1[:, None] + area2[None, :] - intersection
        
        # IoU
        iou = intersection / union.clamp(min=1e-9)
        
        return iou