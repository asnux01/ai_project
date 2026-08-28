# 라이브러리
import torch

from torch.utils.data import DataLoader


# Detection 데이터를 batch 형태로 결합
def detection_collate_fn(batch):

    # 이미지를 하나의 batch Tensor로 결합
    images = torch.stack(
        [sample["img"] for sample in batch],
        dim=0
    )

    # Bbox를 하나의 Tensor로 결합
    bboxes = torch.cat(
        [sample["bboxes"] for sample in batch],
        dim=0
    )

    # Class를 하나의 Tensor로 결합
    classes = torch.cat(
        [sample["cls"] for sample in batch],
        dim=0
    )

    # 객체가 속한 이미지 index 저장
    batch_indices = []

    for batch_index, sample in enumerate(batch):

        # 현재 이미지의 객체 개수 가져오기
        num_objects = sample["bboxes"].shape[0]

        # 각 객체에 현재 이미지 index 부여
        image_indices = torch.full(
            (num_objects,),
            batch_index,
            dtype=torch.long
        )

        batch_indices.append(image_indices)

    # 이미지 index를 하나의 Tensor로 결합
    batch_indices = torch.cat(
        batch_indices,
        dim=0
    )

    # 이미지 ID 저장
    image_ids = [
        sample["image_id"]
        for sample in batch
    ]

    # 이미지 파일 경로 저장
    image_files = [
        sample["im_file"]
        for sample in batch
    ]

    # 원본 이미지 크기 저장
    original_shapes = [
        sample["ori_shape"]
        for sample in batch
    ]

    # Batch 정보 구성
    batch_data = {
        "img": images,
        "bboxes": bboxes,
        "cls": classes,
        "batch_idx": batch_indices,
        "image_id": image_ids,
        "im_file": image_files,
        "ori_shape": original_shapes,
    }

    return batch_data


# Dataset을 DataLoader로 구성
def build_dataloader(
    dataset,
    batch_size,
    shuffle,
    num_workers=4,
    pin_memory=True,
    drop_last=False
):

    # DataLoader 생성
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=detection_collate_fn,
        persistent_workers=num_workers > 0
    )

    return dataloader