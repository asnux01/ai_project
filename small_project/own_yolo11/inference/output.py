"""Ultralytics Results의 시각화 이미지와 YOLO txt 출력을 저장한다."""

# 모든 출력 경로를 운영체제 독립적으로 처리한다.
from pathlib import Path


def increment_path(path, exist_ok=False):
    """Ultralytics runs/predict/exp, exp2 형식으로 충돌 없는 경로를 고른다."""

    path = Path(path)

    # --exist-ok이면 기존 exp를 사용하고, 아니면 새 번호를 찾는다.
    if exist_ok or not path.exists():
        return path

    for index in range(2, 10000):
        candidate = path.with_name(f"{path.name}{index}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"사용 가능한 출력 디렉터리 이름을 찾지 못했습니다: {path}")


def create_save_dir(project="runs/predict", name="exp", exist_ok=False, save_txt=False):
    """현재 실행의 출력 폴더와 선택적인 labels 폴더를 만든다."""

    # 기본 조합: project='runs/predict' + name='exp'
    save_dir = increment_path(Path(project).expanduser() / name, exist_ok=exist_ok)

    # parents=True이므로 runs와 predict가 없어도 한 번에 생성된다.
    save_dir.mkdir(parents=True, exist_ok=True)

    if save_txt:
        # 공식 Ultralytics처럼 텍스트 라벨은 labels 하위 폴더에 모은다.
        (save_dir / "labels").mkdir(parents=True, exist_ok=True)

    return save_dir.resolve()


def _unique_output_path(path):
    """같은 이름의 서로 다른 source가 결과를 덮어쓰지 않게 한다."""

    # 서로 다른 폴더의 source가 같은 stem을 가질 때 결과 덮어쓰기를 막는다.
    if not path.exists():
        return path

    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"사용 가능한 출력 파일명을 찾지 못했습니다: {path}")


def save_results(
    results,
    save_dir,
    save_images=True,
    save_txt=False,
    save_conf=False,
):
    """Results 객체를 공식 plot/save_txt 인터페이스로 기록한다."""

    save_dir = Path(save_dir)
    saved_paths = []

    for result in results:
        # 원본 파일의 stem을 결과 파일명으로 유지한다.
        source_path = Path(result.path)

        if save_images:
            # Results.save는 box/label/confidence가 그려진 이미지를 만든다.
            image_path = _unique_output_path(save_dir / f"{source_path.stem}.jpg")
            result.save(filename=str(image_path))
            saved_paths.append(image_path)

        if save_txt:
            # 한 행은 class_id와 정규화된 xywh이며 --save-conf 시 confidence가 추가된다.
            label_path = _unique_output_path(save_dir / "labels" / f"{source_path.stem}.txt")
            result.save_txt(str(label_path), save_conf=save_conf)
            saved_paths.append(label_path)

    return saved_paths
