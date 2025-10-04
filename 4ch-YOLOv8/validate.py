# import argparse
# import torch
# from ultralytics import YOLO
# from torch.utils.data import DataLoader
# from ultralytics.data.custom_dataset import YOLO4ChannelDataset
# from pathlib import Path
# import yaml
# from types import SimpleNamespace
# from ultralytics.models.yolo.detect.val import DetectionValidator

# def run(weights, data, imgsz, batch, name, workers, hyp):
#     model = YOLO(weights)

#     # 1. 데이터 YAML 로딩
#     with open(data, 'r', encoding='utf-8') as f:
#         data_cfg = yaml.safe_load(f)
#     img_dir = data_cfg['val']

#     # 2. hyp.yaml 또는 args.yaml 전체 로딩
#     hyp_obj = {}
#     args_namespace = {}
#     if hyp:
#         hyp_path = Path(hyp)
#         with open(hyp_path, 'r', encoding='utf-8') as f:
#             hyp_dict = yaml.safe_load(f)
#         hyp_obj = SimpleNamespace(**hyp_dict)
#         args_namespace = hyp_dict
#         print(f"✅ HYP 로딩 완료: {hyp_path.name}")
#     else:
#         print("⚠️ hyp 인자 없이 실행됨: 기본값 또는 오류 발생 가능")

#     # 3. 커스텀 4채널 Dataset 구성
#     dataset = YOLO4ChannelDataset(
#         img_path=img_dir,
#         imgsz=imgsz,
#         batch_size=batch,
#         augment=False,
#         hyp=hyp_obj,
#         rect=False,
#         stride=32,
#         pad=0.5,
#         prefix="val: ",
#         task='detect',
#         data=data_cfg,
#         classes=None
#     )

#     # 4. Dataloader 구성
#     dataloader = DataLoader(
#         dataset=dataset,
#         batch_size=batch,
#         shuffle=False,
#         num_workers=workers,
#         pin_memory=True,
#         collate_fn=dataset.collate_fn
#     )

#     # 5. validator 실행
#     validator = DetectionValidator()
#     validator.args = SimpleNamespace(
#         data=data,
#         imgsz=imgsz,
#         batch=batch,
#         name=name,
#         save=True,
#         save_txt=True,
#         save_conf=True,
#         augment=False,
#         model=weights,
#         iou=args_namespace.get("iou", 0.7),
#         conf=args_namespace.get("conf") if args_namespace.get("conf") is not None else 0.001,  # ✅ 핵심 보정
#         half=args_namespace.get("half", False),
#         device=args_namespace.get("device", None),
#         save_json=args_namespace.get("save_json", False),
#         verbose=args_namespace.get("verbose", True),
#         plots=args_namespace.get("plots", True),
#         max_det=args_namespace.get("max_det", 300),
#         classes=args_namespace.get("classes", None),
#         rect=args_namespace.get("rect", False),
#         split=args_namespace.get("split", 'val'),
#         task='val',
#         mode='val',
#         val=True,
#         dnn=args_namespace.get("dnn", False),
#         retina_masks=args_namespace.get("retina_masks", False),
#         stream_buffer=args_namespace.get("stream_buffer", False),
#         show=args_namespace.get("show", False),
#         save_crop=args_namespace.get("save_crop", False),
#         line_width=args_namespace.get("line_width", None),
#         single_cls=args_namespace.get("single_cls", False),
#         agnostic_nms=args_namespace.get("agnostic_nms", False)
#     )
#     validator.model = model.model
#     validator.dataloader = dataloader

#     validator()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", type=str, required=True)
#     parser.add_argument("--data", type=str, required=True)
#     parser.add_argument("--imgsz", type=int, default=640)
#     parser.add_argument("--batch", type=int, default=1)
#     parser.add_argument("--name", type=str, default="val4ch")
#     parser.add_argument("--workers", type=int, default=0)
#     parser.add_argument("--hyp", type=str, default=None, help="Path to hyp.yaml or args.yaml")
#     opt = parser.parse_args()
#     run(**vars(opt))

import argparse
import torch
from ultralytics import YOLO
from torch.utils.data import DataLoader
from ultralytics.data.custom_dataset import YOLO4ChannelDataset
from pathlib import Path
import yaml
from types import SimpleNamespace
from ultralytics.models.yolo.detect.val import DetectionValidator

def run(weights, data, imgsz, batch, name, workers, hyp):
    model = YOLO(weights)

    # 1. 데이터 YAML 로딩
    with open(data, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    img_dir = data_cfg['val']

    # 2. hyp.yaml 또는 args.yaml 전체 로딩
    hyp_obj = {}
    args_namespace = {}
    if hyp:
        hyp_path = Path(hyp)
        with open(hyp_path, 'r', encoding='utf-8') as f:
            hyp_dict = yaml.safe_load(f)
        hyp_obj = SimpleNamespace(**hyp_dict)
        args_namespace = hyp_dict
        print(f"✅ HYP 로딩 완료: {hyp_path.name}")
    else:
        print("⚠️ hyp 인자 없이 실행됨: 기본값 또는 오류 발생 가능")

    # 3. 커스텀 4채널 Dataset 구성
    dataset = YOLO4ChannelDataset(
        img_path=img_dir,
        imgsz=imgsz,
        batch_size=batch,
        augment=False,
        hyp=hyp_obj,
        rect=False,
        stride=32,
        pad=0.5,
        prefix="val: ",
        task='detect',
        data=data_cfg,
        classes=None
    )

    # 4. Dataloader 구성
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    # 5. validator 설정 및 실행
    validator = DetectionValidator()
    validator.args = SimpleNamespace(
        data=data,
        imgsz=imgsz,
        batch=batch,
        name=name,
        save=True,
        save_txt=True,
        save_conf=True,
        augment=False,
        model=weights,
        iou=args_namespace.get("iou", 0.7),
        conf=args_namespace.get("conf") if args_namespace.get("conf") is not None else 0.001,
        half=args_namespace.get("half", False),
        device=args_namespace.get("device", None),
        save_json=args_namespace.get("save_json", False),
        verbose=args_namespace.get("verbose", True),
        plots=args_namespace.get("plots", True),
        max_det=args_namespace.get("max_det", 300),
        classes=args_namespace.get("classes", None),
        rect=args_namespace.get("rect", False),
        split=args_namespace.get("split", 'val'),
        task='val',
        mode='val',
        val=True,
        dnn=args_namespace.get("dnn", False),
        retina_masks=args_namespace.get("retina_masks", False),
        stream_buffer=args_namespace.get("stream_buffer", False),
        show=args_namespace.get("show", False),
        save_crop=args_namespace.get("save_crop", False),
        line_width=args_namespace.get("line_width", None),
        single_cls=args_namespace.get("single_cls", False),
        agnostic_nms=args_namespace.get("agnostic_nms", False)
    )
    validator.model = model.model.to("cuda" if torch.cuda.is_available() else "cpu")
    validator.dataloader = dataloader

    # 6. 검증 수행
    validator()

    # 7. 모든 배치에 대해 예측 결과 이미지 저장
    for ni, batch in enumerate(validator.dataloader):
        batch = validator.preprocess(batch)
        preds = validator.model(batch["img"], augment=False)
        preds = validator.postprocess(preds)
        validator.plot_predictions(batch, preds, ni)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--name", type=str, default="val4ch")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--hyp", type=str, default=None, help="Path to hyp.yaml or args.yaml")
    opt = parser.parse_args()
    run(**vars(opt))