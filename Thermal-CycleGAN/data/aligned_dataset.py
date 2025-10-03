import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch


class AlignedDataset(BaseDataset):
    """
    - 원본 이미지: opt.dataroot/{phase}/[A or B] (ex: TrainA, TrainB, TestA, TestB)
    - 세그 이미지: opt.dataroot/{phase}/SegA (ex: TrainSegA)
    
    TrainSegB 폴더는 없음.
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_SegA = os.path.join(opt.dataroot, opt.phase + 'SegA')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.SegA_paths = sorted(make_dataset(self.dir_SegA, opt.max_dataset_size)) if os.path.exists(self.dir_SegA) else []

        assert(opt.load_size >= opt.crop_size)

        self.input_nc = opt.input_nc   # 예: 4 (RGB + Seg)
        self.output_nc = opt.output_nc  # 예: 1 (열화상 1채널)

        self.dataset_size = min(len(self.A_paths), len(self.B_paths))

    def __getitem__(self, index):
        index = index % self.dataset_size

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('L')  # 1채널 grayscale로 변경

        A_name = os.path.basename(A_path)

        # SegA 이미지 불러오기 (없으면 검은색 1채널)
        if self.SegA_paths:
            SegA_path = os.path.join(self.dir_SegA, A_name)
            if os.path.exists(SegA_path):
                SegA_img = Image.open(SegA_path).convert('L')
            else:
                SegA_img = Image.new('L', A_img.size)
        else:
            SegA_img = Image.new('L', A_img.size)

        transform_params = get_params(self.opt, A_img.size)

        A_transform = get_transform(self.opt, transform_params, grayscale=False)
        B_transform = get_transform(self.opt, transform_params, grayscale=True)  # 1채널에 맞게 변경
        SegA_transform = get_transform(self.opt, transform_params, grayscale=True)

        A_tensor = A_transform(A_img)
        B_tensor = B_transform(B_img)
        SegA_tensor = SegA_transform(SegA_img)

        # A는 4채널 (RGB + Seg), B는 1채널 (열화상)
        A_4ch = torch.cat([A_tensor, SegA_tensor], dim=0)

        return {'A': A_4ch, 'B': B_tensor, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return self.dataset_size

