import os
import re
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def extract_num(path):
    m = re.search(r"(\d+)", path)  # 从文件名中提取数字
    return int(m.group(1)) if m else 0

def pitchyaw_to_vector(pitchyaw):
    """将俯仰角转换为3D方向向量"""
    # pitchyaw: (B,2), [pitch, yaw]
    pitch = pitchyaw[:, 0]
    yaw = pitchyaw[:, 1]
    
    x = -torch.cos(pitch) * torch.sin(yaw)
    y = -torch.sin(pitch)
    z = -torch.cos(pitch) * torch.cos(yaw)
    
    return torch.stack([x, y, z], dim=1)  # (B,3)

class ETHXGaze_Gazehub_Dataset(Dataset):
    def __init__(
        self,
        root,
        split="train",              # train / test
        T_sample: int = 32,
        stride: int = 1,
        img_resize=(244, 244),
        eye_resize=(60, 36),
        transform=None,
        pad_short: bool = False,
        enableINFO: bool = True,
        test_subjects: list = None 
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.img_root = os.path.join(root, "Image")
        self.label_path = os.path.join(root, "Label", f"train.label")

        self.T_sample = T_sample
        self.stride = stride
        self.img_resize = img_resize
        self.eye_resize = eye_resize
        self.transform = transform
        self.pad_short = pad_short
        self.enableINFO = enableINFO

        if enableINFO:
            print("[Data] Using dataset: ETH-Gazehub")
            print(f"[Data] Split: {split}, T_sample: {T_sample}")

        if not os.path.exists(self.label_path):
            raise RuntimeError(f"Label file not found: {self.label_path}")

        self.subject_data = {}
        self._parse_label_file()
        if test_subjects is None:
            test_subjects = []

        test_subjects = set(test_subjects)

        if split == "train":
            selected_subjects = [
                s for s in self.subject_data.keys()
                if s not in test_subjects
            ]
        elif split == "test":
            selected_subjects = [
                s for s in self.subject_data.keys()
                if s in test_subjects
            ]
        else:
            raise ValueError(f"Unsupported split for ETH: {split}")

        if self.enableINFO: print(f"[Data] **{split}** Using {len(selected_subjects)} subjects")
            
        # sliding window index
        self.index = []
        for subject in selected_subjects:
            info = self.subject_data[subject]
            N = info["n"]
            if N >= T_sample:
                for s in range(0, N - T_sample + 1, stride):
                    self.index.append((subject, s))
            elif pad_short:
                self.index.append((subject, 0))
        
        if not self.index:
            raise RuntimeError("No sliding windows created.")
        
    def _parse_label_file(self):
        with open(self.label_path, "r") as fh:
            lines = [l.strip() for l in fh.readlines() if l.strip()]
            # if self.enableINFO:
            #     print("[Data] Detected subjects:")
            #     for s in sorted(self.subject_data.keys()):
            #         print(s,end=" ")
            #     print()

        # header
        header = lines[0].split()
        lines = lines[1:]

        # detect train/test format
        is_train = "gaze" in header

        def to_floats(s): 
            return [float(x) for x in s.split(",")]

        temp = {}

        for line in lines:
            cols = line.split()
            face_rel = cols[0]
            left_rel = cols[1]
            right_rel = cols[2]

            gaze_pitchyaw = torch.tensor(to_floats(cols[3]), dtype=torch.float32)
            head = torch.tensor(to_floats(cols[4]), dtype=torch.float32)

            origin_rel = cols[5]
            cam_index  = int(cols[6])
            frame_idx  = int(cols[7])
            normmat    = torch.tensor(to_floats(cols[8]), dtype=torch.float32)

            # 提取subject（Linux路径分隔符）
            subject = face_rel.split("/")[0]

            if subject not in temp:
                temp[subject] = []

            temp[subject].append({
                "face": face_rel,
                "left": left_rel,
                "right": right_rel,
                "gaze_pitchyaw": gaze_pitchyaw,              # 转换后的3D向量
                "head3d": head,
                "origin": torch.zeros(3),       # placeholder
                "rmat": normmat,                # ETH的normmat对齐到rmat
                "smat": torch.zeros(9),         # placeholder
                "gaze2d": torch.zeros(2),
                "head2d": torch.zeros(2),
                "frame_idx": frame_idx,
            })

        # sort & store
        for subject, entries in temp.items():
            entries.sort(key=lambda e: extract_num(e["face"]))
            face_paths = [e["face"] for e in entries]
            left_paths  = [e["left"] for e in entries]
            right_paths = [e["right"] for e in entries]
            labels = {}

            # 堆叠所有标签
            gaze_pitchyaw_stack = torch.stack([e["gaze_pitchyaw"] for e in entries], dim=0)
            head_stack = torch.stack([e["head3d"] for e in entries], dim=0)
            rmat_stack = torch.stack([e["rmat"] for e in entries], dim=0)
            smat_stack = torch.stack([e["smat"] for e in entries], dim=0)
            # gaze2d_stack = torch.stack([e["gaze2d"] for e in entries], dim=0)
            gaze2d_stack = gaze_pitchyaw_stack
            # head2d_stack = torch.stack([e["head2d"] for e in entries], dim=0)
            head2d_stack = head_stack
            origin_stack = torch.stack([e["origin"] for e in entries], dim=0)
            
            gaze3d_vector = pitchyaw_to_vector(gaze_pitchyaw_stack)
            head3d_vector = pitchyaw_to_vector(head_stack)
                
            rmat_stack = rmat_stack.view(-1, 3, 3)
            rmat_inv = rmat_stack.transpose(1, 2)
            gaze3d_vector_head = torch.bmm(rmat_inv, gaze3d_vector.unsqueeze(-1)).squeeze(-1)
            head3d_vector_head = torch.bmm(rmat_inv, head3d_vector.unsqueeze(-1)).squeeze(-1)
            
            labels["gaze3d"] = gaze3d_vector_head
            labels["head3d"] = head3d_vector_head
            labels["gaze2d"] = gaze2d_stack
            labels["head2d"] = head2d_stack
            labels["rmat"] = rmat_stack
            labels["smat"] = smat_stack
            labels["origin"] = origin_stack

            self.subject_data[subject] = {
                "face_paths": face_paths,
                "left_paths": left_paths,
                "right_paths": right_paths,
                "labels": labels,
                "n": len(face_paths),
            }

    def __len__(self):
        return len(self.index)

    def load_face(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize(self.img_resize)
        img = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
        return img

    def load_eye(self, path):
        img = Image.open(path).convert("L")
        img = img.resize(self.eye_resize)
        tensor = torch.from_numpy(np.array(img)).float() / 255.0
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        return tensor

    def __getitem__(self, idx):
        """获取一个样本"""
        subject, start = self.index[idx]
        info = self.subject_data[subject]
        N = info["n"]

        indices = (list(range(start, start+self.T_sample))
                   if N >= self.T_sample
                   else list(range(N)) + [N-1]*(self.T_sample-N))

        faces, left_eyes, right_eyes, eyes_concat = [], [], [], []

        for i in indices:
            face_path = os.path.join(self.img_root, info["face_paths"][i])
            left_path = os.path.join(self.img_root, info["left_paths"][i])
            right_path = os.path.join(self.img_root, info["right_paths"][i])

            face = self.load_face(face_path)                # [3,224,224]
            le   = self.load_eye(left_path)                 # [1,36,60]
            re   = self.load_eye(right_path)                # [1,36,60]

            # 左右拼接
            # le: [1,36,60], re: [1,36,60]
            # axis=2 → W方向拼接 → [1,36,120]
            eye_lr = torch.cat([le, re], dim=2)

            # 复制成3通道 → [3,36,120]
            eye_lr = eye_lr.repeat(3, 1, 1)

            faces.append(face)
            left_eyes.append(le)
            right_eyes.append(re)
            eyes_concat.append(eye_lr)

        faces = torch.stack(faces, dim=0)
        left_eyes = torch.stack(left_eyes, dim=0)
        right_eyes = torch.stack(right_eyes, dim=0)
        eyes_concat = torch.stack(eyes_concat, dim=0)

        # slice labels accordingly
        labels = {}
        for k,v in info["labels"].items():
            if N >= self.T_sample:
                labels[k] = v[start:start+self.T_sample].clone()
            else:
                labels[k] = torch.cat([v, v[-1:].repeat(self.T_sample-N,1)], dim=0)

        return {
            "face": faces,
            "left_eye": left_eyes,
            "right_eye": right_eyes,
            "eyes_concat": eyes_concat,
            "labels": labels
        }


if __name__ == "__main__":
    print("[INFO] Checking dataset: ETHGaze-gazehub preprocessed by gazehub-eyeplus")
    dataset_test = ETHXGaze_Gazehub_Dataset(
        root="/home/NWE/miniconda3/envs/worksacpe_gaze/dataset/ETHGaze-eyeplus",
        T_sample=4,
        stride=1,
        img_resize=(256, 256),
        eye_resize=(60, 36),
        transform=None,
        pad_short=False,
        enableINFO=True,
        split="test",
        test_subjects=["subject0008", "subject0012"],
    )
    
    print(f"Test Dataset length: {len(dataset_test)} samples")    
    dataset = ETHXGaze_Gazehub_Dataset(
        root="/home/NWE/miniconda3/envs/worksacpe_gaze/dataset/ETHGaze-eyeplus",
        T_sample=4,
        stride=1,
        img_resize=(256, 256),
        eye_resize=(60, 36),
        transform=None,
        pad_short=False,
        enableINFO=True,
        split="train",
        test_subjects=["subject0008", "subject0012"],
    )
    print(f"Dataset length: {len(dataset)} samples")
    
    # 取一个样本测试
    sample = dataset[0]
    print("Face shape:", sample["face"].shape)
    print("Gaze3d shape:", sample["labels"]["gaze3d"].shape)
    print("Sample head2d vector:", sample["labels"]["head2d"][0])
    print("labels:", sample["labels"])
    
    # 验证向量是单位长度
    gaze_norm = torch.norm(sample["labels"]["gaze3d"], dim=-1)
    print(f"Gaze vector norms (should be ~1.0): {gaze_norm}")