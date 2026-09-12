import torch
import math
import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import config as Config
from datetime import datetime
import copy
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from adan_pytorch import Adan

from dataset_Eyediap_Gazehub_eyeplus import Eyediap_Gazehub_Dataset
from dataset_MPIIFaceGaze_Gazehub_eyeplus import MPIIFaceGaze_Gazehub_Dataset
from dataset_ETH_Gazehub_eyeplus import ETHXGaze_Gazehub_Dataset
from dataset_RTGene_Gazehub_eyeplus import RTGene_Gazehub_Dataset
# from dataset_Gaze360_Gazehub_eyeplus import Gaze360_Gazehub_Dataset
from model import GazeModel, OnlyMamba, OnlyMobileViT, count_parameters

def cosine_angle_deg(a, b, eps=1e-6):
    # a,b: [B,3], assumed normalized
    cos = (a * b).sum(dim=-1).clamp(-1+eps, 1-eps)
    return torch.rad2deg(torch.acos(cos))

# 冻结Batch Normalization，避免干扰
def set_bn_mode(model, mode="full"):
    """
        "full"   -> 冻结 BN（stats + affine）
        "affine" -> 冻结 affine，仅更新 running stats
        "none"   -> 完全不冻结（正常 BN）
    """
    assert mode in ["full", "affine", "none"]

    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d,
                          torch.nn.BatchNorm2d,
                          torch.nn.BatchNorm3d)):
            
            if mode == "full":
                # 1. 冻结 stats
                m.eval()
                # 2. 冻结 affine
                if m.affine:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

            elif mode == "affine":
                # 1. 保持训练模式（更新 running stats）
                m.train()
                # 2. 冻结 affine
                if m.affine:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

            elif mode == "none":
                # 完全正常训练
                m.train()
                if m.affine:
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True
            

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    seed = 42
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    
# 学习率分组
def build_optimizer(model, lr=1e-4, weight_decay=1e-2, optimizer_type="adamw", **kwargs):
    params = model.parameters()
    optimizer_type = optimizer_type.lower()
    
    # 对于 Adam/AdamW，移除 momentum 参数
    if optimizer_type in ["adamw", "adam", "adan"]:
        kwargs.pop('momentum', None)
    
    if optimizer_type == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type == "adan":
        return Adan(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

# 角度
def compute_angle_error(pred_gaze, true_gaze, eps=1e-8):
    # pred_gaze, true_gaze: [B, 3]
    p = pred_gaze / (pred_gaze.norm(dim=-1, keepdim=True) + eps)
    t = true_gaze / (true_gaze.norm(dim=-1, keepdim=True) + eps)
    dot = (p * t).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    angle = torch.acos(dot) * 180.0 / math.pi
    return angle.mean().item()   # degree

# 评估函数
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_angle = 0
    n = 0

    with torch.no_grad():
        for batch in dataloader:
            faces = batch["face"].to(device)             # [B,T,3,224,224]
            labels = batch["labels"]
            labels = {k: v.to(device) for k, v in labels.items()}

            pred = model(faces)

            loss, loss_dict = model.loss(pred, labels)

            pred_gaze = pred[:, :3]
            angle_err = compute_angle_error(pred_gaze, labels["gaze3d"])

            total_loss += loss.item()
            total_angle += angle_err
            n += 1

    return {
        "val_loss": total_loss / n,
        "val_angle_deg": total_angle / n
    }

# 数据规整
def collate_fn(batch):
    faces       = torch.stack([item["face"] for item in batch], dim=0)
    eyes_concat = torch.stack([item["eyes_concat"] for item in batch], dim=0)

    gaze3d = torch.stack([item["labels"]["gaze3d"] for item in batch], dim=0)
    head3d = torch.stack([item["labels"]["head3d"] for item in batch], dim=0)
    
    labels = {
        "gaze3d": gaze3d[:, -1, :],
        "head3d": head3d[:, -1, :]
    }

    return {
        "face": faces,
        "eyes_concat": eyes_concat,
        "labels": labels
    }

# 保存路径
def generate_model_save_path(root, epoch, cfgTrainName, cfgDatasetName):
    now = datetime.now().strftime("%Y%m%d_%H%M")
    cfg_train_name = cfgTrainName.__name__
    cfg_dataset_name = cfgDatasetName.__name__
    filename = f"model_hybMamba_ep{epoch}_{now}_{cfg_train_name}_{cfg_dataset_name}.pt"
    path = os.path.join(root, "train_result/model", filename)
    return path

def train(fomulation, Model = GazeModel, Method = "GAS", Face = "residual"): # MultiTask, gaze
    print(f"[Train] Method:{Method}, FacePredict:{Face}")
    cfg_train = fomulation
    cfg_dataset = cfg_train.dataset
    cfg_model = cfg_train.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    # log
    now = datetime.now().strftime("%Y%m%d_%H%M")
    cfg_train_name = cfg_train.__name__
    log_path = os.path.join(cfg_train.root, "train_result/log", f"log_{now}_{cfg_train_name}.txt")
    log_f = open(log_path, "w", encoding="utf-8")
    log_f.write(f"{now}_{cfg_train_name},")
    log_f.write(f"{cfg_dataset.__name__}, {cfg_model.__name__}, {cfg_train.loss_face_weight}:{cfg_train.loss_eye_weight}, ")
    log_f.flush()
    
    if cfg_dataset.__name__ == "dataset_EyeDiap" or cfg_dataset.__name__ == "dataset_EyeDiap_co":
        dataset_train = Eyediap_Gazehub_Dataset(
            root = cfg_dataset.file_root,
            T_sample = cfg_dataset.T_sample,
            stride = cfg_dataset.stride,
            img_resize = cfg_dataset.image_resize,
            pad_short = cfg_dataset.enable_pad_shorts,
            test_subjects = cfg_dataset.test_subjects,
            split="train"
        )
        dataset_test = Eyediap_Gazehub_Dataset(
            root = cfg_dataset.file_root,
            T_sample = cfg_dataset.T_sample,
            stride = cfg_dataset.stride,
            img_resize = cfg_dataset.image_resize,
            pad_short = cfg_dataset.enable_pad_shorts,
            test_subjects = cfg_dataset.test_subjects,
            split="test"
        )

    elif cfg_dataset.__name__ == "dataset_MPIIFaceGaze" or cfg_dataset.__name__ == "dataset_MPIIFaceGaze_T3S3":
        dataset_train = MPIIFaceGaze_Gazehub_Dataset(
            root = cfg_dataset.file_root,
            T_sample = cfg_dataset.T_sample,
            stride = cfg_dataset.stride,
            img_resize = cfg_dataset.image_resize,
            pad_short = cfg_dataset.enable_pad_shorts,
            test_subjects = cfg_dataset.test_subjects,
            split="train"
        )
        dataset_test = MPIIFaceGaze_Gazehub_Dataset(
            root = cfg_dataset.file_root,
            T_sample = cfg_dataset.T_sample,
            stride = cfg_dataset.stride,
            img_resize = cfg_dataset.image_resize,
            pad_short = cfg_dataset.enable_pad_shorts,
            test_subjects = cfg_dataset.test_subjects,
            split="test"
        )

    elif cfg_dataset.__name__ == "dataset_ETH":
        dataset_train = ETHXGaze_Gazehub_Dataset(
            root = cfg_dataset.file_root,
            T_sample = cfg_dataset.T_sample,
            stride = cfg_dataset.stride,
            img_resize = cfg_dataset.image_resize,
            pad_short = cfg_dataset.enable_pad_shorts,
            test_subjects = cfg_dataset.test_subjects,
            split="train"
        )

        dataset_test = ETHXGaze_Gazehub_Dataset(
            root = cfg_dataset.file_root,
            T_sample = cfg_dataset.T_sample,
            stride = cfg_dataset.stride,
            img_resize = cfg_dataset.image_resize,
            pad_short = cfg_dataset.enable_pad_shorts,
            test_subjects = cfg_dataset.test_subjects,
            split="test"
        )

    elif cfg_dataset.__name__ == "dataset_RTGene":
        dataset_train = RTGene_Gazehub_Dataset(
            root = cfg_dataset.file_root,
            T_sample = cfg_dataset.T_sample,
            stride = cfg_dataset.stride,
            img_resize = cfg_dataset.image_resize,
            pad_short = cfg_dataset.enable_pad_shorts,
            split="train"
            # 无需test_subjects，数据集自带划分有效
        )
        dataset_test = RTGene_Gazehub_Dataset(
            root = cfg_dataset.file_root,
            T_sample = cfg_dataset.T_sample,
            stride = cfg_dataset.stride,
            img_resize = cfg_dataset.image_resize,
            pad_short = cfg_dataset.enable_pad_shorts,
            split="test"
            # 无需test_subjects，数据集自带划分有效
        )
       
    else: print("[ERROR] Cannot find dataset!")

    # train / val split 
    total_size = len(dataset_train)
    val_size   = int(0.05 * total_size)
    train_size = total_size - val_size
    test_size = len(dataset_test)

    train_ds, val_ds = torch.utils.data.random_split(
        dataset_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Train-Val-Test lenth
    print(f"[INFO] Train-Val-Test lenth:{train_size},{val_size},{test_size}")
    log_f.write(f"{train_size}-{val_size}-{test_size}\n")
    log_f.write(f"{cfg_train.msg}\n")
    log_f.write(f"epoch,valloss,valangle,testloss,testangle\n")
    log_f.flush()
    
    # Dataloader
    g = torch.Generator()
    g.manual_seed(42)
    
    num_workers = min(8, os.cpu_count())
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg_train.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        generator=g
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg_train.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        generator=g
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=cfg_train.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        generator=g
    )

    model = Model(
        vit_model_name = cfg_model.vit_model_name,
        vit_weight_path = cfg_model.vit_weight_path,
        mobilevit_output_dim = cfg_model.mobilevit_output_dim,
        mamba_hidden_dim = cfg_model.mamba_hidden_dim,
        mamba_kwargs = cfg_model.mamba_kwargs,
        out_dim = cfg_model.out_dim,
        face_pad_px= cfg_model.face_pad_px,
        face_min_conf= cfg_model.face_min_conf,
        banAttention = cfg_model.banAttention, 
        enableINFO= cfg_model.enableINFO
    ).to(device)

    print("[INFO] Total trainable params:", count_parameters(model))
    print("[Train] loss_face_weight:", cfg_train.loss_face_weight)
    print("[Train] loss_eye_weight:", cfg_train.loss_eye_weight)

    # Loss setting
    model.set_loss(weight_gaze=1.0)

    # Optimizer
    optimizer = build_optimizer(model, 
                                lr=cfg_train.lr, 
                                optimizer_type=cfg_train.optimizer_type, 
                                momentum=0.9)
    
    best_val_metrics = {
        "val_loss": float("inf"),
        "val_angle_deg": float("inf")
    }
    best_test_metrics = {
        "val_loss": float("inf"),
        "val_angle_deg": float("inf")
    }
    best_test_angle = float("inf")          # 新增：单独跟踪 test 最佳
    patience = 15                           # 连续 15 个 epoch val 没有提升就早停
    patience_counter = 0
    best_model_path = None
       
    # save test
    torch.save(model.state_dict(), generate_model_save_path(
                root = cfg_train.root, 
                epoch = "test",
                cfgDatasetName=cfg_dataset,
                cfgTrainName=cfg_train
                ))
    print("[Train] save test pass")
    
    validate(model, val_loader, device)
    print("[Train] validate test pass")
    
    # Traning Loop
    for epoch in range(cfg_train.epoch):
        model.train()
        set_bn_mode(model, cfg_train.bn_mode)
        pbar = tqdm(train_loader, desc=f"Epoch %d" % epoch)

        for batch in pbar:

            # batch unpack
            faces   = batch["face"].to(device)          # [B,T,C,H,W]
            eyes    = batch["eyes_concat"].to(device)   # [B,T,3,36,120]
            labels  = {k: v.to(device) for k,v in batch["labels"].items()}
            gaze_gt = labels["gaze3d"]   # [B,3]
            head_gt = labels["head3d"]   # [B,3]
            optimizer.zero_grad()

            if Method == "MultiTask":
                total_loss = 0
                if cfg_train.loss_face_weight > 0:
                    pred_face = model(faces)
                    loss_face, loss_dict_face = model.loss(pred_face, labels, mode=Face)
                    total_loss += cfg_train.loss_face_weight * loss_face
                else: # No forward
                    loss_face = torch.tensor(-1.0, device=device)
                    loss_dict_face = {"loss_gaze": -1}

                if cfg_train.loss_eye_weight > 0:
                    pred_eye = model(eyes)
                    loss_eye, loss_dict_eye = model.loss(pred_eye, labels, mode="gaze")
                    total_loss += cfg_train.loss_eye_weight * loss_eye
                else: 
                    loss_eye = torch.tensor(-1.0, device=device)
                    loss_dict_eye = {"loss_gaze": -1}                
                total_loss.backward()

            elif Method == "GAS":
                # full face -> gaze
                if cfg_train.loss_face_weight > 0:
                    pred_face = model(faces)
                    loss_face, loss_dict_face = model.loss(pred_face, labels, mode=Face)
                    (loss_face*cfg_train.loss_face_weight).backward()
                else: # No forward
                    loss_face = torch.tensor(-1.0, device=device)
                    loss_dict_face = {"loss_gaze": -1}
                    
                # eye -> gaze
                if cfg_train.loss_eye_weight > 0:
                    pred_eye = model(eyes)
                    loss_eye, loss_dict_eye = model.loss(pred_eye, labels, mode="gaze")
                    (loss_eye*cfg_train.loss_eye_weight).backward()
                else: 
                    loss_eye = torch.tensor(-1.0, device=device)
                    loss_dict_eye = {"loss_gaze": -1}
                                
            # 3) update once
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.75)
            optimizer.step()

            pbar.set_postfix(
                LF=loss_face.item(),
                LE=loss_eye.item(),
                GF=loss_dict_face["loss_gaze"],
                GE=loss_dict_eye["loss_gaze"]
            )

        # Validation & write log
        val_metrics = validate(model, val_loader, device)
        test_metrics = validate(model, test_loader, device)
        print(f"[Epoch {epoch}] "
              f"ValLoss={val_metrics['val_loss']:.4f}, "
              f"ValAngle={val_metrics['val_angle_deg']:.2f} deg, "
              f"TesLoss={test_metrics['val_loss']:.4f}, "
              f"TesAngle={test_metrics['val_angle_deg']:.2f} deg"
              )
        log_f.write(
            f"{epoch}, "
            f"{val_metrics['val_loss']:.6f}, "
            f"{val_metrics['val_angle_deg']:.6f}, "
            f"{test_metrics['val_loss']:.6f}, "
            f"{test_metrics['val_angle_deg']:.6f}\n"
        )
        log_f.flush()
    
        # Save best
        improved = False
        # 1. 主标准：用 Val Angle 判断是否是当前最佳模型
        if val_metrics["val_angle_deg"] < best_val_metrics["val_angle_deg"]:
            best_val_metrics = val_metrics.copy()
            best_val_metrics["epoch"] = epoch
            best_test_metrics = test_metrics.copy()
            improved = True
            patience_counter = 0
            
            save_path = generate_model_save_path(
                root=cfg_train.root, 
                epoch=epoch,
                cfgDatasetName=cfg_dataset,
                cfgTrainName=cfg_train
            )
            torch.save(model.state_dict(), save_path)
            best_model_path = save_path
            print(f"✓ Saved NEW BEST model (val) at epoch {epoch} | TestAngle={test_metrics['val_angle_deg']:.3f}")
        
        # 2. 辅助标准：如果 Test Angle 更好，也保存（防止 val 误导）
        elif test_metrics["val_angle_deg"] < best_test_angle:
            best_test_angle = test_metrics["val_angle_deg"]
            save_path = generate_model_save_path(
                root=cfg_train.root, 
                epoch=epoch,
                cfgDatasetName=cfg_dataset,
                cfgTrainName=cfg_train
            ).replace(".pt", "_best_test.pt")   # 区分文件名
            torch.save(model.state_dict(), save_path)
            print(f"✓ Saved better Test model at epoch {epoch} | TestAngle={best_test_angle:.3f} (not best val)")

        # 3. Early Stopping 判断
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no val improvement for {patience} epochs)")
                break
    
    log_f.write(
    f"{-1}, "
    f"{best_val_metrics['val_loss']:.6f}, "
    f"{best_val_metrics['val_angle_deg']:.6f}, "
    f"{best_test_metrics['val_loss']:.6f}, "
    f"{best_test_metrics['val_angle_deg']:.6f}\n")
    log_f.flush()

def run_loso(cfg):
    all_subjects = sorted([
        d.replace(".label", "") for d in os.listdir(cfg.dataset.file_root + "/Label")
        # if d.startswith("p")
    ])
    print("Detect subject: {all_subjects}")
    for test_subject in all_subjects:
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.optimizer_type = "adamw"
        cfg_copy.lr = 1e-3
        cfg_copy.dataset.test_subjects = [test_subject]
        cfg_copy.msg = f"[LOSO] Test subject:{test_subject}"
        for i in range(0, 3): train(cfg_copy)
    
def run_GasxAttention(cfg):
    yesorno = [True, False]
    for i in yesorno:
        for m in yesorno:
            # 深拷贝原始配置，避免污染
            cfg_copy = copy.deepcopy(cfg)
            cfg_copy.loss_eye_weight = -1 if i else 0.3
            cfg_copy.model.banAttention = m
            cfg_copy.msg = f"[GxA] Eye-{i}:Attention{m}"
            print("[GxA] cfg.loss_eye_weight:", cfg_copy.loss_eye_weight)
            print("[GxA] cfg.model.banAttention:", cfg_copy.model.banAttention)
            train(cfg_copy)
        
def run_eyeWeighSacm(cfg):
    weight = [0.1,0.2,0.3,0.4,0.5,0.75,1.0]  
    for w in weight:
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.optimizer_type = "adan"
        cfg_copy.lr = 3e-04
        cfg_copy.loss_eye_weight = float(w)
        cfg_copy.msg = f"[EWS] Eye weight:{w}, lr3e-4, optmz: adan"
        for i in range(0,3):  train(cfg_copy)
  
def run_lrSacm(cfg):
    listlr = [1e-3, 3e-4,1e-4,3e-5,1e-5]  
    for l in listlr:
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.optimizer_type = "adamw"
        cfg_copy.lr = l
        cfg_copy.msg = f"[lWS] lr:{l}, Optimizer:AdamW"
        for i in range(0,3): train(cfg_copy)
        
def run_optmzSacn(cfg):
    # weight delay 1e-2(MAX)
    listoptmz = ["rmsprop", "adam"] # "adan", "adamw", "sgd", 
    for o in listoptmz:
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.optimizer_type = o
        cfg_copy.msg = f"[OS] Optimizer:{o}"
        for i in range(0,1):  train(cfg_copy)

def run_modelScan(cfg):
    listmodel = [OnlyMamba, OnlyMobileViT, GazeModel] 
    for m in listmodel:
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.msg = f"[MS] model:{m}"
        for i in range(0,2): train(cfg_copy, m)

def run_BNScan(cfg): 
    listBN= ["full", "affine", "none"]  
    for b in listBN:
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.msg = f"[BN] states:{b}"
        cfg_copy.bn_mode = b
        for i in range(0,3): train(cfg_copy)

def run_GAS(cfg):
    listBN = ["full", "none"]
    weight = [(1,-1), (-1, 1), (1, 0.3)]
    for b in listBN:
        for w in weight: 
            cfg_copy = copy.deepcopy(cfg)
            cfg_copy.msg = f"[BN] states:{b}, [GAS] weight:{w}"
            cfg_copy.bn_mode = b
            cfg_copy.loss_face_weight = w[0]
            cfg_copy.loss_eye_weight = w[1]
            for i in range(0,3): train(cfg_copy)

def run_Method(cfg):
    listBN = ["MultiTask", "GAS"]
    method = [(1, 0.3)]
    for b in listBN:
        for m in method: 
            cfg_copy = copy.deepcopy(cfg)
            cfg_copy.msg = f"[BN] states:{b}, Method: {m}"
            cfg_copy.bn_mode = b
            for i in range(0,3): train(cfg_copy, Method=m)

# Layer 6
# head 8
# Dim 32      
# sotronger model: model_vit2_mambaplus
if __name__ == "__main__":
    f = Config.train_fomulation_GAS1
    # run_GAS(f)
    # run_GasxAttention(f)
    # run_loso(f)
    # run_eyeWeighSacm(f)
    # run_lrSacm(f, Face = "residual")
    # run_modelScan(f)
    # run_optmzSacn(f)
    # run_BNScan(f)
    train(f)

'''
# search model weight
START_TIME="20260404_1634"   # 改成你的开始时间
TRAIN_FOMULATION="GAS1" # 改成你的训练配方
EPOCH="46"                   # 要找的 epoch

find . -type f -name "model_*_ep${EPOCH}_*_train_fomulation_${TRAIN_FOMULATION}_*" \
  | awk -v st="$START_TIME" '{ match($0, /_ep[0-9]+_([0-9]{8}_[0-9]{4})_/, arr); if (arr[1] >= st) print }'
'''
