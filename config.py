# ==============================
# 数据集配置
    # 即使使用gazehub提供的预处理代码，每个数据集的文件格式与标记信息不完全一致，每个数据集类一对一处理
    # 在2060的机子上训练，替换路径如下：
        # /home/NWE/miniconda3/envs/worksacpe_gaze 
        # /root/miniconda3/envs/workspace_gaze
    # 在4090的机子上训练，请拷贝数据集到工作目录下的dataset文件夹，并替换路径如下：
        # /home/NWE/miniconda3/envs/worksacpe_gaze
        # /home/mon3tr/miniconda3/envs/hybridGaze
    # 在从windows迁移数据集后，请将label文件的路径分隔符修改成linux
        # \
        # /
# ==============================
# eyediap数据集
class dataset_EyeDiap:
    # 数据集文件夹绝对地址
    file_root = "/home/NWE/miniconda3/envs/worksacpe_gaze/dataset/Eyediap-gazehub"
    # 每个样本包含的静态帧数量，不影响训练参数量，但是大幅影响训练显存占用，溢出到内存中
    T_sample = 6
    # 滑动窗口大小，减少总样本数，增加样本间不同
    stride = 3
    # 图片重缩放
    image_resize = (224,224)
    # 帧的数量不足以构建T时的操作
    enable_pad_shorts = False
    # 是否启用日志打印
    enableINFO: bool = True
    # 测试集指定subject              
    test_subjects: list = ["p14"]
# eyediap,cluster label
class dataset_EyeDiap_c0:
    # 数据集文件夹绝对地址
    file_root = "/home/NWE/miniconda3/envs/worksacpe_gaze/dataset/Eyediap-gazehub"
    T_sample = 6
    stride = 3
    image_resize = (224,224)
    enable_pad_shorts = False
    enableINFO: bool = True
    test_subjects: list = ["Cluster0"]
    
# mpiifacegaze数据集
class dataset_MPIIFaceGaze:
    file_root = "/home/NWE/miniconda3/envs/worksacpe_gaze/dataset/MPIIFaceGaze-gazehub"
    T_sample = 6
    stride = 3
    image_resize = (224,224)
    enable_pad_shorts = False
    enableINFO: bool = True         
    test_subjects: list = ["p01"]  
# mpiifacegaze数据集，降低时序影响
class dataset_MPIIFaceGaze_T3S3:
    file_root = "/home/NWE/miniconda3/envs/worksacpe_gaze/dataset/MPIIFaceGaze-gazehub"
    T_sample = 3
    stride = 3
    image_resize = (224,224)
    enable_pad_shorts = False
    enableINFO: bool = True         
    test_subjects: list = ["p00"]  
    
# ETHgaze数据集
class dataset_ETH:
    # gaze需要从俯仰角转换出
    # val-test差距极大！
    file_root="/home/NWE/miniconda3/envs/worksacpe_gaze/dataset/ETHGaze-eyeplus"
    T_sample = 6
    stride = 3
    image_resize = (224,224)
    enable_pad_shorts = False
    enableINFO: bool = True
    test_subjects: list = ["subject0008", "subject0013", "subject0063", "subject0027", "subject0038", "subject0045",
    "subject0009", "subject0014", "subject0065", "subject0028", "subject0039", "subject0046"]
# ETHgaze数据集，降低时序影响
class dataset_ETH_T3S3:
    # gaze需要从俯仰角转换出
    # val-test差距极大！
    file_root="/home/NWE/miniconda3/envs/worksacpe_gaze/dataset/ETHGaze-eyeplus"
    T_sample = 3
    stride = 3
    image_resize = (224,224)
    enable_pad_shorts = False
    enableINFO: bool = True
    test_subjects: list = ["subject0008", "subject0013", "subject0063", "subject0027", "subject0038", "subject0045",
    "subject0009", "subject0014", "subject0065", "subject0028", "subject0039", "subject0046"]

# RTGene数据集   
class dataset_RTGene:
    file_root="/home/NWE/miniconda3/envs/worksacpe_gaze/dataset/RTGene"
    T_sample = 6
    stride = 3
    image_resize = (224,224)
    enable_pad_shorts = False
    enableINFO: bool = True
    test_subject = 1

# Gaze360数据集
class dataset_Gaze360:
    file_root="/mnt/d/Zhiyang Wang/gaze360-gazehub-subject"
    T_sample = 6
    stride = 3
    image_resize = (224,224)
    enable_pad_shorts = False
    enableINFO: bool = True

# ==============================
# 模型配置
# ==============================
# 基线：使用train_default.py
class model_vit_mamba:
    # 使用的模型名称，mobilevit有多种尺寸多个版本可供使用，贡献了大量的参数量
    vit_model_name: str = "mobilevit_xs.cvnets_in1k"
    # safetensor路径
    vit_weight_path: str = "model_use/mobilevit_xs/model.safetensors"
    # mobile-visionTransformer的输出维度
    mobilevit_output_dim: int = 256
    # mamba隐藏维度
    mamba_hidden_dim = 256
    # mamba模型配置
    mamba_kwargs: dict = dict(
        n_layers = 3,
        d_state = 64,
        expand = 2,
    )
    # mamba输出维度，取决于你想要预测的输出
    out_dim: int = 6
    # 是否启用blazeface掩盖背景，这取决于数据集
    enable_blazeface: bool = False
    # 识别框多少px外视作背景被mask掉
    face_pad_px:int = 40
    # 面部识别置信度阈值
    face_min_conf: float = 0.5
    # 是否启用日志打印
    enableINFO: bool = True
    banAttention = False

# 大一号的编码器
class model_vitplus_mamba:
    vit_model_name: str = "mobilevit_s.cvnets_in1k"
    vit_weight_path: str = "model_use/mobilevit_s/model.safetensors"
    mobilevit_output_dim: int = 256
    mamba_hidden_dim = 256
    mamba_kwargs: dict = dict(
        n_layers = 3,
        d_state = 64,
        expand = 2,
    )
    out_dim: int = 6
    enable_blazeface: bool = False
    face_pad_px:int = 40
    face_min_conf: float = 0.5
    enableINFO: bool = True
    
# 第二代编码器，并增加mamba层数
class model_vit2_mambaplus:
    vit_model_name: str = "mobilevitv2_075.cvnets_in1k"
    vit_weight_path: str = "model_use/mobilevitv2_075/model.safetensors"
    mobilevit_output_dim: int = 256
    mamba_hidden_dim = 256
    mamba_kwargs: dict = dict(
        n_layers = 4,
        d_state = 64,
        expand = 2,
    )
    out_dim: int = 6
    enable_blazeface: bool = False
    face_pad_px:int = 40
    face_min_conf: float = 0.5
    enableINFO: bool = True
# ==============================
# 训练配方：主要区分数据集与模型，轮询某个参数单独写函数配置
# # tips: 巨大改进：预测一半帧
# ==============================
# 为梯度累计监督的测试，eye不参与传播 
class train_fomulation_GAS1_face_only:
    msg = ""
    root = "/home/NWE/miniconda3/envs/worksacpe_gaze/"
    model = model_vit_mamba
    dataset = dataset_EyeDiap
    val_split: float = 0.85
    batch_size: int = 16
    optimizer_type: str = "adamw"
    lr = 1e-4
    epoch: int = 50
    loss_face_weight = 1                     
    loss_eye_weight = -1
    bn_mode = "full"
    
# 为梯度累计监督的测试，face不参与传播 
class train_fomulation_GAS1_eye_only:
    msg = ""
    root = "/home/NWE/miniconda3/envs/worksacpe_gaze/"
    model = model_vit_mamba
    dataset = dataset_EyeDiap
    val_split: float = 0.85
    batch_size: int = 16
    optimizer_type: str = "adamw"
    lr = 1e-4
    epoch: int = 50
    loss_face_weight = -1                    
    loss_eye_weight = 1
    bn_mode = "full"
# 梯度累计监督系数扫描0.2(default)
class train_fomulation_GAS1:
    msg = ""
    root = "/home/NWE/miniconda3/envs/worksacpe_gaze/"
    model = model_vit_mamba
    dataset = dataset_EyeDiap
    batch_size: int = 32
    optimizer_type: str = "adamw"
    lr = 1e-4
    epoch: int = 50
    loss_face_weight = 1                     
    loss_eye_weight = 0.3
    bn_mode = "full"
# 5.228932 + 6.479946 + 5.251804 + 2.906526 + 3.973654 + 4.255223 + 4.387901 + 4.971185 + 7.185961 + 6.623563 + 5.736266 + 6.713506 + 6.338916 del non-convergent subject 8.045575 5.57 -> 5.38
# 大一号编码器，eyediap
class train_fomulation_GAS1_plus:
    msg = ""
    root = "/home/NWE/miniconda3/envs/worksacpe_gaze/"
    model = model_vitplus_mamba
    dataset = dataset_EyeDiap
    batch_size: int = 32
    optimizer_type: str = "adamw"
    lr = 1e-4
    epoch: int = 50
    loss_face_weight = 1                     
    loss_eye_weight = 0.2
    bn_mode = "full"

# mpiiface
class train_fomulation_GAS2:
    msg = ""
    root = "/home/NWE/miniconda3/envs/worksacpe_gaze/"
    model = model_vit_mamba
    dataset = dataset_MPIIFaceGaze
    batch_size: int = 32
    optimizer_type: str = "adamw"
    lr = 1e-4
    # 训练次数
    epoch: int = 50
    loss_face_weight = 1                     
    loss_eye_weight = 0.2
    bn_mode = "full"
# mpiiface
class train_fomulation_GAS2_T3S3:
    msg = ""
    root = "/home/NWE/miniconda3/envs/worksacpe_gaze/"
    model = model_vit_mamba
    dataset = dataset_MPIIFaceGaze_T3S3
    batch_size: int = 32
    optimizer_type: str = "adamw"
    lr = 1e-4
    epoch: int = 50
    loss_face_weight = 1                     
    loss_eye_weight = 0.1
    bn_mode = "full"

# ETH-Xgaze，使用mediapipe分割出的eyecrop大概有5%的错误率     
class train_fomulation_GAS3: 
    msg = ""
    root = "/home/NWE/miniconda3/envs/worksacpe_gaze/"
    model = model_vit_mamba
    dataset = dataset_ETH
    batch_size: int = 32
    optimizer_type: str = "adamw"
    lr = 1e-4
    epoch: int = 50
    loss_face_weight = 1                     
    loss_eye_weight = 0.2
    head_filter_deg = 25
    bn_mode = "full"
# ETH-Xgaze，eyeloss.1
class train_fomulation_GAS3_eye_weight_min: 
    root = "/home/NWE/miniconda3/envs/worksacpe_gaze/"
    model = model_vit_mamba
    dataset = dataset_ETH
    batch_size: int = 16
    optimizer_type: str = "adamw"
    lr = 1e-4
    epoch: int = 50
    loss_face_weight = 1                     
    loss_eye_weight = 0.1
    head_filter_deg = 25
    bn_mode = "full"
# ETH-Xgaze，不使用eyecrop
class train_fomulation_GAS3_face_only:
    root = "/home/NWE/miniconda3/envs/worksacpe_gaze/"
    model = model_vit_mamba
    dataset = dataset_ETH
    batch_size: int = 16
    optimizer_type: str = "adamw"
    lr = 1e-4
    epoch: int = 50
    loss_face_weight = 1                     
    loss_eye_weight = -1
    head_filter_deg = 25   
    bn_mode = "full"

# RT-Gene数据集
class train_fomulation_GAS4:
    msg = ""
    root = "/home/mon3tr/miniconda3/envs/hybridGaze/"
    model = model_vit_mamba
    dataset = dataset_RTGene
    batch_size: int = 16
    optimizer_type: str = "adamw"
    lr = 1e-4
    epoch: int = 50
    loss_face_weight = 1                     
    loss_eye_weight = 0.2
    bn_mode = "full"
    
# Gaze360(等待处理)
class train_fomulation_GAS5:
    msg = ""
    root = "/mnt/d/Zhiyang Wang/gaze360-gazehub-subject"
    model = model_vit_mamba
    dataset = dataset_Gaze360
    batch_size: int = 16
    lr = 1e-4
    epoch: int = 50
    loss_face_weight = 1                     
    loss_eye_weight = 0.2
    pass
