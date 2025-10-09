import torch
from torch import nn
from torch.nn import functional as F
from torch_utils.models.randlanet import RandLANet  # 替换成你的 import 路径
import utils
# -------------------------
# 1. 加载 RandLANet 模型
# -------------------------
cfg = utils.Config.load_from_file("/home/zxhc/Workspace/Open3D-ML/configs/config.yaml")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg.model['seed'] = 0

model = RandLANet(cfg.model)
checkpoint = torch.load('randlanet.pth', map_location='cpu')

# 根据 checkpoint 结构加载
if isinstance(checkpoint, dict):
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
else:
    model = checkpoint

model.to(device)
model.eval()

# -------------------------
# 2. 定义 Wrapper
# -------------------------
class RandLANetWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, coords0, coords1,
                neighbor0, neighbor1,
                sub_idx0, sub_idx1,
                interp_idx0, interp_idx1):
        inputs = {
            'features': features,
            'coords': [coords0, coords1],
            'neighbor_indices': [neighbor0, neighbor1],
            'sub_idx': [sub_idx0, sub_idx1],
            'interp_idx': [interp_idx0, interp_idx1]
        }
        return self.model(inputs)

wrapper = RandLANetWrapper(model)
wrapper.to(device)
wrapper.eval()

# -------------------------
# 3. 构造 dummy 输入
# -------------------------
B, N, C = 1, 8192, cfg['in_channels']
K = 16  # 假设每点邻居数

dummy_features = torch.randn(B, N, C).to(device)
dummy_coords0 = torch.randn(B, N, 3).to(device)
dummy_coords1 = torch.randn(B, N//2, 3).to(device)

dummy_neighbor0 = torch.randint(0, N, (B, N, K)).to(device)
dummy_neighbor1 = torch.randint(0, N//2, (B, N//2, K)).to(device)

dummy_sub_idx0 = torch.randint(0, N, (B, N//2)).to(device)
dummy_sub_idx1 = torch.randint(0, N//2, (B, N//4)).to(device)

dummy_interp_idx0 = torch.randint(0, N, (B, N)).to(device)
dummy_interp_idx1 = torch.randint(0, N//2, (B, N//2)).to(device)

# -------------------------
# 4. 导出 ONNX
# -------------------------
onnx_file = "randlanet.onnx"
torch.onnx.export(
    wrapper,
    (dummy_features, dummy_coords0, dummy_coords1,
     dummy_neighbor0, dummy_neighbor1,
     dummy_sub_idx0, dummy_sub_idx1,
     dummy_interp_idx0, dummy_interp_idx1),
    onnx_file,
    opset_version=17,
    input_names=[
        'features', 'coords0', 'coords1',
        'neighbor0', 'neighbor1',
        'sub_idx0', 'sub_idx1',
        'interp_idx0', 'interp_idx1'
    ],
    output_names=['scores'],
    dynamic_axes={
        'features': {0: 'batch_size', 1: 'num_points'},
        'coords0': {0: 'batch_size', 1: 'num_points'},
        'coords1': {0: 'batch_size', 1: 'num_points'},
        'neighbor0': {0: 'batch_size', 1: 'num_points'},
        'neighbor1': {0: 'batch_size', 1: 'num_points'},
        'sub_idx0': {0: 'batch_size'},
        'sub_idx1': {0: 'batch_size'},
        'interp_idx0': {0: 'batch_size'},
        'interp_idx1': {0: 'batch_size'},
        'scores': {0: 'batch_size', 1: 'num_points'}
    },
    verbose=True
)

print(f"ONNX 导出完成: {onnx_file}")

# -------------------------
# 5. 验证 ONNX
# -------------------------
import onnx
import onnxruntime as ort
onnx_model = onnx.load(onnx_file)
onnx.checker.check_model(onnx_model)
print("ONNX 模型检查通过")

ort_session = ort.InferenceSession(onnx_file)
outputs = ort_session.run(
    None,
    {
        'features': dummy_features.cpu().numpy(),
        'coords0': dummy_coords0.cpu().numpy(),
        'coords1': dummy_coords1.cpu().numpy(),
        'neighbor0': dummy_neighbor0.cpu().numpy(),
        'neighbor1': dummy_neighbor1.cpu().numpy(),
        'sub_idx0': dummy_sub_idx0.cpu().numpy(),
        'sub_idx1': dummy_sub_idx1.cpu().numpy(),
        'interp_idx0': dummy_interp_idx0.cpu().numpy(),
        'interp_idx1': dummy_interp_idx1.cpu().numpy()
    }
)
print("ONNX 推理输出 shape:", outputs[0].shape)
