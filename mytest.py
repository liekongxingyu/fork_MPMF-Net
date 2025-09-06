import torch
import numpy as np

class Tester:
    def positional_encoding(self, input, L):
        shape = input.shape
        # 如果没有GPU，请将下一行中的 .cuda() 删除或改为在CPU上
        # freq = 2 ** torch.arange(L, dtype=torch.float32).cuda() * np.pi
        freq = 2 ** torch.arange(L, dtype=torch.float32) * np.pi
        spectrum = input[..., None] * freq
        sin, cos = spectrum.sin(), spectrum.cos()
        input_enc = torch.stack([sin, cos], dim=-2)
        input_enc = input_enc.view(*shape[:-1], -1)
        return input_enc

def make_2x3_coords(batch=1, num_points=6, device="cpu"):
    # 生成一个 2x3 网格的归一化坐标 [-1,1]
    H, W = 2, 3
    ys = torch.linspace(-1, 1, H, device=device)  # 高度方向
    xs = torch.linspace(-1, 1, W, device=device)  # 宽度方向
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # [H,W]
    grid = torch.stack([xx, yy], dim=-1).view(-1, 2)  # [H*W, 2] -> [6,2]
    # 若 num_points != 6，可从 grid 中选取前 num_points 个点
    grid = grid[:num_points]  # [num_points, 2]
    # 扩展 batch 维度 -> [B, N, 2]
    return grid.unsqueeze(0).expand(batch, -1, -1)  # [B, N, 2]

def test_positional_encoding():
    L = 5  # 频率层数，可改为 1/3/5/10 做对比
    B = 1  # batch
    N = 6  # 2x3 网格 -> 6 个点
    device = "cpu"  # 若使用GPU，改为 "cuda" 并把上面函数中的 .cuda() 打开

    tester = Tester()

    # 1) 构造 2x3 的坐标输入 [B, N, 2]
    coord = make_2x3_coords(batch=B, num_points=N, device=device)  # [-1,1] 归一化坐标
    print("输入坐标 coord 形状:", coord.shape)          # [B, N, 2]
    print("输入坐标 coord 数值:\n", coord)

    # 2) 调用位置编码
    enc = tester.positional_encoding(coord, L=L)
    print("\n位置编码输出 enc 形状:", enc.shape)       # [B, N, 4*L]
    print("理论每点编码维度 = 4*L =", 4*L)

    # 3) 展示其中一个点的编码（例如第一个点）
    idx = 0
    print(f"\n第 {idx} 个点原始坐标:", coord[0, idx])
    print(f"第 {idx} 个点编码向量长度:", enc.shape[-1])
    print(f"第 {idx} 个点编码前 10 项示例:\n", enc[0, idx, :10])

    # 4) 验证与手工计算是否一致（拿一个点做check）
    # 选择第一个点 (x, y)
    x, y = coord[0, 0]  # tensor
    # 构造频率
    freq = (2 ** torch.arange(L, dtype=torch.float32)) * np.pi  # [L]
    # 对 x 和 y 分别计算 sin/cos
    sin_x = torch.sin(x * freq)  # [L]
    cos_x = torch.cos(x * freq)  # [L]
    sin_y = torch.sin(y * freq)  # [L]
    cos_y = torch.cos(y * freq)  # [L]
    manual = torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)  # [4L]
    print("\n手工计算的首点编码向量长度:", manual.numel())
    print("手工计算的首点编码前 10 项示例:\n", manual[:10])

    # 5) 与函数输出对比（数值应一致）
    diff = torch.abs(enc[0, 0] - manual).max().item()
    print("\n自动实现 vs 手工计算 最大绝对误差:", diff)

if __name__ == "__main__":
    test_positional_encoding()
