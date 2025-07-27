import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class VideoFrameDecoder(nn.Module):
    def __init__(self, input_dim=768, output_channels=3, image_size=(64, 64)):
        super(VideoFrameDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)  # 从 768 到 1024
        self.fc2 = nn.Linear(4096, 8192)  # 从 1024 到 2048
        self.fc3 = nn.Linear(8192, 16384)  # 从 2048 到 4096

        # 反卷积层，恢复为图像
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # (256, 8, 8) -> (128, 16, 16)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # (128, 16, 16) -> (64, 32, 32)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # (64, 32, 32) -> (32, 64, 64)
        self.deconv4 = nn.ConvTranspose2d(32, output_channels, 4, stride=2, padding=1)  # (32, 64, 64) -> (3, 64, 64)

    def forward(self, x):
        # 将输入通过全连接层处理
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        # 重塑为卷积层所需的形状
        x = x.view(x.size(0), 256, 8, 8)  # 将结果变为 (batch, 128, 8, 8) 适合反卷积层

        # 逐层通过反卷积层生成图像
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))  # 输出 RGB 图像

        return x

class VideoShow(nn.Module):
    def __init__(self, input_dim, output_channels=3, batch_size=16):
        super(VideoShow, self).__init__()
        # input_dim = input_dim
        # output_channels = output_channels
        self.batch_size = batch_size
        self.decoder = VideoFrameDecoder(input_dim=input_dim, output_channels=output_channels)

    def forward(self, frame_features):
        # 创建解码器
        # decoder = VideoFrameDecoder(input_dim=768)

        # 解码器输出每帧的恢复图像
        # 我们首先将每个视频的 50 帧展平为 [batch_size * 50, 768] 然后恢复图像
        reconstructed_frames = self.decoder(frame_features.view(self.batch_size * 50, 768))

        # 将恢复的图像重塑为 [batch_size, 50, 3, 64, 64] 形状
        reconstructed_frames = reconstructed_frames.view(self.batch_size, 200, 3, 64, 64)

        print(f"Reconstructed frame shape: {reconstructed_frames.shape}")

        # 可视化第一个视频的第一帧（你可以根据需要选择其他帧）
        frame_to_show = reconstructed_frames[0, 0].cpu().detach().numpy()  # 选择第一个视频的第一帧
        # 检查输出图像的值范围
        print("Min value:", reconstructed_frames.min().item())
        print("Max value:", reconstructed_frames.max().item())
        frame_to_show = np.clip(frame_to_show, 0, 1)

        # 使用 matplotlib 显示图像
        plt.imshow(frame_to_show.transpose(1, 2, 0))  # 转换为 HWC 格式以适应 imshow
        plt.axis('off')  # 不显示坐标轴
        plt.show()

        # # 可视化第一个视频的前10帧
        # for frame_idx in range(3):  # 只显示前3帧
        #     frame = reconstructed_frames[0, frame_idx]
        #     plt.imshow(frame.permute(1, 2, 0).detach().cpu().numpy())  # permute 调整维度为 [64, 64, 3]
        #     plt.axis('off')  # 关闭坐标轴
        #     plt.show()  # 每次显示一帧图像，直到用户关闭图像窗口后继续
        pass


# # 假设生成器输出的形状是 [batch, 50, 768]
# batch_size = 16
# frame_feature = torch.randn(batch_size, 50, 768)  # 每个视频有 50 帧，每帧 768 维特征
#
# Videoshow = VideoShow(input_dim=768)
# Videoshow(frame_feature)
# # 创建解码器
# decoder = VideoFrameDecoder(input_dim=768)
#
# # 解码器输出每帧的恢复图像
# # 我们首先将每个视频的 50 帧展平为 [batch_size * 50, 768] 然后恢复图像
# reconstructed_frames = decoder(frame_feature.view(batch_size * 50, 768))
#
# # 将恢复的图像重塑为 [batch_size, 50, 3, 64, 64] 形状
# reconstructed_frames = reconstructed_frames.view(batch_size, 200, 3, 64, 64)
#
# print(f"Reconstructed frame shape: {reconstructed_frames.shape}")
#
# # 可视化第一个视频的第一帧（你可以根据需要选择其他帧）
# frame_to_show = reconstructed_frames[0, 0].cpu().detach().numpy()  # 选择第一个视频的第一帧
#
# # 使用 matplotlib 显示图像
# plt.imshow(frame_to_show.transpose(1, 2, 0))  # 转换为 HWC 格式以适应 imshow
# plt.axis('off')  # 不显示坐标轴
# plt.show()
#
# # 可视化第一个视频的前10帧
# for frame_idx in range(10):  # 只显示前10帧
#     frame = reconstructed_frames[0, frame_idx]
#     plt.imshow(frame.permute(1, 2, 0).detach().cpu().numpy())  # permute 调整维度为 [64, 64, 3]
#     plt.axis('off')  # 关闭坐标轴
#     plt.show()  # 每次显示一帧图像，直到用户关闭图像窗口后继续