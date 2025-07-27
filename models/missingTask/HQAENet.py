# -*- coding: utf-8 -*-
"""
@CreateTime :       2025/06/12 09:21
@File       :       HQAELayer.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2025/07/30 23:35
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.missingTask.AttentionLayer import GQAFusion

class UNet1D(nn.Module):
    """1D UNet for text denoising"""

    def __init__(self, in_channels=768):
        super(UNet1D, self).__init__()
        # Encoder
        self.enc1 = nn.Conv1d(in_channels, 512, kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.enc3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        # Decoder
        self.up1 = nn.ConvTranspose1d(128, 256, kernel_size=2, stride=2)
        self.dec1 = nn.Conv1d(384, 256, kernel_size=3, padding=1)
        self.up2 = nn.ConvTranspose1d(256, 512, kernel_size=2, stride=2)
        self.dec2 = nn.Conv1d(768, 512, kernel_size=3, padding=1)
        self.up3 = nn.ConvTranspose1d(512, 768, kernel_size=2, stride=2)
        self.dec3 = nn.Conv1d(1280, 768, kernel_size=3, padding=1)

        # Output
        self.outc = nn.Conv1d(768, 768, kernel_size=1)

    def forward(self, x):
        # x shape: [batch, channels, seq_len]
        # Encoder
        e1 = F.relu(self.enc1(x))  # [batch, 512, 50]
        p1 = self.pool(e1)  # [batch, 512, 25]

        e2 = F.relu(self.enc2(p1))  # [batch, 256, 25]
        p2 = self.pool(e2)  # [batch, 256, 12]

        e3 = F.relu(self.enc3(p2))  # [batch, 128, 12]
        p3 = self.pool(e3)  # [batch, 128, 6]

        # Decoder
        d1 = self.up1(p3)  # [batch, 256, 12]
        d1 = torch.cat([d1, e3], dim=1)  # [batch, 384, 12]
        d1 = F.relu(self.dec1(d1))  # [batch, 256, 12]

        d2 = self.up2(d1)  # [batch, 512, 24]
        d2 = torch.cat([d2, e2[:, :, :24]], dim=1)  # [batch, 768, 24]
        d2 = F.relu(self.dec2(d2))  # [batch, 512, 24]

        d3 = self.up3(d2)  # [batch, 768, 48]
        d3 = torch.cat([d3, e1[:, :, :48]], dim=1)  # [batch, 1024, 48] -> [batch, 1280, 48]
        d3 = F.relu(self.dec3(d3))  # [batch, 768, 48]

        # Output - use padding to match original length
        output = F.pad(d3, (0, 2))  # [batch, 768, 50]
        return self.outc(output)  # [batch, 768, 50]

class HQAENet(nn.Module):
    def __init__(self, text_dim=768, visual_dim=32, acoustic_dim=16,
                 noise_std=0.1, alpha=0.7, max_retrain=3, gqa_hidden=256, gqa_heads=8, gqa_groups=4):
        super(HQAENet, self).__init__()
        self.noise_std = noise_std
        self.alpha = alpha  # simairity
        self.max_retrain = max_retrain  # max retrain

        # UNet for text denoising
        self.unet = UNet1D()

        # Projection layers for noise adaptation
        self.visual_noise_proj = nn.Linear(text_dim, visual_dim)
        self.acoustic_noise_proj = nn.Linear(text_dim, acoustic_dim)

        # Fusion and FFN
        self.fusion = nn.Linear(text_dim + visual_dim + acoustic_dim, 512)

    def add_noise(self, x):
        """add gaussian noise"""
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x

    def compute_cosine_similarity(self, original, enhanced):
        """
        compute cosine similarity
        Input:
            original: [batch, seq_len, dim]
            enhanced: [batch, seq_len, dim]
        Output:
            Average similarity: [batch]
        """
        batch_size, seq_len, dim = original.shape
        original_flat = original.reshape(-1, dim)
        enhanced_flat = enhanced.reshape(-1, dim)
        cosine_sim = F.cosine_similarity(original_flat, enhanced_flat, dim=-1)
        cosine_sim = cosine_sim.reshape(batch_size, seq_len)
        return torch.mean(cosine_sim, dim=1)  # [batch]

    def compute_pearson_correlation(self, original, enhanced):
        """
        compute pearson similarity (Unfinished)
        Input:
            original: [batch, seq_len, dim]
            enhanced: [batch, seq_len, dim]
        Output:
            Average correlation: [batch]
        """
        batch_size, seq_len, dim = original.shape
        correlations = []
        for i in range(batch_size):
            orig_sample = original[i]  # [seq_len, dim]
            enh_sample = enhanced[i]  # [seq_len, dim]

            seq_corrs = []
            for t in range(seq_len):

                orig_vec = orig_sample[t]  # [dim]
                enh_vec = enh_sample[t]  # [dim]

                orig_mean = torch.mean(orig_vec)
                enh_mean = torch.mean(enh_vec)

                cov = torch.sum((orig_vec - orig_mean) * (enh_vec - enh_mean))

                orig_std = torch.sqrt(torch.sum((orig_vec - orig_mean) ** 2))
                enh_std = torch.sqrt(torch.sum((enh_vec - enh_mean) ** 2))

                eps = 1e-8
                if orig_std < eps or enh_std < eps:
                    corr = 0.0
                else:
                    corr = cov / (orig_std * enh_std)

                seq_corrs.append(corr)

            correlations.append(torch.mean(torch.stack(seq_corrs)))

        return torch.stack(correlations)  # [batch]

    def forward(self, text, visual, acoustic):

        visual_orig = visual.detach().clone()
        acoustic_orig = acoustic.detach().clone()

        # Step : Learning Noise
        noisy_text = self.add_noise(text)
        noisy_text_perm = noisy_text.permute(0, 2, 1)
        learned_noise = self.unet(noisy_text_perm)
        learned_noise = learned_noise.permute(0, 2, 1)
        T = noisy_text - learned_noise

        V_enhanced, A_enhanced = None, None
        retrain_count = 0

        while retrain_count < self.max_retrain:
            # Step 1: Generating Noise
            noisy_visual = self.add_noise(visual)
            noisy_acoustic = self.add_noise(acoustic)

            # Step 2: Learning Noise
            learned_noise = learned_noise

            # Step 3: Enhancing Features
            E_visual = self.visual_noise_proj(learned_noise)
            E_acoustic = self.acoustic_noise_proj(learned_noise)
            V_enhanced = noisy_visual - E_visual
            A_enhanced = noisy_acoustic - E_acoustic

            # Step 4: Comparing Features
            visual_sim = self.compute_cosine_similarity(visual_orig, V_enhanced)
            # visual_sim = self.compute_pearson_correlation(visual_orig, V_enhanced)
            acoustic_sim = self.compute_cosine_similarity(acoustic_orig, A_enhanced)
            # acoustic_sim = self.compute_pearson_correlation(acoustic_orig, A_enhanced)
            if torch.all(visual_sim > self.alpha) and torch.all(acoustic_sim > self.alpha):
                break
            with torch.no_grad():
                # Re-Learning Noise
                new_noisy_text = self.add_noise(text)
                new_noisy_text_perm = new_noisy_text.permute(0, 2, 1)
                learned_noise = self.unet(new_noisy_text_perm)
                learned_noise = learned_noise.permute(0, 2, 1)
            retrain_count += 1

        final_visual_sim = visual_sim.mean()
        final_acoustic_sim = acoustic_sim.mean()

        return T, V_enhanced, A_enhanced, final_visual_sim, final_acoustic_sim
        # return y.squeeze(-1), T, V_enhanced, A_enhanced, final_visual_sim, final_acoustic_sim


class EmotionLoss(nn.Module):
    def __init__(self):
        super(EmotionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        return self.mse_loss(predictions, targets)


# Example
if __name__ == "__main__":
    batch_size = 32
    seq_len = 50

    # Input
    text = torch.randn(batch_size, seq_len, 768)
    visual = torch.randn(batch_size, seq_len, 32)
    acoustic = torch.randn(batch_size, seq_len, 16)
    labels = torch.rand(batch_size) * 4 - 2

    # Model
    model = HQAENet(noise_std=0.1, alpha=0.7, max_retrain=3)
    criterion = EmotionLoss()

    # forward
    output, T, V_enhanced, A_enhanced, visual_sim, acoustic_sim = model(text, visual, acoustic)

    # Loss_HQAENet
    loss = criterion(output, labels)

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        optimizer.zero_grad()

        # forward
        output, T, V_enhanced, A_enhanced, visual_sim, acoustic_sim = model(text, visual, acoustic)

        # compute loss
        loss = criterion(output, labels)

        # backward
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1} | Loss: {loss.item():.4f} | "
              f"Visual Sim: {visual_sim.item():.4f} | "
              f"Acoustic Sim: {acoustic_sim.item():.4f}")