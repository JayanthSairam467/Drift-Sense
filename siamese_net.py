import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class TinyEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(TinyEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # IMPORTANT: adaptive pooling to a SPATIAL grid, not a single value.
        # A global 1x1 pool erases the small defect fingerprint (missing contact,
        # bright particle, scratch) that is the ONLY signal separating periodic
        # duplicates at deployment. Keeping a 4x4 spatial map preserves it.
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.flatten = nn.Flatten()
        self.head = nn.Linear(128 * 4 * 4, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.head(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class SiameseRanker(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseRanker, self).__init__()
        self.encoder = TinyEncoder(embedding_dim=embedding_dim)

    def forward(self, x):
        return self.encoder(x)
        
    @torch.no_grad()
    def rank_candidates(self, reference_np, candidate_patches_np):
        device = next(self.parameters()).device
        self.eval()
        
        # Convert reference to tensor
        ref_h, ref_w = reference_np.shape
        ref_tensor = torch.from_numpy(reference_np).float().unsqueeze(0).unsqueeze(0).to(device)
        ref_emb = self.forward(ref_tensor)
        
        scores_list = []
        for patch in candidate_patches_np:
            # Resize if necessary
            if patch.shape != (ref_h, ref_w):
                patch_resized = cv2.resize(patch, (ref_w, ref_h))
            else:
                patch_resized = patch
                
            patch_tensor = torch.from_numpy(patch_resized).float().unsqueeze(0).unsqueeze(0).to(device)
            patch_emb = self.forward(patch_tensor)
            
            # Compute similarity (e.g. cosine similarity since embeddings are L2 normalized)
            sim = torch.sum(ref_emb * patch_emb, dim=1).item()
            scores_list.append(sim)
            
        ranking = np.argsort(scores_list)[::-1].tolist()
        return scores_list, ranking

def load_model(weights_path, device='cpu'):
    try:
        model = SiameseRanker()
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
        return model
    except Exception:
        return None

def load_siamese_model(weights_path, device='cpu'):
    return load_model(weights_path, device)

def export_to_onnx(model, output_path, input_h=100, input_w=100):
    model.to('cpu')
    model.eval()
    dummy_input = torch.randn(1, 1, input_h, input_w, device='cpu')
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {output_path}")

def get_model_info():
    model = SiameseRanker()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    }

if __name__ == '__main__':
    info = get_model_info()
    print("Model Info:", info)
    
    # Quick inference test
    model = SiameseRanker()
    model.eval()
    x = torch.randn(2, 1, 100, 100)
    out = model(x)
    print("Output shape:", out.shape)
