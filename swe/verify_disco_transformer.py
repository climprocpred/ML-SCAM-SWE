
import torch
from DiscoMlpTransformer import DiscoMlpTransformer, DiscoMlpTransformerConfig

def verify():
    # Use smaller grid for faster verification
    cfg = DiscoMlpTransformerConfig(
        in_channels=2,
        embed_dim=16,
        in_shape=(32, 64),
        out_shape=(32, 64),
        kernel_shape=(3, 3),
        add_lat_sincos=False, 
        pos_embed="spectral"
    )
    
    print("Instantiating model...")
    model = DiscoMlpTransformer(cfg)
    print("Model instantiated successfully!")
    
    print("\nModel Structure:")
    # print(model) # Skip verbose print
    
    print("\nRunning dummy forward pass...")
    x = torch.randn(1, 2, 32, 64)
    out = model(x)
    print(f"Output shape: {out.shape}")
    
    expected_channels = 2 * 16 # 2*embed_dim + 0 (extra_lat)
    assert out.shape == (1, expected_channels, 32, 64)
    print("Verification passed!")

if __name__ == "__main__":
    verify()
