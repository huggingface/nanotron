#!/usr/bin/env python3
"""
Simple test script for FlashAttention
"""

import torch

def test_flash_attention():
    """Test if FlashAttention is working correctly"""
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. FlashAttention requires CUDA.")
        return False
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    
    try:
        # Try to import FlashAttention
        from flash_attn import flash_attn_func
        print("✓ FlashAttention imported successfully!")
    except ImportError as e:
        print(f"✗ Failed to import FlashAttention: {e}")
        return False
    
    # Create dummy tensors
    batch_size = 1
    seq_len = 128
    num_heads = 8
    head_dim = 64
    
    device = torch.device('cuda')
    dtype = torch.float16
    
    # Create Q, K, V tensors
    Q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    print(f"Input shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")
    
    try:
        # Run FlashAttention
        output = flash_attn_func(Q, K, V, dropout_p=0.0, softmax_scale=None, causal=False)
        print("✓ FlashAttention ran successfully!")
        print(f"Output shape: {output.shape}")
        
        # Test with causal attention
        output_causal = flash_attn_func(Q, K, V, dropout_p=0.0, softmax_scale=None, causal=True)
        print("✓ Causal FlashAttention ran successfully!")
        print(f"Causal output shape: {output_causal.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ FlashAttention failed to run: {e}")
        return False

def test_memory_usage():
    """Test memory usage with different sequence lengths"""
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        print("FlashAttention not available, skipping memory test")
        return
    
    device = torch.device('cuda')
    dtype = torch.float16
    
    # Test different sequence lengths
    seq_lengths = [512, 1024, 2048]
    batch_size = 1
    num_heads = 8
    head_dim = 64
    
    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        
        # Clear cache before test
        torch.cuda.empty_cache()
        
        # Get initial memory
        initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        
        try:
            # Create tensors
            Q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
            K = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
            V = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
            
            # Run attention
            output = flash_attn_func(Q, K, V, dropout_p=0.0, softmax_scale=None, causal=False)
            
            # Get peak memory
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            print(f"  Initial memory: {initial_memory:.2f} GB")
            print(f"  Peak memory: {peak_memory:.2f} GB")
            print(f"  Memory used: {peak_memory - initial_memory:.2f} GB")
            
        except Exception as e:
            print(f"  Failed: {e}")
        
        # Clean up
        del Q, K, V, output
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("=== FlashAttention Test ===\n")
    
    # Basic functionality test
    success = test_flash_attention()
    
    if success:
        print("\n=== Memory Usage Test ===")
        test_memory_usage()
    
    print("\n=== Test Complete ===") 