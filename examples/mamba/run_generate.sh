if [ -n "$DEBUG" ]; then
    debugpy-run -m torch.distributed.run -- --nproc_per_node=1 run_generate.py --ckpt-path /fsx/ferdinandmom/ferdinand-hf/nanotron/examples/checkpoints/100
else
    torchrun --nproc_per_node=1 run_generate.py --ckpt-path /fsx/ferdinandmom/ferdinand-hf/nanotron/examples/checkpoints/100
fi