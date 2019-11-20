cd ~/leelenscorer/
python3 multi_client.py \
  --clients-per-gpu=3 \
  --engine-path=/root/binaries/lc0 \
  --weights-path=/root/binaries/ls-n11-1.pb.gz \
  --host=173.67.18.127 \
  --port=9000 \
  --backend=cudnn-fp16 \
  --num-nodes=128 \
  --minibatchsize=16 \
  --client-name=mb16N128 \
  --chunk-size=5