cd ~/leelenscorer/
python3 multi_client.py \
  --clients-per-gpu=4 \
  --engine-path=/root/binaries/lc0 \
  --weights-path=/root/binaries/ls-n11-1.pb.gz \
  --host=173.67.18.127 \
  --port=19888 \
  --backend=cudnn-fp16 \
  --chunk-size=10