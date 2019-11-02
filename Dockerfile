FROM nvidia/cuda:10.0-cudnn7-devel
RUN touch ~/.no_auto_tmux
RUN apt-get update && apt-get dist-upgrade -y && apt clean all
RUN apt-get install -y curl wget supervisor git clang-6.0 ninja-build protobuf-compiler libprotobuf-dev python3-pip
RUN apt-get install -y qt5-default libblas-dev libopenblas-base libopenblas-dev unzip wget htop
RUN pip3 install meson

WORKDIR /root
RUN mkdir binaries

WORKDIR /root
RUN git clone --recurse-submodules https://github.com/LeelaChessZero/lc0.git
WORKDIR /root/lc0
RUN /root/lc0/build.sh
RUN cp /root/lc0/build/release/lc0 /root/binaries

WORKDIR /root
RUN wget https://stockfishchess.org/files/stockfish-10-linux.zip
RUN unzip stockfish-10-linux.zip
RUN chmod 755 /root/stockfish-10-linux/Linux/stockfish_10_x64_modern
RUN cp /root/stockfish-10-linux/Linux/stockfish_10_x64_modern /root/binaries

RUN echo "setw -g mode-keys vi" > /root/.tmux.conf
RUN apt-get install -y python3.7-dev vim
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

ADD . /root/leelenscorer
RUN pip3 install -r /root/leelenscorer/requirements.txt

WORKDIR /root/binaries
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sLFGu0pnw7PHrRmyGbIVFHIjbB4X-qNp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sLFGu0pnw7PHrRmyGbIVFHIjbB4X-qNp" -O ls-n11-1.pb.gz && rm -rf /tmp/cookies.txt
