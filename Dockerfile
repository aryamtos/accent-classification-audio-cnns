
FROM nvcr.io/nvidia/pytorch:22.11-py3

WORKDIR /segmentation/src/


ENV SHELL=/bin/bash
#RUN pip install --upgrade pip
#RUN pip install jupyterlab-horizon-theme
#RUN pip install jupyterlab-git
#RUN pip install jupyterlab-lsp
#RUN pip install 'python-lsp-server[all]'
#RUN pip install jupyterlab-system-monitor


COPY . .

#CMD [ "jupyter","lab","--ip=0.0.0.0","--port=8900","--allow-root","--no-browser"]
