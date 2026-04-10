FROM condaforge/miniforge3:latest

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

COPY scmodal/ ./scmodal/

ENV PATH=/opt/conda/envs/scmodal/bin:$PATH
ENV PYTHONPATH=/app

CMD ["python"]
