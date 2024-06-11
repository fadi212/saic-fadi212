FROM python:3.10

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY \

PYTHONUNBUFFERED=1

WORKDIR /app

COPY src /app/src
COPY main.py Car-models-overview.md requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt
# Download and set up the frpc binary needed by Gradio for public links
ADD https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_amd64 /usr/local/lib/python3.10/site-packages/gradio/frpc_linux_amd64_v0.2
RUN chmod +x /usr/local/lib/python3.10/site-packages/gradio/frpc_linux_amd64_v0.2

EXPOSE 7860
CMD ["python", "main.py"]