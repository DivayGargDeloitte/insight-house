FROM gcr.io/deeplearning-platform-release/base-cpu
COPY requirements.txt /app/requirements.txt
COPY . /app/.
WORKDIR /app
RUN pip install -r requirements.txt
CMD ['python','preprocessing.py']
