from fastapi import FastAPI

app = FastAPI()

#domain where this api is hosted for example : localhost:5000/docs to see swagger documentation automagically generated.


@app.get("/")
def home():
    return [{"target": "my_metric", "datapoints": [[2, 1525320000000], [1, 1525320000000], [19, 1525320000000], [0, 1525320000000], [2, 1525665600000], [1, 1525665600000], [17, 1525665600000], [0, 1525665600000], [3, 1525838400000], [1, 1525838400000], [17, 1525838400000], [0, 1525838400000], [0, 1525838400000]]}]
