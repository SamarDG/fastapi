from fastapi import FastAPI

app = FastAPI()

#domain where this api is hosted for example : localhost:5000/docs to see swagger documentation automagically generated.


@app.get("/")
def home():
    return {"message":"Hello TutLinks.com"}

@app.get("/query")
def func1():
    return {"status": "success",
    "data": {
        "resultType": "matrix",
        "result": [
            {
                "metric": {
                    "__name__": "node_memory_Active_bytes",
                    "instance": "localhost:9100",
                    "job": "node"
                },
                "values": [
                    [
                        1598341663,
                        "522907648mhm"
                    ],
                    [
                        1598341673,
                        "522907648"
                    ],
                    [
                        1598341683,
                        "523993088"
                    ],
                    [
                        1598341693,
                        "524374016"
                    ],
                    [
                        1598341703,
                        "524374016"
                    ],
                    [
                        1598341713,
                        "523743232"
                    ],
                    [
                        1598341723,
                        "523784192"
                    ],
                    [
                        1598341733,
                        "523784192"
                    ],
                    [
                        1598341743,
                        "523472896"
                    ],
                    [
                        1598341753,
                        "523481088"
                    ],
                    [
                        1598341763,
                        "523481088"
                    ],
                    [
                        1598341773,
                        "523526144"
                    ],
                    [
                        1598341783,
                        "523345920"
                    ],
                    [
                        1598341793,
                        "523345920"
                    ]
                ]
            }
        ]
    }
}
