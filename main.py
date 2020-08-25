from fastapi import FastAPI

app = FastAPI()

#domain where this api is hosted for example : localhost:5000/docs to see swagger documentation automagically generated.


@app.get("/")
def home():
    return { timezone: ‘browser’, panelId: 6, range: { from: ‘2018-02-23T04:14:50.889Z’, to: ‘2018-02-23T10:14:50.889Z’, raw: { from: ‘now-6h’, to: ‘now’ } }, rangeRaw: { from: ‘now-6h’, to: ‘now’ }, interval: ‘30s’, intervalMs: 30000, targets: [ { target: ‘’, refId: ‘A’, type: ‘timeserie’ } ], maxDataPoints: 755, scopedVars: { __interval: { text: ‘30s’, value: ‘30s’ }, __interval_ms: { text: 30000, value: 30000 } } }
