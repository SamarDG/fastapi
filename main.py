from fastapi import FastAPI

app = FastAPI()

#domain where this api is hosted for example : localhost:5000/docs to see swagger documentation automagically generated.


@app.get("/")
def home():
    return {
  "type": "graph",
  "title": "pro json",
  "gridPos": {
    "x": 0,
    "y": 0,
    "w": 12,
    "h": 8
  },
  "id": 23763571993,
  "fieldConfig": {
    "defaults": {
      "custom": {},
      "thresholds": {
        "mode": "absolute",
        "steps": [
          {
            "value": null,
            "color": "green"
          },
          {
            "value": 80,
            "color": "red"
          }
        ]
      },
      "mappings": []
    },
    "overrides": []
  },
  "pluginVersion": "7.1.4",
  "renderer": "flot",
  "yaxes": [
    {
      "label": null,
      "show": true,
      "logBase": 1,
      "min": null,
      "max": null,
      "format": "short"
    },
    {
      "label": null,
      "show": true,
      "logBase": 1,
      "min": null,
      "max": null,
      "format": "short"
    }
  ],
  "xaxis": {
    "show": true,
    "mode": "time",
    "name": null,
    "values": [],
    "buckets": null
  },
  "yaxis": {
    "align": false,
    "alignLevel": null
  },
  "lines": true,
  "fill": 1,
  "linewidth": 1,
  "dashLength": 10,
  "spaceLength": 10,
  "pointradius": 2,
  "legend": {
    "show": true,
    "values": false,
    "min": false,
    "max": false,
    "current": false,
    "total": false,
    "avg": false
  },
  "nullPointMode": "null",
  "tooltip": {
    "value_type": "individual",
    "shared": true,
    "sort": 0
  },
  "aliasColors": {},
  "seriesOverrides": [],
  "thresholds": [],
  "timeRegions": [],
  "datasource": "Prometheus",
  "targets": [
    {
      "expr": "node_memory_Active_bytes{instance=\"localhost:9100\",job=\"node\"}",
      "legendFormat": "",
      "interval": "",
      "refId": "A"
    }
  ],
  "timeFrom": null,
  "timeShift": null,
  "fillGradient": 0,
  "dashes": false,
  "hiddenSeries": false,
  "points": false,
  "bars": false,
  "stack": false,
  "percentage": false,
  "steppedLine": false
}
