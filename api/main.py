from fastapi import FastAPI

app = FastAPI(title="Anomaly Detection API")


@app.get("/")
def health_check():
    return {"status": "ok"}
