from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "PickMeUp API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "0.0.0.0", port = 6545)