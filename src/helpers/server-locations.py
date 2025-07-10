from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from schema import schema
import uvicorn

app = FastAPI()

graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/")
def read_root():
    return {"message": "GraphQL server is running. Visit /graphql"}

uvicorn.run(app, host="127.0.0.1", port=8000)