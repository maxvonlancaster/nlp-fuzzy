import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from strawberry.fastapi import GraphQLRouter

from .schema import schema


app = FastAPI(
    title="GraphQL Evolutionary Optimization Server",
    description=(
        "Experimental GraphQL server for "
        "evolutionary query optimization"
    ),
    version="1.0.0"
)


# ============================================================
# GRAPHQL
# ============================================================

graphql_app = GraphQLRouter(schema)

app.include_router(
    graphql_app,
    prefix="/graphql"
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
async def root():

    return {
        "service": "GraphQL Evolutionary Optimization Server",
        "status": "running",
        "graphql_endpoint": "/graphql"
    }


@app.get("/health")
async def health():

    return {
        "status": "healthy"
    }